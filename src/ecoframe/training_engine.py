"""
TrainingEngine — wires BrainProtocol + EnvironmentProtocol with automatic
CPU-GPU overlap. Callers just call step() in a loop.

Overlap mechanism (backward overlaps env step):
  step T:
    obs[T]   = env.step_wait()                   # collect from workers
    actions  = brain.forward(obs[T]) → actions[T], loss[T]
    env.step_async(actions[T])                   # workers start stepping (non-blocking)
    metrics  = brain.backward(loss[T])           # GPU backward overlaps CPU env step
    ← overlap here ─────────────────────────────────────────────────────

Why no stale actions:
  actions[T] are computed from obs[T] and sent immediately.
  Workers execute actions[T] to produce obs[T+1].
  No buffering of old actions.

Why no prime() needed:
  First call: no env.step_wait() yet — calls env.reset() instead.
  loss[0] = LossHandle(None) → backward() is no-op.
  Pipeline starts without any caller setup.

Why profiling stays transparent:
  step() is one unit of work: wait → forward → send → backward → done.
  Timings still meaningful: step_wait catches up CPU, backward runs GPU.
  No hidden state machines.
"""
from __future__ import annotations

from typing import Iterator

from ecoframe.protocol import (
    BrainProtocol, EnvironmentProtocol, LossHandle,
    Session, TrainingMetrics,
)


class TrainingEngine:
    """
    Automatic CPU-GPU overlap. Call step() in a loop.

    Usage:
        engine = TrainingEngine(brain, env)
        for step, metrics in engine.run(n_steps=2_000_000):
            log(step, metrics)
    """

    def __init__(
        self,
        brain:       BrainProtocol,
        env:         EnvironmentProtocol,
        explore_mag: float = 0.0,
    ):
        self.brain       = brain
        self.env         = env
        self.explore_mag = explore_mag
        self._session:   Session | None = None
        self._obs        = None
        self._started    = False
        self._step_count = 0

    def start(self) -> None:
        """Enter the environment and collect initial observations."""
        self._session = self.brain.enter(self.env)
        self._obs     = self.env.reset(self._session)
        self._started = True

    def step(self) -> TrainingMetrics:
        """
        One training step with automatic backward/env overlap.

        On first call: auto-starts if start() wasn't called.
        Returns empty TrainingMetrics on first step (no prev h_cond yet).
        """
        if not self._started:
            self.start()

        self._step_count += 1

        # ── collect obs (blocks until env workers done) ────────────────────────
        if self._step_count > 1:
            self._obs = self.env.step_wait()

        # ── forward: obs → actions + live loss tensor (~2ms) ──────────────────
        actions, loss = self.brain.forward(
            self._obs, self.env.manifest, explore_mag=self.explore_mag)

        # ── step_async: send fresh actions to workers (non-blocking) ──────────
        self.env.step_async(actions)

        # ── backward: overlaps with env step on CPU (~3ms) ────────────────────
        metrics = self.brain.backward(loss)

        return metrics

    def run(self, n_steps: int) -> Iterator[tuple[int, TrainingMetrics]]:
        """Yield (step_number, metrics) for n_steps steps. step_number is 1-indexed."""
        if not self._started:
            self.start()
        for _ in range(n_steps):
            metrics = self.step()
            yield self._step_count, metrics

    def stop(self) -> None:
        """Exit the environment cleanly."""
        if self._session is not None:
            self.brain.exit(self.env)
            self._session = None
        self._started = False
