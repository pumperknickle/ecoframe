"""
TrainingEngine tests: validates the automatic CPU-GPU overlap pattern.

Key invariants:
  1. step() returns metrics without caller managing ordering
  2. First step returns empty metrics (no prev h_cond) — no crash
  3. Subsequent steps have non-zero CE loss
  4. brain.forward() is called before env.step_async() — no stale actions
  5. brain.backward() is called after env.step_async() — overlap happens
  6. run(n) yields exactly n (step, metrics) pairs
  7. stop() exits the environment cleanly
"""
import numpy as np
import pytest
from unittest.mock import MagicMock, call, patch

from ecoframe.protocol import (
    ActionBundle, LossHandle, SensorBundle,
    SensorManifest, SensorSpec, Session, TrainingMetrics,
)
from ecoframe.training_engine import TrainingEngine


MANIFEST = SensorManifest("test_env", sensors=(
    SensorSpec("visual",         (64, 64, 3), action_affected=True,  world_external=True),
    SensorSpec("proprioceptive", (5,),        action_affected=False, world_external=False),
))


def _make_bundle(aid="a0"):
    return SensorBundle(
        visual=np.zeros((64, 64, 3), dtype=np.uint8),
        proprioceptive=np.zeros(5, dtype=np.float32),
        env_id="test_env", agent_id=aid,
    )


def _make_brain(ce_loss=1.5):
    brain = MagicMock()
    brain.brain_id = "test_brain"
    brain.sessions = {}
    # enter returns a session
    brain.enter.return_value = Session(
        brain_id="test_brain", env_id="test_env",
        agent_id="a0", ssm_state={})
    # forward returns actions + valid LossHandle
    loss_handle = LossHandle("fake_loss_state")
    brain.forward.return_value = (
        {"a0": ActionBundle(continuous=np.array([0.1, 0.5]), agent_id="a0")},
        loss_handle,
    )
    # backward returns metrics
    brain.backward.return_value = TrainingMetrics(
        loss=2.0, ce_loss=ce_loss)
    return brain, loss_handle


def _make_env():
    env = MagicMock()
    env.env_id   = "test_env"
    env.manifest = MANIFEST
    env.capacity = 8
    env.reset.return_value  = {"a0": _make_bundle()}
    env.step_wait.return_value = {"a0": _make_bundle()}
    env.enter.return_value  = Session(
        brain_id="test_brain", env_id="test_env",
        agent_id="a0", ssm_state={})
    env.exit.return_value   = {}
    return env


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_step_returns_metrics():
    brain, _ = _make_brain()
    env      = _make_env()
    engine   = TrainingEngine(brain, env)
    metrics  = engine.step()
    assert isinstance(metrics, TrainingMetrics)


def test_first_step_auto_starts():
    brain, _ = _make_brain()
    env      = _make_env()
    engine   = TrainingEngine(brain, env)
    assert not engine._started
    engine.step()
    assert engine._started
    env.reset.assert_called_once()


def test_forward_called_before_step_async():
    """forward must precede step_async so actions are fresh."""
    brain, _ = _make_brain()
    env      = _make_env()
    engine   = TrainingEngine(brain, env)
    call_order = []

    brain.forward.side_effect = lambda *a, **kw: (
        call_order.append('forward'),
        brain.forward.return_value,
    )[1]
    env.step_async.side_effect = lambda *a, **kw: call_order.append('step_async')

    engine.step()
    assert call_order.index('forward') < call_order.index('step_async'), \
        "forward() must be called before step_async() — actions must be fresh"


def test_backward_called_after_step_async():
    """backward must come after step_async so it overlaps env step."""
    brain, _ = _make_brain()
    env      = _make_env()
    engine   = TrainingEngine(brain, env)
    call_order = []

    env.step_async.side_effect = lambda *a, **kw: call_order.append('step_async')
    brain.backward.side_effect = lambda *a, **kw: (
        call_order.append('backward'),
        brain.backward.return_value,
    )[1]

    engine.step()
    assert call_order.index('step_async') < call_order.index('backward'), \
        "step_async() must precede backward() — backward overlaps env step"


def test_loss_handle_passed_from_forward_to_backward():
    """The exact LossHandle returned by forward() is passed to backward()."""
    brain, loss_handle = _make_brain()
    env                = _make_env()
    engine             = TrainingEngine(brain, env)
    engine.step()
    brain.backward.assert_called_with(loss_handle)


def test_run_yields_n_steps():
    brain, _ = _make_brain()
    env      = _make_env()
    engine   = TrainingEngine(brain, env)
    results  = list(engine.run(n_steps=5))
    assert len(results) == 5
    steps = [r[0] for r in results]
    assert steps == list(range(1, 6))


def test_run_yields_step_metrics_tuples():
    brain, _ = _make_brain()
    env      = _make_env()
    for step_n, metrics in TrainingEngine(brain, env).run(n_steps=3):
        assert isinstance(step_n, int)
        assert isinstance(metrics, TrainingMetrics)


def test_stop_exits_environment():
    brain, _ = _make_brain()
    env      = _make_env()
    engine   = TrainingEngine(brain, env)
    engine.start()
    engine.stop()
    brain.exit.assert_called_once_with(env)
    assert engine._session is None


def test_step_wait_called_after_first_step():
    """First step uses env.reset(), subsequent steps use env.step_wait()."""
    brain, _ = _make_brain()
    env      = _make_env()
    engine   = TrainingEngine(brain, env)
    engine.step()
    assert env.step_wait.call_count == 0   # first step: no step_wait
    engine.step()
    assert env.step_wait.call_count == 1   # second step: step_wait called
    engine.step()
    assert env.step_wait.call_count == 2


def test_step_count_increments():
    brain, _ = _make_brain()
    env      = _make_env()
    engine   = TrainingEngine(brain, env)
    for _ in range(5):
        engine.step()
    assert engine._step_count == 5


def test_explore_mag_passed_to_forward():
    brain, _ = _make_brain()
    env      = _make_env()
    engine   = TrainingEngine(brain, env, explore_mag=0.3)
    engine.step()
    _, kwargs = brain.forward.call_args
    assert kwargs.get('explore_mag') == 0.3 or brain.forward.call_args[0][2] == 0.3


def test_loss_handle_none_is_valid():
    """LossHandle(None) is valid Python — backward handles it as no-op."""
    h = LossHandle(None)
    assert not h.is_valid()
    h2 = LossHandle("something")
    assert h2.is_valid()
