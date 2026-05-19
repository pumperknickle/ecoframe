"""
Phase 3: MetaEnvironment — the always-on environment every brain lives in.

The brain never exists "outside" an environment. Between task environments,
it lives in the MetaEnvironment. Entering/exiting task environments happens
from within the MetaEnvironment.

Observations: the curiosity field gradient — what environments are nearby,
how much each one is teaching currently, how full each one is.

Actions: navigation decisions — which environment to move toward.
When the brain's action selects a known environment and its curiosity is
above threshold, the brain automatically enters that environment.

Curriculum is navigation: the brain learns which environments reduce its
surprise most efficiently. No designer schedule — pure prediction error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Callable

import numpy as np

from ecoframe.protocol import (
    ActionBundle, CapacityError, EnvironmentProtocol,
    SensorBundle, SensorManifest, SensorSpec, Session, TrainingMetrics,
)
from ecoframe.signal import EnvironmentSignal
from ecoframe.field import Field


# MetaEnvironment sensor manifest
# The brain observes: what environments are available, their learning signal
META_MANIFEST = SensorManifest(
    env_id="meta",
    sensors=(
        SensorSpec(
            name            = "env_field",
            shape           = (64,),      # flattened field observations
            dtype           = "float32",
            action_affected = True,       # brain's navigation changes what it sees
            world_external  = True,       # the ecology is the world at meta level
            temporal_res    = 1.0,
        ),
        SensorSpec(
            name            = "self_state",
            shape           = (4,),       # [ce_ema, session_count, steps_in_meta, energy]
            dtype           = "float32",
            action_affected = False,
            world_external  = False,
        ),
    ),
)


class MetaEnvironment:
    """
    Always-on environment. Every brain lives here perpetually.

    capacity = unlimited (no spawn point constraint — meta is everywhere)

    The MetaEnvironment:
    - observes the Field to see available task environments
    - returns SensorBundles describing the ecology state
    - receives ActionBundles describing navigation intent
    - when brain navigates toward a known env: auto-calls brain.enter(env)
    - when brain navigates away or env is mastered: auto-calls brain.exit(env)
    """

    env_id   = "meta"
    capacity = 2**31    # effectively unlimited
    manifest = META_MANIFEST

    def __init__(
        self,
        field:            Field | None  = None,
        enter_threshold:  float         = 2.0,   # min curiosity to auto-enter
        exit_threshold:   float         = 1.0,   # max curiosity to auto-exit
        verbose:          bool          = False,
    ):
        self._field            = field or Field(backend='local')
        self._enter_threshold  = enter_threshold
        self._exit_threshold   = exit_threshold
        self._verbose          = verbose

        # Active sessions: brain_id → Session
        self._sessions: dict[str, Session] = {}

        # Known task environments (registered via register_env)
        self._known_envs: dict[str, EnvironmentProtocol] = {}

        # Active task environment — when set, step_async/step_wait proxy here.
        # None means brain is between task envs, navigating via field gradient.
        self._active_env: EnvironmentProtocol | None = None

        # Per-brain state
        self._brain_pos:    dict[str, tuple[float, float]] = {}
        self._brain_ce_ema: dict[str, float]               = {}
        self._step_count = 0

    # ── Manifest: switches when proxying a task env ───────────────────────────

    @property
    def manifest(self) -> SensorManifest:
        """When proxying a task env, expose its manifest so brain encodes correctly."""
        if self._active_env is not None:
            return self._active_env.manifest
        return META_MANIFEST

    # ── Environment registration ───────────────────────────────────────────────

    def register_env(self, env: EnvironmentProtocol) -> None:
        """
        Register a task environment as navigable from meta.
        The brain can auto-enter this env when it navigates toward it.
        """
        self._known_envs[env.env_id] = env
        self._field.register_agent(env.env_id, pos=(
            len(self._known_envs) * 2.0, 0.0))

    # ── EnvironmentProtocol ────────────────────────────────────────────────────

    def start(self) -> None:
        pass   # MetaEnvironment is always ready

    def close(self) -> None:
        pass   # MetaEnvironment doesn't own resources

    def enter(self, brain_id: str,
              ssm_state: dict | None = None) -> Session:
        """
        Brain enters the meta environment.
        Always succeeds — meta has unlimited capacity.
        """
        session = Session(
            brain_id   = brain_id,
            env_id     = self.env_id,
            agent_id   = brain_id,     # 1:1 mapping in meta
            ssm_state  = ssm_state or {},
            entered_at = self._step_count,
        )
        self._sessions[brain_id] = session
        self._brain_pos[brain_id]    = (0.0, 0.0)
        self._brain_ce_ema[brain_id] = 5.5

        self._field.register_agent(brain_id, pos=(0.0, 0.0))
        if self._verbose:
            print(f"MetaEnv.enter: brain={brain_id}", flush=True)
        return session

    def exit(self, session: Session) -> dict:
        brain_id = session.brain_id
        self._sessions.pop(brain_id, None)
        self._brain_pos.pop(brain_id, None)
        self._brain_ce_ema.pop(brain_id, None)
        return session.ssm_state

    def reset(self, session: Session) -> dict[str, SensorBundle]:
        """
        Return initial observations.
        Auto-enters the first registered task env on first call — no designer
        schedule, just 'start learning immediately if a task env is available.'
        In the future, field gradient determines which env to enter.
        """
        if self._active_env is None and self._known_envs:
            self._enter_task_env(next(iter(self._known_envs.values())), session)
        if self._active_env is not None:
            return self._active_env.reset(session)
        return self.step_wait()

    def _enter_task_env(self, env: EnvironmentProtocol, session: Session) -> None:
        """Enter a task env on the brain's behalf. Start it if needed."""
        if not hasattr(env, '_started') or not env._started:
            env.start()
            if hasattr(env, '_started'):
                env._started = True
        env.enter(session.brain_id, ssm_state=session.ssm_state)
        self._active_env = env
        if self._verbose:
            print(f"MetaEnv: brain entered task env '{env.env_id}'", flush=True)

    def _exit_task_env(self, session: Session) -> None:
        """Exit the current task env, carry SSM state back to meta."""
        if self._active_env is not None:
            from ecoframe.protocol import Session as _Session
            task_session = _Session(
                brain_id  = session.brain_id,
                env_id    = self._active_env.env_id,
                agent_id  = session.agent_id,
                ssm_state = session.ssm_state,
            )
            self._active_env.exit(task_session)
            if self._verbose:
                print(f"MetaEnv: brain exited '{self._active_env.env_id}'", flush=True)
            self._active_env = None

    def step_async(self, actions: dict[str, ActionBundle]) -> None:
        """
        Proxy to active task env if one is running.
        Otherwise process navigation actions in meta.
        """
        if self._active_env is not None:
            self._active_env.step_async(actions)
        else:
            self._pending_actions = actions

    def step_wait(self) -> dict[str, SensorBundle]:
        """
        Proxy to active task env when one is running.
        Otherwise return field gradient observations (brain is navigating meta).
        """
        self._step_count += 1

        if self._active_env is not None:
            bundles = self._active_env.step_wait()
            if self._step_count % 100 == 0:
                self._publish_env_signal(self._active_env)
            return bundles

        self._field.step()

        actions  = getattr(self, '_pending_actions', {}) or {}
        bundles: dict[str, SensorBundle] = {}

        for brain_id, session in self._sessions.items():
            pos      = self._brain_pos.get(brain_id, (0.0, 0.0))
            ce_ema   = self._brain_ce_ema.get(brain_id, 5.5)

            # Query field for nearby environments
            signals  = self._field.query(pos=pos, radius=5.0)
            env_sigs = [s for s in signals if isinstance(s, EnvironmentSignal)]

            # Build field observation: top-4 envs by curiosity
            env_sigs.sort(key=lambda s: -s.curiosity)
            field_obs = np.zeros(64, dtype=np.float32)
            for i, sig in enumerate(env_sigs[:8]):
                base = i * 8
                field_obs[base + 0] = sig.curiosity / 6.0
                field_obs[base + 1] = sig.load_fraction
                field_obs[base + 2] = sig.difficulty
                dx = sig.position[0] - pos[0]
                dz = sig.position[1] - pos[1]
                dist = max(1e-6, math.sqrt(dx**2 + dz**2))
                field_obs[base + 3] = min(1.0, 1.0 / dist)
                field_obs[base + 4] = dx / (dist + 1e-6)
                field_obs[base + 5] = dz / (dist + 1e-6)

            self_state = np.array([
                ce_ema / 6.0,
                len(self._sessions) / 100.0,
                (self._step_count - session.entered_at) / 1000.0,
                0.0,   # energy placeholder
            ], dtype=np.float32)

            bundles[brain_id] = SensorBundle(
                proprioceptive = self_state,
                extra          = {'env_field': field_obs},   # manifest-declared sensor
                reward         = 0.0,
                done           = False,
                env_id         = self.env_id,
                agent_id       = brain_id,
                step           = self._step_count,
                info           = {'env_signals': env_sigs},  # non-sensor metadata
            )

        self._pending_actions = {}
        return bundles

    # ── Signal publishing ─────────────────────────────────────────────────────

    def _publish_env_signal(self, env: EnvironmentProtocol) -> None:
        """Publish the active task env's state into the Field."""
        curiosity = getattr(env, '_ce_ema', 5.5)
        load      = getattr(env, '_n_active', 0) / max(1, getattr(env, 'capacity', 1))
        hw        = getattr(env, 'hardware_spec', None)
        sig = EnvironmentSignal(
            position      = (2.0, 0.0),
            timestamp     = self._step_count,
            publisher     = env.env_id,
            curiosity     = float(curiosity),
            load_fraction = float(load),
            env_type      = env.env_id,
            manifest_hash = env.manifest.hash,
            device_type   = hw.device_type if hw else "cpu",
            memory_gb     = hw.memory_gb   if hw else 0.0,
        )
        self._field.register_agent(env.env_id, pos=(2.0, 0.0))
        self._field.publish(env.env_id, sig)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def hardware_spec(self):
        from ecoframe.protocol import HardwareSpec
        return HardwareSpec.cpu()  # MetaEnvironment itself is CPU-only

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._active_env is not None:
            try:
                self._active_env.close()
            except Exception:
                pass
            self._active_env = None

    # ── Convenience: auto-navigate brains ─────────────────────────────────────

    def navigate(
        self,
        brain_id:  str,
        target_env_id: str | None,
        brain_obj = None,    # optional BrainProtocol — for auto enter/exit
    ) -> str | None:
        """
        Move brain toward target_env_id.
        If brain has a BrainProtocol reference, auto-calls enter/exit.
        Returns the env_id the brain entered (or None).
        """
        if target_env_id is None or target_env_id not in self._known_envs:
            return None

        env = self._known_envs[target_env_id]

        # Check field signal for this env
        pos      = self._brain_pos.get(brain_id, (0.0, 0.0))
        signals  = self._field.query(pos=(0.0, 0.0), radius=100.0)
        env_sigs = [s for s in signals
                    if isinstance(s, EnvironmentSignal)
                    and s.publisher == target_env_id]
        curiosity = env_sigs[0].curiosity if env_sigs else 0.0

        if curiosity >= self._enter_threshold and brain_obj is not None:
            try:
                brain_obj.enter(env)
                if self._verbose:
                    print(f"MetaEnv: brain={brain_id} auto-entered {target_env_id} "
                          f"(curiosity={curiosity:.2f})", flush=True)
                return target_env_id
            except CapacityError:
                if self._verbose:
                    print(f"MetaEnv: {target_env_id} at capacity", flush=True)
        return None
