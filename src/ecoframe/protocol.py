"""
EcoFrame Protocol — Phase 0: types and contracts only.

No torch. No MetaDrive. No brain model code.
Only numpy + stdlib. Both brain and environment processes import this.

The three invariants this file enforces structurally:
  1. SensorSpec.action_affected tags what to predict (no hardcoded logic)
  2. proprioceptive is never a prediction target (world_external=False)
  3. Session carries SSM state — neither brain nor env resets it unilaterally
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


# ── Errors ─────────────────────────────────────────────────────────────────────

class CapacityError(Exception):
    """Raised when a brain tries to enter a full environment."""


class ManifestMismatchError(Exception):
    """Raised when brain's expected manifest differs from env's manifest."""


# ── Sensor schema ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SensorSpec:
    """
    Description of one sensor channel.

    action_affected + world_external together determine whether this sensor
    is a valid prediction target. The brain predicts only sensors where
    both are True — everything else is SSM input only.

    Why this matters: the static attractor problem. If the brain predicts
    proprioception (action_affected=False), it learns "stay still = low loss"
    without needing to model the world. Tagging prevents this structurally.
    """
    name:            str
    shape:           tuple[int, ...]
    dtype:           str              = "float32"
    action_affected: bool             = True   # different action → different obs
    world_external:  bool             = True   # describes world, not self
    temporal_res:    float            = 1.0    # seconds per observation

    @property
    def is_prediction_target(self) -> bool:
        return self.action_affected and self.world_external


@dataclass(frozen=True)
class SensorManifest:
    """
    Schema for all sensors in one environment.

    The manifest is the authoritative source of what to predict.
    Brain adapts to manifest — not the other way around.
    """
    env_id:  str
    sensors: tuple[SensorSpec, ...]

    @property
    def prediction_targets(self) -> list[SensorSpec]:
        return [s for s in self.sensors if s.is_prediction_target]

    @property
    def ssm_inputs(self) -> list[SensorSpec]:
        return list(self.sensors)   # all sensors feed the SSM

    def get(self, name: str) -> SensorSpec | None:
        for s in self.sensors:
            if s.name == name:
                return s
        return None

    @property
    def hash(self) -> str:
        import hashlib, json
        spec = [(s.name, s.shape, s.action_affected, s.world_external)
                for s in self.sensors]
        return hashlib.sha1(json.dumps(spec).encode()).hexdigest()[:12]


# ── Hardware declaration ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class HardwareSpec:
    """
    Hardware requirements for one environment instance.

    Published in EnvironmentSignal so brains navigate only to reachable envs.
    Used by ecoframe-runtime's DeviceContext to enforce CUDA isolation.

    This is data only — no OS calls, no CUDA imports. The runtime library
    reads HardwareSpec and performs the actual device management.
    """
    device_type:   str   = "cpu"   # "cpu" | "cuda" | "mps"
    device_id:     int   = 0       # GPU device index (ignored for cpu)
    memory_gb:     float = 0.0     # GPU memory required
    n_cpu_workers: int   = 1       # CPU cores for subprocess workers
    accelerator:   str   = ""      # "panda3d" | "isaac_gym" | "brax_jax" | ""

    def is_compatible_with(self, other: 'HardwareSpec') -> bool:
        """
        True if these two specs can run simultaneously without CUDA conflict.

        CPU envs never conflict with anything.
        CUDA envs conflict only if they share a device_id.
        """
        if self.device_type == "cpu" or other.device_type == "cpu":
            return True
        return self.device_id != other.device_id

    def cuda_visible_devices(self) -> str:
        """Value to set CUDA_VISIBLE_DEVICES to for this spec."""
        if self.device_type == "cpu":
            return ""
        return str(self.device_id)

    @classmethod
    def cpu(cls, n_workers: int = 1, accelerator: str = "") -> 'HardwareSpec':
        return cls(device_type="cpu", n_cpu_workers=n_workers,
                   accelerator=accelerator)

    @classmethod
    def cuda(cls, device_id: int = 0, memory_gb: float = 4.0,
             accelerator: str = "") -> 'HardwareSpec':
        return cls(device_type="cuda", device_id=device_id,
                   memory_gb=memory_gb, accelerator=accelerator)


# ── Universal data types ───────────────────────────────────────────────────────

@dataclass
class SensorBundle:
    """
    Universal observation format. All sensors from one agent at one timestep.

    World sensors (action_affected=True, world_external=True):
      → SSM inputs AND prediction targets
      → different action leads to different observation
      → examples: camera frame, audio, LIDAR

    Self sensors (world_external=False):
      → SSM inputs ONLY — never prediction targets
      → trivially predictable from momentum / not world-external
      → examples: ego velocity, joint angles

    reward is one named sensor channel — not a privileged return value.
    done signals episode boundary — does NOT reset SSM.
    """
    # World sensors
    visual:      np.ndarray | None = None   # (H, W, 3) uint8
    audio:       np.ndarray | None = None   # (T,) float32
    text_tokens: np.ndarray | None = None   # (L,) int32

    # Self sensors (SSM input only)
    proprioceptive: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32))

    # Additional manifest-defined sensors (keyed by SensorSpec.name)
    # Use this for sensors beyond visual/audio/text/proprioceptive.
    # Serializable through EnvironmentProxy (numpy arrays only).
    extra:   dict  = field(default_factory=dict)

    # Episode metadata
    reward:  float = 0.0   # one channel, not privileged
    done:    bool  = False  # episode boundary — SSM persists across this
    info:    dict  = field(default_factory=dict)

    # Routing
    env_id:   str = ""
    agent_id: str = ""
    step:     int = 0

    def has(self, sensor_name: str) -> bool:
        return getattr(self, sensor_name, None) is not None


@dataclass
class ActionBundle:
    """Universal action format."""
    continuous:  np.ndarray | None = None   # steering, speed, joint angles
    discrete:    int | None        = None   # categorical action
    text_tokens: np.ndarray | None = None   # language output

    env_id:   str = ""
    agent_id: str = ""


@dataclass
class TrainingMetrics:
    """What the brain reports back after each process() call."""
    loss:            float = 0.0
    ce_loss:         float = 0.0
    ce_ema:          float = 0.0
    avg_reward:      float = 0.0
    crash_rate:      float = 0.0
    completion_rate: float = 0.0
    sensor_freq:     float = 1.0
    explore_mag:     float = 0.0
    extra:           dict  = field(default_factory=dict)


# ── Session — brain's tenancy in one environment ───────────────────────────────

@dataclass
class Session:
    """
    Records a brain's active tenancy in one environment.

    ssm_state is owned by the brain — the environment stores it during the
    session but never modifies it. On exit(), the environment returns it
    unchanged (plus any updates the brain wrote during the session).

    Neither brain nor environment resets ssm_state. It accumulates since
    training began and persists across episode boundaries and env switches.
    """
    brain_id:   str
    env_id:     str
    agent_id:   str        # assigned by env on enter
    ssm_state:  dict       # owned by brain, survives resets
    entered_at: int = 0    # step number when brain joined
    metadata:   dict = field(default_factory=dict)


# ── Protocols ──────────────────────────────────────────────────────────────────

@runtime_checkable
class EnvironmentProtocol(Protocol):
    """
    Long-running process with its own GPU scope.

    Produces SensorBundles, consumes ActionBundles.
    Owns: physics simulation, rendering, NPC models, its own CUDA context.

    Async step interface (step_async / step_wait) enables GPU/env overlap:
      actions, loss = brain.forward(obs, manifest)  # forward + build loss
      env.step_async(actions)                        # send fresh actions, non-blocking
      metrics = brain.backward(loss)                 # backward overlaps env step
      new_obs = env.step_wait()                      # collect when GPU done
    """
    env_id:        str
    capacity:      int
    manifest:      SensorManifest
    hardware_spec: HardwareSpec     # what hardware this env occupies

    def start(self) -> None: ...
    def close(self) -> None: ...

    def enter(self, brain_id: str,
              ssm_state: dict | None = None) -> Session:
        """
        Allocate an agent slot for this brain.
        Brain brings its SSM state — env never resets it.
        Raises CapacityError if no slots available.
        """
        ...

    def exit(self, session: Session) -> dict:
        """
        Free the agent slot.
        Returns final SSM state for brain to carry to next environment.
        """
        ...

    def reset(self, session: Session) -> dict[str, SensorBundle]:
        """
        Episode boundary reset — resets physics, NOT SSM.
        Called by brain when it wants a new episode in the same env.
        """
        ...

    def step_async(self, actions: dict[str, ActionBundle]) -> None:
        """Send actions non-blocking. Returns immediately."""
        ...

    def step_wait(self) -> dict[str, SensorBundle]:
        """Collect results. Blocks until env step complete."""
        ...


@dataclass
class LossHandle:
    """
    Opaque handle returned by BrainProtocol.forward().

    Carries the live loss tensor (with grad_fn) and step metadata.
    Pass unchanged to BrainProtocol.backward().

    None loss means no gradient available (first step, or train=False).
    Implementation-defined internals — callers must not inspect _state.
    """
    _state: Any = None

    def is_valid(self) -> bool:
        """True if backward() will compute a real gradient."""
        return self._state is not None


@runtime_checkable
class BrainProtocol(Protocol):
    """
    Long-running process with its own GPU scope.

    Two-phase interface enables automatic CPU-GPU overlap in TrainingEngine:

      actions, loss = brain.forward(obs, manifest)  # forward pass + build loss (~2ms)
      env.step_async(actions)                        # env steps on CPU (non-blocking)
      metrics = brain.backward(loss)                 # backward overlaps env step (~3ms)
      obs = env.step_wait()                          # collect when GPU done

    Why this ordering eliminates stale actions:
      - forward() computes fresh actions from current obs
      - actions sent to env immediately (no staleness)
      - backward() runs while env executes those fresh actions
      - overlap is on backward, not forward

    Why no prime() needed:
      - first call: forward() returns LossHandle(None) — no previous h_cond
      - backward(LossHandle(None)) is a no-op, returns empty metrics
      - pipeline starts cleanly without caller managing init state

    The brain's SSM is never reset. It accumulates since training began.
    enter/exit carry it between environments.
    """
    brain_id: str
    sessions: dict[str, Session]

    def enter(self, env: EnvironmentProtocol) -> Session:
        """Join env. Brain brings current SSM state."""
        ...

    def exit(self, env: EnvironmentProtocol) -> None:
        """Leave env. Brain keeps SSM state for next environment."""
        ...

    def forward(
        self,
        sensors:     dict[str, SensorBundle],
        manifest:    SensorManifest,
        explore_mag: float = 0.0,
    ) -> tuple[dict[str, ActionBundle], LossHandle]:
        """
        Forward pass: sensors → actions + live loss tensor.

        Computes:
          1. Encode sensors per manifest (world + self → SSM inputs)
          2. forward_stateful — persistent SSM, never reset
          3. trajectory_head + efference → actions
          4. CE loss: prev_h_cond predicts current world sensors (no next obs needed)

        Does NOT call .backward(). Returns LossHandle for caller to decide when.
        Call backward() after env.step_async() to overlap backward with env step.
        """
        ...

    def backward(self, loss: LossHandle) -> TrainingMetrics:
        """
        Backward pass: loss.backward() + optimizer.step().

        No-op if loss.is_valid() is False (first step or train=False).
        Call after env.step_async() so backward overlaps env step on CPU.
        """
        ...

    def get_state(self) -> dict: ...
    def set_state(self, state: dict) -> None: ...
