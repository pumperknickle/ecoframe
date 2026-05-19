# ecoframe

Standard brain/environment protocol for persistent-state AI agents.

EcoFrame defines the contracts and transport layer that connect brains to environments. No training logic, no model code — only types, interfaces, and wiring. Both sides import this package; neither side knows about the other's internals.

## Core idea

A brain is a long-running process with a persistent SSM (state space model) that is **never reset**. It accumulates state across episode boundaries and across environment switches. Environments are also long-running processes, each owning their own GPU context, physics simulation, and rendering.

Training is a loop over `(forward, step_async, backward, step_wait)`. The overlap between `backward` (GPU) and `step_async` (CPU) is the free lunch — it's automatic inside `TrainingEngine`.

## Installation

```bash
pip install ecoframe
# optional extras
pip install ecoframe[zmq]   # distributed env server/proxy
pip install ecoframe[nats]  # NATS gossip backend
pip install ecoframe[redis] # Redis gossip backend
```

Core depends only on `numpy`. No torch required.

## Packages

### `ecoframe.protocol`

All the contracts:

| Type | Role |
|------|------|
| `SensorSpec` | Describes one sensor channel. `action_affected` + `world_external` determine whether it's a prediction target. Proprioceptive sensors are SSM input only — never predicted. |
| `SensorManifest` | Schema for all sensors in one environment. Brain adapts to manifest, not vice versa. |
| `HardwareSpec` | Hardware requirements published by environments so brains route to reachable envs. Data only — no OS or CUDA calls. |
| `SensorBundle` | One timestep of observations from one agent. World sensors (visual, audio, text) + self sensors (proprioceptive) + extras. |
| `ActionBundle` | One timestep of actions from one agent. Continuous + discrete + text. |
| `Session` | A brain's tenancy in one environment. Carries SSM state — the environment stores it but never modifies it. |
| `LossHandle` | Opaque handle from `forward()`. Pass to `backward()`. `None` loss = no-op on first step. |
| `EnvironmentProtocol` | `start / close / enter / exit / reset / step_async / step_wait` |
| `BrainProtocol` | `enter / exit / forward / backward / get_state / set_state` |

**Why `step_async` / `step_wait`:**  
Actions are fresh (computed from current obs, sent immediately). Backward runs while the environment executes those actions on CPU. No stale actions, no wasted GPU time.

**Why SSM is never reset:**  
An agent that loses memory at episode boundaries can't learn temporal structure. The SSM accumulates from training step zero. Episode `done=True` is metadata, not a reset trigger.

### `ecoframe.signal`

Typed scalar signals for gossip between agents and environments.

```
Signal (base)
├── CuriositySignal   — published by brains: visual, body, dynamics, kl, contact
├── EnvironmentSignal — published by envs: curiosity, load_fraction, difficulty, address, certs
├── CertSignal        — published by cert envs: passed/failed + retry_after_steps
└── BrainSignal       — published by brains: ce_ema, surprise, steps, load
```

Channel values are always Python floats. Assigning a tensor to a signal channel raises `TypeError` immediately — signals carry measurements, not representations.

### `ecoframe.field`

Swappable gossip transport. All backends implement the same interface.

```python
field = Field()                                         # local in-memory (default)
field = Field(backend='mock_remote', latency_ms=2.0)   # simulated network
field = Field(backend='nats', url='nats://broker:4222') # optional

field.publish(agent_id, signal)
signals = field.query(pos=(x, z), radius=2.0)
dx, dz  = field.gradient(agent_id)
```

### `ecoframe.training_engine`

Wires one `BrainProtocol` to one `EnvironmentProtocol` with automatic backward/env overlap.

```python
engine = TrainingEngine(brain, env, explore_mag=0.1)

for step, metrics in engine.run(n_steps=2_000_000):
    log(step, metrics)
```

On the first step, `forward()` returns `LossHandle(None)` and `backward()` is a no-op. No `prime()` call needed.

### `ecoframe.environment`

Abstract base class for all training environments. Provides `capacity`, `step`, `reset`, `enter`, `exit`, `emit`, `register_observer`, and `contains` (nested sub-envs for hierarchical composition).

### `ecoframe.ecology`

`Ecology(Environment, ABC)` — abstract base for multi-agent ecologies. Provides `_gossip_cycle()` and `motor_bias()` as shared coordination infrastructure so subclasses don't reimplement gossip.

`SimpleEcology` — sequential ecology for conventional agents. Calls `observe / predict / act / surprise` by convention. Good for prototypes and non-GPU domains.

### `ecoframe.env_server` / `ecoframe.env_proxy`

Expose any `EnvironmentProtocol` over ZMQ so brain and environment run on separate machines (and separate GPU contexts).

```python
# Environment machine (GPU 1):
server = EnvironmentServer(env, ctrl_port=5555, obs_port=5556)
server.serve_forever()

# Brain machine (GPU 0):
proxy = EnvironmentProxy("tcp://gpu1")
session = proxy.enter("brain0")
proxy.step_async(actions)   # non-blocking
# ... GPU backward here ...
obs = proxy.step_wait()     # collect when ready
```

`EnvironmentProxy` conforms to `EnvironmentProtocol`. The brain cannot distinguish local from remote.

Wire format: msgpack with numpy arrays; pickle fallback for complex objects.

## Architecture

```
         ┌─────────────┐        Field (gossip)       ┌──────────────────┐
         │    Brain    │◄──── EnvironmentSignal ─────│  Environment     │
         │  (BrainProtocol)                          │  (Env Protocol)  │
         │             │──── CuriositySignal ───────►│                  │
         └──────┬──────┘                             └────────┬─────────┘
                │  forward() → actions, loss                  │
                │  step_async(actions) ──────────────────────►│
                │  backward(loss)  ◄── overlaps env step ─────│
                │  step_wait() ◄──────────────────────────────┘
                │
         TrainingEngine (wires the loop)
```

## Invariants

1. `SensorSpec.action_affected` tags what to predict — no hardcoded logic.
2. Proprioceptive is never a prediction target (`world_external=False`).
3. `Session` carries SSM state — neither brain nor env resets it unilaterally.
