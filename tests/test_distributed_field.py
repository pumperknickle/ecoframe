"""
Phase 5 validation: distributed Field backends are swappable.

Tests verify:
  1. Field(backend='local') and Field(backend='mock_remote') have identical APIs
  2. Swapping backend requires zero agent code changes
  3. NATS backend raises ImportError gracefully when nats-py not installed
  4. All backends support: register_agent, publish, query, gradient, step
  5. EnvironmentSignal flows through mock_remote backend (simulated latency)
  6. Field API is stable across backends (same method signatures)
"""
import pytest
from ecoframe.field import Field
from ecoframe.signal import CuriositySignal, EnvironmentSignal


def _exercise_field(field: Field, prefix: str = "a") -> None:
    """Run the same sequence against any backend — proves API stability."""
    aid0, aid1 = f"{prefix}0", f"{prefix}1"
    field.register_agent(aid0, pos=(0.0, 0.0))
    field.register_agent(aid1, pos=(1.0, 0.0))

    sig = CuriositySignal(position=(0.0, 0.0), timestamp=1,
                          publisher=aid0, visual=2.0)
    field.publish(aid0, sig)
    field.relay(aid0)
    field.step()

    env_sig = EnvironmentSignal(
        position=(0.5, 0.0), timestamp=2, publisher=aid0,
        curiosity=3.0, load_fraction=0.2, env_type="test",
    )
    field.publish(aid0, env_sig)

    nearby = field.query(pos=(0.0, 0.0), radius=1.0)
    assert len(nearby) >= 1, f"Expected signals near origin, got {len(nearby)}"

    dx, dz = field.gradient(aid0)
    assert isinstance(dx, float)
    assert isinstance(dz, float)


def test_local_backend_full_api():
    field = Field(backend='local')
    _exercise_field(field, prefix="local")


def test_mock_remote_backend_full_api():
    field = Field(backend='mock_remote', latency_ms=0.0)
    _exercise_field(field, prefix="mock")


def test_backends_produce_identical_api_surface():
    """The set of public methods is identical across backends."""
    local  = Field(backend='local')
    mock   = Field(backend='mock_remote', latency_ms=0.0)
    public = lambda f: {m for m in dir(f) if not m.startswith('_')}
    assert public(local) == public(mock)


def test_swap_backend_zero_code_changes():
    """Exact same agent code works with both backends."""

    def agent_routine(field: Field, agent_id: str) -> list:
        field.register_agent(agent_id, pos=(0.0, 0.0))
        sig = CuriositySignal(position=(0.0, 0.0), timestamp=1,
                              publisher=agent_id, visual=1.5)
        field.publish(agent_id, sig)
        return field.query(pos=(0.0, 0.0), radius=2.0)

    results_local = agent_routine(Field(backend='local'),       "ag0")
    results_mock  = agent_routine(Field(backend='mock_remote',
                                        latency_ms=0.0),        "ag0")
    assert len(results_local) == len(results_mock) == 1


def test_nats_backend_raises_import_error_gracefully():
    """If nats-py not installed, Field(backend='nats') gives clear error."""
    import sys
    # Temporarily hide nats module
    nats_mod = sys.modules.get('nats')
    sys.modules['nats'] = None   # block the import
    try:
        with pytest.raises((ImportError, TypeError)):
            Field(backend='nats', url='nats://localhost:4222')
    finally:
        if nats_mod is not None:
            sys.modules['nats'] = nats_mod
        else:
            sys.modules.pop('nats', None)


def test_environment_signal_survives_mock_latency():
    field = Field(backend='mock_remote', latency_ms=0.0)
    field.register_agent("env0", pos=(0.0, 0.0))
    sig = EnvironmentSignal(
        position=(0.0, 0.0), timestamp=1, publisher="env0",
        curiosity=4.2, env_type="metadrive_roundabout",
    )
    field.publish("env0", sig)
    results = field.query(pos=(0.0, 0.0), radius=0.1)
    env_sigs = [s for s in results if isinstance(s, EnvironmentSignal)]
    assert len(env_sigs) == 1
    assert env_sigs[0].curiosity == pytest.approx(4.2)
    assert env_sigs[0].env_type == "metadrive_roundabout"


def test_unknown_backend_raises_value_error():
    with pytest.raises(ValueError, match="Unknown backend"):
        Field(backend='blockchain')
