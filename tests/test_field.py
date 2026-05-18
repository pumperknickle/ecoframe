"""Tests for Field with local and mock_remote backends."""
import pytest
from ecoframe.field import Field
from ecoframe.signal import CuriositySignal, EnvironmentSignal


def _make_signal(publisher="a0", pos=(0.0, 0.0), visual=1.0):
    return CuriositySignal(position=pos, timestamp=1,
                           publisher=publisher, visual=visual)


def test_field_local_publish_receive():
    field = Field(backend='local')
    field.register_agent("a0", pos=(0.0, 0.0))
    field.register_agent("a1", pos=(1.0, 0.0))
    field.publish("a0", _make_signal())
    field.relay("a0")
    msgs = field.receive("a1")
    assert len(msgs) == 1


def test_field_query_by_radius():
    field = Field(backend='local')
    field.register_agent("a0", pos=(0.0, 0.0))
    field.publish("a0", _make_signal(pos=(0.5, 0.0)))
    field.publish("a0", _make_signal(pos=(10.0, 0.0)))
    nearby = field.query(pos=(0.0, 0.0), radius=1.0)
    assert len(nearby) == 1


def test_field_environment_signal_discoverable():
    field = Field(backend='local')
    field.register_agent("env0", pos=(5.0, 0.0))
    env_sig = EnvironmentSignal(
        position=(5.0, 0.0), timestamp=1, publisher="env0",
        curiosity=2.5, load_fraction=0.3, address="localhost:5555",
    )
    field.publish("env0", env_sig)
    results = field.query(pos=(5.0, 0.0), radius=1.0)
    assert any(isinstance(s, EnvironmentSignal) for s in results)


def test_field_mock_remote_same_api():
    field = Field(backend='mock_remote', latency_ms=0.0)
    field.register_agent("a0", pos=(0.0, 0.0))
    field.register_agent("a1", pos=(0.5, 0.0))
    field.publish("a0", _make_signal())
    field.relay("a0")
    assert len(field.receive("a1")) == 1


def test_field_unknown_backend_raises():
    with pytest.raises(ValueError):
        Field(backend='unknown_backend')
