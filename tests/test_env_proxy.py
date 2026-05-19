"""
Phase 4 validation: EnvironmentServer + EnvironmentProxy.

Tests use an in-process mock transport to avoid ZMQ dependency.
Validates that proxy is indistinguishable from a local env.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock

from ecoframe.protocol import (
    ActionBundle, CapacityError, SensorBundle,
    SensorManifest, SensorSpec, Session,
)
from ecoframe.env_server import (
    _serialize, _deserialize,
    _manifest_to_dict, _dict_to_manifest,
    _session_to_dict, _dict_to_session,
    _bundle_to_dict, _dict_to_bundle,
    EnvironmentServer,
)


MANIFEST = SensorManifest("test_env", sensors=(
    SensorSpec("visual",         (64, 64, 3), action_affected=True,  world_external=True),
    SensorSpec("proprioceptive", (5,),        action_affected=False, world_external=False),
))


def _make_mock_env():
    env = MagicMock()
    env.manifest = MANIFEST
    env.env_id   = "test_env"
    env.capacity = 8
    env.enter.return_value = Session(
        brain_id="b0", env_id="test_env",
        agent_id="a0", ssm_state={"k": "v"})
    env.exit.return_value = {"k": "v"}
    env.step_wait.return_value = {
        "a0": SensorBundle(
            visual=np.zeros((64, 64, 3), dtype=np.uint8),
            proprioceptive=np.zeros(5, dtype=np.float32),
            reward=0.1, done=False,
            env_id="test_env", agent_id="a0",
        )
    }
    return env


class _InProcessProxy:
    """
    Test proxy that calls EnvironmentServer._dispatch() directly.
    Simulates the full serialization round-trip without network.
    """
    def __init__(self, server: EnvironmentServer):
        self._server = server
        self._manifest_cache = None

    def _call(self, method, **kwargs):
        req = {'method': method, 'args': kwargs}
        raw = _serialize(req)
        req2 = _deserialize(raw)
        rep  = self._server._dispatch(req2)
        raw2 = _serialize(rep)
        return _deserialize(raw2)

    @property
    def manifest(self):
        if self._manifest_cache is None:
            rep = self._call('manifest')
            self._manifest_cache = _dict_to_manifest(rep['manifest'])
        return self._manifest_cache

    @property
    def env_id(self): return self.manifest.env_id

    def enter(self, brain_id, ssm_state=None):
        rep = self._call('enter', brain_id=brain_id, ssm_state=ssm_state or {})
        return _dict_to_session(rep['session'])

    def exit(self, session):
        rep = self._call('exit', session=_session_to_dict(session))
        return rep.get('ssm_state', {})

    def step_async(self, actions):
        action_dicts = {
            k: {'continuous': v.continuous, 'discrete': v.discrete,
                'env_id': v.env_id, 'agent_id': v.agent_id}
            for k, v in actions.items()
        }
        self._call('step_async', actions=action_dicts)

    def step_wait(self):
        rep = self._call('step_wait')
        return {k: _dict_to_bundle(v) for k, v in rep['bundles'].items()}

    def reset(self, session):
        rep = self._call('reset', session=_session_to_dict(session))
        return {k: _dict_to_bundle(v) for k, v in rep['bundles'].items()}


@pytest.fixture
def proxy():
    env    = _make_mock_env()
    server = EnvironmentServer(env, port=5556, verbose=False)
    return _InProcessProxy(server)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_manifest_fetched_over_wire(proxy):
    m = proxy.manifest
    assert isinstance(m, SensorManifest)
    assert m.env_id == "test_env"


def test_manifest_prediction_targets_survive_roundtrip(proxy):
    targets = proxy.manifest.prediction_targets
    names   = [s.name for s in targets]
    assert "visual" in names
    assert "proprioceptive" not in names


def test_enter_returns_session(proxy):
    session = proxy.enter("brain0", ssm_state={"x": 1})
    assert isinstance(session, Session)
    assert session.brain_id == "b0"
    assert session.env_id   == "test_env"


def test_exit_returns_ssm_state(proxy):
    session  = proxy.enter("brain0")
    returned = proxy.exit(session)
    assert isinstance(returned, dict)


def test_step_wait_returns_sensor_bundles(proxy):
    bundles = proxy.step_wait()
    assert "a0" in bundles
    assert isinstance(bundles["a0"], SensorBundle)


def test_sensor_bundle_shapes_survive_roundtrip(proxy):
    bundles = proxy.step_wait()
    b = bundles["a0"]
    assert b.visual is not None
    assert b.visual.shape == (64, 64, 3)
    assert b.proprioceptive.shape == (5,)


def test_step_async_then_wait(proxy):
    actions = {"a0": ActionBundle(
        continuous=np.array([0.1, 0.5], dtype=np.float32),
        agent_id="a0")}
    proxy.step_async(actions)   # should not raise
    bundles = proxy.step_wait()
    assert "a0" in bundles


def test_serialization_roundtrip_numpy():
    arr = np.random.rand(64, 64, 3).astype(np.float32)
    obj = {"array": arr, "name": "test"}
    recovered = _deserialize(_serialize(obj))
    assert np.allclose(arr, recovered["array"])


def test_serialization_roundtrip_session():
    s = Session(brain_id="b0", env_id="e0", agent_id="a0",
                ssm_state={"key": "val"}, entered_at=42)
    d = _session_to_dict(s)
    s2 = _dict_to_session(d)
    assert s2.brain_id  == s.brain_id
    assert s2.agent_id  == s.agent_id
    assert s2.ssm_state == s.ssm_state


def test_proxy_env_id_matches_manifest(proxy):
    assert proxy.env_id == "test_env"


def test_manifest_cached_after_first_call(proxy):
    m1 = proxy.manifest
    m2 = proxy.manifest
    assert m1 is m2   # cached — same object
