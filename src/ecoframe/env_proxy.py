"""
Phase 4: EnvironmentProxy — remote environment, same protocol as local.

The brain can't tell local from remote. All calls go through the proxy.

Usage:
    proxy = EnvironmentProxy("tcp://gpu1:5555")
    session = proxy.enter("brain0", ssm_state={})
    proxy.step_async(actions)
    obs = proxy.step_wait()
    proxy.exit(session)

Manifest is fetched once on connect and cached — brain uses it for
sensor encoding without repeated network calls.
"""
from __future__ import annotations

from ecoframe.protocol import (
    ActionBundle, CapacityError, EnvironmentProtocol,
    SensorBundle, SensorManifest, Session,
)
from ecoframe.env_server import (
    _serialize, _deserialize,
    _manifest_to_dict, _dict_to_manifest,
    _session_to_dict, _dict_to_session,
    _bundle_to_dict, _dict_to_bundle,
)


class EnvironmentProxy:
    """
    Conforms to EnvironmentProtocol. All calls forwarded to EnvironmentServer.

    Indistinguishable from a local environment from the brain's perspective.
    """

    def __init__(
        self,
        address:   str  = "tcp://localhost:5555",
        transport: str  = 'zmq',
        timeout_ms: int = 5000,
    ):
        self._address   = address
        self._transport = transport
        self._timeout   = timeout_ms
        self._socket    = None
        self._manifest_cache: SensorManifest | None = None

        self._connect()

    # ── Connection ─────────────────────────────────────────────────────────────

    def _connect(self) -> None:
        if self._transport == 'zmq':
            try:
                import zmq
            except ImportError:
                raise ImportError("zmq required: pip install ecoframe[zmq]")
            ctx = zmq.Context.instance()
            self._socket = ctx.socket(zmq.REQ)
            self._socket.setsockopt(zmq.RCVTIMEO, self._timeout)
            self._socket.connect(self._address)
        else:
            raise ValueError(f"Unknown transport: {self._transport!r}")

    def _call(self, method: str, **kwargs) -> dict:
        req = {'method': method, 'args': kwargs}
        self._socket.send(_serialize(req))
        rep = _deserialize(self._socket.recv())
        if 'error' in rep:
            raise RuntimeError(f"EnvironmentProxy: server error: {rep['error']}")
        return rep

    # ── EnvironmentProtocol ────────────────────────────────────────────────────

    @property
    def env_id(self) -> str:
        return self.manifest.env_id

    @property
    def manifest(self) -> SensorManifest:
        if self._manifest_cache is None:
            rep = self._call('manifest')
            self._manifest_cache = _dict_to_manifest(rep['manifest'])
        return self._manifest_cache

    @property
    def capacity(self) -> int:
        return 2**31   # proxy doesn't know capacity; CapacityError comes from server

    def start(self) -> None:
        pass   # server is already running

    def close(self) -> None:
        try:
            self._call('close')
        except Exception:
            pass
        if self._socket:
            self._socket.close()

    def enter(self, brain_id: str,
              ssm_state: dict | None = None) -> Session:
        rep = self._call('enter', brain_id=brain_id, ssm_state=ssm_state or {})
        if 'error' in rep:
            if 'capacity' in rep['error'].lower():
                raise CapacityError(rep['error'])
            raise RuntimeError(rep['error'])
        return _dict_to_session(rep['session'])

    def exit(self, session: Session) -> dict:
        rep = self._call('exit', session=_session_to_dict(session))
        return rep.get('ssm_state', {})

    def reset(self, session: Session) -> dict[str, SensorBundle]:
        rep = self._call('reset', session=_session_to_dict(session))
        return {k: _dict_to_bundle(v) for k, v in rep['bundles'].items()}

    def step_async(self, actions: dict[str, ActionBundle]) -> None:
        action_dicts = {
            k: {'continuous': v.continuous, 'discrete': v.discrete,
                'env_id': v.env_id, 'agent_id': v.agent_id}
            for k, v in actions.items()
        }
        self._call('step_async', actions=action_dicts)

    def step_wait(self) -> dict[str, SensorBundle]:
        rep = self._call('step_wait')
        return {k: _dict_to_bundle(v) for k, v in rep['bundles'].items()}
