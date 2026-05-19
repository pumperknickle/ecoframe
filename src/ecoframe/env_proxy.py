"""
Phase 4: EnvironmentProxy — remote environment, same protocol as local.

Uses two socket channels:
  ctrl: REQ/REP for enter/exit/manifest (synchronous, infrequent)
  step: PUSH actions → server, PULL observations ← server (genuinely async)

step_async sends actions and returns immediately.
Brain does GPU forward+backward while server steps the environment.
step_wait blocks only when GPU work is done and observations are needed.
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
    Conforms to EnvironmentProtocol. Forwards all calls to EnvironmentServer.
    Brain cannot distinguish local from remote.
    """

    def __init__(
        self,
        host:        str  = "tcp://localhost",
        ctrl_port:   int  = 5555,
        obs_port:    int  = 5556,
        action_port: int  = 5557,
        timeout_ms:  int  = 5000,
    ):
        self._host        = host
        self._ctrl_port   = ctrl_port
        self._obs_port    = obs_port
        self._action_port = action_port
        self._timeout     = timeout_ms
        self._manifest_cache: SensorManifest | None = None
        self._ctrl   = None
        self._push   = None   # client → server (actions)
        self._pull   = None   # server → client (observations)
        self._connect()

    def _connect(self) -> None:
        try:
            import zmq
        except ImportError:
            raise ImportError("zmq required: pip install ecoframe[zmq]")
        ctx = zmq.Context.instance()

        self._ctrl = ctx.socket(zmq.REQ)
        self._ctrl.setsockopt(zmq.RCVTIMEO, self._timeout)
        self._ctrl.connect(f"{self._host}:{self._ctrl_port}")

        self._push = ctx.socket(zmq.PUSH)
        self._push.connect(f"{self._host}:{self._action_port}")

        self._pull = ctx.socket(zmq.PULL)
        self._pull.setsockopt(zmq.RCVTIMEO, self._timeout)
        self._pull.connect(f"{self._host}:{self._obs_port}")

    def _ctrl_call(self, method: str, **kwargs) -> dict:
        req = {'method': method, 'args': kwargs}
        self._ctrl.send(_serialize(req))
        rep = _deserialize(self._ctrl.recv())
        if 'error' in rep:
            raise RuntimeError(f"EnvironmentProxy server error: {rep['error']}")
        return rep

    # ── EnvironmentProtocol ────────────────────────────────────────────────────

    @property
    def env_id(self) -> str:
        return self.manifest.env_id

    @property
    def manifest(self) -> SensorManifest:
        if self._manifest_cache is None:
            rep = self._ctrl_call('manifest')
            self._manifest_cache = _dict_to_manifest(rep['manifest'])
        return self._manifest_cache

    @property
    def capacity(self) -> int:
        return 2**31

    def start(self) -> None:
        pass

    def close(self) -> None:
        try:
            self._ctrl_call('close')
        except Exception:
            pass
        for s in (self._ctrl, self._push, self._pull):
            if s:
                try: s.close()
                except Exception: pass

    def enter(self, brain_id: str, ssm_state: dict | None = None) -> Session:
        rep = self._ctrl_call('enter', brain_id=brain_id, ssm_state=ssm_state or {})
        return _dict_to_session(rep['session'])

    def exit(self, session: Session) -> dict:
        rep = self._ctrl_call('exit', session=_session_to_dict(session))
        return rep.get('ssm_state', {})

    def reset(self, session: Session) -> dict[str, SensorBundle]:
        rep = self._ctrl_call('reset', session=_session_to_dict(session))
        return {k: _dict_to_bundle(v) for k, v in rep['bundles'].items()}

    def step_async(self, actions: dict[str, ActionBundle]) -> None:
        """
        Send actions to server — returns immediately.
        Server begins stepping the environment.
        Call step_wait() to collect results after GPU work.
        """
        action_dicts = {
            k: {'continuous': v.continuous, 'discrete': v.discrete,
                'env_id': v.env_id, 'agent_id': v.agent_id}
            for k, v in actions.items()
        }
        self._push.send(_serialize({'actions': action_dicts}))

    def step_wait(self) -> dict[str, SensorBundle]:
        """
        Collect observations from server.
        Blocks only until server's env step is complete.
        """
        rep = _deserialize(self._pull.recv())
        return {k: _dict_to_bundle(v) for k, v in rep['bundles'].items()}
