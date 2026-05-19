"""
Phase 4: EnvironmentServer — exposes any EnvironmentProtocol over the network.

Pairs with EnvironmentProxy: brain can't tell local from remote.
Transport is swappable (ZMQ, gRPC, shared memory).

Default transport: ZMQ REQ/REP (simple, no broker needed, works across machines).
ZMQ is an optional dependency: pip install ecoframe[zmq]

Wire format: msgpack (fast, cross-language, handles numpy arrays).

Usage:
    # On GPU 1 (environment machine):
    env = MetaDriveEnvironment(...)
    server = EnvironmentServer(env, port=5555)
    server.serve_forever()   # blocks

    # On GPU 0 (brain machine):
    proxy = EnvironmentProxy("tcp://gpu1:5555")
    session = proxy.enter("brain0", ssm_state={})
    proxy.step_async(actions)
    obs = proxy.step_wait()
"""
from __future__ import annotations

import threading
from typing import Any

from ecoframe.protocol import (
    ActionBundle, EnvironmentProtocol, SensorBundle,
    SensorManifest, Session,
)


class EnvironmentServer:
    """
    Wraps any EnvironmentProtocol and serves it over ZMQ REP socket.

    Thread-safe: requests are serialized by the ZMQ event loop.
    """

    def __init__(
        self,
        env:        EnvironmentProtocol,
        port:       int  = 5555,
        transport:  str  = 'zmq',
        verbose:    bool = False,
    ):
        self._env       = env
        self._port      = port
        self._transport = transport
        self._verbose   = verbose
        self._running   = False

    def serve_forever(self) -> None:
        """Block and serve requests. Call in a thread or separate process."""
        if self._transport == 'zmq':
            self._serve_zmq()
        else:
            raise ValueError(f"Unknown transport: {self._transport!r}")

    def serve_background(self) -> threading.Thread:
        """Start server in a daemon thread. Returns the thread."""
        t = threading.Thread(target=self.serve_forever, daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._running = False

    # ── ZMQ transport ─────────────────────────────────────────────────────────

    def _serve_zmq(self) -> None:
        try:
            import zmq
        except ImportError:
            raise ImportError("zmq required: pip install ecoframe[zmq]")

        ctx    = zmq.Context()
        socket = ctx.socket(zmq.REP)
        socket.bind(f"tcp://*:{self._port}")
        self._running = True

        if self._verbose:
            print(f"EnvironmentServer listening on port {self._port}", flush=True)

        while self._running:
            try:
                if not socket.poll(timeout=100):   # 100ms timeout for stop check
                    continue
                raw = socket.recv()
                req = _deserialize(raw)
                rep = self._dispatch(req)
                socket.send(_serialize(rep))
            except Exception as e:
                socket.send(_serialize({'error': str(e)}))

        socket.close()
        ctx.term()

    def _dispatch(self, req: dict) -> dict:
        method = req.get('method')
        args   = req.get('args', {})

        if method == 'manifest':
            return {'manifest': _manifest_to_dict(self._env.manifest)}

        elif method == 'enter':
            session = self._env.enter(args['brain_id'], args.get('ssm_state'))
            return {'session': _session_to_dict(session)}

        elif method == 'exit':
            state = self._env.exit(_dict_to_session(args['session']))
            return {'ssm_state': state}

        elif method == 'step_async':
            actions = {k: _dict_to_action(v) for k, v in args['actions'].items()}
            self._env.step_async(actions)
            return {'ok': True}

        elif method == 'step_wait':
            bundles = self._env.step_wait()
            return {'bundles': {k: _bundle_to_dict(b) for k, b in bundles.items()}}

        elif method == 'reset':
            bundles = self._env.reset(_dict_to_session(args['session']))
            return {'bundles': {k: _bundle_to_dict(b) for k, b in bundles.items()}}

        elif method == 'close':
            self._env.close()
            self._running = False
            return {'ok': True}

        else:
            return {'error': f"Unknown method: {method!r}"}


# ── Serialization helpers ──────────────────────────────────────────────────────

def _serialize(obj: Any) -> bytes:
    try:
        import msgpack
        import numpy as np
        def _enc(x):
            if isinstance(x, np.ndarray):
                return {'__ndarray__': True, 'data': x.tobytes(),
                        'dtype': str(x.dtype), 'shape': list(x.shape)}
            raise TypeError(type(x))
        return msgpack.packb(obj, default=_enc, use_bin_type=True)
    except ImportError:
        import pickle
        return pickle.dumps(obj)


def _deserialize(data: bytes) -> Any:
    try:
        import msgpack
        import numpy as np
        def _dec(obj):
            if '__ndarray__' in obj:
                arr = np.frombuffer(obj['data'], dtype=obj['dtype'])
                return arr.reshape(obj['shape'])
            return obj
        return msgpack.unpackb(data, object_hook=_dec, raw=False)
    except ImportError:
        import pickle
        return pickle.loads(data)


def _manifest_to_dict(m: SensorManifest) -> dict:
    return {
        'env_id':  m.env_id,
        'sensors': [
            {'name': s.name, 'shape': list(s.shape), 'dtype': s.dtype,
             'action_affected': s.action_affected,
             'world_external':  s.world_external,
             'temporal_res':    s.temporal_res}
            for s in m.sensors
        ],
    }


def _dict_to_manifest(d: dict) -> SensorManifest:
    from ecoframe.protocol import SensorSpec
    return SensorManifest(
        env_id  = d['env_id'],
        sensors = tuple(
            SensorSpec(name=s['name'], shape=tuple(s['shape']),
                       dtype=s['dtype'],
                       action_affected=s['action_affected'],
                       world_external=s['world_external'],
                       temporal_res=s['temporal_res'])
            for s in d['sensors']
        ),
    )


def _session_to_dict(s: Session) -> dict:
    return {'brain_id': s.brain_id, 'env_id': s.env_id,
            'agent_id': s.agent_id, 'ssm_state': s.ssm_state,
            'entered_at': s.entered_at}


def _dict_to_session(d: dict) -> Session:
    return Session(brain_id=d['brain_id'], env_id=d['env_id'],
                   agent_id=d['agent_id'], ssm_state=d.get('ssm_state', {}),
                   entered_at=d.get('entered_at', 0))


def _bundle_to_dict(b: SensorBundle) -> dict:
    import numpy as np
    return {
        'visual':         b.visual,
        'audio':          b.audio,
        'text_tokens':    b.text_tokens,
        'proprioceptive': b.proprioceptive,
        'reward':         b.reward,
        'done':           b.done,
        'env_id':         b.env_id,
        'agent_id':       b.agent_id,
        'step':           b.step,
    }


def _dict_to_bundle(d: dict) -> SensorBundle:
    return SensorBundle(
        visual         = d.get('visual'),
        audio          = d.get('audio'),
        text_tokens    = d.get('text_tokens'),
        proprioceptive = d.get('proprioceptive'),
        reward         = d.get('reward', 0.0),
        done           = d.get('done', False),
        env_id         = d.get('env_id', ''),
        agent_id       = d.get('agent_id', ''),
        step           = d.get('step', 0),
    )


def _dict_to_action(d: dict) -> ActionBundle:
    return ActionBundle(
        continuous  = d.get('continuous'),
        discrete    = d.get('discrete'),
        text_tokens = d.get('text_tokens'),
        env_id      = d.get('env_id', ''),
        agent_id    = d.get('agent_id', ''),
    )
