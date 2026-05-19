"""
Phase 4: EnvironmentServer — exposes any EnvironmentProtocol over the network.

Transport: ZMQ with two socket pairs:
  - Control socket (REQ/REP): enter, exit, reset, manifest, close
  - Async socket (PUSH/PULL pair): step_async sends actions (PUSH),
    step_wait receives observations (PULL from server's PUSH)

This provides genuine overlap: client sends actions, does GPU compute,
then collects observations. Server steps the environment concurrently.

Wire format: msgpack with numpy + pickle fallback for complex objects.

Usage:
    # Environment machine (GPU 1):
    server = EnvironmentServer(env, ctrl_port=5555, obs_port=5556)
    server.serve_forever()

    # Brain machine (GPU 0):
    proxy = EnvironmentProxy("tcp://gpu1", ctrl_port=5555, obs_port=5556)
    session = proxy.enter("brain0")
    proxy.step_async(actions)      # non-blocking: sends to server
    # ... GPU forward+backward here ...
    obs = proxy.step_wait()        # collect when ready
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
    Serves any EnvironmentProtocol over ZMQ.

    Two socket channels:
      ctrl_port: REQ/REP for enter/exit/manifest (infrequent, synchronous)
      obs_port:  PUSH (server→client) + action_port PULL (client→server)
                 for async step_async/step_wait
    """

    def __init__(
        self,
        env:          EnvironmentProtocol,
        ctrl_port:    int  = 5555,
        obs_port:     int  = 5556,
        action_port:  int  = 5557,
        verbose:      bool = False,
    ):
        self._env         = env
        self._ctrl_port   = ctrl_port
        self._obs_port    = obs_port
        self._action_port = action_port
        self._verbose     = verbose
        self._running     = False

    def serve_forever(self) -> None:
        try:
            import zmq
        except ImportError:
            raise ImportError("zmq required: pip install ecoframe[zmq]")

        ctx = zmq.Context()

        # Control: synchronous REQ/REP
        ctrl = ctx.socket(zmq.REP)
        ctrl.bind(f"tcp://*:{self._ctrl_port}")

        # Async step: server pulls actions, pushes observations
        action_pull = ctx.socket(zmq.PULL)
        action_pull.bind(f"tcp://*:{self._action_port}")

        obs_push = ctx.socket(zmq.PUSH)
        obs_push.bind(f"tcp://*:{self._obs_port}")

        self._running = True
        if self._verbose:
            print(f"EnvironmentServer: ctrl={self._ctrl_port} "
                  f"obs={self._obs_port} action={self._action_port}", flush=True)

        poller = zmq.Poller()
        poller.register(ctrl, zmq.POLLIN)
        poller.register(action_pull, zmq.POLLIN)

        while self._running:
            ready = dict(poller.poll(timeout=100))

            if ctrl in ready:
                raw = ctrl.recv()
                req = _deserialize(raw)
                rep = self._dispatch_ctrl(req)
                ctrl.send(_serialize(rep))

            if action_pull in ready:
                raw     = action_pull.recv()
                req     = _deserialize(raw)
                actions = {k: _dict_to_action(v)
                           for k, v in req.get('actions', {}).items()}
                self._env.step_async(actions)
                bundles = self._env.step_wait()
                obs_push.send(_serialize(
                    {'bundles': {k: _bundle_to_dict(b)
                                 for k, b in bundles.items()}}))

        ctrl.close()
        action_pull.close()
        obs_push.close()
        ctx.term()

    def serve_background(self) -> threading.Thread:
        t = threading.Thread(target=self.serve_forever, daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self._running = False

    def _dispatch_ctrl(self, req: dict) -> dict:
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
        elif method == 'reset':
            bundles = self._env.reset(_dict_to_session(args['session']))
            return {'bundles': {k: _bundle_to_dict(b) for k, b in bundles.items()}}
        elif method == 'close':
            self._env.close()
            self._running = False
            return {'ok': True}
        else:
            return {'error': f"Unknown method: {method!r}"}


# ── Serialization ──────────────────────────────────────────────────────────────

def _serialize(obj: Any) -> bytes:
    try:
        import msgpack
        import numpy as np
        def _enc(x):
            if isinstance(x, np.ndarray):
                return {'__ndarray__': True, 'data': x.tobytes(),
                        'dtype': str(x.dtype), 'shape': list(x.shape)}
            # Torch tensors: convert to numpy first
            try:
                import torch
                if isinstance(x, torch.Tensor):
                    arr = x.detach().cpu().numpy()
                    return {'__ndarray__': True, '__torch__': True,
                            'data': arr.tobytes(), 'dtype': str(arr.dtype),
                            'shape': list(arr.shape)}
            except ImportError:
                pass
            raise TypeError(f"Cannot serialize type {type(x)}")
        return msgpack.packb(obj, default=_enc, use_bin_type=True)
    except (ImportError, TypeError):
        # Fallback: pickle handles anything including torch tensors
        import pickle
        return b'\x00' + pickle.dumps(obj)  # prefix byte distinguishes from msgpack


def _deserialize(data: bytes) -> Any:
    if data[:1] == b'\x00':
        import pickle
        return pickle.loads(data[1:])
    try:
        import msgpack
        import numpy as np
        def _dec(obj):
            if b'__ndarray__' in obj or '__ndarray__' in obj:
                key = b'__ndarray__' if b'__ndarray__' in obj else '__ndarray__'
                dt_key  = b'dtype'  if b'dtype'  in obj else 'dtype'
                shp_key = b'shape'  if b'shape'  in obj else 'shape'
                dat_key = b'data'   if b'data'   in obj else 'data'
                torch_key = b'__torch__' if b'__torch__' in obj else '__torch__'
                arr = np.frombuffer(obj[dat_key],
                                    dtype=np.dtype(obj[dt_key])).reshape(obj[shp_key])
                if torch_key in obj and obj[torch_key]:
                    try:
                        import torch
                        return torch.from_numpy(arr.copy())
                    except ImportError:
                        pass
                return arr
            return obj
        return msgpack.unpackb(data, object_hook=_dec, raw=False)
    except ImportError:
        import pickle
        return pickle.loads(data)


def _manifest_to_dict(m: SensorManifest) -> dict:
    return {
        'env_id': m.env_id,
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
    return {
        'visual':         b.visual,
        'audio':          b.audio,
        'text_tokens':    b.text_tokens,
        'proprioceptive': b.proprioceptive,
        'extra':          b.extra,
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
        extra          = d.get('extra') or {},
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
