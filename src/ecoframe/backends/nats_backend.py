"""
Phase 5: NATS backend for distributed Field.

NATS is a lightweight pub/sub message broker — no central registry,
peer-to-peer friendly, exactly what the Field needs for decentralized discovery.

Install: pip install ecoframe[nats]   (pulls nats-py)

Architecture:
  - Each agent/env publishes signals to subject: "ecoframe.signals.<agent_id>"
  - Any subscriber can query by subject pattern
  - NATS JetStream provides persistence for late-joining agents
  - No coordinator: NATS cluster is the only shared infrastructure

Usage:
    field = Field(backend='nats', url='nats://broker:4222')

Zero agent/env code changes vs 'local' backend.
"""
from __future__ import annotations

import asyncio
import json
import threading
from collections import defaultdict, deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ecoframe.signal import Signal


class NatsBackend:
    """
    Distributed Field backend over NATS pub/sub.

    Implements the same interface as LocalBackend.
    Runs async NATS client on a background thread.

    Note: full async NATS support requires nats-py >= 2.3
    """

    def __init__(
        self,
        url:        str  = 'nats://localhost:4222',
        neighbor_k: int  = 4,
        relay_ttl:  int  = 3,
        max_queue:  int  = 32,
    ):
        try:
            import nats
        except ImportError:
            raise ImportError(
                "nats-py required: pip install ecoframe[nats]  "
                "(or: pip install nats-py)"
            )
        self._url        = url
        self._k          = neighbor_k
        self._relay_ttl  = relay_ttl
        self._positions:  dict[str, tuple[float, float]] = {}
        self._published:  dict[str, deque]               = defaultdict(lambda: deque(maxlen=max_queue))
        self._received:   dict[str, list]                = defaultdict(list)
        self._trust:      dict[str, float]               = defaultdict(lambda: 1.0)

        self._loop   = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._nc = None
        asyncio.run_coroutine_threadsafe(self._connect(), self._loop).result(timeout=10)

    async def _connect(self) -> None:
        import nats
        self._nc = await nats.connect(self._url)

    def register_agent(self, agent_id: str,
                       pos: tuple[float, float] = (0.0, 0.0)) -> None:
        self._positions[agent_id] = pos

    def update_position(self, agent_id: str,
                        pos: tuple[float, float]) -> None:
        self._positions[agent_id] = pos

    def publish(self, agent_id: str, signal: 'Signal') -> None:
        self._published[agent_id].append(signal)
        if self._nc is not None:
            subject = f"ecoframe.signals.{agent_id}"
            payload = _signal_to_bytes(signal)
            asyncio.run_coroutine_threadsafe(
                self._nc.publish(subject, payload), self._loop)

    def receive(self, agent_id: str) -> list['Signal']:
        msgs = list(self._received.get(agent_id, []))
        self._received[agent_id] = []
        return msgs

    def relay(self, agent_id: str, threshold: float = None) -> None:
        from ecoframe.backends.local import LocalBackend
        LocalBackend.relay(self, agent_id, threshold)

    def gradient(self, agent_id: str) -> tuple[float, float]:
        from ecoframe.backends.local import LocalBackend
        return LocalBackend.gradient(self, agent_id)

    def update_trust_from_surprise(self, agent_id: str, surp: float) -> None:
        alpha = 0.05
        self._trust[agent_id] = (1 - alpha) * self._trust[agent_id] + alpha * surp

    def trust_variance(self) -> float:
        vals = list(self._trust.values())
        if len(vals) < 2:
            return 0.0
        mean = sum(vals) / len(vals)
        return sum((v - mean)**2 for v in vals) / len(vals)

    def step(self) -> None:
        pass

    def _k_nearest(self, agent_id: str) -> list[str]:
        from ecoframe.backends.local import LocalBackend
        return LocalBackend._k_nearest(self, agent_id)


def _signal_to_bytes(signal: 'Signal') -> bytes:
    d = {
        'type':      type(signal).__name__,
        'position':  list(signal.position),
        'timestamp': signal.timestamp,
        'publisher': signal.publisher,
    }
    for ch in signal.__class__.channels:
        d[ch] = getattr(signal, ch, 0.0)
    return json.dumps(d).encode()
