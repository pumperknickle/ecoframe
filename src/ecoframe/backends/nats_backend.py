"""
Phase 5: NATS backend for distributed Field.

Extends LocalBackend: adds NATS pub/sub on top of in-memory state.
Agents can discover each other across machines via NATS subjects.

Install: pip install ecoframe[nats]

Usage:
    field = Field(backend='nats', url='nats://broker:4222')
"""
from __future__ import annotations

import asyncio
import json
import threading
from typing import TYPE_CHECKING

from ecoframe.backends.local import LocalBackend

if TYPE_CHECKING:
    from ecoframe.signal import Signal


class NatsBackend(LocalBackend):
    """
    Distributed Field backend over NATS pub/sub.

    Inherits all local state management from LocalBackend.
    Adds async NATS publishing on top — remote agents receive signals
    via NATS subscriptions.
    """

    def __init__(
        self,
        url:        str  = 'nats://localhost:4222',
        neighbor_k: int  = 4,
        relay_ttl:  int  = 3,
        max_queue:  int  = 32,
    ):
        try:
            import nats as _nats  # noqa: F401 — check at init time
        except ImportError:
            raise ImportError(
                "nats-py required: pip install ecoframe[nats]  "
                "(or: pip install nats-py)"
            )
        super().__init__(neighbor_k=neighbor_k, max_queue=max_queue,
                         relay_ttl=relay_ttl)
        self._url    = url
        self._nc     = None
        self._loop   = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True)
        self._thread.start()
        asyncio.run_coroutine_threadsafe(
            self._connect(), self._loop).result(timeout=10)

    async def _connect(self) -> None:
        import nats
        self._nc = await nats.connect(self._url)

    def publish(self, agent_id: str, signal: 'Signal') -> None:
        """Publish locally AND broadcast over NATS."""
        super().publish(agent_id, signal)
        if self._nc is not None:
            payload = _signal_to_bytes(signal)
            asyncio.run_coroutine_threadsafe(
                self._nc.publish(f"ecoframe.signals.{agent_id}", payload),
                self._loop,
            )

    # relay, gradient, step — all inherited from LocalBackend (no override needed)


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
