"""
LocalBackend: in-memory A2A gossip field.

Single-machine, zero network deps. Default backend for Field.
Implements publish/receive/relay/gradient over an in-process signal store.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ecoframe.signal import Signal


class LocalBackend:
    """
    In-memory gossip backend. No external dependencies.

    Each agent has a position in 2D world space. Signals propagate to
    k-nearest neighbors with TTL-based decay.
    """

    def __init__(self, neighbor_k: int = 4, max_queue: int = 32,
                 relay_ttl: int = 3):
        self._k         = neighbor_k
        self._max_q     = max_queue
        self._relay_ttl = relay_ttl

        self._positions:  dict[str, tuple[float, float]] = {}
        self._published:  dict[str, deque]               = defaultdict(lambda: deque(maxlen=max_queue))
        self._received:   dict[str, list]                = defaultdict(list)
        self._trust:      dict[str, float]               = defaultdict(lambda: 1.0)

    def register_agent(self, agent_id: str,
                       pos: tuple[float, float] = (0.0, 0.0)) -> None:
        self._positions[agent_id] = pos

    def update_position(self, agent_id: str,
                        pos: tuple[float, float]) -> None:
        self._positions[agent_id] = pos

    def publish(self, agent_id: str, signal: 'Signal') -> None:
        self._published[agent_id].append(signal)

    def receive(self, agent_id: str) -> list['Signal']:
        msgs = list(self._received.get(agent_id, []))
        self._received[agent_id] = []
        return msgs

    def relay(self, agent_id: str, threshold: float = None) -> None:
        if agent_id not in self._positions:
            return
        pos     = self._positions[agent_id]
        signals = list(self._published.get(agent_id, []))
        peers   = self._k_nearest(agent_id)
        for sig in signals:
            if threshold is not None and sig.R < threshold:
                continue
            for peer_id in peers:
                if peer_id != sig.publisher:
                    self._received[peer_id].append(sig)

    def gradient(self, agent_id: str) -> tuple[float, float]:
        if agent_id not in self._positions:
            return (0.0, 0.0)
        pos     = self._positions[agent_id]
        signals = self.receive(agent_id)
        if not signals:
            return (0.0, 0.0)
        dx, dz, total = 0.0, 0.0, 0.0
        for sig in signals:
            sx, sz = sig.position
            dist   = max(1e-6, math.sqrt((sx - pos[0])**2 + (sz - pos[1])**2))
            weight = sig.R / dist
            dx    += weight * (sx - pos[0]) / dist
            dz    += weight * (sz - pos[1]) / dist
            total += weight
        if total > 0:
            dx /= total
            dz /= total
        return (dx, dz)

    def update_trust_from_surprise(self, agent_id: str, surp: float) -> None:
        alpha              = 0.05
        self._trust[agent_id] = (1 - alpha) * self._trust[agent_id] + alpha * surp

    def trust_variance(self) -> float:
        vals = list(self._trust.values())
        if len(vals) < 2:
            return 0.0
        mean = sum(vals) / len(vals)
        return sum((v - mean)**2 for v in vals) / len(vals)

    def step(self) -> None:
        pass   # decay / TTL management can be added here

    def _k_nearest(self, agent_id: str) -> list[str]:
        pos   = self._positions.get(agent_id, (0.0, 0.0))
        peers = [(aid, p) for aid, p in self._positions.items() if aid != agent_id]
        peers.sort(key=lambda ap: (ap[1][0]-pos[0])**2 + (ap[1][1]-pos[1])**2)
        return [aid for aid, _ in peers[:self._k]]
