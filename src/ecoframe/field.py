"""
Field: swappable communication backend for inter-agent and brain/environment gossip.

Usage:
    field = Field()                              # local in-memory
    field = Field(backend='mock_remote', latency_ms=2.0)  # simulated network

    field.publish(agent_id, signal)
    signals = field.query(pos=(x, z), radius=2.0)
    dx, dz  = field.gradient(agent_id)

Swap backend — zero agent or environment code changes:
    field = Field(backend='nats', url='nats://broker:4222')   # future
    field = Field(backend='redis', url='redis://cache:6379')  # future
    field = Field(backend='dht', bootstrap=['peer:8000'])     # future
"""
from __future__ import annotations

from ecoframe.signal import Signal


class Field:
    """
    Swappable gossip transport for curiosity and environment signals.

    All backends implement the same interface:
        register_agent, update_position, publish, receive,
        relay, gradient, update_trust_from_surprise, trust_variance, step
    """

    def __init__(self, backend: str = 'local', **kwargs):
        if backend == 'local':
            from ecoframe.backends.local import LocalBackend
            self._backend = LocalBackend(
                neighbor_k=kwargs.get('neighbor_k', 4),
                max_queue=kwargs.get('max_queue', 32),
                relay_ttl=kwargs.get('relay_ttl', 3),
            )
        elif backend == 'mock_remote':
            from ecoframe.backends.mock import MockRemoteBackend
            self._backend = MockRemoteBackend(
                neighbor_k=kwargs.get('neighbor_k', 4),
                relay_ttl=kwargs.get('relay_ttl', 3),
                latency_ms=kwargs.get('latency_ms', 1.0),
            )
        elif backend == 'nats':
            from ecoframe.backends.nats_backend import NatsBackend  # optional dep
            self._backend = NatsBackend(**kwargs)
        elif backend == 'redis':
            from ecoframe.backends.redis_backend import RedisBackend  # optional dep
            self._backend = RedisBackend(**kwargs)
        else:
            raise ValueError(
                f"Unknown backend: {backend!r}. "
                "Available: 'local', 'mock_remote'. "
                "Optional (install extras): 'nats', 'redis'."
            )
        self._backend_name = backend

    def register_agent(self, agent_id: str,
                       pos: tuple[float, float] = (0.0, 0.0)) -> None:
        self._backend.register_agent(agent_id, pos)

    def update_position(self, agent_id: str,
                        pos: tuple[float, float]) -> None:
        self._backend.update_position(agent_id, pos)

    def publish(self, agent_id: str, signal: Signal) -> None:
        self._backend.publish(agent_id, signal)

    def receive(self, agent_id: str) -> list[Signal]:
        return self._backend.receive(agent_id)

    def relay(self, agent_id: str, threshold: float = None) -> None:
        self._backend.relay(agent_id, threshold)

    def gradient(self, agent_id: str) -> tuple[float, float]:
        return self._backend.gradient(agent_id)

    def query(self, pos: tuple[float, float],
              radius: float = 2.0) -> list[Signal]:
        """Return all recently published signals within radius of pos."""
        import math
        results = []
        for aid, q in self._backend._published.items():
            for sig in q:
                sx, sz = sig.position
                px, pz = pos
                if math.sqrt((sx-px)**2 + (sz-pz)**2) <= radius:
                    results.append(sig)
        return results

    def update_trust_from_surprise(self, agent_id: str, surp: float) -> None:
        self._backend.update_trust_from_surprise(agent_id, surp)

    def trust_variance(self) -> float:
        return self._backend.trust_variance()

    def step(self) -> None:
        self._backend.step()

    @property
    def backend_name(self) -> str:
        return self._backend_name
