"""
MockRemoteBackend: LocalBackend with artificial latency.

Used for testing distributed field behavior without a real message bus.
Agent code requires zero changes when swapping from 'local' to 'mock_remote'.
"""
from __future__ import annotations

import time
from .local import LocalBackend


class MockRemoteBackend(LocalBackend):
    def __init__(self, neighbor_k: int = 4, relay_ttl: int = 3,
                 latency_ms: float = 1.0):
        super().__init__(neighbor_k=neighbor_k, relay_ttl=relay_ttl)
        self._latency_s = latency_ms / 1000.0

    def _sleep(self):
        if self._latency_s > 0:
            time.sleep(self._latency_s)

    def publish(self, agent_id, signal):
        self._sleep()
        super().publish(agent_id, signal)

    def receive(self, agent_id):
        self._sleep()
        return super().receive(agent_id)

    def register_agent(self, agent_id, pos=(0.0, 0.0)):
        self._sleep()
        super().register_agent(agent_id, pos)
