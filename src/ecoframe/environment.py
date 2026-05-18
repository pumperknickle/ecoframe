"""
Environment ABC: unified interface for WorldPool, EcoFrame ecologies, and Courses.

All training environments implement the same contract:
  capacity   — max agents that can inhabit this environment simultaneously
  step()     — advance one tick
  reset()    — optional per-agent or full reset
  enter()    — agent requests entry (may be refused if at capacity)
  exit()     — agent leaves
  emit()     — broadcast event to registered observers
  register_observer() — subscribe a callable to receive events
  contains   — nested sub-environments (for hierarchical compositions)

This enables MetaCourse composition: any Environment can contain other
Environments. CertProgressEvents flow upward through the observer chain
without any designer-prescribed routing.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Environment(ABC):
    """
    Abstract base class for all training environments.

    Design invariants:
    - capacity is relative to the environment (no global pool size constant)
    - emit() delegates to observers; no global event bus (local/relative)
    - enter/exit are soft gates; subclasses decide policy (default: always allow)
    - contains lists nested environments (tree structure, not graph)
    """

    @property
    @abstractmethod
    def capacity(self) -> int:
        """Maximum simultaneous agents. Derived from environment state, not hardcoded."""
        ...

    @abstractmethod
    def step(self, *args, **kwargs):
        """Advance the environment one tick. Returns per-agent results."""
        ...

    def reset(self, agent=None) -> None:
        """Reset environment or one agent's state. No-op by default."""
        pass

    def enter(self, agent) -> bool:
        """Agent requests entry. Returns True if admitted. Default: always admit."""
        return True

    def exit(self, agent) -> None:
        """Agent leaves the environment. No-op by default."""
        pass

    def emit(self, event: Any) -> None:
        """
        Broadcast event to all registered observers.

        Observer exceptions are logged (not swallowed) so signal failures
        are diagnosable. A broken observer must not silently kill signal flow
        for all subsequent observers in the chain.
        """
        import logging
        _log = logging.getLogger(__name__)
        for obs in getattr(self, '_observers', []):
            try:
                obs(event)
            except Exception as exc:
                _log.warning("Environment.emit: observer %r raised %s: %s",
                             obs, type(exc).__name__, exc)

    def register_observer(self, fn) -> None:
        """Subscribe a callable to receive events emitted by this environment."""
        if not hasattr(self, '_observers'):
            self._observers = []
        self._observers.append(fn)

    @property
    def contains(self) -> list:
        """Nested sub-environments. Empty list for leaf environments."""
        return getattr(self, '_sub_envs', [])
