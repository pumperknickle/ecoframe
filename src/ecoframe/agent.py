from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Observation:
    data: Any
    timestamp: int = 0
    agent_id: str = ''

@dataclass
class Prediction:
    data: Any
    confidence: float = 1.0

@dataclass
class Action:
    data: Any
    motor_bias: tuple[float, float] = (0.0, 0.0)

@dataclass
class BrainState:
    hidden: Any = None
    step: int = 0


class EcoAgent:
    """
    Coordination protocol base for all EcoFrame agents.

    Carries only the invariants: field reference, surprise EMA, motor bias.
    No execution model is prescribed — observe/predict/act/surprise are
    conventions, not abstract contracts. Subclasses define what they need.

    The real framework is Signal + Field. This class is the minimal hook
    that connects an agent to those two things.
    """

    def __init__(self, agent_id: str = ''):
        self.agent_id      = agent_id or f'agent_{id(self)}'
        self._field        = None   # injected by Ecology at registration
        self._surprise_ema: float = 1.0
        self._step_count   = 0

    def motor_bias(self) -> tuple[float, float]:
        """Read gossip gradient from field. Returns (0, 0) before first gossip cycle."""
        if self._field is not None:
            return self._field.gradient(self.agent_id)
        return (0.0, 0.0)
