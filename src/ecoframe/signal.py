"""
Signal type system for EcoFrame.

The key principle: inter-agent signals must be MEASUREMENTS (typed scalars
with fixed semantics), not representations (learned embeddings). The type
system makes this non-negotiable: attempting to set a channel to a tensor
raises TypeError immediately.

All Signal subclasses define their channel semantics at class creation time
via the `channels` ClassVar. Channel values are always Python floats.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import ClassVar, Any


@dataclass
class Signal:
    """
    Base class for all inter-agent signals. Schema fixed at definition time.

    Required fields (position, timestamp, publisher) must be provided at
    construction. Channel values are floats; tensor assignment is rejected.
    """
    position:  tuple[float, float]   # world coordinates (required)
    timestamp: int                    # step number (required)
    publisher: str                    # agent ID (required)

    channels: ClassVar[dict[str, str]] = {}  # subclass fills this

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__class__.channels:
            try:
                import torch
                if isinstance(value, torch.Tensor):
                    raise TypeError(
                        f"{self.__class__.__name__}.{name} must be a Python float, "
                        "not a tensor. Signals carry measurements, not representations."
                    )
            except ImportError:
                pass
            try:
                import numpy as np
                if isinstance(value, np.ndarray):
                    raise TypeError(
                        f"{self.__class__.__name__}.{name} must be a Python float, "
                        "not a numpy array. Signals carry measurements, not representations."
                    )
            except ImportError:
                pass
        object.__setattr__(self, name, value)

    def validate(self) -> None:
        """Raise TypeError if any channel value contains a tensor or array."""
        try:
            import torch
            for ch in self.__class__.channels:
                v = getattr(self, ch, None)
                if isinstance(v, torch.Tensor):
                    raise TypeError(
                        f"{self.__class__.__name__}.{ch} contains a tensor: {v}"
                    )
        except ImportError:
            pass

    @property
    def R(self) -> float:
        """Overall signal magnitude: sum of all channel values."""
        return sum(getattr(self, ch, 0.0) for ch in self.__class__.channels)


@dataclass
class CuriositySignal(Signal):
    """
    Standard curiosity signal for physics-based agents.

    Each channel is an R-1 value: positive means above that agent's EMA
    baseline, zero means at or below baseline. Binary contact is an exception.
    """
    channels: ClassVar[dict[str, str]] = {
        'visual':   'raw visual MSE prediction error (consequence-space measurement)',
        'body':     'raw proprioceptive MSE prediction error',
        'dynamics': 'raw dynamics prediction error',
        'kl':       'raw KL(posterior||prior) value',
        'contact':  'binary: physics contact event this step',
    }

    visual:   float = 0.0
    body:     float = 0.0
    dynamics: float = 0.0
    kl:       float = 0.0
    contact:  float = 0.0


@dataclass
class EnvironmentSignal(Signal):
    """
    Published by environments into the Field so brains can discover them.

    Brains navigate toward high curiosity + low load.
    No central registry — discovery is emergent from field gradients.
    """
    channels: ClassVar[dict[str, str]] = {
        'curiosity':     'mean brain CE EMA — how much brains are learning here',
        'load_fraction': 'current_agents / capacity',
        'difficulty':    'current config_head difficulty dial [0, 1]',
    }

    curiosity:     float = 0.0
    load_fraction: float = 0.0
    difficulty:    float = 0.0
    address:       str   = ""    # host:port to connect (EnvironmentProxy)
    env_type:      str   = ""    # "metadrive_roundabout", "navsim", etc.
    manifest_hash: str   = ""    # SensorManifest.hash — brain checks compatibility
