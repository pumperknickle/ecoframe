"""
EcoFrame — standard brain/environment protocol for persistent-state AI agents.

Quick start:
    from ecoframe.protocol import (
        SensorBundle, ActionBundle, Session,
        SensorSpec, SensorManifest,
        EnvironmentProtocol, BrainProtocol,
    )
    from ecoframe.signal import CuriositySignal, EnvironmentSignal
    from ecoframe.field import Field
"""
from ecoframe.protocol import (
    SensorBundle, ActionBundle, Session,
    SensorSpec, SensorManifest, TrainingMetrics,
    LossHandle, HardwareSpec,
    EnvironmentProtocol, BrainProtocol,
    CapacityError, ManifestMismatchError,
)
from ecoframe.training_engine import TrainingEngine
from ecoframe.signal import Signal, CuriositySignal, EnvironmentSignal
from ecoframe.field import Field
from ecoframe.environment import Environment

__version__ = "0.1.0"
__all__ = [
    "SensorBundle", "ActionBundle", "Session",
    "SensorSpec", "SensorManifest", "TrainingMetrics",
    "LossHandle", "HardwareSpec",
    "EnvironmentProtocol", "BrainProtocol",
    "CapacityError", "ManifestMismatchError",
    "Signal", "CuriositySignal", "EnvironmentSignal",
    "Field",
    "Environment",
    "TrainingEngine",
]
