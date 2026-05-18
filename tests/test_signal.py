"""Tests for Signal types including new EnvironmentSignal."""
import pytest
from ecoframe.signal import CuriositySignal, EnvironmentSignal, Signal


def test_curiosity_signal_channels():
    sig = CuriositySignal(position=(0.0, 0.0), timestamp=1, publisher="a0",
                          visual=1.5, body=0.3)
    assert sig.R == pytest.approx(1.5 + 0.3)


def test_environment_signal_fields():
    sig = EnvironmentSignal(
        position=(1.0, 2.0), timestamp=10, publisher="env0",
        curiosity=3.2, load_fraction=0.25, difficulty=0.1,
        address="host:5555", env_type="metadrive_roundabout",
    )
    assert sig.curiosity == 3.2
    assert sig.address == "host:5555"
    assert sig.env_type == "metadrive_roundabout"


def test_signal_rejects_tensor():
    try:
        import torch
        sig = CuriositySignal(position=(0.0, 0.0), timestamp=1, publisher="a0")
        with pytest.raises(TypeError):
            sig.visual = torch.tensor(1.0)
    except ImportError:
        pytest.skip("torch not installed")


def test_signal_rejects_array():
    import numpy as np
    sig = CuriositySignal(position=(0.0, 0.0), timestamp=1, publisher="a0")
    with pytest.raises(TypeError):
        sig.visual = np.array(1.0)
