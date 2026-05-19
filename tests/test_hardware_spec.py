"""
HardwareSpec tests: data-only, no OS deps, pure protocol layer.
"""
import pytest
from ecoframe.protocol import HardwareSpec
from ecoframe.signal import EnvironmentSignal


def test_cpu_factory():
    s = HardwareSpec.cpu(n_workers=4)
    assert s.device_type   == "cpu"
    assert s.n_cpu_workers == 4
    assert s.memory_gb     == 0.0


def test_cuda_factory():
    s = HardwareSpec.cuda(device_id=1, memory_gb=8.0)
    assert s.device_type == "cuda"
    assert s.device_id   == 1
    assert s.memory_gb   == 8.0


def test_cpu_compatible_with_everything():
    cpu  = HardwareSpec.cpu()
    cuda = HardwareSpec.cuda(device_id=0)
    assert cpu.is_compatible_with(cuda)
    assert cuda.is_compatible_with(cpu)
    assert cpu.is_compatible_with(cpu)


def test_cuda_same_device_incompatible():
    a = HardwareSpec.cuda(device_id=0, memory_gb=4.0)
    b = HardwareSpec.cuda(device_id=0, memory_gb=4.0)
    assert not a.is_compatible_with(b)


def test_cuda_different_devices_compatible():
    a = HardwareSpec.cuda(device_id=0)
    b = HardwareSpec.cuda(device_id=1)
    assert a.is_compatible_with(b)
    assert b.is_compatible_with(a)


def test_cuda_visible_devices_cpu():
    s = HardwareSpec.cpu()
    assert s.cuda_visible_devices() == ""


def test_cuda_visible_devices_gpu():
    s = HardwareSpec.cuda(device_id=2)
    assert s.cuda_visible_devices() == "2"


def test_hardware_spec_is_frozen():
    s = HardwareSpec.cpu()
    with pytest.raises((TypeError, AttributeError)):
        s.device_type = "cuda"  # type: ignore


def test_hardware_spec_in_environment_signal():
    """EnvironmentSignal carries hardware fields from HardwareSpec."""
    spec = HardwareSpec.cuda(device_id=0, memory_gb=8.0)
    sig  = EnvironmentSignal(
        position=(0., 0.), timestamp=1, publisher="env0",
        curiosity=3.0,
        device_type=spec.device_type,
        device_id=spec.device_id,
        memory_gb=spec.memory_gb,
    )
    assert sig.device_type == "cuda"
    assert sig.device_id   == 0
    assert sig.memory_gb   == pytest.approx(8.0)


def test_hardware_aware_routing():
    """Brain on cuda:0 should only see envs on cpu or cuda:1+."""
    brain_spec = HardwareSpec.cuda(device_id=0, memory_gb=16.0)

    envs = [
        ("env_cpu",    HardwareSpec.cpu()),
        ("env_gpu0",   HardwareSpec.cuda(device_id=0, memory_gb=4.0)),
        ("env_gpu1",   HardwareSpec.cuda(device_id=1, memory_gb=4.0)),
    ]

    reachable = [
        name for name, spec in envs
        if brain_spec.is_compatible_with(spec)
    ]
    assert "env_cpu"  in reachable
    assert "env_gpu0" not in reachable   # same GPU as brain
    assert "env_gpu1" in reachable


def test_accelerator_field():
    s = HardwareSpec.cpu(accelerator="panda3d")
    assert s.accelerator == "panda3d"


def test_default_spec_is_cpu():
    s = HardwareSpec()
    assert s.device_type == "cpu"
    assert s.memory_gb   == 0.0
