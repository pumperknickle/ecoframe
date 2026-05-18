"""
Phase 0 validation: protocol types work with zero non-stdlib deps.
"""
import numpy as np
import pytest
from ecoframe.protocol import (
    SensorBundle, ActionBundle, Session,
    SensorSpec, SensorManifest, TrainingMetrics,
    CapacityError,
)


def test_sensor_spec_prediction_target():
    cam = SensorSpec("visual", (64, 64, 3), action_affected=True, world_external=True)
    ego = SensorSpec("proprioceptive", (5,), action_affected=False, world_external=False)
    assert cam.is_prediction_target
    assert not ego.is_prediction_target


def test_manifest_prediction_targets():
    manifest = SensorManifest("test_env", sensors=(
        SensorSpec("visual",         (64, 64, 3), action_affected=True,  world_external=True),
        SensorSpec("audio",          (1024,),     action_affected=True,  world_external=True),
        SensorSpec("proprioceptive", (5,),        action_affected=False, world_external=False),
    ))
    targets = manifest.prediction_targets
    assert len(targets) == 2
    assert all(s.name in ("visual", "audio") for s in targets)
    # proprioceptive must never be a prediction target
    assert all(s.name != "proprioceptive" for s in targets)


def test_manifest_hash_stable():
    m1 = SensorManifest("env", sensors=(
        SensorSpec("visual", (64, 64, 3)),
    ))
    m2 = SensorManifest("env", sensors=(
        SensorSpec("visual", (64, 64, 3)),
    ))
    assert m1.hash == m2.hash


def test_manifest_hash_differs_on_shape_change():
    m1 = SensorManifest("env", sensors=(SensorSpec("visual", (64, 64, 3)),))
    m2 = SensorManifest("env", sensors=(SensorSpec("visual", (128, 128, 3)),))
    assert m1.hash != m2.hash


def test_sensor_bundle_defaults():
    b = SensorBundle()
    assert b.visual is None
    assert b.audio is None
    assert b.done is False
    assert b.reward == 0.0
    assert b.proprioceptive.shape == (0,)


def test_sensor_bundle_has():
    b = SensorBundle(visual=np.zeros((64, 64, 3), dtype=np.uint8))
    assert b.has("visual")
    assert not b.has("audio")


def test_session_ssm_state_preserved():
    state = {"layer0": np.ones((1, 32))}
    session = Session(
        brain_id="brain0", env_id="env0",
        agent_id="agent0", ssm_state=state,
    )
    assert session.ssm_state is state   # same object — no copy


def test_action_bundle_defaults():
    a = ActionBundle()
    assert a.continuous is None
    assert a.discrete is None


def test_capacity_error_is_exception():
    with pytest.raises(CapacityError):
        raise CapacityError("env is full")


def test_training_metrics_defaults():
    m = TrainingMetrics()
    assert m.loss == 0.0
    assert m.sensor_freq == 1.0
