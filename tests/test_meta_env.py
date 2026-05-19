"""
Phase 3 validation: MetaEnvironment — brain always lives here.

Tests verify:
  1. enter() always succeeds (unlimited capacity)
  2. step_wait() returns SensorBundles for all registered brains
  3. Field gradient visible as env_field in proprioceptive info
  4. EnvironmentSignal from task env appears in field observation
  5. navigate() auto-enters brain into task env when curiosity > threshold
  6. CapacityError from task env is caught gracefully in navigate()
  7. exit() removes brain from sessions
  8. SSM state never reset (preserved across meta steps)
  9. META_MANIFEST prediction target = env_field (action_affected=True)
"""
import math
import numpy as np
import pytest
from unittest.mock import MagicMock

from ecoframe.meta_env import MetaEnvironment, META_MANIFEST
from ecoframe.field import Field
from ecoframe.signal import EnvironmentSignal
from ecoframe.protocol import Session, CapacityError, SensorBundle


@pytest.fixture
def meta():
    field = Field(backend='local')
    return MetaEnvironment(field=field, verbose=False)


def test_manifest_env_field_is_prediction_target():
    targets = META_MANIFEST.prediction_targets
    names   = [s.name for s in targets]
    assert "env_field" in names


def test_manifest_self_state_not_predicted():
    targets = META_MANIFEST.prediction_targets
    assert all(s.name != "self_state" for s in targets)


def test_enter_always_succeeds(meta):
    for i in range(10):
        session = meta.enter(f"brain_{i}")
        assert isinstance(session, Session)
    assert len(meta._sessions) == 10


def test_enter_preserves_ssm_state(meta):
    state   = {"layer0": "my_state"}
    session = meta.enter("brain_x", ssm_state=state)
    assert session.ssm_state is state


def test_exit_removes_brain(meta):
    session = meta.enter("brain_z")
    assert "brain_z" in meta._sessions
    meta.exit(session)
    assert "brain_z" not in meta._sessions


def test_exit_returns_ssm_state(meta):
    state   = {"k": "v"}
    session = meta.enter("brain_r", ssm_state=state)
    returned = meta.exit(session)
    assert returned is state


def test_step_wait_returns_bundles_for_all_brains(meta):
    meta.enter("b0")
    meta.enter("b1")
    bundles = meta.step_wait()
    assert "b0" in bundles and "b1" in bundles


def test_step_wait_bundle_is_sensor_bundle(meta):
    meta.enter("b0")
    bundles = meta.step_wait()
    assert isinstance(bundles["b0"], SensorBundle)
    assert bundles["b0"].env_id == "meta"


def test_env_field_in_extra(meta):
    """env_field is a manifest-declared sensor — goes in extra, not info."""
    meta.enter("b0")
    bundles = meta.step_wait()
    extra = bundles["b0"].extra
    assert "env_field" in extra
    assert extra["env_field"].shape == (64,)


def test_info_does_not_contain_env_field(meta):
    """info is for non-sensor metadata only."""
    meta.enter("b0")
    bundles = meta.step_wait()
    assert "env_field" not in bundles["b0"].info


def test_environment_signal_visible_in_field(meta):
    """Task env's EnvironmentSignal appears in meta observations."""
    # Simulate a task env publishing to field
    meta._field.register_agent("task_env", pos=(1.0, 0.0))
    sig = EnvironmentSignal(
        position=(1.0, 0.0), timestamp=1, publisher="task_env",
        curiosity=3.5, load_fraction=0.25, env_type="metadrive_roundabout",
    )
    meta._field.publish("task_env", sig)

    meta.enter("brain_seeker", ssm_state={})
    bundles = meta.step_wait()

    info = bundles["brain_seeker"].info
    env_sigs = info.get("env_signals", [])
    types    = [s.env_type for s in env_sigs]
    assert "metadrive_roundabout" in types


def test_navigate_enters_env_above_threshold(meta):
    """Brain auto-enters env when curiosity > enter_threshold."""
    # Register a mock task env
    mock_env = MagicMock()
    mock_env.env_id = "task_env"
    mock_env.enter.return_value = Session(
        brain_id="brain_nav", env_id="task_env",
        agent_id="agent0", ssm_state={})
    meta.register_env(mock_env)

    # Publish high-curiosity signal for task_env
    meta._field.register_agent("task_env", pos=(0.5, 0.0))
    meta._field.publish("task_env", EnvironmentSignal(
        position=(0.5, 0.0), timestamp=1, publisher="task_env",
        curiosity=3.0,   # above enter_threshold=2.0
        load_fraction=0.1,
    ))

    # Mock brain object
    brain_obj = MagicMock()
    meta.enter("brain_nav")
    result = meta.navigate("brain_nav", "task_env", brain_obj=brain_obj)
    assert result == "task_env"
    brain_obj.enter.assert_called_once_with(mock_env)


def test_navigate_skips_env_below_threshold(meta):
    """Brain does NOT enter env when curiosity < enter_threshold."""
    mock_env = MagicMock()
    mock_env.env_id = "boring_env"
    meta.register_env(mock_env)

    meta._field.register_agent("boring_env", pos=(0.5, 0.0))
    meta._field.publish("boring_env", EnvironmentSignal(
        position=(0.5, 0.0), timestamp=1, publisher="boring_env",
        curiosity=0.5,   # below enter_threshold=2.0
    ))

    brain_obj = MagicMock()
    meta.enter("brain_bored")
    result = meta.navigate("brain_bored", "boring_env", brain_obj=brain_obj)
    assert result is None
    brain_obj.enter.assert_not_called()


def test_navigate_handles_capacity_error(meta):
    """CapacityError from task env is caught gracefully."""
    mock_env = MagicMock()
    mock_env.env_id = "full_env"
    mock_env.enter.side_effect = CapacityError("full")
    meta.register_env(mock_env)

    meta._field.register_agent("full_env", pos=(0.5, 0.0))
    meta._field.publish("full_env", EnvironmentSignal(
        position=(0.5, 0.0), timestamp=1, publisher="full_env",
        curiosity=5.0, load_fraction=1.0,
    ))

    brain_obj = MagicMock()
    brain_obj.enter.side_effect = CapacityError("full")
    meta.enter("brain_blocked")
    result = meta.navigate("brain_blocked", "full_env", brain_obj=brain_obj)
    assert result is None   # graceful handling, no crash


def test_ssm_never_reset_across_meta_steps(meta):
    state   = {"persistent": True}
    session = meta.enter("brain_p", ssm_state=state)
    for _ in range(10):
        meta.step_async({})
        meta.step_wait()
    # SSM state must still be the same object
    assert meta._sessions["brain_p"].ssm_state is state
