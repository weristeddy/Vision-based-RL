"""Robot contracts used by tasks with a fixed, closed gripper."""

from __future__ import annotations

import numpy as np
import pytest


pytest.importorskip("mjlab")


@pytest.mark.parametrize(
  "robot_name",
  ["trossen", "trossen_realistic", "yam"],
)
def test_robot_definitions_are_colocated_and_fresh(robot_name: str) -> None:
  from vbrl.asset_zoo.robots import ROBOTS, get_robot

  first = get_robot(robot_name)
  second = get_robot(robot_name)

  assert ROBOTS[robot_name]().name == robot_name
  assert first.xml_path.is_file()
  assert first is not second
  assert first.action_scale is not second.action_scale
  assert first.home_joint_pos is not second.home_joint_pos
  assert first.cameras is not second.cameras
  assert first == second


@pytest.mark.parametrize(
  "robot_name",
  ["trossen", "trossen_realistic", "yam"],
)
def test_arm_only_contract_has_six_actions_and_keeps_gripper_actuated(
  robot_name: str,
) -> None:
  from vbrl.asset_zoo.robots import get_robot

  robot = get_robot(robot_name)

  assert len(robot.action_scale) == 7
  assert len(robot.arm_action_scale) == 6
  assert robot.arm_actuator_names == tuple(robot.arm_action_scale)
  assert set(robot.arm_action_scale) < set(robot.action_scale)
  assert set(robot.arm_action_scale).isdisjoint(robot.closed_gripper_joint_pos)

  # The policy excludes the gripper actuator, but the robot retains it so its
  # position controller can physically hold the fingers at the closed target.
  entity_cfg = robot.make_entity_cfg(
    action_delay=False,
    fixed_closed_gripper=True,
  )
  actuator_targets = {
    target
    for actuator in entity_cfg.articulation.actuators
    for target in actuator.target_names_expr
  }
  gripper_action_names = set(robot.action_scale) - set(robot.arm_action_scale)
  assert gripper_action_names <= actuator_targets

  if robot_name == "yam":
    from mjlab.entity import Entity

    model = Entity(entity_cfg).spec.compile()
  else:
    model = entity_cfg.spec_fn().compile()
  assert model.nu == 7


@pytest.mark.parametrize(
  "robot_name",
  ["trossen", "trossen_realistic", "yam"],
)
def test_fixed_closed_gripper_does_not_change_default_robot_state(
  robot_name: str,
) -> None:
  from vbrl.asset_zoo.robots import get_robot

  robot = get_robot(robot_name)
  default_before = robot.make_entity_cfg(action_delay=False)
  default_joint_pos = dict(default_before.init_state.joint_pos)

  closed = robot.make_entity_cfg(
    action_delay=False,
    fixed_closed_gripper=True,
  )
  for joint_name, position in robot.closed_gripper_joint_pos.items():
    assert position == 0.0
    assert closed.init_state.joint_pos[joint_name] == 0.0

  # In particular, constructing the push variant must not mutate the shared
  # standalone robot defaults.
  default_after = robot.make_entity_cfg(action_delay=False)
  assert default_after.init_state.joint_pos == default_joint_pos
  assert any(
    default_joint_pos[joint_name] != 0.0
    for joint_name in robot.closed_gripper_joint_pos
  )


def test_action_scales_preserve_the_trained_policy_contract() -> None:
  from vbrl.asset_zoo.robots import get_robot

  trossen = {
    "joint_0": 0.25,
    "joint_1": 0.25,
    "joint_2": 0.25,
    "joint_3": 0.25,
    "joint_4": 0.25,
    "joint_5": 0.25,
    "left_carriage_joint": 0.01,
  }
  yam = {
    "joint1": 0.3599426554453139,
    "joint2": 0.1597918534090456,
    "joint3": 0.1904427157497401,
    "joint4": 0.5250193985879242,
    "joint5": 1.7347616639294616,
    "joint6": 5.520026131457555,
    "left_finger": 0.08657395590844981,
  }

  assert dict(get_robot("trossen").action_scale) == trossen
  assert dict(get_robot("trossen_realistic").action_scale) == trossen
  assert dict(get_robot("yam").action_scale) == yam


@pytest.mark.parametrize(
  ("robot_name", "action_delay", "expected_max_lag"),
  [
    ("trossen", False, 0),
    ("trossen", True, 1),
    ("trossen_realistic", False, 0),
    ("trossen_realistic", True, 1),
    ("yam", False, 0),
    ("yam", True, 0),
  ],
)
def test_actuator_delay_preserves_the_robot_and_modality_contract(
  robot_name: str,
  action_delay: bool,
  expected_max_lag: int,
) -> None:
  from vbrl.asset_zoo.robots import get_robot

  cfg = get_robot(robot_name).make_entity_cfg(action_delay=action_delay)
  assert {
    (actuator.delay_min_lag, actuator.delay_max_lag)
    for actuator in cfg.articulation.actuators
  } == {(0, expected_max_lag)}


def test_yam_collision_overlay_preserves_gripper_only_contact() -> None:
  import mujoco

  from mjlab.entity import Entity

  from vbrl.asset_zoo.robots import get_robot

  cfg = get_robot("yam").make_entity_cfg(action_delay=False)
  model = Entity(cfg).spec.compile()

  def geom_id(name: str) -> int:
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    assert index >= 0
    return index

  base = geom_id("base_collision")
  gripper = geom_id("link6_1_collision")
  fingertip = geom_id("lf_down6_collision")

  assert model.geom_contype[base] == 0
  assert model.geom_conaffinity[base] == 0
  assert model.geom_contype[gripper] == 1
  assert model.geom_conaffinity[gripper] == 1
  assert model.geom_condim[gripper] == 3
  assert model.geom_friction[gripper, 0] == pytest.approx(0.6)

  assert model.geom_condim[fingertip] == 6
  np.testing.assert_allclose(
    model.geom_friction[fingertip],
    (1.0, 5.0e-3, 5.0e-4),
  )
  np.testing.assert_allclose(model.geom_solref[fingertip], (0.01, 1.0))
  assert model.geom_priority[fingertip] == 1
