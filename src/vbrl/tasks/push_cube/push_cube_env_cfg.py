"""Planar Push-Cube built from MJLab's Lift-Cube primitives."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from vbrl.asset_zoo.robots.definition import RobotDefinition
from vbrl.tasks.utils import make_tabletop_env_cfg

from . import mdp
from .mdp import OBJECT_HALF_EXTENT


OBJECT_REST_HEIGHT = 0.02
_COMMAND = "push_goal"


def build_env_cfg(
  *,
  robot: RobotDefinition,
  object_name: str,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_tabletop_env_cfg(
    robot,
    fixed_closed_gripper=True,
    keep_rewards=("action_rate_l2", "joint_pos_limits", "joint_vel_hinge"),
  )
  regularization = dict(cfg.rewards)
  common = {"command_name": _COMMAND, "object_name": object_name}
  push_point = {
    **common,
    "object_half_extent": OBJECT_HALF_EXTENT,
    "asset_cfg": SceneEntityCfg("robot", site_names=(robot.ee_site,)),
  }

  base_terms = cfg.observations["actor"].terms
  terms = {
    "joint_pos": base_terms["joint_pos"],
    "joint_vel": base_terms["joint_vel"],
    "ee_to_push_point": ObservationTermCfg(
      func=mdp.ee_to_push_point,
      params=push_point,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "cube_to_goal": ObservationTermCfg(
      func=mdp.object_to_goal_distance,
      params=common,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "actions": base_terms["actions"],
  }
  cfg.observations["actor"].terms = terms
  cfg.observations["critic"].terms = {**terms}

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.actuator_names = robot.arm_actuator_names
  action.scale = dict(robot.arm_action_scale)

  command = mdp.PushingCommandCfg(
    entity_name=object_name,
    difficulty="dynamic",
    resampling_time_range=(8.0, 12.0),
    debug_vis=True,
    object_pose_range=mdp.PushingCommandCfg.ObjectPoseRangeCfg(
      x=(0.2, 0.4),
      y=(-0.2, 0.2),
      z=(OBJECT_REST_HEIGHT, 0.05),
      yaw=(-3.14, 3.14),
    ),
    target_position_range=mdp.PushingCommandCfg.TargetPositionRangeCfg(
      x=(0.3, 0.5),
      y=(-0.2, 0.2),
      z=(OBJECT_REST_HEIGHT, OBJECT_REST_HEIGHT),
    ),
    min_xy_separation=0.15,
  )
  cfg.commands = {_COMMAND: command}
  cfg.rewards = {
    "object_goal_distance": RewardTermCfg(
      func=mdp.object_goal_distance, weight=-1.0, params=common
    ),
    "ee_push_point_distance": RewardTermCfg(
      func=mdp.ee_push_point_distance, weight=-0.5, params=push_point
    ),
    "ee_table_contact": RewardTermCfg(
      func=mdp.illegal_contact,
      weight=-1.0,
      params={"sensor_name": "ee_ground_collision", "force_threshold": 10.0},
    ),
    **regularization,
  }

  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    command.resampling_time_range = (4.0, 4.0)
  return cfg


__all__ = ["build_env_cfg"]
