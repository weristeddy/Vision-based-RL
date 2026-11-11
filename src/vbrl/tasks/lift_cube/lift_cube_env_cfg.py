from __future__ import annotations

import copy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
  CurriculumTermCfg,
  EventTermCfg,
  ObservationTermCfg,
  RewardTermCfg,
  SceneEntityCfg,
  TerminationTermCfg,
)
from mjlab.viewer import ViewerConfig

from vbrl.asset_zoo.robots.definition import RobotDefinition
from vbrl.tasks.utils import EE_GROUND_CONTACT_SENSOR, make_tabletop_env_cfg

from . import mdp

_COMMAND = "lift_height"
DECIMATION = 10
ACTION_SCALE = 0.025
WRIST_ROLL_SCALE = 0.02
RAMP_S = 0.05
EPISODE_LENGTH_S = 20.0
ACTION_RATE_WEIGHT = -0.1
JOINT_VEL_WEIGHT = -0.05
POSTURE_WEIGHT = 1.0
POSTURE_STD = 0.5
# After every absolute-gripper run had learned the grasp (iteration 200-400); from
# the start, b36q4eo3 hovered at the cube near home and never grasped.
POSTURE_START_ITERATION = 500
STEPS_PER_ITERATION = 24
OBJECT_REST_HEIGHT = 0.02


def build_env_cfg(
  *,
  robot: RobotDefinition,
  object_name: str,
  rgb: bool = False,
  distill: bool = False,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_tabletop_env_cfg(robot, action_delay=True)
  arm = robot.arm_actuator_names
  gripper = next(name for name in robot.action_scale if name not in arm)
  robot_ee = SceneEntityCfg("robot", site_names=(robot.ee_site,))
  common = {"command_name": _COMMAND, "object_name": object_name}

  terms = {
    "qpos": ObservationTermCfg(func=mdp.qpos),
    "qvel": ObservationTermCfg(func=mdp.qvel),
    "target_qpos": ObservationTermCfg(func=mdp.target_qpos),
    "tcp_pose": ObservationTermCfg(func=mdp.tcp_pose, params={"asset_cfg": robot_ee}),
    "goal_position": ObservationTermCfg(
      func=mdp.target_position,
      params={"command_name": _COMMAND, "asset_cfg": robot_ee},
    ),
    "obj_pose": ObservationTermCfg(
      func=mdp.obj_pose, params={"object_name": object_name}
    ),
  }
  cfg.observations["actor"].terms = terms
  cfg.observations["critic"].terms = {
    **terms,
    "ee_to_cube": ObservationTermCfg(
      func=mdp.ee_to_object_distance,
      params={"object_name": object_name, "asset_cfg": robot_ee},
    ),
    "cube_to_goal": ObservationTermCfg(func=mdp.object_to_goal_distance, params=common),
  }
  if distill:
    cfg.observations["teacher"] = copy.deepcopy(cfg.observations["actor"])

  cfg.actions = {
    "joint_pos": mdp.TargetRelativeJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(*arm, gripper),
      scale=dict.fromkeys(arm, ACTION_SCALE),
      wrist_roll_scale=WRIST_ROLL_SCALE,
      absolute_names=(gripper,),
      ramp_s=RAMP_S,
      preserve_order=True,
    )
  }
  cfg.commands = {
    _COMMAND: mdp.LiftingCommandCfg(
      entity_name=object_name,
      resampling_time_range=(1.0e9, 1.0e9) if play else (8.0, 12.0),
      debug_vis=True,
      difficulty="dynamic",
      object_pose_range=mdp.LiftingCommandCfg.ObjectPoseRangeCfg(
        z=(OBJECT_REST_HEIGHT, 0.05)
      ),
    )
  }
  cfg.rewards = {
    "lift": RewardTermCfg(
      func=mdp.staged_position_reward,
      weight=1.0,
      params={
        **common,
        "reaching_std": 0.2,
        "bringing_std": 0.3,
        "asset_cfg": robot_ee,
      },
    ),
    "lift_precise": RewardTermCfg(
      func=mdp.bring_object_reward, weight=1.0, params={**common, "std": 0.05}
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=ACTION_RATE_WEIGHT),
    "joint_vel_l2": RewardTermCfg(
      func=mdp.joint_vel_l2,
      weight=JOINT_VEL_WEIGHT,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=arm)
      },
    ),
    "posture": RewardTermCfg(
      func=mdp.posture,
      weight=0.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=arm),
        "std": {".*": POSTURE_STD},
      },
    ),
    "joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
  }
  cfg.terminations.update(
    ee_ground_collision=TerminationTermCfg(
      func=mdp.illegal_contact,
      params={"sensor_name": EE_GROUND_CONTACT_SENSOR, "force_threshold": 10.0},
    ),
    nan_detection=TerminationTermCfg(func=mdp.nan_detection),
  )
  cfg.curriculum = {
    "posture_weight": CurriculumTermCfg(
      func=mdp.reward_weight_from_step,
      params={
        "reward_name": "posture",
        "step": POSTURE_START_ITERATION * STEPS_PER_ITERATION,
        "weight": POSTURE_WEIGHT,
      },
    )
  }
  cfg.events["arm_joint_position_noise"] = EventTermCfg(
    func=mdp.reset_joints_with_gaussian_offset,
    mode="reset",
    params={
      "position_std": mdp.ROBOT_JOINT_POSITION_STD_RAD,
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=arm, preserve_order=True
      ),
    },
  )
  for suffix, axis, distribution, ranges in (
    ("slide", 0, "uniform", (0.3, 1.5)),
    ("spin", 1, "log_uniform", (1e-4, 2e-2)),
    ("roll", 2, "log_uniform", (1e-5, 5e-3)),
  ):
    cfg.events[f"fingertip_friction_{suffix}"] = EventTermCfg(
      func=mdp.dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=robot.fingertip_geom_pattern),
        "operation": "abs",
        "distribution": distribution,
        "axes": [axis],
        "ranges": ranges,
      },
    )
  cfg.episode_length_s = EPISODE_LENGTH_S
  cfg.decimation = DECIMATION
  cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_ROOT
  cfg.viewer.entity_name = "robot"
  cfg.viewer.max_extra_envs = 0
  cfg.viewer.distance = 1.45
  cfg.viewer.elevation = -30.0
  cfg.viewer.azimuth = 150.0

  if rgb:
    cfg.observations["actor"].terms.pop("obj_pose")
  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
  return cfg


__all__ = ["OBJECT_REST_HEIGHT", "STEPS_PER_ITERATION", "build_env_cfg"]
