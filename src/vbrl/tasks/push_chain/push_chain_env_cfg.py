from __future__ import annotations

import copy
import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
  EventTermCfg,
  MetricsTermCfg,
  ObservationTermCfg,
  RewardTermCfg,
  SceneEntityCfg,
  TerminationTermCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.viewer import ViewerConfig

from vbrl.asset_zoo.robots.definition import RobotDefinition
from vbrl.tasks.push_t.push_t_env_cfg import (
  ACTION_RATE_WEIGHT,
  ACTION_SCALE,
  AT_GOAL_STATIC_WEIGHT,
  DECIMATION,
  EE_VERTICAL_WEIGHT,
  JOINT_VEL_WEIGHT,
  RAMP_S,
  SUCCESS_THRESHOLD,
  TABLE_TOUCH_WEIGHT,
  WORKSPACE_X,
  WORKSPACE_Y,
  WRIST_ROLL_SCALE,
)
from vbrl.tasks.utils import make_tabletop_env_cfg

from . import mdp
from .geometry import REST_HEIGHT
from .goal_marker import GOAL_ENTITY_NAMES

COMMAND = "push_chain_goal"
_CONTACT_SENSOR = "ee_object_contact"
_TABLE_SENSOR = "robot_table_contact"
EPISODE_LENGTH_S = 15.0
LANDMARK_SCALE = 0.10
NCONMAX = 128


def _command(
  object_name: str,
  ee_site: str,
  rgb: bool,
  goal_observation_noise: tuple[float, float],
) -> mdp.PushChainCommandCfg:
  return mdp.PushChainCommandCfg(
    entity_name=object_name,
    ee_site=ee_site,
    goal_marker_names=GOAL_ENTITY_NAMES if rgb else (),
    resampling_time_range=(1.0e9, 1.0e9),
    debug_vis=True,
    success_threshold=SUCCESS_THRESHOLD,
    object_pose_range=mdp.PushChainCommandCfg.ObjectPoseRangeCfg(
      x=WORKSPACE_X,
      y=WORKSPACE_Y,
      z=(REST_HEIGHT, REST_HEIGHT),
      yaw=(-math.pi, math.pi),
    ),
    observation_position_noise=goal_observation_noise[0],
    observation_yaw_noise=goal_observation_noise[1],
  )


def build_env_cfg(
  *,
  robot: RobotDefinition,
  object_name: str,
  rgb: bool = False,
  distill: bool = False,
  play: bool = False,
  goal_observation_noise: tuple[float, float] = (0.0, 0.0),
) -> ManagerBasedRlEnvCfg:
  cfg = make_tabletop_env_cfg(robot, action_delay=True, fixed_closed_gripper=True)
  robot_ee = SceneEntityCfg("robot", site_names=(robot.ee_site,))
  arm = SceneEntityCfg("robot", joint_names=robot.arm_actuator_names)
  command = {"command_name": COMMAND}

  terms = {
    "qpos": ObservationTermCfg(func=mdp.qpos),
    "qvel": ObservationTermCfg(func=mdp.qvel),
    "target_qpos": ObservationTermCfg(func=mdp.target_qpos),
    "tcp_pose": ObservationTermCfg(func=mdp.tcp_pose, params={"asset_cfg": robot_ee}),
    "target_pose": ObservationTermCfg(func=mdp.target_pose, params={**command}),
    "obj_pose": ObservationTermCfg(func=mdp.chain_pose, params={**command}),
  }
  cfg.observations["actor"].terms = terms
  cfg.observations["critic"].terms = {
    **terms,
    "target_pose": ObservationTermCfg(
      func=mdp.target_pose, params={**command, "calibration_bias": False}
    ),
    "chain_errors": ObservationTermCfg(func=mdp.chain_errors, params={**command}),
    "link_overlaps": ObservationTermCfg(func=mdp.link_overlaps, params={**command}),
    "tcp_to_chain": ObservationTermCfg(func=mdp.tcp_to_chain, params={**command}),
  }
  cfg.observations["actor"].nan_policy = "sanitize"
  cfg.observations["critic"].nan_policy = "sanitize"
  if distill:
    teacher = copy.deepcopy(cfg.observations["actor"])
    teacher.terms["target_pose"].params["calibration_bias"] = False
    cfg.observations["teacher"] = teacher

  cfg.actions = {
    "joint_pos": mdp.TargetRelativeJointPositionActionCfg(
      entity_name="robot",
      actuator_names=robot.arm_actuator_names,
      scale=ACTION_SCALE,
      wrist_roll_scale=WRIST_ROLL_SCALE,
      ramp_s=RAMP_S,
      preserve_order=True,
    )
  }
  cfg.commands = {
    COMMAND: _command(object_name, robot.ee_site, rgb, goal_observation_noise)
  }
  cfg.rewards = {
    "chain_dense": RewardTermCfg(
      func=mdp.chain_dense_reward,
      weight=1.0,
      params={**command, "distance_scale": LANDMARK_SCALE},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=ACTION_RATE_WEIGHT),
    "joint_vel_l2": RewardTermCfg(
      func=mdp.joint_vel_l2, weight=JOINT_VEL_WEIGHT, params={"asset_cfg": arm}
    ),
    "at_goal_static": RewardTermCfg(
      func=mdp.at_goal_static,
      weight=AT_GOAL_STATIC_WEIGHT,
      params={**command, "asset_cfg": arm},
    ),
    "table_touch": RewardTermCfg(
      func=mdp.table_touch,
      weight=TABLE_TOUCH_WEIGHT,
      params={"sensor_name": _TABLE_SENSOR},
    ),
    "ee_vertical": RewardTermCfg(
      func=mdp.ee_tilt, weight=EE_VERTICAL_WEIGHT, params={"asset_cfg": robot_ee}
    ),
  }
  cfg.metrics = {
    "peak_top_face_force": MetricsTermCfg(
      func=mdp.max_contact_force_on_face,
      reduce="max",
      params={"sensor_name": _CONTACT_SENSOR, "vertical": True},
    ),
    "peak_side_face_force": MetricsTermCfg(
      func=mdp.max_contact_force_on_face,
      reduce="max",
      params={"sensor_name": _CONTACT_SENSOR, "vertical": False},
    ),
  }
  cfg.terminations.update(
    object_off_table=TerminationTermCfg(
      func=mdp.object_off_table, params={"object_name": object_name}
    ),
    invalid_object_state=TerminationTermCfg(
      func=mdp.invalid_object_state, params={"object_name": object_name}
    ),
    nan_detection=TerminationTermCfg(func=mdp.nan_detection),
  )
  cfg.curriculum = {}
  cfg.events["arm_joint_position_noise"] = EventTermCfg(
    func=mdp.reset_joints_with_gaussian_offset,
    mode="reset",
    params={
      "position_std": mdp.ROBOT_JOINT_POSITION_STD_RAD,
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=robot.arm_actuator_names, preserve_order=True
      ),
    },
  )
  cfg.events["fingertip_friction_slide"] = EventTermCfg(
    func=mdp.dr.geom_friction,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=robot.fingertip_geom_pattern),
      "operation": "abs",
      "distribution": "uniform",
      "axes": [0],
      "ranges": (0.3, 1.5),
    },
  )
  cfg.events["object_friction"] = EventTermCfg(
    func=mdp.randomize_object_table_friction,
    mode="reset",
    params={
      "mean": mdp.OBJECT_TABLE_FRICTION_MEAN,
      "std": mdp.OBJECT_TABLE_FRICTION_STD,
      "object_asset_cfg": SceneEntityCfg(
        object_name, geom_names=mdp.OBJECT_COLLISION_GEOMS
      ),
      "table_asset_cfg": SceneEntityCfg("table", geom_names=mdp.TABLE_COLLISION_GEOM),
    },
  )
  hinges = SceneEntityCfg(object_name, joint_names=mdp.HINGE_JOINTS)
  for name, func in (
    ("hinge_friction", mdp.dr.joint_friction),
    ("hinge_damping", mdp.dr.joint_damping),
  ):
    cfg.events[name] = EventTermCfg(
      func=func,
      mode="reset",
      params={
        "asset_cfg": hinges,
        "operation": "scale",
        "distribution": "uniform",
        "ranges": mdp.HINGE_SCALE_RANGE,
      },
    )
  cfg.scene.sensors += (
    ContactSensorCfg(
      name=_CONTACT_SENSOR,
      primary=ContactMatch(
        mode="subtree", pattern=robot.collision_body_pattern, entity="robot"
      ),
      secondary=ContactMatch(
        mode="body", pattern=r"push_chain_link\d", entity=object_name
      ),
      fields=("found", "force", "normal"),
      reduce="maxforce",
      num_slots=1,
    ),
    ContactSensorCfg(
      name=_TABLE_SENSOR,
      primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
      secondary=ContactMatch(mode="body", pattern="table", entity="table"),
      fields=("found", "force"),
      reduce="maxforce",
      num_slots=1,
    ),
  )
  cfg.episode_length_s = EPISODE_LENGTH_S
  cfg.decimation = DECIMATION
  cfg.sim.nconmax = NCONMAX
  cfg.scale_rewards_by_dt = False
  cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_ROOT
  cfg.viewer.entity_name = "robot"
  cfg.viewer.max_extra_envs = 0
  cfg.viewer.distance = 1.45
  cfg.viewer.elevation = -42.0
  cfg.viewer.azimuth = 170.0
  cfg.viewer.width = 640
  cfg.viewer.height = 480

  if rgb:
    cfg.observations["actor"].terms.pop("obj_pose")
  if not play:
    cfg.commands[COMMAND].near_object_start = True
  if not (distill or play):
    cfg.terminations["top_face_contact"] = TerminationTermCfg(
      func=mdp.top_face_contact,
      params={"sensor_name": _CONTACT_SENSOR, "min_force_n": 1.0},
    )
  if play:
    cfg.observations["actor"].enable_corruption = False
  return cfg


__all__ = ["COMMAND", "EPISODE_LENGTH_S", "LANDMARK_SCALE", "build_env_cfg"]
