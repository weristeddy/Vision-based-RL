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
from vbrl.tasks.utils import make_tabletop_env_cfg

from . import mdp
from .geometry import HALF_HEIGHT, REST_HEIGHT
from .goal_marker import GOAL_ENTITY_NAME

_COMMAND = "push_t_goal"
_CONTACT_SENSOR = "ee_object_contact"
_OBJECT_TABLE_SENSOR = "object_table_contact"
_TABLE_SENSOR = "robot_table_contact"
# 20 Hz: one decision per arm response time. At 50 Hz the measured shoulder
# (110-200 ms lag, 4-14 mrad friction deadband) swallowed most per-step actions,
# and the same recipe reached 0.64 mean-action success against 0.96 here.
DECIMATION = 10
ACTION_SCALE = 0.025
WRIST_ROLL_SCALE = 0.02
RAMP_S = 0.05
EPISODE_LENGTH_S = 10.0
SUCCESS_THRESHOLD = 0.90
EE_VERTICAL_WEIGHT = -0.1
TABLE_TOUCH_WEIGHT = -1.0
# The printed object weighed 50.7 g, against the 172.8 g the MJCF used to carry; sliding
# distance goes as 1/m^2, so the old mass travelled 11.6x less for the same push.
OBJECT_WEIGHT_N = 0.497
ACTION_RATE_WEIGHT = -0.01
JOINT_VEL_WEIGHT = -0.05
AT_GOAL_STATIC_WEIGHT = 0.5


WORKSPACE_X = (0.2, 0.5)
WORKSPACE_Y = (-0.2, 0.2)


def _command(
  object_name: str,
  goal_marker_name: str | None = None,
  goal_observation_noise: tuple[float, float] = (0.0, 0.0),
) -> mdp.PushTCommandCfg:
  return mdp.PushTCommandCfg(
    goal_marker_name=goal_marker_name,
    entity_name=object_name,
    difficulty="dynamic",
    resampling_time_range=(1.0e9, 1.0e9),
    debug_vis=True,
    success_threshold=SUCCESS_THRESHOLD,
    object_pose_range=mdp.PushTCommandCfg.ObjectPoseRangeCfg(
      x=WORKSPACE_X,
      y=WORKSPACE_Y,
      z=(REST_HEIGHT, REST_HEIGHT),
      yaw=(-math.pi, math.pi),
    ),
    target_position_range=mdp.PushTCommandCfg.TargetPositionRangeCfg(
      x=WORKSPACE_X,
      y=WORKSPACE_Y,
      z=(HALF_HEIGHT, HALF_HEIGHT),
    ),
    target_yaw_range=(-math.pi, math.pi),
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
  cfg = make_tabletop_env_cfg(
    robot, action_delay=True, fixed_closed_gripper=True
  )
  robot_ee = SceneEntityCfg("robot", site_names=(robot.ee_site,))
  common = {"command_name": _COMMAND, "object_name": object_name}

  terms = {
    "qpos": ObservationTermCfg(func=mdp.qpos),
    "qvel": ObservationTermCfg(func=mdp.qvel),
    "target_qpos": ObservationTermCfg(func=mdp.target_qpos),
    "tcp_pose": ObservationTermCfg(func=mdp.tcp_pose, params={"asset_cfg": robot_ee}),
    "target_pose": ObservationTermCfg(
      func=mdp.target_pose,
      params={"command_name": _COMMAND, "asset_cfg": SceneEntityCfg("robot")},
    ),
    "obj_pose": ObservationTermCfg(
      func=mdp.obj_pose, params={"object_name": object_name}
    ),
  }
  cfg.observations["actor"].terms = terms
  cfg.observations["critic"].terms = {
    **terms,
    "target_pose": ObservationTermCfg(
      func=mdp.target_pose,
      params={
        "command_name": _COMMAND,
        "asset_cfg": SceneEntityCfg("robot"),
        "calibration_bias": False,
      },
    ),
    "relative_yaw": ObservationTermCfg(func=mdp.relative_yaw, params=common),
    "object_to_goal": ObservationTermCfg(
      func=mdp.object_to_goal_distance, params=common
    ),
    "ee_to_object": ObservationTermCfg(
      func=mdp.ee_to_object_distance,
      params={"object_name": object_name, "asset_cfg": robot_ee},
    ),
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
    _COMMAND: _command(
      object_name,
      goal_marker_name=GOAL_ENTITY_NAME if rgb else None,
      goal_observation_noise=goal_observation_noise,
    )
  }
  cfg.rewards = {
    "maniskill_dense": RewardTermCfg(
      func=mdp.maniskill_dense_reward,
      weight=1.0,
      params={
        **common,
        "asset_cfg": robot_ee,
      },
    ),
    "action_rate_l2": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=ACTION_RATE_WEIGHT,
    ),
    "joint_vel_l2": RewardTermCfg(
      func=mdp.joint_vel_l2,
      weight=JOINT_VEL_WEIGHT,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=robot.arm_actuator_names)
      },
    ),
    "at_goal_static": RewardTermCfg(
      func=mdp.at_goal_static,
      weight=AT_GOAL_STATIC_WEIGHT,
      params={
        "command_name": _COMMAND,
        "asset_cfg": SceneEntityCfg("robot", joint_names=robot.arm_actuator_names),
      },
    ),
    "table_touch": RewardTermCfg(
      func=mdp.table_touch,
      weight=TABLE_TOUCH_WEIGHT,
      params={"sensor_name": _TABLE_SENSOR},
    ),
    "ee_vertical": RewardTermCfg(
      func=mdp.ee_tilt,
      weight=EE_VERTICAL_WEIGHT,
      params={"asset_cfg": robot_ee},
    ),
  }
  # Hardware-safety readouts. Metrics carry no weight and never enter the return.
  cfg.metrics = {
    "peak_table_force": MetricsTermCfg(
      func=mdp.max_contact_force,
      reduce="max",
      params={"sensor_name": _TABLE_SENSOR},
    ),
    # A lateral push is bounded by the task (the T slides at ~0.7 N); a press
    # into the top face is bounded only by the arm.
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
    # Weight subtracted, so 0.0 means "not pressing".
    "peak_object_press": MetricsTermCfg(
      func=mdp.peak_object_press,
      reduce="max",
      params={
        "sensor_name": _OBJECT_TABLE_SENSOR,
        "weight_n": OBJECT_WEIGHT_N,
      },
    ),
    "top_contact_share": MetricsTermCfg(
      func=mdp.top_contact_share,
      reduce="mean",
      params={"sensor_name": _CONTACT_SENSOR},
    ),
    "at_goal_share": MetricsTermCfg(
      func=mdp.at_goal_share,
      reduce="mean",
      params={"command_name": _COMMAND},
    ),
  }
  cfg.terminations.update(
    object_off_table=TerminationTermCfg(
      func=mdp.object_off_table,
      params={"object_name": object_name},
    ),
    invalid_object_state=TerminationTermCfg(
      func=mdp.invalid_object_state,
      params={"object_name": object_name},
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
        "robot",
        joint_names=robot.arm_actuator_names,
        preserve_order=True,
      ),
    },
  )
  cfg.events["fingertip_friction_slide"] = EventTermCfg(
    func=mdp.dr.geom_friction,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", geom_names=robot.fingertip_geom_pattern
      ),
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
      "table_asset_cfg": SceneEntityCfg(
        "table", geom_names=mdp.TABLE_COLLISION_GEOM
      ),
    },
  )
  cfg.scene.sensors += (
    ContactSensorCfg(
      name=_CONTACT_SENSOR,
      primary=ContactMatch(
        mode="subtree",
        pattern=robot.collision_body_pattern,
        entity="robot",
      ),
      secondary=ContactMatch(
        mode="body",
        pattern="push_t",
        entity=object_name,
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
    # `netforce`: the exact quantity is the *total* vertical load the table
    # carries, not the largest of the T's several footprint contacts.
    ContactSensorCfg(
      name=_OBJECT_TABLE_SENSOR,
      primary=ContactMatch(mode="body", pattern="push_t", entity=object_name),
      secondary=ContactMatch(mode="geom", pattern="table_top", entity="table"),
      fields=("found", "force"),
      reduce="netforce",
    ),
  )
  cfg.episode_length_s = EPISODE_LENGTH_S
  cfg.decimation = DECIMATION
  cfg.scale_rewards_by_dt = False
  # Azimuth 180 is the +x side the robot faces. ASSET_ROOT, not the robot's
  # `viewer_body`: that body is the gripper, so the view would swing with the arm.
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
    cfg.commands[_COMMAND].near_object_start = True
  if not (distill or play):
    cfg.terminations["top_face_contact"] = TerminationTermCfg(
      func=mdp.top_face_contact,
      params={"sensor_name": _CONTACT_SENSOR, "min_force_n": 1.0},
    )

  if play:
    cfg.observations["actor"].enable_corruption = False
  return cfg


__all__ = ["build_env_cfg"]
