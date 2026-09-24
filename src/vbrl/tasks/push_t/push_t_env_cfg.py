from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
  CurriculumTermCfg,
  EventTermCfg,
  MetricsTermCfg,
  ObservationTermCfg,
  RewardTermCfg,
  SceneEntityCfg,
  TerminationTermCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from vbrl.asset_zoo.robots.definition import RobotDefinition
from vbrl.tasks.utils import EE_GROUND_CONTACT_SENSOR, make_tabletop_env_cfg

from . import mdp
from .geometry import HALF_HEIGHT, REST_HEIGHT
from .goal_marker import GOAL_ENTITY_NAME

_COMMAND = "push_t_goal"
_CONTACT_SENSOR = "ee_object_contact"
_OBJECT_TABLE_SENSOR = "object_table_contact"
ACTION_SCALE = 0.015
# The object's own height, so it follows the T. Linear, not quadratic: a constant
# gradient pulls the arm down from any height; a quadratic is weakest at the ceiling.
EE_HEIGHT_CEILING_M = 2.0 * HALF_HEIGHT
# Raising it is measured and reverted: -0.1 takes success to 0.008, and -0.05
# moved the lowest pad only 27.6 -> 26.2 mm for 8-19% of overlap.
EE_HEIGHT_WEIGHT = -0.02
# A starting size, not a measured one: the penalties cost ~3% of task reward, so this
# puts the bonus on the same scale.
SIDE_CONTACT_ALIGN_WEIGHT = 0.05
# No onset: the T is 24 mm tall, so avoiding the table is a height adjustment rather
# than a change of strategy, which is what makes it safe to apply hard.
# Both contact penalties are linear: a target-relative action integrates, so a wound-up
# target presses up to the torque limit, and a quadratic turned that into -170/s.
TABLE_CONTACT_ONSET_N = 0.0
TABLE_CONTACT_SCALE_N = 5.0
TABLE_CONTACT_WEIGHT = -0.01
# The printed object weighed 50.7 g, against the 172.8 g the MJCF used to carry; sliding
# distance goes as 1/m^2, so the old mass travelled 11.6x less for the same push.
OBJECT_WEIGHT_N = 0.497
OBJECT_PRESS_ONSET_N = OBJECT_WEIGHT_N + 1.0
OBJECT_PRESS_SCALE_N = 5.0
# 5.4% of task reward on the behaviour it corrects -- the largest penalty here,
# intended: it is the only one the policy can zero out without giving up the task.
OBJECT_PRESS_WEIGHT = -0.01
# `action_rate_l2` is upstream Lift-Cube's -0.01 at a fifth, because for a Gaussian
# policy consecutive actions differ by 2*sigma^2 even when the mean never moves.
ACTION_PATH_LENGTH_WEIGHT = -0.002
ACTION_RATE_WEIGHT = -0.002
# Cut post-success drift 55% (2.33 -> 1.05 mm per step). -0.2 was tried and is wrong:
# the term cannot distort behaviour *at* goal, but it lowers the goal state's value.
AT_GOAL_ACTION_WEIGHT = -0.05
# Terminating on forceful top contact is deliberately not wired in, though
# `mdp.forceful_top_contact` stays reachable.
# Environment steps at num_steps_per_env=16: pinned for 3,000 iterations, then 8
# rungs of 22.5 degrees every 250, full circle at 4,750. Both numbers are
# measured. The pin has to outlast incompetence -- at 1,500 iterations overlap
# was 0.051 and widening from there went nowhere -- and 45-degree rungs made yaw
# error worse in 8 of the 15 runs trained on them.
GOAL_YAW_STAGES = (
  {"step": 0, "half_range": 0.0},
  {"step": 48_000, "half_range": math.pi * 1 / 8},
  {"step": 52_000, "half_range": math.pi * 2 / 8},
  {"step": 56_000, "half_range": math.pi * 3 / 8},
  {"step": 60_000, "half_range": math.pi * 4 / 8},
  {"step": 64_000, "half_range": math.pi * 5 / 8},
  {"step": 68_000, "half_range": math.pi * 6 / 8},
  {"step": 72_000, "half_range": math.pi * 7 / 8},
  {"step": 76_000, "half_range": math.pi * 8 / 8},
)


_PRIVILEGED_ACTOR_TERMS = (
  "ee_to_object",
  "object_to_goal",
  "object_heading",
  "relative_yaw",
)


# Offset x ranges held 15 cm apart, so an episode never starts near the goal.
WORKSPACE_Y = (-0.2, 0.2)
OBJECT_X = (0.2, 0.4)
TARGET_X = (0.3, 0.5)
MIN_XY_SEPARATION = 0.15


def _command(
  object_name: str,
  success_threshold: float,
  goal_marker_name: str | None = None,
  fixed_target: tuple[float, float, float] | None = None,
  goal_observation_noise: tuple[float, float] = (0.0, 0.0),
) -> mdp.PushTCommandCfg:
  return mdp.PushTCommandCfg(
    goal_marker_name=goal_marker_name,
    entity_name=object_name,
    difficulty="dynamic",
    resampling_time_range=(1.0e9, 1.0e9),
    debug_vis=True,
    success_threshold=success_threshold,
    object_pose_range=mdp.PushTCommandCfg.ObjectPoseRangeCfg(
      x=OBJECT_X,
      y=WORKSPACE_Y,
      z=(REST_HEIGHT, REST_HEIGHT),
      yaw=(-math.pi, math.pi),
    ),
    target_position_range=mdp.PushTCommandCfg.TargetPositionRangeCfg(
      x=TARGET_X,
      y=WORKSPACE_Y,
      z=(HALF_HEIGHT, HALF_HEIGHT),
    ),
    target_yaw_range=(-math.pi, math.pi),
    min_xy_separation=MIN_XY_SEPARATION,
    fixed_target=fixed_target,
    observation_position_noise=goal_observation_noise[0],
    observation_yaw_noise=goal_observation_noise[1],
  )


def build_env_cfg(
  *,
  robot: RobotDefinition,
  object_name: str,
  rgb: bool = False,
  play: bool = False,
  success_threshold: float = 0.90,
  goal_yaw_stages: Sequence[Mapping[str, float]] | None = None,
  visual_goal: bool = False,
  goal_in_observation: bool = True,
  fixed_target: tuple[float, float, float] | None = None,
  action_scale: float = ACTION_SCALE,
  episode_length_s: float = 16.0,
  goal_outline: bool = False,
  goal_observation_noise: tuple[float, float] = (0.0, 0.0),
) -> ManagerBasedRlEnvCfg:
  cfg = make_tabletop_env_cfg(
    robot, action_delay=True, fixed_closed_gripper=True
  )
  robot_ee = SceneEntityCfg("robot", site_names=(robot.ee_site,))
  common = {"command_name": _COMMAND, "object_name": object_name}

  base_terms = cfg.observations["actor"].terms
  terms = {
    "joint_pos": base_terms["joint_pos"],
    "joint_vel": base_terms["joint_vel"],
    "joint_target": ObservationTermCfg(func=mdp.joint_target),
    "ee_to_object": ObservationTermCfg(
      func=mdp.ee_to_object_distance,
      params={"object_name": object_name, "asset_cfg": robot_ee},
      noise=Unoise(n_min=-0.01, n_max=0.01),
      clip=(-2.0, 2.0),
    ),
    "object_to_goal": ObservationTermCfg(
      func=mdp.object_to_goal_distance,
      params={**common, "asset_cfg": SceneEntityCfg("robot")},
      noise=Unoise(n_min=-0.01, n_max=0.01),
      clip=(-2.0, 2.0),
    ),
    "object_heading": ObservationTermCfg(
      func=mdp.object_heading,
      params={"object_name": object_name},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "relative_yaw": ObservationTermCfg(
      func=mdp.relative_yaw,
      params=common,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "actions": base_terms["actions"],
  }
  cfg.observations["actor"].terms = terms
  cfg.observations["critic"].terms = {**terms}
  cfg.observations["actor"].nan_policy = "sanitize"
  cfg.observations["critic"].nan_policy = "sanitize"

  cfg.actions = {
    "joint_pos": mdp.TargetRelativeJointPositionActionCfg(
      entity_name="robot",
      actuator_names=robot.arm_actuator_names,
      scale=action_scale,
      preserve_order=True,
    )
  }
  cfg.commands = {
    _COMMAND: _command(
      object_name,
      success_threshold,
      goal_marker_name=GOAL_ENTITY_NAME if visual_goal else None,
      fixed_target=fixed_target,
      goal_observation_noise=goal_observation_noise,
    )
  }
  cfg.rewards = {
    "maniskill_dense": RewardTermCfg(
      func=mdp.maniskill_dense_reward,
      weight=1.0,
      params={**common, "asset_cfg": robot_ee},
    ),
    "side_contact_align": RewardTermCfg(
      func=mdp.side_contact_align,
      weight=SIDE_CONTACT_ALIGN_WEIGHT,
      params={"sensor_name": _CONTACT_SENSOR},
    ),
    "action_path_length": RewardTermCfg(
      func=mdp.action_path_length_l1,
      weight=ACTION_PATH_LENGTH_WEIGHT,
    ),
    "action_rate_l2": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=ACTION_RATE_WEIGHT,
    ),
    "at_goal_action": RewardTermCfg(
      func=mdp.at_goal_action_l1,
      weight=AT_GOAL_ACTION_WEIGHT,
      params={"command_name": _COMMAND},
    ),
    "table_contact_force": RewardTermCfg(
      func=mdp.contact_force_hinge,
      weight=TABLE_CONTACT_WEIGHT,
      params={
        "sensor_name": EE_GROUND_CONTACT_SENSOR,
        "onset": TABLE_CONTACT_ONSET_N,
        "scale": TABLE_CONTACT_SCALE_N,
      },
    ),
    "object_table_press": RewardTermCfg(
      func=mdp.object_table_press,
      weight=OBJECT_PRESS_WEIGHT,
      params={
        "sensor_name": _OBJECT_TABLE_SENSOR,
        "onset": OBJECT_PRESS_ONSET_N,
        "scale": OBJECT_PRESS_SCALE_N,
      },
    ),
    "ee_height_ceiling": RewardTermCfg(
      func=mdp.fingertip_height_excess,
      weight=EE_HEIGHT_WEIGHT,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", geom_names=robot.fingertip_geom_pattern
        ),
        "ceiling": EE_HEIGHT_CEILING_M,
        "sensor_name": _CONTACT_SENSOR,
      },
    ),
  }
  # Hardware-safety readouts. Metrics carry no weight and never enter the return.
  cfg.metrics = {
    "peak_table_force": MetricsTermCfg(
      func=mdp.max_contact_force,
      reduce="max",
      params={"sensor_name": EE_GROUND_CONTACT_SENSOR},
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
  if goal_yaw_stages is not None:
    cfg.curriculum["goal_yaw_range"] = CurriculumTermCfg(
      func=mdp.goal_yaw_curriculum,
      params={"command_name": _COMMAND, "stages": list(goal_yaw_stages)},
    )
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
  for sensor in cfg.scene.sensors:
    if sensor.name == EE_GROUND_CONTACT_SENSOR:
      sensor.reduce = "maxforce"
  cfg.episode_length_s = episode_length_s
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
    actor = cfg.observations["actor"]
    for name in _PRIVILEGED_ACTOR_TERMS:
      actor.terms.pop(name)
    target_pose_term = ObservationTermCfg(
      func=mdp.target_pose,
      params={
        "command_name": _COMMAND,
        "asset_cfg": SceneEntityCfg("robot"),
      },
      clip=(-2.0, 2.0),
    )
    cfg.observations["critic"].terms["target_pose"] = target_pose_term
    if goal_in_observation:
      actor.terms["target_pose"] = target_pose_term

  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
  return cfg


__all__ = ["build_env_cfg"]
