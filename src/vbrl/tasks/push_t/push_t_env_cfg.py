"""State and RGB Push-T configuration shared by every robot and scene."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg
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
_ACTION_DELTA = 0.1
VERTICAL_CONTACT_FORCE_CURRICULUM_STEP = 3200

# --- safety and motion shaping ----------------------------------------------
#
# Every term below regularizes *how* the arm moves; none of them touches the
# task. They are sized against the measured task margin: over a 250-step
# episode a do-nothing policy returns ~11-16 and a working one ~86, so the
# signal worth protecting is about 0.28 per step. A clean push pays under 5% of
# that, while thrashing, scratching and unsafe force grow steeply.

# The L1 travel penalty is the one term with a genuine do-nothing incentive, so
# it starts near zero and only bites once the policy can already push. 8,000
# steps is 500 iterations at the registered num_steps_per_env of 16.
MOTION_PENALTY_CURRICULUM_STEP = 8000
ACTION_PATH_LENGTH_WEIGHTS = (-0.001, -0.003)
# Half MJLab's own 10 N "illegal contact" level for this end-effector, with the
# normalizer chosen so 10 N costs exactly the -0.02 the retired binary
# `illegal_contact` reward paid. 20 N then costs -0.18 per step.
TABLE_CONTACT_ONSET_N = 5.0
TABLE_CONTACT_SCALE_N = 5.0
# Measured on this task over 64 envs: sustained full-scale commands peak at
# 3.1-4.4 rad/s, while per-step sign flips peak at 7.9-11.6. So 5 rad/s is zero
# for any decisive motion and only jitter crosses it. Mean joint speed does not
# separate the two (0.77 against 0.68) -- the peak does, which is why this is a
# hinge and not `joint_vel_l2`.
JOINT_SPEED_LIMIT_RAD_S = 5.0
# Goal-yaw schedule, in environment steps. A 3000-iteration run at
# num_steps_per_env=16 covers 48,000 steps, so the goal is fixed for the first
# 500 iterations and fully random for the last 1,000.
GOAL_YAW_CURRICULUM_STAGES = (
  {"step": 0, "half_range": 0.0},
  {"step": 8_000, "half_range": math.pi / 4},
  {"step": 16_000, "half_range": math.pi / 2},
  {"step": 24_000, "half_range": 3 * math.pi / 4},
  {"step": 32_000, "half_range": math.pi},
)
# The same idea, but the goal stays fixed for 3,000 iterations rather than 500,
# and then widens in 22.5-degree steps instead of 45.
#
# Both numbers are measured, not chosen. The long fixed phase is necessary: under
# the 500-iteration schedule every architecture except DINOv2 was still at
# 1.44-1.48 rad when widening began, so the fixed phase never did its job. But
# 45-degree rungs then undid it. Across the 15 runs trained on the coarse
# version, yaw error between the end of the fixed phase and the end of training
# got *worse* in 8, stayed level in 6, and improved in 1. DinoV2-Afa6 is the
# clearest loss: 0.155 rad on the fixed goal -- 9 degrees, essentially solved --
# collapsing to 1.510 by the end.
#
# Halving the step doubles the number of transitions but makes each a smaller
# distribution shift, and the full circle still arrives at iteration 4,750,
# leaving 1,250 of a 6,000-iteration run to consolidate.
GOAL_YAW_SLOW_STAGES = (
  {"step": 0, "half_range": 0.0},
  {"step": 48_000, "half_range": math.pi / 8},
  {"step": 52_000, "half_range": math.pi / 4},
  {"step": 56_000, "half_range": 3 * math.pi / 8},
  {"step": 60_000, "half_range": math.pi / 2},
  {"step": 64_000, "half_range": 5 * math.pi / 8},
  {"step": 68_000, "half_range": 3 * math.pi / 4},
  {"step": 72_000, "half_range": 7 * math.pi / 8},
  {"step": 76_000, "half_range": math.pi},
)
_PRIVILEGED_ACTOR_TERMS = (
  "ee_to_object",
  "object_to_goal",
  "object_heading",
  "relative_yaw",
)


# Object and goal are drawn from offset x ranges and held 15 cm apart, so an
# episode never *starts* near the goal. Since reaching 0.90 overlap requires fine
# adjustment at close range, the policy only meets those states after already
# transporting the T there -- and the sparse at-goal bonus, which replaces the
# whole reward with 3.0, therefore never fires early in training.
#
# `FREE_START_*` is the alternative: the object is drawn uniformly over one
# rectangle and the goal on a radius about it, redrawn whenever it lands off the
# rectangle. The floor is derived, not chosen -- at 1 cm the worst-case initial
# overlap is 0.834 against the 0.90 threshold, so no episode can begin already
# solved at any yaw, while 0 cm would begin at exactly 1.0. Drawing the radius
# uniformly puts more mass near the object than a uniform goal position would
# (area grows with the radius, so a uniform radius has density proportional to
# 1/r): 29% of episodes start inside 5 cm against 8% for a uniform goal. That is
# a smooth continuum, not the bimodal split a mixture or a schedule imposes, and
# it is the *same* mechanism the other two variants use -- they differ from this
# one only in the radius band.
WORKSPACE_X = (0.25, 0.45)
WORKSPACE_Y = (-0.2, 0.2)
FREE_START_X = WORKSPACE_X
FREE_START_MIN_SEPARATION = 0.01
# The rectangle's diagonal: the widest separation the workspace can hold. It is
# what an unbounded ceiling resolves to, and where the GrowStart ramp ends -- so
# GrowStart's final stretch is FreeStart itself rather than an approximation.
FREE_START_MAX_SEPARATION = math.hypot(
  WORKSPACE_X[1] - WORKSPACE_X[0], WORKSPACE_Y[1] - WORKSPACE_Y[0]
)
# Just short of the 0.90 overlap threshold, which needs roughly 5 mm and 5
# degrees together: at 6 mm and perfect alignment overlap is 0.891.
NEAR_GOAL_SEPARATION_RANGE = (0.006, 0.015)
NEAR_GOAL_YAW_RANGE = (math.radians(5.0), math.radians(20.0))
SEPARATION_CURRICULUM_ITERATIONS = 4000


def _command(
  object_name: str,
  success_threshold: float,
  goal_marker_name: str | None = None,
  free_start: bool = False,
  near_goal_probability: float = 0.0,
) -> mdp.PushTCommandCfg:
  object_x = FREE_START_X if free_start else (0.2, 0.4)
  target_x = FREE_START_X if free_start else (0.3, 0.5)
  separation = FREE_START_MIN_SEPARATION if free_start else 0.15
  return mdp.PushTCommandCfg(
    goal_marker_name=goal_marker_name,
    entity_name=object_name,
    difficulty="dynamic",
    resampling_time_range=(1.0e9, 1.0e9),
    debug_vis=True,
    success_threshold=success_threshold,
    object_pose_range=mdp.PushTCommandCfg.ObjectPoseRangeCfg(
      x=object_x,
      y=WORKSPACE_Y,
      z=(REST_HEIGHT, REST_HEIGHT),
      yaw=(-math.pi, math.pi),
    ),
    target_position_range=mdp.PushTCommandCfg.TargetPositionRangeCfg(
      x=target_x,
      y=WORKSPACE_Y,
      z=(HALF_HEIGHT, HALF_HEIGHT),
    ),
    target_yaw_range=(-math.pi, math.pi),
    min_xy_separation=separation,
    near_goal_probability=near_goal_probability,
    near_goal_separation_range=NEAR_GOAL_SEPARATION_RANGE,
    near_goal_yaw_range=NEAR_GOAL_YAW_RANGE,
  )


def build_env_cfg(
  *,
  robot: RobotDefinition,
  object_name: str,
  rgb: bool = False,
  play: bool = False,
  success_threshold: float = 0.98,
  goal_yaw_stages: Sequence[Mapping[str, float]] | None = None,
  quadratic_orientation: bool = False,
  visual_goal: bool = False,
  free_start: bool = False,
  near_goal_probability: float = 0.0,
  separation_curriculum: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build the ManiSkill3-inspired Push-T MDP.

  ``quadratic_orientation`` swaps only the orientation factor of the dense
  reward, for the variants that drop the goal-yaw curriculum and therefore start
  episodes anywhere on the circle.
  """
  cfg = make_tabletop_env_cfg(
    robot, action_delay=True, fixed_closed_gripper=True
  )
  robot_ee = SceneEntityCfg("robot", site_names=(robot.ee_site,))
  common = {"command_name": _COMMAND, "object_name": object_name}
  # The gripper is held closed, so the safety terms watch the arm joints only.
  arm_joints = robot.arm_actuator_names

  base_terms = cfg.observations["actor"].terms
  terms = {
    "joint_pos": base_terms["joint_pos"],
    "joint_vel": base_terms["joint_vel"],
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

  delta_clip = {
    name: (-_ACTION_DELTA, _ACTION_DELTA)
    for name in robot.arm_actuator_names
  }
  cfg.actions = {
    "joint_pos": RelativeJointPositionActionCfg(
      entity_name="robot",
      actuator_names=robot.arm_actuator_names,
      scale=_ACTION_DELTA,
      clip=delta_clip,
      preserve_order=True,
    )
  }
  # Drawing the target is what every published Push-T does; VBRL's default of
  # numbers-only is the deviation. The entity itself is installed by the caller,
  # which owns the scene.
  cfg.commands = {
    _COMMAND: _command(
      object_name,
      success_threshold,
      goal_marker_name=GOAL_ENTITY_NAME if visual_goal else None,
      free_start=free_start,
      near_goal_probability=near_goal_probability,
    )
  }
  cfg.rewards = {
    "maniskill_dense": RewardTermCfg(
      func=(
        mdp.quadratic_orientation_reward
        if quadratic_orientation
        else mdp.maniskill_dense_reward
      ),
      weight=1.0,
      params={**common, "asset_cfg": robot_ee},
    ),
    # Total commanded travel, which is what a forward/backward correction cycle
    # doubles. L1 so splitting one motion into many is never cheaper.
    "action_path_length": RewardTermCfg(
      func=mdp.action_path_length_l1,
      weight=ACTION_PATH_LENGTH_WEIGHTS[0],
    ),
    # Oscillation and jitter, on the raw command. Zero for a constant action.
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.005),
    # Jerky reversals specifically; weak because it correlates with the rate.
    "action_acc_l2": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.001),
    # Graded table force. Replaces a binary illegal_contact that paid the same
    # at 10 N as at 100 N, so nothing pushed the force back down.
    "table_contact_force": RewardTermCfg(
      func=mdp.contact_force_hinge,
      weight=-0.02,
      params={
        "sensor_name": EE_GROUND_CONTACT_SENSOR,
        "onset": TABLE_CONTACT_ONSET_N,
        "scale": TABLE_CONTACT_SCALE_N,
      },
    ),
    # Top-down contact on the T. Unchanged: |normal_z| * tanh(F/10) is already
    # ~1 for a press and ~0 for a clean side push.
    "vertical_contact_force": RewardTermCfg(
      func=mdp.vertical_contact_force,
      weight=0.0,
      params={"sensor_name": _CONTACT_SENSOR, "force_scale": 10.0},
    ),
    # Joint-limit protection for the real arm. A hinge on the *soft* limits, so
    # it is exactly zero anywhere inside them.
    "joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-0.25,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=arm_joints)},
    ),
    # Peak joint speed, hinged above anything a decisive push reaches.
    "joint_speed_hinge": RewardTermCfg(
      func=mdp.joint_velocity_hinge_penalty,
      weight=-0.001,
      params={
        "max_vel": JOINT_SPEED_LIMIT_RAD_S,
        "asset_cfg": SceneEntityCfg("robot", joint_names=arm_joints),
      },
    ),
  }
  # Peak forces in newtons, for judging whether this is safe on hardware.
  # Metrics carry no weight and never enter the return.
  cfg.metrics = {
    "peak_table_force": MetricsTermCfg(
      func=mdp.max_contact_force,
      reduce="max",
      params={"sensor_name": EE_GROUND_CONTACT_SENSOR},
    ),
    "peak_object_force": MetricsTermCfg(
      func=mdp.max_contact_force,
      reduce="max",
      params={"sensor_name": _CONTACT_SENSOR},
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
  cfg.curriculum = {
    "vertical_contact_force_weight": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "vertical_contact_force",
        "stages": [
          {"step": 0, "weight": 0.0},
          {"step": VERTICAL_CONTACT_FORCE_CURRICULUM_STEP, "weight": -0.05},
        ],
      },
    ),
    "action_path_length_weight": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "action_path_length",
        "stages": [
          {"step": 0, "weight": ACTION_PATH_LENGTH_WEIGHTS[0]},
          {
            "step": MOTION_PENALTY_CURRICULUM_STEP,
            "weight": ACTION_PATH_LENGTH_WEIGHTS[1],
          },
        ],
      },
    ),
  }
  if separation_curriculum:
    cfg.curriculum["separation_range"] = CurriculumTermCfg(
      func=mdp.separation_curriculum,
      params={
        "command_name": _COMMAND,
        "start": 0.05,
        "end": FREE_START_MAX_SEPARATION,
        "iterations": SEPARATION_CURRICULUM_ITERATIONS,
        # Overwritten from the rollout length at launch; see train.py. A literal
        # here is only the registered default and mistimes the ramp for any
        # other `num_steps_per_env`, because the term counts environment steps.
        "steps_per_iteration": 16,
        "pin_iterations": 0,
      },
    )
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
  )
  # The table term bounds peak force, so the one retained contact has to be the
  # strongest rather than an arbitrary one. Inherited from Lift-Cube as "none".
  for sensor in cfg.scene.sensors:
    if sensor.name == EE_GROUND_CONTACT_SENSOR:
      sensor.reduce = "maxforce"
  cfg.episode_length_s = 5.0
  cfg.scale_rewards_by_dt = False
  # Framing for the recorded training video and the Viser view, which share
  # `cfg.viewer`. Lift-Cube looks along the table at -5 degrees from 1.5 m,
  # which hides the T behind the arm and draws two neighbouring envs. Look down
  # on the workspace from the front instead -- azimuth 180 is the +x side the
  # robot faces -- and render the tracked env alone. Verified to show the robot,
  # the tabletop, the red T and the goal overlay together.
  # ASSET_ROOT rather than the robot's declared `viewer_body`: that body is the
  # gripper, so the view swings with the arm and the goal leaves frame. The
  # root is the fixed base, which keeps the whole workspace steady, and naming
  # the entity rather than a body keeps this robot-agnostic.
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
    actor.terms["target_pose"] = ObservationTermCfg(
      func=mdp.target_pose,
      params={
        "command_name": _COMMAND,
        "asset_cfg": SceneEntityCfg("robot"),
      },
      clip=(-2.0, 2.0),
    )

  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
  return cfg


__all__ = ["build_env_cfg"]
