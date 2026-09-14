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
# The fingertip height ceiling: the soft form of the planar constraint every
# published Push-T imposes in its action space. Both numbers are the object's
# own height rather than free parameters -- the fingertip should be no higher
# than the thing it is pushing -- so they follow the T if its geometry changes.
#
# Measured on a trained policy: the lowest fingertip sits at a median of 47 mm
# above a 24 mm object and 65% of steps are above 40 mm, so there is roughly
# 20 mm of pure headroom before this can cost the task anything. Linear rather
# than quadratic: a constant gradient keeps pulling the arm down from any height,
# where a quadratic is weakest exactly at the ceiling and explodes at the reset
# pose, which starts 243 mm up.
#
# `vertical_contact_force` used to sit here and was removed: at -0.05 it left
# top-face contact at 0.105 against the no-penalty baseline's 0.107 -- no effect
# at all -- and every weight that did move it (-0.10, -0.15, -0.25) cost 73-93%
# of episode success. It penalised an emergent contact normal; this penalises a
# height the policy chooses directly.
EE_HEIGHT_CEILING_M = 2.0 * HALF_HEIGHT
# -0.02, the value every run that worked carried: avv1us6e, 3mw7oyvo and
# 2458edlt reached 0.368 / 0.414 / 0.632 episode success with it. At this weight
# the term is close to inert -- it costs about 3% of a 14+ task reward and run
# avv1us6e kept a fingertip median of 46.3 mm against this 24 mm ceiling -- but
# -0.1 is the only weight measured to bind (top-face contact 0.054) and it took
# episode success to 0.008. The usable window, if there is one, lies between,
# and finding it needs seed replicates rather than single probes: two runs of an
# identical config scored 0.414 and 0.632.
EE_HEIGHT_WEIGHT = -0.02
# The one *positive* shaping term, and the replacement for the whole family of
# top-contact penalties: `vertical_contact_force` at four weights, the
# exponential force barrier, the height ceiling and the no-fly cylinder are all
# gone. See `mdp.side_contact_align` for why the sign is the argument.
#
# 0.05 is a starting size, not a measured one. The entire penalty family costs
# about 3% of task reward, so this puts the bonus on the same scale -- large
# enough to choose between two contact geometries, too small to outrank placing
# the T. Read `Episode_Reward/side_contact_align` against
# `Episode_Reward/maniskill_dense` at iteration ~50 and adjust once if it is not
# near 5%.
SIDE_CONTACT_ALIGN_WEIGHT = 0.05
# Table contact, targeted at zero. No onset: any contact is charged, because the
# T stands 24 mm tall and the gripper has that much clearance to push a side face
# without ever reaching the surface -- so "do not touch the table" is a small
# height adjustment, not a change of strategy. That is what separates this from
# the top-contact term: a penalty the policy can satisfy by lifting the wrist is
# safe to apply hard, while one it can only satisfy by abandoning contact is not.
#
# Quadratic, so a numerical graze is nearly free (1 N costs 0.0008 per step) and
# a real press is not (10 N costs 0.04, 16 N costs 0.10). The weight is halved
# against the old 5 N-onset version to offset removing the onset; measured peak
# table force during healthy pushing was 9.7-16.4 N.
TABLE_CONTACT_ONSET_N = 0.0
TABLE_CONTACT_SCALE_N = 5.0
TABLE_CONTACT_WEIGHT = -0.01
# Total commanded travel (L1) and MJLab's own action-rate term. `action_rate_l2`
# is upstream Lift-Cube's, at -0.01; it is kept here at a fifth of that because
# for a Gaussian policy whose mean never changes consecutive actions still differ
# by 2*sigma^2 per joint, so at the initial sigma of 0.975 it charges pure
# exploration noise -- 0.023 per step at this weight, against 0.114 at upstream's.
# The L1 travel term has no such problem: it scales as sigma rather than sigma^2,
# so it stays meaningful once the policy converges instead of vanishing.
# `action_acc_l2` is deliberately absent: it is not an upstream term, it is the
# sharpest sigma^2 penalty of the three, and it duplicates the rate term.
ACTION_PATH_LENGTH_WEIGHT = -0.002
ACTION_RATE_WEIGHT = -0.002
# Motion charged only once the object is at its goal, where ManiSkill's reward
# has gone flat and the task offers no gradient at all. 25x the travel weight
# above, and affordable precisely because it applies nowhere else: at goal the
# task pays a constant 1.0 per step, so 1.75 of L1 action costs 0.088 -- about
# a tenth of what being at goal is worth -- and cannot make the goal unattractive.
# -0.05 is the value with a measurement behind it: it cut post-success drift 55%
# (2.33 -> 1.05 mm per step). It was briefly -0.2 on the theory that the flat
# task reward made it free. That was wrong in the way that matters -- the
# gradient is flat at goal, so the term cannot distort behaviour *there*, but it
# lowers the value of the goal state and gamma carries that backwards into every
# state leading to it. Measured on the best policy under the current rewards,
# -0.2 costs 0.166 per step at goal, 17% of the at-goal reward and the largest
# single penalty in the config; the margin for being at goal fell from 0.72 to
# 0.56 per step.
AT_GOAL_ACTION_WEIGHT = -0.05
# `forceful_top_contact` used to be wired in here and is deliberately not any
# more. It is retained in `mdp.terminations` because the measurement is worth
# keeping reachable, but no task installs it.
#
# It was the last untried mechanism against dragging, on the argument that a
# penalty is a price the task reward can pay while a termination is not. That
# argument was right and the mechanism still failed, for a reason force cannot
# fix. Run 4fmml3fl at 5 N against its control 2458edlt: identical to iteration
# 25, then at iteration 50 the control's peak object force jumps 3.6 -> 20.1 N
# and its task reward 4.78 -> 6.04, which is a policy discovering that touching
# the T pays. The first contacts it discovers are hard top-face presses, so
# under a 5 N rule every one of those discoveries is an episode ending. The
# terminated run stayed at 0.86 N of peak object force for iterations 40-200 --
# not touching the object at all -- and sigma decayed against that flat reward
# from 0.44 to 0.097, which PPO cannot undo. It reached at iteration 425 what
# the control had at 75, and finished at 0.000 success against 0.626.
#
# Note the firing rate through the dead phase was only 3.2%: the collapse was
# not the termination firing, it was the policy having already learned to avoid
# it. The pressure also moved rather than disappearing -- with the top face
# closed, peak *table* force finished at 32.4 N against the control's 5.1.
#
# No threshold separates the two cases. Dragging runs 17-78 N (p50 17.4) and
# learning to push peaks at ~20 N, so they are the same forces; above the
# converged plateau of 39.8 N the term stops binding at all. What does separate
# them is time, not force -- discovery is at iteration 50, dragging is converged
# behaviour -- so this would need constraint annealing, and this task carries no
# curriculum. `side_contact_align` is the mechanism instead, and it cannot fail
# this way: it never removes reward from contact, so its worst case is being
# ignored.
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
  success_threshold: float = 0.90,
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
    # The one positive shaping term: push a side face, not the top face.
    # See SIDE_CONTACT_ALIGN_WEIGHT and mdp.side_contact_align.
    "side_contact_align": RewardTermCfg(
      func=mdp.side_contact_align,
      weight=SIDE_CONTACT_ALIGN_WEIGHT,
      params={"sensor_name": _CONTACT_SENSOR},
    ),
    # Total commanded travel: an L1 path penalty, so splitting one motion into
    # many is never cheaper and a correction cycle costs double.
    "action_path_length": RewardTermCfg(
      func=mdp.action_path_length_l1,
      weight=ACTION_PATH_LENGTH_WEIGHT,
    ),
    # MJLab's own action-rate term, at a fifth of upstream Lift-Cube's weight.
    "action_rate_l2": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=ACTION_RATE_WEIGHT,
    ),
    # Settle once the T is placed; see AT_GOAL_ACTION_WEIGHT.
    "at_goal_action": RewardTermCfg(
      func=mdp.at_goal_action_l1,
      weight=AT_GOAL_ACTION_WEIGHT,
      params={"command_name": _COMMAND},
    ),
    # Table contact, charged from the first newton -- the goal is zero.
    "table_contact_force": RewardTermCfg(
      func=mdp.contact_force_hinge,
      weight=TABLE_CONTACT_WEIGHT,
      params={
        "sensor_name": EE_GROUND_CONTACT_SENSOR,
        "onset": TABLE_CONTACT_ONSET_N,
        "scale": TABLE_CONTACT_SCALE_N,
      },
    ),
    # Keep the fingertip in the object's own height band, so a side push is the
    # only geometry available; see EE_HEIGHT_CEILING_M.
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
    # Fraction of the episode spent pressing a horizontal face of the T. The
    # drag-versus-push behaviour measure; 0.134 on run 8z5zwqj8's policy.
    "top_contact_share": MetricsTermCfg(
      func=mdp.top_contact_share,
      reduce="mean",
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
  cfg.curriculum = {}
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
