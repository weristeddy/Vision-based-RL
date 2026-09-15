"""State and RGB Stack-Cubes configuration, shared by every robot and camera.

Stack all four cubes into one tower at a fixed place, then withdraw to the
observation pose and hold it. Cube identity and order do not matter, a finished
tower is never a termination, and the episode keeps running -- so the policy
meets its own tower being knocked down while it is standing by, which is what
the real rig has to survive.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
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


# 40 s at 50 Hz: 2,000 policy steps. Lift-Cube's 20 s covers one pick and
# place; this has up to four of them, plus the withdrawal, plus whatever a
# mid-episode scenario change or a disturbance asks for afterwards.
EPISODE_LENGTH_S = 40.0

# 2.5 ms substeps at decimation 8, which is 50 Hz -- Lift-Cube's control rate
# to the hertz, and its controller untouched. Only the physics substep halves,
# and it has to: MuJoCo Warp's box-box contact cannot hold a cube tower up at
# Lift-Cube's 5 ms. Measured on the four-cube stack with random per-level yaw,
# 5 s of settling with no robot in contact -- 4 of 5 towers collapse at 5 ms
# and 5 of 5 at 4 ms, against 0 of 5 at 2.5 ms and at 2 ms. It is a solver
# artifact, not the contact tuning: the same model, the same solref/solimp and
# the same 5 ms step is perfectly stable in stock MuJoCo (0.2 mm of sag, no
# drift), and neither softening the contact, raising nconmax/njmax, nor going
# to 50 solver iterations recovers it under Warp. Aligning every cube's yaw
# also fixes it, which is the tell: the failure is the yaw-mismatched box-box
# manifold. Aligning them is not an option here -- cube orientation is
# deliberately free -- so the substep is what moves. It costs 2x the physics
# work per control step.
SIM_TIMESTEP_S = 0.0025
DECIMATION = 8

# The observation pose, and the only thing in this task that is not Lift-Cube's.
# Chosen by search rather than by hand, against three competing demands. With
# the arm here the wrist camera sees the whole loose-cube workspace *and* the
# full height of a four-cube tower with 6.6 px of margin in a 224 px frame; the
# gripper sits at (0.178, 0, 0.369), withdrawn behind and above the workspace
# and above the top edge of the external camera's frame, so it occludes
# nothing; `joint_3` stops 0.054 rad short of its soft limit, which matters
# because `joint_pos_limits` carries weight -10 and would otherwise fight the
# home reward for the same pose; and it is 0.74 rad (L1) from the robot's own
# home, so holding it costs actions inside +-2.2 units.
#
# The trade is real: the maximum-coverage pose reaches 8.5 px but needs -2.84
# on joint_1, and the closest-to-home pose gets only 5.0 px. This is the middle.
OBSERVATION_JOINT_POS = {
  "joint_0": 0.0,
  "joint_1": 0.79,
  "joint_2": 1.56,
  "joint_3": -1.36,
  "joint_4": 0.0,
  "joint_5": 0.0,
}

# Exactly one disturbance per episode, at a uniformly random moment, and that
# is a designed property rather than a rate. MJLab resamples an interval
# timer on every env reset, so the first firing lands at U[lower, upper] and a
# second would land at or after 2*lower. Keeping
#
#     2 * lower > episode_length >= upper
#
# therefore guarantees one firing and forbids a second;
# `tests/test_stack_cubes.py` pins the inequality for every interval term here.
#
# Once, not more. A four-cube build costs the robot roughly 15-25 s, so a
# policy disturbed every 12-20 s -- the first rate tried -- would never be
# allowed to finish one, and would never hold the observation pose long enough
# for the last tenth of the reward to mean anything. The shape wanted from an
# episode is build -> home -> one disturbance -> repair -> home, and these
# windows leave 8-19 s after the collapse to do the second half of it.
SCENARIO_INTERVAL_S = (21.0, 32.0)
# How much of the tower a collapse can take. Up to three courses, so a
# four-cube tower can lose everything but its base -- but the top course alone
# is the common case; see `mdp.events.STACK_DECAY`.
MAX_FALLING_CUBES = 3
# Four cubes, a tower, and a gripper in among them need more contact and
# constraint room than upstream Lift-Cube's single cube.
NCONMAX = 200
NJMAX = 1800

_PRIVILEGED_ACTOR_TERMS = (
  "cube_positions",
  "ee_to_target_cube",
  "tower_progress",
)


def build_env_cfg(
  *,
  robot: RobotDefinition,
  rgb: bool = False,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build the Stack-Cubes MDP for one robot and modality."""
  cfg = make_tabletop_env_cfg(
    robot,
    action_delay=rgb,
    keep_rewards=("action_rate_l2", "joint_pos_limits", "joint_vel_hinge"),
  )
  # The gripper has to open and close, so the whole actuator set is driven and
  # the scale is the robot's own -- Lift-Cube's controller, unchanged.
  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = dict(robot.action_scale)

  arm_joints = robot.arm_actuator_names
  home_joint_pos = tuple(OBSERVATION_JOINT_POS[name] for name in arm_joints)
  # One cfg carrying everything the task logic needs to know about the robot:
  # where its fingertips are and how far the gripper is open.
  task_cfg = SceneEntityCfg(
    "robot",
    site_names=(robot.ee_site,),
    joint_names=("left_carriage_joint",),
  )
  arm_cfg = SceneEntityCfg("robot", joint_names=arm_joints, preserve_order=True)
  gripper_cfg = SceneEntityCfg("robot", joint_names=("left_carriage_joint",))
  home = {
    "arm_cfg": arm_cfg,
    "gripper_cfg": gripper_cfg,
    "home_joint_pos": home_joint_pos,
  }

  # Where the next course goes. It is fixed in xy and rises with the tower, so
  # it is published rather than sampled -- which is what gives evaluation its
  # `episode_success` and the viewer a drawn target.
  cfg.commands = {
    "stack_target": mdp.StackCommandCfg(asset_cfg=task_cfg, debug_vis=True)
  }

  base_terms = cfg.observations["actor"].terms
  proprioception = {
    "joint_pos": base_terms["joint_pos"],
    "joint_vel": base_terms["joint_vel"],
  }
  privileged = {
    "cube_positions": ObservationTermCfg(
      func=mdp.cube_positions,
      params={"asset_cfg": task_cfg},
      noise=Unoise(n_min=-0.005, n_max=0.005),
    ),
    "ee_to_target_cube": ObservationTermCfg(
      func=mdp.ee_to_target_cube,
      params={"asset_cfg": task_cfg},
      noise=Unoise(n_min=-0.005, n_max=0.005),
    ),
    "tower_progress": ObservationTermCfg(
      func=mdp.tower_progress, params={"asset_cfg": task_cfg}
    ),
  }
  actor_terms = {**proprioception, **privileged, "actions": base_terms["actions"]}
  cfg.observations["actor"].terms = dict(actor_terms)
  cfg.observations["critic"].terms = dict(actor_terms)

  cfg.rewards = {
    # The whole task, normalized to roughly [0, 1]; see mdp.rewards.
    "stack_progress": RewardTermCfg(
      func=mdp.stack_progress,
      weight=1.0,
      params={"asset_cfg": task_cfg, **home},
    ),
    # Everything below is upstream Lift-Cube's regularization at upstream's
    # weights, kept by `make_tabletop_env_cfg`: smooth commanded motion, the
    # soft joint-limit hinge, and the velocity hinge its curriculum tightens.
    **cfg.rewards,
  }
  cfg.metrics = {
    "tower_height_peak": MetricsTermCfg(
      func=mdp.tower_height, reduce="max", params={"asset_cfg": task_cfg}
    ),
    "home_reached": MetricsTermCfg(
      func=mdp.home_reached,
      reduce="mean",
      params={"asset_cfg": task_cfg, **home},
    ),
    # Peak fingertip force on any cube, in newtons. Push-T's whole dragging
    # investigation ran on a metric like this one; without it a policy that
    # slams cubes together looks identical to one that sets them down.
    "peak_grasp_force": MetricsTermCfg(func=mdp.peak_grasp_force, reduce="max"),
    "reward_stage": MetricsTermCfg(
      func=mdp.reward_stage, reduce="mean", params={"asset_cfg": task_cfg}
    ),
  }
  # A finished tower is not a termination, and neither is one that falls over.
  cfg.terminations["cube_out_of_reach"] = TerminationTermCfg(
    func=mdp.cube_out_of_reach, params={"asset_cfg": task_cfg}
  )
  cfg.terminations["ee_table_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": EE_GROUND_CONTACT_SENSOR, "force_threshold": 10.0},
  )
  cfg.curriculum = {
    "joint_vel_hinge_weight": CurriculumTermCfg(
      func=mdp.reward_curriculum,
      params={
        "reward_name": "joint_vel_hinge",
        "stages": [
          {"step": 0, "weight": -0.01},
          {"step": 500 * 24, "weight": -0.1},
          {"step": 1000 * 24, "weight": -1.0},
        ],
      },
    ),
  }

  cfg.events["stack_scenario"] = EventTermCfg(
    func=mdp.reset_stack_scenario,
    mode="reset",
    # No `asset_cfg`: MJLab's own `reset_robot_joints` is putting the arm back
    # to its start pose in this same reset, so where it happens to be now says
    # nothing about where a cube would be in its way.
    params={},
  )
  cfg.events["collapse_tower"] = EventTermCfg(
    func=mdp.collapse_tower,
    mode="interval",
    interval_range_s=SCENARIO_INTERVAL_S,
    params={"asset_cfg": task_cfg, "max_falling": MAX_FALLING_CUBES},
  )
  cfg.events["cube_friction"] = EventTermCfg(
    func=mdp.randomize_cube_friction,
    mode="reset",
    params={
      "mean": mdp.CUBE_FRICTION_MEAN,
      "std": mdp.CUBE_FRICTION_STD,
    },
  )
  for suffix, distribution, axis, ranges in (
    ("slide", "uniform", 0, (0.3, 1.5)),
    ("spin", "log_uniform", 1, (1e-4, 2e-2)),
    ("roll", "log_uniform", 2, (1e-5, 5e-3)),
  ):
    cfg.events[f"fingertip_friction_{suffix}"] = EventTermCfg(
      func=mdp.dr.geom_friction,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", geom_names=robot.fingertip_geom_pattern
        ),
        "operation": "abs",
        "distribution": distribution,
        "axes": [axis],
        "ranges": ranges,
      },
    )

  # One contact sensor per cube, the fingertip pads against that cube's body.
  # `tower.held` reads it for a two-finger grasp test, and `peak_grasp_force`
  # logs it -- the only window onto how hard the policy handles a cube, which
  # is the number that decides whether a sim placement is safe on hardware.
  cfg.scene.sensors += tuple(
    ContactSensorCfg(
      name=sensor,
      primary=ContactMatch(
        mode="geom", pattern=robot.fingertip_geom_pattern, entity="robot"
      ),
      secondary=ContactMatch(mode="body", pattern="cube", entity=cube),
      fields=("found", "force"),
      reduce="maxforce",
      num_slots=1,
    )
    for sensor, cube in zip(mdp.CONTACT_SENSORS, mdp.CUBE_NAMES, strict=True)
  )

  cfg.episode_length_s = EPISODE_LENGTH_S
  cfg.sim.mujoco.timestep = SIM_TIMESTEP_S
  cfg.decimation = DECIMATION
  cfg.sim.nconmax = NCONMAX
  cfg.sim.njmax = NJMAX
  # Look down on the whole workspace from the front, as Push-T does: azimuth
  # 180 is the +x side the robot faces, and the fixed base keeps the tower in
  # frame instead of swinging with the gripper.
  cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_ROOT
  cfg.viewer.entity_name = "robot"
  cfg.viewer.max_extra_envs = 0
  cfg.viewer.distance = 1.05
  cfg.viewer.elevation = -34.0
  cfg.viewer.azimuth = 170.0
  cfg.viewer.width = 640
  cfg.viewer.height = 480

  if rgb:
    # The visual actor keeps proprioception and its one image. No cube poses,
    # no active mask, no tower height: the deployed policy has to read "there
    # is a cube on the table that is not in the tower" off pixels.
    actor = cfg.observations["actor"]
    for name in _PRIVILEGED_ACTOR_TERMS:
      actor.terms.pop(name)

  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
  return cfg


__all__ = [
  "DECIMATION",
  "EPISODE_LENGTH_S",
  "MAX_FALLING_CUBES",
  "OBSERVATION_JOINT_POS",
  "SCENARIO_INTERVAL_S",
  "SIM_TIMESTEP_S",
  "build_env_cfg",
]
