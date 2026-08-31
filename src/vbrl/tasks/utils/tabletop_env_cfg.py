"""The tabletop skeleton every VBRL manipulation task starts from.

Built on MJLab's ``make_lift_cube_env_cfg()`` rather than declared from
scratch, so solver settings, ``decimation``, ``env_spacing``, reset-event
ranges, and proprioception noise are inherited and cannot drift from upstream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.terrains import TerrainEntityCfg

if TYPE_CHECKING:
  import mujoco

  from vbrl.asset_zoo.robots.definition import RobotDefinition


EE_GROUND_CONTACT_SENSOR = "ee_ground_collision"

# MJLab reads the grid of env origins off the terrain entity, so a scene
# without one leaves every env at the world origin: correct physics, because
# MuJoCo Warp gives each env its own world, but a multi-env view or video draws
# N robots inside each other. These scenes build their own finite table, so the
# terrain is a floor for the human views and an origin source, never something
# the policy sees -- see ``_floor_for_the_human_views_only``.
ORIGIN_PLANE_GEOM = "terrain"
# Group 1 holds nothing else, and no camera renders it: the visual generation
# is (0, 2) and the retained CollisionCam tasks are (0, 3). So the floor reaches
# the Viser view and the recorder -- which `add_rgb_camera` opts into for this
# group alone -- while every RGB observation keeps the background it was
# trained against.
ORIGIN_PLANE_GROUP = 1
# The tabletop's top face is z=0 and its slab reaches z=-0.04, so the floor sits
# a hair under that: the tables stand *on* it instead of hovering above a
# detached drop shadow. A hair rather than exactly flush, to keep the plane out
# of the slab's own bottom face. It stays collidable, so an object pushed off
# the table lands instead of falling forever.
ORIGIN_PLANE_HEIGHT_M = -0.041
# The table spans 0.9 x 0.7 m, so this leaves a clear gap between neighbours.
ENV_SPACING_M = 1.2


def _contact_sensor(cfg: ManagerBasedRlEnvCfg) -> ContactSensorCfg:
  sensor = next(
    sensor
    for sensor in cfg.scene.sensors
    if sensor.name == EE_GROUND_CONTACT_SENSOR
  )
  assert isinstance(sensor, ContactSensorCfg)
  return sensor


def _floor_for_the_human_views_only(spec: mujoco.MjSpec) -> None:
  """Move MJLab's ground plane to the group only the human views draw.

  Runs as ``SceneCfg.spec_fn``, after the entities are attached and before the
  model compiles, which is the only point where the terrain's geom exists as a
  spec object. Written as a module-level function so TorchrunX can pickle it
  into a fresh worker.

  Viser ignores the group entirely: mjviser replaces any plane geom with its
  own infinite fading grid, so the viewer gets a floor from the plane simply
  existing.
  """
  for geom in spec.geoms:
    if geom.name.rsplit("/", 1)[-1] == ORIGIN_PLANE_GEOM:
      geom.group = ORIGIN_PLANE_GROUP
      geom.pos = (0.0, 0.0, ORIGIN_PLANE_HEIGHT_M)


def lay_out_envs_on_a_grid(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Give the scene the env origins a multi-env view and video need.

  Every position the MDP reads is already relative -- the commands add
  ``env_origins`` to the object and goal poses, the terminations subtract it,
  and every observation and reward is a difference of two world positions -- so
  each env is the same scene rigidly translated and the camera, which rides the
  robot's base body, translates with it. The observation is unchanged pixel for
  pixel; only the world coordinates it sits at differ.

  MJLab's own ``reset_base`` already puts the robot at its env origin. The
  table is a VBRL entity that upstream does not have, so it needs the matching
  term; both are fixed-base mocap entities, which is the case
  ``reset_root_state_uniform`` handles by writing the mocap pose.
  """
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="plane",
    # MJLab's own checker groundplane, which is what a MuJoCo scene looks like.
    # Only the sun is dropped: this scene builds its own lighting, and a second
    # one would reach the camera.
    lights=(),
  )
  cfg.scene.env_spacing = ENV_SPACING_M
  cfg.scene.spec_fn = _floor_for_the_human_views_only
  cfg.events["reset_table_base"] = EventTermCfg(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {},
      "velocity_range": {},
      "asset_cfg": SceneEntityCfg("table"),
    },
  )
  return cfg


def attach_robot(
  cfg: ManagerBasedRlEnvCfg,
  robot: RobotDefinition,
  *,
  action_delay: bool = False,
  fixed_closed_gripper: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Install one robot and retarget everything that names its bodies."""
  lay_out_envs_on_a_grid(cfg)
  cfg.scene.entities = {
    "robot": robot.make_entity_cfg(
      action_delay=action_delay,
      fixed_closed_gripper=fixed_closed_gripper,
    )
  }
  contact = _contact_sensor(cfg)
  contact.primary.pattern = robot.collision_body_pattern
  contact.secondary = ContactMatch(mode="body", pattern="table", entity="table")
  cfg.viewer.body_name = robot.viewer_body
  return cfg


def make_tabletop_env_cfg(
  robot: RobotDefinition,
  *,
  action_delay: bool = False,
  fixed_closed_gripper: bool = False,
  keep_rewards: tuple[str, ...] = (),
) -> ManagerBasedRlEnvCfg:
  """Return MJLab's tabletop skeleton with the Lift-Cube MDP removed.

  Retained: simulation settings, the robot, the end-effector/table contact
  sensor, the three reset events, the ``time_out`` termination, the
  proprioception observation terms, and any reward named in ``keep_rewards``.
  Callers own observations beyond ``joint_pos``/``joint_vel``/``actions``, and
  own commands, terminations, and curriculum entirely.
  """
  cfg = make_lift_cube_env_cfg()
  attach_robot(
    cfg,
    robot,
    action_delay=action_delay,
    fixed_closed_gripper=fixed_closed_gripper,
  )
  cfg.scene.sensors = (_contact_sensor(cfg),)
  cfg.events = {
    name: cfg.events[name]
    for name in ("reset_base", "reset_table_base", "reset_robot_joints")
  }
  cfg.terminations = {"time_out": cfg.terminations["time_out"]}
  cfg.rewards = {name: cfg.rewards[name] for name in keep_rewards}
  return cfg


__all__ = [
  "EE_GROUND_CONTACT_SENSOR",
  "ENV_SPACING_M",
  "ORIGIN_PLANE_GEOM",
  "ORIGIN_PLANE_GROUP",
  "ORIGIN_PLANE_HEIGHT_M",
  "attach_robot",
  "lay_out_envs_on_a_grid",
  "make_tabletop_env_cfg",
]
