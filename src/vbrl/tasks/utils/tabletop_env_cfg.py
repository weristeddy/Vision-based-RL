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

ORIGIN_PLANE_GEOM = "terrain"
# No camera renders group 1 -- the visual generation is (0, 2) and CollisionCam is (0,
# 3) -- so the floor reaches the viewer and the recorder while every RGB observation.
ORIGIN_PLANE_GROUP = 1
# A hair under the tabletop slab's z=-0.04 bottom face, so tables stand *on* the floor
# rather than above a detached shadow.
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
  for geom in spec.geoms:
    if geom.name.rsplit("/", 1)[-1] == ORIGIN_PLANE_GEOM:
      geom.group = ORIGIN_PLANE_GROUP
      geom.pos = (0.0, 0.0, ORIGIN_PLANE_HEIGHT_M)


def lay_out_envs_on_a_grid(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  cfg.scene.terrain = TerrainEntityCfg(
    terrain_type="plane",
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
