"""Generic robot adaptation of MJLab's Lift-Cube task."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationTermCfg, SceneEntityCfg
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg

from vbrl.asset_zoo.robots.definition import RobotDefinition
from vbrl.tasks.utils import attach_robot

from . import mdp


OBJECT_REST_HEIGHT = 0.02


def build_env_cfg(
  *,
  robot: RobotDefinition,
  object_name: str,
  rgb: bool = False,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Adapt MJLab Lift-Cube to one registered robot and modality."""
  cfg = make_lift_cube_env_cfg()
  attach_robot(cfg, robot, action_delay=rgb)

  action = cfg.actions["joint_pos"]
  assert isinstance(action, JointPositionActionCfg)
  action.scale = dict(robot.action_scale)

  for group in cfg.observations.values():
    group.terms["ee_to_cube"].params.update(
      object_name=object_name,
      asset_cfg=SceneEntityCfg("robot", site_names=(robot.ee_site,)),
    )
    group.terms["cube_to_goal"].params["object_name"] = object_name

  command = cfg.commands["lift_height"]
  assert isinstance(command, mdp.LiftingCommandCfg)
  command.entity_name = object_name
  assert command.object_pose_range is not None
  command.object_pose_range.z = (OBJECT_REST_HEIGHT, 0.05)
  cfg.rewards["lift"].params.update(
    asset_cfg=SceneEntityCfg("robot", site_names=(robot.ee_site,)),
    object_name=object_name,
  )
  cfg.rewards["lift_precise"].params["object_name"] = object_name

  for suffix in ("slide", "spin", "roll"):
    cfg.events[f"fingertip_friction_{suffix}"].params[
      "asset_cfg"
    ].geom_names = robot.fingertip_geom_pattern

  if rgb:
    actor = cfg.observations["actor"]
    actor.terms.pop("ee_to_cube")
    actor.terms.pop("cube_to_goal")
    actor.terms["goal_position"] = ObservationTermCfg(
      func=mdp.target_position,
      params={
        "command_name": "lift_height",
        "asset_cfg": SceneEntityCfg("robot", site_names=(robot.ee_site,)),
      },
    )

  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    command.resampling_time_range = (4.0, 4.0)
  return cfg


__all__ = ["OBJECT_REST_HEIGHT", "build_env_cfg"]
