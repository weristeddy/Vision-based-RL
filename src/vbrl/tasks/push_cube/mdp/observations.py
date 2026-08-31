"""Push-Cube observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from .commands import pushing_command


if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


OBJECT_HALF_EXTENT = 0.02
_ROBOT = SceneEntityCfg("robot")


def push_point_position(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  object_half_extent: float = OBJECT_HALF_EXTENT,
  clearance: float = 0.005,
) -> torch.Tensor:
  """Return the goal-conditioned point immediately behind the cube."""
  command = pushing_command(env, command_name)
  obj: Entity = env.scene[object_name]
  object_pos = obj.data.root_link_pos_w
  direction = torch.nn.functional.normalize(
    command.target_pos[:, :2] - object_pos[:, :2],
    dim=-1,
    eps=torch.finfo(object_pos.dtype).eps,
  )
  xy = object_pos[:, :2] - direction * (object_half_extent + clearance)
  return torch.cat((xy, command.target_pos[:, 2:3]), dim=-1)


def ee_to_push_point(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
  object_half_extent: float = OBJECT_HALF_EXTENT,
  clearance: float = 0.005,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  ee_pos = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  delta = push_point_position(
    env, command_name, object_name, object_half_extent, clearance
  ) - ee_pos
  return quat_apply(quat_inv(robot.data.root_link_quat_w), delta)


__all__ = ["OBJECT_HALF_EXTENT", "ee_to_push_point", "push_point_position"]
