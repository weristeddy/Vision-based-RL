"""Push-Cube rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers import SceneEntityCfg
from .commands import pushing_command
from .observations import OBJECT_HALF_EXTENT, ee_to_push_point


if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_ROBOT = SceneEntityCfg("robot")


def ee_push_point_distance(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
  object_half_extent: float = OBJECT_HALF_EXTENT,
  clearance: float = 0.005,
) -> torch.Tensor:
  return torch.linalg.vector_norm(
    ee_to_push_point(
      env,
      command_name,
      object_name,
      asset_cfg,
      object_half_extent,
      clearance,
    ),
    dim=-1,
  )


def object_goal_distance(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
) -> torch.Tensor:
  command = pushing_command(env, command_name)
  obj: Entity = env.scene[object_name]
  return torch.linalg.vector_norm(
    command.target_pos - obj.data.root_link_pos_w, dim=-1
  )


__all__ = ["ee_push_point_distance", "object_goal_distance"]
