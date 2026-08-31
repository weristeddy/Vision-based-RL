"""Push-T terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity

from vbrl.scenes.presets import TABLE_CENTER, TABLE_HALF_EXTENTS


if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv



def object_off_table(
  env: ManagerBasedRlEnv,
  object_name: str,
  min_height: float = -0.05,
) -> torch.Tensor:
  """Whether the object center has left the physical tabletop."""
  obj: Entity = env.scene[object_name]
  position = obj.data.root_link_pos_w - env.scene.env_origins
  return (
    (position[:, 0] < TABLE_CENTER[0] - TABLE_HALF_EXTENTS[0])
    | (position[:, 0] > TABLE_CENTER[0] + TABLE_HALF_EXTENTS[0])
    | (position[:, 1] < TABLE_CENTER[1] - TABLE_HALF_EXTENTS[1])
    | (position[:, 1] > TABLE_CENTER[1] + TABLE_HALF_EXTENTS[1])
    | (position[:, 2] < min_height)
  )


def invalid_object_state(
  env: ManagerBasedRlEnv,
  object_name: str,
  max_height: float = 0.25,
  max_linear_speed: float = 5.0,
  max_angular_speed: float = 50.0,
) -> torch.Tensor:
  """Catch implausible object height or velocity."""
  obj: Entity = env.scene[object_name]
  position = obj.data.root_link_pos_w - env.scene.env_origins
  velocity = obj.data.root_link_vel_w
  return (
    (position[:, 2] > max_height)
    | (
      torch.linalg.vector_norm(velocity[:, :3], dim=-1)
      > max_linear_speed
    )
    | (
      torch.linalg.vector_norm(velocity[:, 3:], dim=-1)
      > max_angular_speed
    )
  )


__all__ = ["invalid_object_state", "object_off_table"]
