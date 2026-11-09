from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity

from vbrl.scenes.presets import TABLE_CENTER, TABLE_HALF_EXTENTS
from vbrl.tasks.push_t.mdp.terminations import top_face_contact

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def object_off_table(
  env: ManagerBasedRlEnv, object_name: str, min_height: float = -0.05
) -> torch.Tensor:
  obj: Entity = env.scene[object_name]
  position = obj.data.body_link_pos_w - env.scene.env_origins[:, None]
  centre = position.new_tensor(TABLE_CENTER[:2])
  half = position.new_tensor(TABLE_HALF_EXTENTS[:2])
  off = ((position[..., :2] - centre).abs() > half).any(-1)
  return (off | (position[..., 2] < min_height)).any(-1)


def invalid_object_state(
  env: ManagerBasedRlEnv,
  object_name: str,
  max_height: float = 0.25,
  max_linear_speed: float = 5.0,
  max_angular_speed: float = 50.0,
) -> torch.Tensor:
  obj: Entity = env.scene[object_name]
  height = obj.data.body_link_pos_w[..., 2] - env.scene.env_origins[:, None, 2]
  velocity = obj.data.body_link_vel_w
  return (
    (height > max_height)
    | (torch.linalg.vector_norm(velocity[..., :3], dim=-1) > max_linear_speed)
    | (torch.linalg.vector_norm(velocity[..., 3:], dim=-1) > max_angular_speed)
  ).any(-1)


__all__ = ["invalid_object_state", "object_off_table", "top_face_contact"]
