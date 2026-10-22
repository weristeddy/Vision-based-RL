from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity

from vbrl.scenes.presets import TABLE_CENTER, TABLE_HALF_EXTENTS

from .rewards import max_contact_force_on_face

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def object_off_table(
  env: ManagerBasedRlEnv,
  object_name: str,
  min_height: float = -0.05,
) -> torch.Tensor:
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




# A rule rather than a price: riding on the top face and dragging the T was worth
# more than any penalty a policy could still learn under.
def top_face_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  min_force_n: float = 1.0,
  enabled: bool = False,
) -> torch.Tensor:
  if not enabled:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  return max_contact_force_on_face(env, sensor_name, vertical=True) > min_force_n


__all__ = [
  "invalid_object_state",
  "object_off_table",
  "top_face_contact",
]
