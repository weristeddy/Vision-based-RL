from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.sensor import ContactSensor

from vbrl.scenes.presets import TABLE_CENTER, TABLE_HALF_EXTENTS

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


def forceful_top_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float,
  verticality_threshold: float = 0.7,
) -> torch.Tensor:
  if force_threshold <= 0.0:
    raise ValueError("forceful_top_contact needs force_threshold > 0.")
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.found is None or data.force is None or data.normal is None:
    raise RuntimeError(
      f"Contact sensor {sensor_name!r} requires found, force, and normal."
    )
  vertical = data.normal[..., 2].abs() > verticality_threshold
  force = torch.linalg.vector_norm(data.force, dim=-1)
  return ((data.found > 0) & vertical & (force > force_threshold)).any(dim=-1)


__all__ = [
  "forceful_top_contact",
  "invalid_object_state",
  "object_off_table",
]
