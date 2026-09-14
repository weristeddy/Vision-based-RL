"""Push-T terminations."""

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


def forceful_top_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float,
  verticality_threshold: float = 0.7,
) -> torch.Tensor:
  """End the episode when the gripper presses a horizontal face of the object.

  A termination rather than a penalty, and that is the whole point. Every
  penalty tried against dragging was bought out or broke the task: the direction
  penalty at four weights, an exponential force barrier, a height ceiling at two
  weights, a no-fly cylinder. A penalty is a price the task reward can pay; this
  is not payable.

  It is also narrower than any of them. Contact on a *vertical* face is untouched
  at any force, so pushing hard from the side stays fully available -- the escape
  route is to push properly, not to stop touching. Light top contact stays legal
  too, below ``force_threshold``, so brushing the top while manoeuvring is not
  fatal. Only leaning on the T hard enough to drag it ends the episode.

  ``verticality_threshold`` reads the contact normal, which points from the
  gripper into the object, so 0.7 is roughly a 45-degree cone about vertical --
  the same test ``top_contact_share`` reports as a metric.
  """
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
