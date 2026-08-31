"""Push-T observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv, wrap_to_pi

from ..geometry import yaw_from_quat
from .commands import push_t_command


if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_ROBOT = SceneEntityCfg("robot")


def target_pose(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Target position and heading in the robot base frame."""
  command = push_t_command(env, command_name)
  robot: Entity = env.scene[asset_cfg.name]
  target_position = quat_apply(
    quat_inv(robot.data.root_link_quat_w),
    command.target_pos - robot.data.root_link_pos_w,
  )
  target_yaw = wrap_to_pi(
    command.target_yaw - yaw_from_quat(robot.data.root_link_quat_w)
  )
  return torch.cat(
    (
      target_position,
      torch.stack((torch.sin(target_yaw), torch.cos(target_yaw)), dim=-1),
    ),
    dim=-1,
  )


def object_heading(
  env: ManagerBasedRlEnv,
  object_name: str,
) -> torch.Tensor:
  """World-frame object heading encoded continuously as sine and cosine."""
  obj: Entity = env.scene[object_name]
  yaw = yaw_from_quat(obj.data.root_link_quat_w)
  return torch.stack((torch.sin(yaw), torch.cos(yaw)), dim=-1)


def relative_yaw(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
) -> torch.Tensor:
  """Goal-minus-object yaw encoded continuously as sine and cosine."""
  command = push_t_command(env, command_name)
  obj: Entity = env.scene[object_name]
  error = wrap_to_pi(command.target_yaw - yaw_from_quat(obj.data.root_link_quat_w))
  return torch.stack((torch.sin(error), torch.cos(error)), dim=-1)


__all__ = ["object_heading", "relative_yaw", "target_pose"]
