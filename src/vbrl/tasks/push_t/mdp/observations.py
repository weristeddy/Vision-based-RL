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


# ManiSkill's proprioception carries a target-delta controller's target; without it the
# policy cannot see how far the commanded pose leads the arm, up to 0.155 rad here.
def joint_target(env: ManagerBasedRlEnv, action_name: str = "joint_pos") -> torch.Tensor:
  term = env.action_manager.get_term(action_name)
  robot: Entity = env.scene[term.cfg.entity_name]
  return term.target - robot.data.default_joint_pos[:, term.target_ids]


def target_pose(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  command = push_t_command(env, command_name)
  robot: Entity = env.scene[asset_cfg.name]
  target_position = quat_apply(
    quat_inv(robot.data.root_link_quat_w),
    command.target_pos - robot.data.root_link_pos_w,
  ) + command.observation_offset
  target_yaw = wrap_to_pi(
    command.target_yaw
    + command.observation_yaw_offset
    - yaw_from_quat(robot.data.root_link_quat_w)
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
  obj: Entity = env.scene[object_name]
  yaw = yaw_from_quat(obj.data.root_link_quat_w)
  return torch.stack((torch.sin(yaw), torch.cos(yaw)), dim=-1)


def relative_yaw(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
) -> torch.Tensor:
  command = push_t_command(env, command_name)
  obj: Entity = env.scene[object_name]
  error = wrap_to_pi(command.target_yaw - yaw_from_quat(obj.data.root_link_quat_w))
  return torch.stack((torch.sin(error), torch.cos(error)), dim=-1)


__all__ = ["joint_target", "object_heading", "relative_yaw", "target_pose"]
