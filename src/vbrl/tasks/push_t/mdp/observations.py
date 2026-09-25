from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, wrap_to_pi

from ..geometry import yaw_from_quat
from .commands import push_t_command

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_ROBOT = SceneEntityCfg("robot")


def qpos(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  return env.scene[asset_cfg.name].data.joint_pos


def qvel(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT) -> torch.Tensor:
  return env.scene[asset_cfg.name].data.joint_vel


def target_qpos(env: ManagerBasedRlEnv, action_name: str = "joint_pos") -> torch.Tensor:
  return env.action_manager.get_term(action_name).target


def _in_base(robot: Entity, position: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
  inverse = quat_inv(robot.data.root_link_quat_w)
  return torch.cat(
    (
      quat_apply(inverse, position - robot.data.root_link_pos_w),
      quat_mul(inverse, quat),
    ),
    dim=-1,
  )


def tcp_pose(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  site = asset_cfg.site_ids
  position = robot.data.site_pos_w[:, site][:, 0]
  return _in_base(robot, position, robot.data.site_quat_w[:, site][:, 0])


def obj_pose(env: ManagerBasedRlEnv, object_name: str) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  obj: Entity = env.scene[object_name]
  return _in_base(robot, obj.data.root_link_pos_w, obj.data.root_link_quat_w)


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


__all__ = ["obj_pose", "qpos", "qvel", "target_pose", "target_qpos", "tcp_pose"]
