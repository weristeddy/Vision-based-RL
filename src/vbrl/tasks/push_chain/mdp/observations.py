from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from vbrl.tasks.push_t.geometry import yaw_from_quat

from .commands import push_chain_command

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_ROBOT = SceneEntityCfg("robot")


def _in_base(robot: Entity, position: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
  quat = robot.data.root_link_quat_w[:, None].expand(-1, 3, -1)
  local = quat_apply(quat_inv(quat), position - robot.data.root_link_pos_w[:, None])
  yaw = yaw - yaw_from_quat(robot.data.root_link_quat_w)[:, None]
  pose = torch.cat((local, torch.sin(yaw)[..., None], torch.cos(yaw)[..., None]), -1)
  return pose.flatten(1)


def target_pose(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
  calibration_bias: bool = True,
) -> torch.Tensor:
  command = push_chain_command(env, command_name)
  links = command.observed_links if calibration_bias else command.target_links
  height = command.target_pos[:, None, 2:].expand(-1, 3, -1)
  position = torch.cat((links[..., :2], height), dim=-1)
  return _in_base(env.scene[asset_cfg.name], position, links[..., 2])


def chain_pose(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = push_chain_command(env, command_name)
  position = command.object.data.body_link_pos_w[:, command.link_ids]
  return _in_base(env.scene["robot"], position, command.chain_links()[..., 2])


def chain_errors(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = push_chain_command(env, command_name)
  middle = command.chain_links()[:, 1]
  errors = command.goal_errors()
  yaw = errors[:, 1:2]
  return torch.cat(
    (
      command.target_pos[:, :2] - middle[:, :2],
      torch.sin(yaw),
      torch.cos(yaw),
      errors[:, 2:],
    ),
    dim=-1,
  )


def link_overlaps(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return push_chain_command(env, command_name).get_overlap()


def tcp_to_chain(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return push_chain_command(env, command_name).tcp_distance()[:, None]


__all__ = [
  "chain_errors",
  "chain_pose",
  "link_overlaps",
  "target_pose",
  "tcp_to_chain",
]
