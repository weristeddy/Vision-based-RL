from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import wrap_to_pi

from vbrl.tasks.push_t.mdp.rewards import max_contact_force_on_face, table_touch

from ..geometry import HINGE_LIMIT, chain_signed_distance
from .commands import PushChainCommand, push_chain_command

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_ROBOT = SceneEntityCfg("robot")
_DISTANCE_SCALE = 5.0
_MAX_REWARD = 3.0
_REACH_WEIGHT = 0.05
MIDDLE_LINK_WEIGHT = 0.5
HINGE_WEIGHT = 0.5


def goal_orderings(command: PushChainCommand) -> tuple[torch.Tensor, torch.Tensor]:
  yaw = torch.stack((command.target_yaw, command.target_yaw + torch.pi), dim=-1)
  hinges = torch.stack((command.target_hinges, -command.target_hinges.flip(-1)), 1)
  return yaw, hinges


def ordering_values(command: PushChainCommand) -> torch.Tensor:
  middle = command.chain_links()[:, 1]
  goal_yaw, goal_hinges = goal_orderings(command)
  weight = command.cfg.orientation_weight
  distance = torch.linalg.vector_norm(command.target_pos[:, :2] - middle[:, :2], dim=-1)
  position = (1.0 - torch.tanh(_DISTANCE_SCALE * distance)).square()
  yaw_error = wrap_to_pi(goal_yaw - middle[:, 2:]).abs()
  middle_term = (
    weight * (1.0 - yaw_error / torch.pi) + (1.0 - weight) * position[:, None]
  )
  hinge_error = (goal_hinges - command.chain_hinges()[:, None]).abs()
  hinge_term = (1.0 - hinge_error / (2.0 * HINGE_LIMIT)).mean(-1)
  return MIDDLE_LINK_WEIGHT * middle_term + HINGE_WEIGHT * hinge_term


def tcp_chain_distance(
  env: ManagerBasedRlEnv, command: PushChainCommand, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  tcp = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  height = tcp[:, 2] - command.object.data.root_link_pos_w[:, 2]
  outline = chain_signed_distance(tcp[:, :2], command.chain_links()).abs()
  return torch.hypot(outline, height)


def chain_dense_reward(
  env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg = _ROBOT
) -> torch.Tensor:
  command = push_chain_command(env, command_name)
  tcp_distance = tcp_chain_distance(env, command, asset_cfg)
  reach = torch.sqrt((1.0 - torch.tanh(_DISTANCE_SCALE * tcp_distance)).clamp_min(0.0))
  reward = ordering_values(command).amax(-1) + _REACH_WEIGHT * reach
  reward = torch.where(command.get_at_goal(), _MAX_REWARD, reward)
  return reward / _MAX_REWARD


def at_goal_static(
  env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  command = push_chain_command(env, command_name)
  vel = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
  static = 1.0 - torch.tanh(5.0 * torch.linalg.vector_norm(vel, dim=-1))
  return static * command.get_at_goal().to(static.dtype)


__all__ = [
  "HINGE_WEIGHT",
  "MIDDLE_LINK_WEIGHT",
  "at_goal_static",
  "chain_dense_reward",
  "goal_orderings",
  "max_contact_force_on_face",
  "ordering_values",
  "table_touch",
  "tcp_chain_distance",
]
