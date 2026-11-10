from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers import SceneEntityCfg

from vbrl.tasks.push_t.mdp.rewards import max_contact_force_on_face, table_touch

from .commands import DISTANCE_SCALE, push_chain_command

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_MAX_REWARD = 3.0
_REACH_WEIGHT = 0.05


def chain_dense_reward(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = push_chain_command(env, command_name)
  distance = command.tcp_distance()
  reach = torch.sqrt((1.0 - torch.tanh(DISTANCE_SCALE * distance)).clamp_min(0.0))
  reward = command.ordering_values().amax(-1) + _REACH_WEIGHT * reach
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
  "at_goal_static",
  "chain_dense_reward",
  "max_contact_force_on_face",
  "table_touch",
]
