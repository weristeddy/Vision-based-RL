from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers import SceneEntityCfg

from vbrl.tasks.push_t.mdp.rewards import max_contact_force_on_face, table_touch

from .commands import DISTANCE_SCALE, push_chain_command

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_REACH_WEIGHT = 0.05


def chain_dense_reward(
  env: ManagerBasedRlEnv, command_name: str, distance_scale: float
) -> torch.Tensor:
  command = push_chain_command(env, command_name)
  geometry = torch.exp(-command.landmark_distance() / distance_scale)
  reach = torch.sqrt(
    (1.0 - torch.tanh(DISTANCE_SCALE * command.tcp_distance())).clamp_min(0.0)
  )
  return geometry + _REACH_WEIGHT * reach + command.get_at_goal().float()


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
