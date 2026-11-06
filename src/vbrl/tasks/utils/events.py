from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.envs.mdp import resolve_env_ids
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_gaussian

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# ManiSkill3's vision-based sim-to-real setup.
ROBOT_JOINT_POSITION_STD_RAD = 0.02


def reset_joints_with_gaussian_offset(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  *,
  position_std: float,
  asset_cfg: SceneEntityCfg,
) -> None:
  if position_std < 0.0:
    raise ValueError("position_std must be non-negative.")

  selected_env_ids = resolve_env_ids(env, env_ids).to(
    device=env.device, dtype=torch.int
  )
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  default_joint_vel = asset.data.default_joint_vel
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert default_joint_pos is not None
  assert default_joint_vel is not None
  assert soft_joint_pos_limits is not None

  joint_ids = asset_cfg.joint_ids
  joint_pos = default_joint_pos[selected_env_ids][:, joint_ids].clone()
  joint_pos += sample_gaussian(
    0.0,
    position_std,
    joint_pos.shape,
    device=env.device,
  )
  limits = soft_joint_pos_limits[selected_env_ids][:, joint_ids]
  joint_pos.clamp_(limits[..., 0], limits[..., 1])
  joint_vel = default_joint_vel[selected_env_ids][:, joint_ids].clone()

  write_joint_ids: torch.Tensor | slice
  if isinstance(joint_ids, list):
    write_joint_ids = torch.tensor(joint_ids, device=env.device)
  else:
    write_joint_ids = joint_ids
  asset.write_joint_state_to_sim(
    joint_pos.view(len(selected_env_ids), -1),
    joint_vel.view(len(selected_env_ids), -1),
    env_ids=selected_env_ids,
    joint_ids=write_joint_ids,
  )


__all__ = ["ROBOT_JOINT_POSITION_STD_RAD", "reset_joints_with_gaussian_offset"]
