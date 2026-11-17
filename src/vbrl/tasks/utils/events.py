from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.envs.mdp import resolve_env_ids
from mjlab.managers import SceneEntityCfg
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.utils.lab_api.math import sample_gaussian, sample_uniform

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


@requires_model_fields("mat_rgba")
def shared_material_rgb(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  entity_names: tuple[str, ...],
  ranges: tuple[tuple[float, float], ...],
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  low, high = torch.tensor(ranges, device=env.device).unbind(-1)
  rgb = sample_uniform(low, high, (len(env_ids), 3), env.device)
  materials = torch.cat([env.scene[name].indexing.mat_ids for name in entity_names])
  env.sim.model.mat_rgba[env_ids[:, None], materials, :3] = rgb[:, None]


# One colour per episode for every marker entity: Push-Chain's goal is one per link.
def goal_colour_event(
  entity_names: tuple[str, ...], ranges: tuple[tuple[float, float], ...]
) -> EventTermCfg:
  return EventTermCfg(
    func=shared_material_rgb,
    mode="reset",
    params={"entity_names": entity_names, "ranges": ranges},
  )


__all__ = [
  "ROBOT_JOINT_POSITION_STD_RAD",
  "goal_colour_event",
  "reset_joints_with_gaussian_offset",
  "shared_material_rgb",
]
