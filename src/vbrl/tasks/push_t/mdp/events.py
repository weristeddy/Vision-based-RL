"""Push-T event term functions; ``push_t_env_cfg.py`` wires them up."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp import resolve_env_ids
from mjlab.managers import SceneEntityCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.utils.lab_api.math import sample_gaussian

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# ManiSkill3's vision-based sim-to-real setup.
ROBOT_JOINT_POSITION_STD_RAD = 0.02
OBJECT_TABLE_FRICTION_MEAN = 0.3
OBJECT_TABLE_FRICTION_STD = 0.025

OBJECT_COLLISION_GEOMS = r"push_t_(crossbar|stem)_collision"
TABLE_COLLISION_GEOM = ("table_top",)


class _ModelWithGeomFriction(Protocol):
  geom_friction: torch.Tensor


def reset_joints_with_gaussian_offset(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  *,
  position_std: float,
  asset_cfg: SceneEntityCfg,
) -> None:
  """Reset selected joints around their defaults using Gaussian offsets."""
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


def _geom_indices(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  return asset.indexing.geom_ids[asset_cfg.geom_ids]


@requires_model_fields("geom_friction")
def randomize_object_table_friction(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  *,
  mean: float,
  std: float,
  object_asset_cfg: SceneEntityCfg,
  table_asset_cfg: SceneEntityCfg,
) -> None:
  """Set one sampled effective sliding coefficient on both contact surfaces.

  MuJoCo takes the maximum sliding-friction coefficient of equal-priority
  colliding geoms. Sampling only the object would therefore be masked by the
  table's nominal coefficient of 1.0. Writing the same sample to the T's two
  collision geoms and the table collision geom makes the effective T-table
  coefficient equal to the requested sample.
  """
  if std < 0.0:
    raise ValueError("std must be non-negative.")

  selected_env_ids = resolve_env_ids(env, env_ids).to(
    device=env.device, dtype=torch.int
  )
  sampled = sample_gaussian(
    mean,
    std,
    (len(selected_env_ids), 1),
    device=env.device,
  ).clamp_min_(0.0)
  friction = cast(_ModelWithGeomFriction, env.sim.model).geom_friction

  for asset_cfg in (object_asset_cfg, table_asset_cfg):
    geom_ids = _geom_indices(env, asset_cfg)
    env_grid, geom_grid = torch.meshgrid(
      selected_env_ids,
      geom_ids,
      indexing="ij",
    )
    friction[env_grid, geom_grid, 0] = sampled.expand(-1, len(geom_ids))


__all__ = [
  "OBJECT_COLLISION_GEOMS",
  "OBJECT_TABLE_FRICTION_MEAN",
  "OBJECT_TABLE_FRICTION_STD",
  "ROBOT_JOINT_POSITION_STD_RAD",
  "TABLE_COLLISION_GEOM",
  "randomize_object_table_friction",
  "reset_joints_with_gaussian_offset",
]
