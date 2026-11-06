from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

import torch
from mjlab.envs.mdp import resolve_env_ids
from mjlab.managers import SceneEntityCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.utils.lab_api.math import sample_gaussian

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# Object/table sliding coefficient, not measured on this rig. At 0.55 the real T
# kept sliding after the fingertip left it, which it never did in simulation; 0.35
# still read as too sticky, so the policy sees a slippery T and learns gentle pushes.
OBJECT_TABLE_FRICTION_MEAN = 0.25
OBJECT_TABLE_FRICTION_STD = 0.10

OBJECT_COLLISION_GEOMS = r"push_t_(crossbar|stem)_collision"
TABLE_COLLISION_GEOM = ("table_top",)


class _ModelWithGeomFriction(Protocol):
  geom_friction: torch.Tensor


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
  "TABLE_COLLISION_GEOM",
  "randomize_object_table_friction",
]
