from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.event_manager import requires_model_fields
from mjlab.utils.lab_api.math import sample_uniform

from vbrl.tasks.push_t.mdp.events import (
  OBJECT_TABLE_FRICTION_MEAN,
  OBJECT_TABLE_FRICTION_STD,
  TABLE_COLLISION_GEOM,
  randomize_object_table_friction,
)

OBJECT_COLLISION_GEOMS = r"push_chain_link\d+_.*_collision"
HINGE_JOINTS = r"push_chain_hinge\d"
HINGE_SCALE_RANGE = (0.5, 2.0)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


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


__all__ = [
  "HINGE_JOINTS",
  "HINGE_SCALE_RANGE",
  "OBJECT_COLLISION_GEOMS",
  "OBJECT_TABLE_FRICTION_MEAN",
  "OBJECT_TABLE_FRICTION_STD",
  "TABLE_COLLISION_GEOM",
  "randomize_object_table_friction",
  "shared_material_rgb",
]
