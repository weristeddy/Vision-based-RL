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
# The object/table sliding coefficient, retuned to the real rig rather than
# inherited. ManiSkill3's 0.30 matched one PVC-on-wood measurement (0.296), but
# the standard reference for plastic on wood is 0.40 sliding (0.50 static) and
# printed PLA measures 0.38-0.57 -- and the real table here is a bare wood plate,
# the rougher end of that range. The rig's table is unsanded pine -- not
# laminate, not planed -- so it is rough and, more importantly, rough
# *unevenly*: the coefficient varies from spot to spot across the board.
#
# 0.55 is set from the literature rather than from this table, which has not
# been measured:
#
#   * generic plastic on wood is quoted at mu_s 0.50 / mu_k 0.40, the most
#     widely repeated pair, and MuJoCo's geom_friction is a single sliding
#     coefficient with no static/kinetic split;
#   * PLA itself sits above generic plastic -- reciprocating tests put neat PLA
#     at 0.65-0.70, and PLA wear studies span 0.37-0.75;
#   * print orientation moves it too, transverse higher than longitudinal;
#   * and an unsanded surface pushes it up again from any planed reference.
#
# So the centre belongs above the 0.40 the MJCF carried, which is also what the
# hardware says: the real T is visibly harder to slide than the simulated one.
# 0.12 puts two sigma at 0.31-0.79, spanning the whole published range, and
# three sigma at 0.19-0.91 for the tails.
#
# Still a reference value, not a measurement. A tilt test at five or six spots
# (mu = tan(theta), centre on the median, three sigma over the spread) would
# replace both numbers with the board's own.
#
# Side effect worth naming, since it is not the reason for the change: dragging
# the T from its top face needs mu_table*mg/(mu_grip - mu_table), so raising the
# table from 0.40 to 0.55 moves that threshold with it. The fingertip pads run
# to mu 1.5, and that ratio, not the table, is what makes dragging cheap.
OBJECT_TABLE_FRICTION_MEAN = 0.55
OBJECT_TABLE_FRICTION_STD = 0.12

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
