"""Stack-Cubes scenario events: the reset pool, and the two disturbances.

The reset pool is the point of this module. Starting every parallel env with
four loose cubes would put the whole PPO batch at the same point of a
2,000-step task, so a reset instead draws how much of the tower is already
built. The batch then holds envs at every stage of the problem at once, which
is the property the staggered-reset work (arXiv:2511.21011) argues for on
exactly this task.

Mid-episode the draw is *not* repeated: replacing a tower the policy just built
with an unrelated scenario throws away what it earned. What runs instead is a
partial collapse -- some number of top courses come off and land back on the
table, everything below them untouched -- which is the disturbance the real rig
actually presents, and which leaves the episode continuous.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import math

import torch

from mjlab.entity import Entity
from mjlab.envs.mdp import resolve_env_ids
from mjlab.managers import SceneEntityCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.utils.lab_api.math import (
  quat_from_euler_xyz,
  sample_gaussian,
  sample_uniform,
)

from .tower import (
  CUBE_HALF,
  CUBE_NAMES,
  CUBE_SIZE,
  MAX_CUBES,
  MIN_CUBE_SEPARATION,
  STACK_XY,
  blockers,
  gripper_blocks_the_tower,
  sample_loose_xy,
  tower_state,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# How fast the probability of a taller pre-built tower, and of a bigger
# collapse, falls away. One number for both ladders, because they are the same
# idea seen from either end: the common case is an empty table and a single
# cube coming off the top, and every further course is half as likely as the
# one before it.
#
#   pre-stacked at reset:         0 52%  1 26%  2 13%  3 6%  4 3%
#   collapse, from a full tower:  1 57%  2 29%  3 14%
#
# The finished-tower case falls out of the same ladder rather than needing its
# own probability, which is why there is no `complete_probability` any more.
STACK_DECAY = 0.5


def _decaying_choice(counts: torch.Tensor, ceiling: int, device) -> torch.Tensor:
  """Draw 0..counts per row, weighted ``STACK_DECAY ** k`` and zero above it."""
  levels = torch.arange(ceiling + 1, device=device)
  weights = (STACK_DECAY**levels).expand(len(counts), -1).clone()
  weights[levels.unsqueeze(0) > counts.unsqueeze(1)] = 0.0
  return torch.multinomial(weights, 1).squeeze(1)


def reset_stack_scenario(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  *,
  asset_cfg: SceneEntityCfg | None = None,
) -> None:
  """Draw one valid task state: ``k`` of the four cubes already stacked.

  All four cubes are always on the table. ``k`` runs from 0 to 4 on the
  ``STACK_DECAY`` ladder, so an empty table is the normal start -- about half
  of all draws -- and every further pre-built course is half as likely as the
  one below it. The tail reaches a finished tower a few percent of the time,
  which is how the home behaviour gets any experience before the policy can
  build one itself.

  Which physical cube ends up where is a fresh permutation every time, so
  nothing learns that ``cube_0`` is the bottom one.

  ``asset_cfg`` names the robot when this runs *during* an episode, which is the
  only difference between the two uses. The arm is left alone -- it keeps
  whatever it was doing, exactly as Lift-Cube's own mid-episode resample does --
  so instead the draw works around it: loose cubes avoid the hand, and an env
  whose hand is standing inside the tower column is skipped and picks the next
  firing up. At an episode reset there is nothing to avoid, because the arm is
  on its way back to the start pose.
  """
  ids = resolve_env_ids(env, env_ids).to(env.device)
  device = env.device
  if asset_cfg is not None:
    ids = ids[~gripper_blocks_the_tower(env, ids, asset_cfg)]
  count = len(ids)
  if count == 0:
    return

  stacked = _decaying_choice(
    torch.full((count,), MAX_CUBES, device=device), MAX_CUBES, device
  )

  order = torch.argsort(torch.rand(count, MAX_CUBES, device=device), dim=1)
  rank = torch.arange(MAX_CUBES, device=device)

  tower = torch.zeros(count, MAX_CUBES, 3, device=device)
  tower[..., 0] = STACK_XY[0]
  tower[..., 1] = STACK_XY[1]
  tower[..., 2] = CUBE_HALF + rank * CUBE_SIZE

  loose = torch.zeros(count, MAX_CUBES, 3, device=device)
  loose[..., :2] = sample_loose_xy(
    count, MAX_CUBES, blockers(env, ids, asset_cfg)
  )
  loose[..., 2] = CUBE_HALF

  is_tower = (rank < stacked.unsqueeze(1)).unsqueeze(-1)
  by_rank = torch.where(is_tower, tower, loose)
  yaw_by_rank = sample_uniform(-math.pi, math.pi, (count, MAX_CUBES), device=device)

  position = torch.empty_like(by_rank).scatter_(
    1, order.unsqueeze(-1).expand(-1, -1, 3), by_rank
  )
  yaw = torch.empty_like(yaw_by_rank).scatter_(1, order, yaw_by_rank)
  _write_cubes(env, ids, position, yaw)


def collapse_tower(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  *,
  asset_cfg: SceneEntityCfg,
  max_falling: int,
) -> None:
  """Knock the top of the tower over and scatter those cubes back on the table.

  Between one and ``max_falling`` courses come off -- the top one alone most of
  the time, see ``STACK_DECAY`` -- and **everything below the break is left
  exactly as it stands**. That is what makes this a disturbance
  to recover from rather than a second episode: the policy keeps what it built,
  the tower height drops by the number that fell, the task reward drops with
  it, the home bonus disappears by itself, and the job is to put them back.

  Teleportation, not a simulated topple -- no impulse is applied and nothing is
  pushed. The cubes simply cease to be on the tower and reappear on the table,
  clear of the stack, of the gripper, and of whatever else is already lying
  there.
  """
  ids = resolve_env_ids(env, env_ids).to(env.device)
  state = tower_state(env, asset_cfg)
  height = state.height[ids]
  if not (height >= 1).any():
    return

  # One course off the top is the common case; each further one is half as
  # likely. `_decaying_choice` counts from zero, so this asks it for 0..h-1
  # extra courses on top of the one that always falls.
  ceiling = height.clamp(max=max_falling) - 1
  falling = 1 + _decaying_choice(ceiling.clamp(min=0), max_falling - 1, env.device)
  # The lowest level that comes off. A cube stacked at or above it falls.
  break_at = (height - falling).clamp(min=0)
  toppled = (
    state.stacked[ids]
    & (state.level[ids] >= break_at.unsqueeze(1))
    & (height >= 1).unsqueeze(1)
  )
  _scatter(env, ids, toppled, state, asset_cfg)


def _scatter(
  env: ManagerBasedRlEnv,
  ids: torch.Tensor,
  moving: torch.Tensor,
  state,
  asset_cfg: SceneEntityCfg,
) -> None:
  """Drop every cube ``moving`` selects onto a free patch of table.

  Only those cubes are written; every other cube keeps the pose and the yaw it
  settled into. The spots come from the same sampler a reset uses, extended
  with the cubes already lying on the table so a falling course cannot land on
  one.
  """
  if not moving.any():
    return
  device = env.device
  position = state.position[ids]
  settled = (position[..., 2] < CUBE_HALF + 0.005) & ~moving

  blocked = torch.zeros(len(ids), 2 + MAX_CUBES, 3, device=device)
  blocked[:, :2] = blockers(env, ids, asset_cfg)
  blocked[:, 2:, :2] = position[..., :2]
  blocked[:, 2:, 2] = torch.where(
    settled, torch.full_like(position[..., 2], MIN_CUBE_SEPARATION), 0.0
  )
  spots = sample_loose_xy(len(ids), MAX_CUBES, blocked)
  # Each falling cube takes the spot matching its rank among the falling ones.
  rank = (moving.long().cumsum(dim=1) - 1).clamp(min=0)
  taken = torch.gather(spots, 1, rank.unsqueeze(-1).expand(-1, -1, 2))
  yaw = sample_uniform(-math.pi, math.pi, (len(ids), MAX_CUBES), device=device)

  for index, name in enumerate(CUBE_NAMES):
    mask = moving[:, index]
    if not mask.any():
      continue
    entity: Entity = env.scene[name]
    selected = ids[mask]
    dropped = torch.zeros(len(selected), 3, device=device)
    dropped[:, :2] = taken[mask, index]
    dropped[:, 2] = CUBE_HALF
    dropped += env.scene.env_origins[selected]
    zeros = torch.zeros(len(selected), device=device)
    quat = quat_from_euler_xyz(zeros, zeros, yaw[mask, index])
    entity.write_root_link_pose_to_sim(
      torch.cat((dropped, quat), dim=-1), env_ids=selected
    )
    entity.write_root_link_velocity_to_sim(
      torch.zeros(len(selected), 6, device=device), env_ids=selected
    )


# Sliding friction, resampled every episode and shared by every cube and the
# tabletop. Stacking lives or dies on this: it sets whether a cube placed a
# millimetre off centre settles or slides off, and whether the one below it
# stays put while the gripper lets go. Training at a single coefficient trains
# a policy that only works at that coefficient.
#
# The numbers are Push-T's, which were set for printed plastic on an unsanded
# wooden board from the published range rather than from this table: generic
# plastic on wood is quoted at mu_k 0.40, neat PLA measures 0.65-0.70, and wear
# studies span 0.37-0.75. Two sigma covers 0.31-0.79, three covers 0.19-0.91.
# They are a literature prior, not a measurement of the real cubes -- weighing
# and tilt-testing those is still outstanding and would replace both numbers.
#
# One sample per env written to every surface, for the reason Push-T documents:
# MuJoCo takes the *maximum* coefficient of two equal-priority geoms, so
# randomizing the cubes alone would be masked by the table's own value. Writing
# the same number everywhere makes the effective coefficient equal the sample,
# for cube-table and cube-cube contacts alike.
CUBE_FRICTION_MEAN = 0.55
CUBE_FRICTION_STD = 0.12
TABLE_COLLISION_GEOM = "table_top"


@requires_model_fields("geom_friction")
def randomize_cube_friction(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  *,
  mean: float,
  std: float,
) -> None:
  """Give every cube and the tabletop one shared sliding coefficient."""
  ids = resolve_env_ids(env, env_ids).to(env.device)
  sampled = sample_gaussian(
    mean, std, (len(ids), 1), device=env.device
  ).clamp_min_(0.0)
  friction = env.sim.model.geom_friction

  table = env.scene["table"]
  surfaces = [env.scene[name].indexing.geom_ids for name in CUBE_NAMES]
  surfaces.append(table.indexing.geom_ids[table.find_geoms(TABLE_COLLISION_GEOM)[0]])
  for geoms in surfaces:
    env_grid, geom_grid = torch.meshgrid(ids.to(geoms.dtype), geoms, indexing="ij")
    friction[env_grid, geom_grid, 0] = sampled.expand(-1, len(geoms))


# Every cube is one colour, resampled each episode, and every cube gets the
# *same* one. Cube identity carries no meaning in this task, so four
# distinguishable colours would hand the policy a feature it could use to
# encode an order that does not exist; one shared colour makes them literally
# interchangeable, which is also what a real set of cubes looks like.
#
# The draw is saturated by construction rather than uniform over the RGB cube.
# Push-T measured what uniform costs: a quarter of its resets put the object
# within a 1.2:1 luminance ratio of its tabletop, leaving a silhouette no
# encoder can read. Here that would hide every cube at once. Rescaling each
# sample so its darkest channel lands at CUBE_COLOUR_FLOOR and its brightest at
# CUBE_COLOUR_CEILING keeps the hue uniform while guaranteeing both a bright
# channel and real chroma, so a cube can never come out as table-coloured grey.
CUBE_COLOUR_FLOOR = 0.10
CUBE_COLOUR_CEILING = 0.95


@requires_model_fields("geom_rgba")
def randomize_cube_colour(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
) -> None:
  """Give every cube in an env one shared, saturated colour."""
  ids = resolve_env_ids(env, env_ids).to(env.device)
  rgb = torch.rand(len(ids), 3, device=env.device)
  lowest = rgb.amin(dim=1, keepdim=True)
  highest = rgb.amax(dim=1, keepdim=True)
  rgb = (rgb - lowest) / (highest - lowest).clamp_min(1e-6)
  rgb = CUBE_COLOUR_FLOOR + rgb * (CUBE_COLOUR_CEILING - CUBE_COLOUR_FLOOR)

  geom_rgba = env.sim.model.geom_rgba
  for name in CUBE_NAMES:
    geoms = env.scene[name].indexing.geom_ids
    env_grid, geom_grid = torch.meshgrid(ids.to(geoms.dtype), geoms, indexing="ij")
    geom_rgba[env_grid, geom_grid, :3] = rgb.unsqueeze(1).expand(-1, len(geoms), 3)


def _write_cubes(
  env: ManagerBasedRlEnv,
  ids: torch.Tensor,
  position: torch.Tensor,
  yaw: torch.Tensor,
) -> None:
  """Teleport all four cube slots of ``ids`` and stop them dead."""
  origins = env.scene.env_origins[ids]
  zeros = torch.zeros(len(ids), device=env.device)
  velocity = torch.zeros(len(ids), 6, device=env.device)
  for index, name in enumerate(CUBE_NAMES):
    cube: Entity = env.scene[name]
    quat = quat_from_euler_xyz(zeros, zeros, yaw[:, index])
    cube.write_root_link_pose_to_sim(
      torch.cat((position[:, index] + origins, quat), dim=-1), env_ids=ids
    )
    cube.write_root_link_velocity_to_sim(velocity, env_ids=ids)


__all__ = [
  "CUBE_COLOUR_CEILING",
  "CUBE_COLOUR_FLOOR",
  "CUBE_FRICTION_MEAN",
  "CUBE_FRICTION_STD",
  "STACK_DECAY",
  "collapse_tower",
  "randomize_cube_colour",
  "randomize_cube_friction",
  "reset_stack_scenario",
]
