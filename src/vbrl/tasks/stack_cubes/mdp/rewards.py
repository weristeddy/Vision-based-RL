"""The one Stack-Cubes task reward, plus the stage scalar it is built from.

Three sources, combined: DeepMind RGB-Stacking's idea that the highest stage
reached sets the band (arXiv:2110.06192, eqs. S3-S12), ManiSkill3's
``1 - tanh(k d)`` distance shaping for each band, and a multi-cube
normalization that makes every completed cube worth the same whether there are
two of them or four.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers import SceneEntityCfg

from .tower import (
  CUBE_HALF,
  CUBE_SIZE,
  GRIPPER_OPEN_M,
  MAX_CUBES,
  TowerState,
  gather_rows,
  stack_point,
  tower_state,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# Five equal bands of 0.2. Each band's floor is the band below it at full
# value, so the scalar is continuous across a stage transition and no stage can
# outbid the one after it -- which is what keeps a large reaching term from
# competing with actually placing the cube.
STAGES = 5
BAND = 1.0 / STAGES

# ManiSkill3's Stack-Cube uses 1 - tanh(5 d) for reaching and placement.
REACH_STD = 0.2
HOVER_STD = 0.1
PLACE_STD = 0.05
# Enough clearance to carry a cube over one already lying on the table, and no
# more: the hover stage takes it the rest of the way up.
LIFT_CLEARANCE = CUBE_SIZE
# The cube is asked to pass one cube-height above its target level before
# descending onto it.
HOVER_CLEARANCE = CUBE_SIZE
HOVER_TOL = 0.06
# Reaching the tower is worth 0.9; the last tenth pays for withdrawing to the
# observation pose, so a complete tower is still worth more than an incomplete
# one however the arm is parked.
TASK_SHARE = 0.9
HOME_STD = 0.5


def stage_scalar(state: TowerState) -> torch.Tensor:
  """Normalized progress of the current target cube, in [0, 1].

  ``reach -> lift -> hover -> place -> release and settle``, taken as the
  largest band whose gate holds. Taking the maximum rather than a chain of
  conditions is what stops opening the gripper -- the very thing the last stage
  asks for -- from dropping the policy back into the reaching band.
  """
  cube = gather_rows(state.position, state.target)
  held = gather_rows(state.held, state.target)
  still = gather_rows(state.still, state.target)
  reach = gather_rows(state.reach, state.target)
  at_level = gather_rows(state.at_level, state.target)
  level = gather_rows(state.level, state.target)

  next_level = state.height.clamp(max=MAX_CUBES - 1)
  place = stack_point(next_level)
  hover = place.clone()
  hover[:, 2] += HOVER_CLEARANCE

  lift = ((cube[:, 2] - CUBE_HALF) / LIFT_CLEARANCE).clamp(0.0, 1.0)
  hover_error = torch.linalg.vector_norm(cube - hover, dim=-1)
  place_error = torch.linalg.vector_norm(cube - place, dim=-1)
  placed = at_level & (level == next_level)
  settled = 0.5 * (~held).float() + 0.5 * still.float()
  zero = torch.zeros_like(reach)

  reaching = _kernel(reach, REACH_STD)
  hovering = _kernel(hover_error, HOVER_STD)
  placing = _kernel(place_error, PLACE_STD)
  bands = torch.stack(
    (
      0 * BAND + BAND * reaching,
      torch.where(held, 1 * BAND + BAND * lift, zero),
      torch.where(held & (lift >= 1.0), 2 * BAND + BAND * hovering, zero),
      torch.where(hover_error < HOVER_TOL, 3 * BAND + BAND * placing, zero),
      torch.where(placed, 4 * BAND + BAND * settled, zero),
    ),
    dim=0,
  )
  return bands.amax(dim=0)


def home_pose(
  env: ManagerBasedRlEnv,
  arm_cfg: SceneEntityCfg,
  gripper_cfg: SceneEntityCfg,
  joint_pos: tuple[float, ...],
) -> torch.Tensor:
  """How close the arm is to the observation pose, with the gripper open."""
  robot: Entity = env.scene[arm_cfg.name]
  arm = robot.data.joint_pos[:, arm_cfg.joint_ids]
  error = torch.linalg.vector_norm(arm - arm.new_tensor(joint_pos), dim=-1)
  carriage = robot.data.joint_pos[:, gripper_cfg.joint_ids].squeeze(1)
  opened = (carriage / GRIPPER_OPEN_M).clamp(0.0, 1.0)
  return _kernel(error, HOME_STD) * opened


def stack_progress(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  arm_cfg: SceneEntityCfg,
  gripper_cfg: SceneEntityCfg,
  home_joint_pos: tuple[float, ...],
) -> torch.Tensor:
  """The whole task, normalized to roughly [0, 1].

  ``0.9 * (h + s) / 4`` while cubes are missing from the tower, so every cube
  is worth the same 0.225 and the stage scalar ``s`` of the one in progress
  fills the gap between courses continuously. Once all four are stacked the
  task pays a flat 0.9 and the last tenth buys the withdrawal to the
  observation pose -- which disappears by itself the moment the tower breaks,
  because ``complete`` stops holding.
  """
  state = tower_state(env, asset_cfg)
  progress = (state.height + stage_scalar(state)) / MAX_CUBES
  home = home_pose(env, arm_cfg, gripper_cfg, home_joint_pos)
  reward = torch.where(
    state.complete,
    TASK_SHARE + (1.0 - TASK_SHARE) * home,
    TASK_SHARE * progress,
  )
  # A world whose physics has diverged is terminated in this same step, but the
  # reward is computed before the reset and RSL-RL checks rewards as well as
  # observations -- one NaN here kills the rank. Score it zero and let the
  # termination do its job.
  return torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)


def _kernel(error: torch.Tensor, std: float) -> torch.Tensor:
  return 1.0 - torch.tanh(error / std)


__all__ = [
  "HOVER_CLEARANCE",
  "HOVER_TOL",
  "LIFT_CLEARANCE",
  "STAGES",
  "TASK_SHARE",
  "home_pose",
  "stack_progress",
  "stage_scalar",
]
