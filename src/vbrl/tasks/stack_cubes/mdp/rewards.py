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
# Where each stage starts and ends on the [0, 1] progress scale.
#
# Four of the five boundaries meet exactly, so the scalar is continuous there
# and no stage can outbid the one after it. The exception is deliberate: there
# is a 0.10 **gap** between reaching and grasping, because closing the hand has
# to pay something at the moment it happens.
#
# The first version had them all flush, which looked tidy and was wrong.
# Reaching tops out at 0.20 and grasping started at 0.20, so a policy that had
# driven the gripper onto a cube gained exactly nothing by closing it -- the
# only payoff was lifting, which it could not discover without first closing.
# ManiSkill3 does not do this either: their reaching term maxes at 2 and the
# grasped branch starts at 4, a jump of a quarter of the whole scale. This is
# the same idea at a tenth, which is worth 0.0225 of task reward per step.
BANDS = (
  (0.00, 0.20),  # reach
  (0.30, 0.45),  # grasped, lifting clear of the table
  (0.45, 0.60),  # carrying to the hover pose
  (0.60, 0.80),  # descending onto the tower
  (0.80, 1.00),  # released and settled
)
STAGES = len(BANDS)

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
  def band(index: int, fraction: torch.Tensor, gate: torch.Tensor | None = None):
    low, high = BANDS[index]
    value = low + (high - low) * fraction
    return value if gate is None else torch.where(gate, value, zero)

  bands = torch.stack(
    (
      band(0, reaching),
      band(1, lift, held),
      band(2, hovering, held & (lift >= 1.0)),
      band(3, placing, hover_error < HOVER_TOL),
      band(4, settled, placed),
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
  "BANDS",
  "HOVER_CLEARANCE",
  "HOVER_TOL",
  "LIFT_CLEARANCE",
  "STAGES",
  "TASK_SHARE",
  "home_pose",
  "stack_progress",
  "stage_scalar",
]
