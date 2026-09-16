"""The one Stack-Cubes task reward, plus the stage scalar it is built from.

Three sources, combined: DeepMind RGB-Stacking's idea that the highest stage
reached sets the band (arXiv:2110.06192; the released code is
``rgb_stacking/stack_rewards.py`` and ``rgb_stacking/reward_functions.py``),
ManiSkill3's ``1 - tanh(k d)`` distance shaping for each band, and a multi-cube
normalization that makes every completed cube worth the same whether there are
two of them or four.

Two of the five bands are DeepMind's structure rather than only their idea:
the reach band is ``ConditionalAnd(reach, grasp, 0.9)`` over
``Max((close_fingers, 0.5), (grasp, 1.0))``, and the settle band carries their
``_get_reward_above_red`` retreat factor. Both are corrections, and both are
measured -- see `_reach_and_grasp` and `_retreat`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers import SceneEntityCfg

from .tower import (
  GRIPPER_OPEN_M,
  MAX_CUBES,
  TowerState,
  stack_point,
  tower_state,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# ManiSkill3's `StackCube-v1`, `mani_skill/envs/tasks/tabletop/stack_cube.py`.
# Their ladder in full, and the only published dense reward for this exact task:
#
#     reward = 2 * (1 - tanh(5 * d(tcp, cubeA)))
#     reward[is_cubeA_grasped] = 4 + (1 - tanh(5 * d(cubeA, goal)))
#     reward[is_cubeA_on_cubeB] = 6 + (ungrasp + static) / 2
#     reward[success] = 8
#
# Note these are *assignments*, not maxima: each branch overwrites the one above
# it, so a grasped cube scores 4 however far it is from the goal. `goal` is
# cubeB's top face, which is `stack_point`. `REACH_K = 5` is `1 / 0.2`, the same
# width MJLab Lift-Cube uses for its own reaching kernel.
STACK_SCALE = 8.0
REACH_K = 5.0
PLACE_K = 5.0
STATIC_K = 10.0
# Reaching the tower is worth 0.9; the last tenth pays for withdrawing to the
# observation pose, so a complete tower is still worth more than an incomplete
# one however the arm is parked.
TASK_SHARE = 0.9
HOME_STD = 0.5


def stage_scalar(state: TowerState) -> torch.Tensor:
  """How far the best-placed loose cube is through its course, in [0, 1].

  ManiSkill3's `StackCube-v1` dense reward, branch for branch, divided by their
  success value of 8 so one course spans [0, 1] and four of them compose. The
  mapping onto this task is exact: their ``is_cubeA_grasped`` is ``held``,
  their ``is_cubeA_on_cubeB`` is ``at_level``, their ``goal_xyz`` -- cubeB's top
  face -- is ``stack_point``, and their ``success`` is ``seated``.

  Two deliberate departures, both forced by this task rather than chosen:

  * **The maximum over loose cubes.** They have one cube to place; this has up
    to four, and the reward has to say which. A per-step nearest-cube ``argmin``
    let the target flip between two cubes as the hand passed between them,
    stepping the reward with it. MJLab's own ``MultiCubeLiftingCommand`` latches
    its choice at reset for the same reason; a maximum of continuous functions
    is continuous and needs no latch.
  * **``ungrasp`` is measured against ``GRIPPER_OPEN_M``**, the width that clears
    a cube, not against the carriage's mechanical limit. Theirs divides by the
    Panda's joint limit and carries a ``# NOTE: hard-coded with panda`` beside
    it; this gripper's release happens at 22 mm of a 44 mm travel, so the limit
    would score a hand that has fully let go at only one half.

  Their ``reward[success] = 8`` has no branch here because it cannot be reached:
  a seated cube is part of the tower in the same ``tower_state`` that seats it,
  so it leaves the pool this maximum runs over. The course is paid for by
  ``height`` rising instead, which is worth ``0.9 / 4`` for good -- more than
  the 8 would have been, and it does not decay.
  """
  goal = stack_point(state.height.clamp(max=MAX_CUBES - 1))
  place_error = torch.linalg.vector_norm(state.position - goal.unsqueeze(1), dim=-1)

  reward = 2.0 * (1.0 - torch.tanh(REACH_K * state.reach))

  place = 1.0 - torch.tanh(PLACE_K * place_error)
  reward = torch.where(state.held, 4.0 + place, reward)

  ungrasp = torch.where(
    state.held, state.opening.unsqueeze(1), torch.ones_like(state.reach)
  )
  static = 1.0 - torch.tanh(STATIC_K * state.speed + state.spin)
  on_level = state.at_level & (
    state.level == state.height.clamp(max=MAX_CUBES - 1).unsqueeze(1)
  )
  reward = torch.where(on_level, 6.0 + (ungrasp + static) / 2.0, reward)

  stage = reward / STACK_SCALE
  return stage.masked_fill(state.stacked, -1.0).amax(dim=1).clamp(min=0.0)


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
  return torch.exp(-torch.square(error) / HOME_STD**2) * opened


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


def _tanh_squared(error: torch.Tensor, margin: float) -> torch.Tensor:
  """DeepMind's shaping loss, ``1 - tanh(w d)^2`` with ``loss_at_margin`` 0.95.

  ``reward_functions.tanh_squared``, verbatim: ``w`` is chosen so the loss is
  0.95 exactly at ``margin``. It is flat near zero and falls off hard past the
  margin, where ``1 - tanh(d / std)`` is steep near zero and has a long tail.
  """
  weight = math.atanh(math.sqrt(0.95)) / margin
  return 1.0 - torch.tanh(weight * error) ** 2


__all__ = [
  "PLACE_K",
  "REACH_K",
  "STACK_SCALE",
  "STATIC_K",
  "TASK_SHARE",
  "home_pose",
  "stack_progress",
  "stage_scalar",
]
