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
# Two of the four boundaries meet exactly; the other two are deliberate
# **gaps**, at the two moments that are achievements rather than progress --
# closing the hand on a cube, and getting that cube onto the tower. Each has to
# pay something the instant it happens or the policy cannot discover the stage
# that follows it.
#
# The first version had every boundary flush, which looked tidy and was wrong:
# reaching topped out at exactly the value grasping started at, so a policy
# already touching a cube gained nothing by closing, and the only payoff was
# lifting -- which it could not reach without first closing.
#
# ManiSkill3 puts gaps in the same two places and larger ones. Normalized by
# their success bonus of 8: reaching spans [0, 0.25], the grasped branch starts
# at 0.5, the on-the-tower branch starts at 0.75, and success is 1.0 -- 37% of
# their scale is gaps. These two are 0.10 each, a fifth of the scale, which is
# the same structure at a gentler size.
BANDS = (
  (0.00, 0.18),  # reach
  (0.28, 0.42),  # grasped, lifting clear of the table
  (0.42, 0.56),  # carrying to the hover pose
  (0.56, 0.72),  # descending onto the tower
  (0.82, 1.00),  # on the tower, letting go and settling
)
STAGES = len(BANDS)

# ManiSkill3's `1 - tanh(5 d)`, with DeepMind's flat top inside
# `_REACH_POSITION_TOLERANCE = 0.02` so the term genuinely reaches 1.0 and the
# `GRASP_GATE` below is crossable -- the end-effector site cannot sit at a
# cube's centre, so an ungated kernel tops out near 0.90.
#
# DeepMind's shaping tolerance of 0.15 m was tried here and is **wrong for this
# workspace**. Measured over 3,072 resets, an episode starts with the target
# cube a median 0.207 m from the end-effector (q01 0.123, q99 0.263): their
# basket keeps the pinch close to the objects, this table does not. Through
# their `tanh_squared` that median pays 0.011 against this kernel's 0.226 -- a
# factor of 20, and 144 at 0.30 m -- so the band that has to get the hand to a
# cube in the first place was flat across the entire workspace. Both visual
# runs then diverged outright (value loss 4e9, `action_rate_l2` -48,000, the
# arm slamming joint limits) and both state runs spent 83% of their episodes
# ending on table contact at 45% of full length. One kernel, two failure modes,
# same cause.
REACH_STD = 0.2
REACH_TOLERANCE = 0.02
# `ConditionalAnd(reach_red, grasp, 0.9)`: the reach term has to clear 0.9
# before the grasp half of the band unlocks.
GRASP_GATE = 0.9
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
# DeepMind's `_get_reward_above_red`: `offset=(0, 0, _HOVER_OFFSET)` with
# `_HOVER_OFFSET = 0.1`, `position_tolerance=(1, 1, 0.03)` -- anywhere
# horizontally -- and `shaping_tolerance=0.05`.
RETREAT_HEIGHT = 0.10
RETREAT_MARGIN = 0.05
RETREAT_TOLERANCE = (1.0, 1.0, 0.03)


def stage_scalar(state: TowerState) -> torch.Tensor:
  """Normalized progress of the current target cube, in [0, 1].

  ``reach -> lift -> hover -> place -> release and settle``, taken as the
  largest band whose gate holds. Taking the maximum rather than a chain of
  conditions is what stops opening the gripper -- the very thing the last stage
  asks for -- from dropping the policy back into the reaching band.
  """
  cube = gather_rows(state.position, state.target)
  held = gather_rows(state.held, state.target)
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
  # ManiSkill3's `(ungrasp_reward + static_reward) / 2`, continuous in both like
  # theirs, plus DeepMind's third factor. The first version made each half a
  # step function -- released or not, still or not -- which left the last stage,
  # the one that has to teach letting go gently, with no gradient at all.
  opening = torch.where(held, state.opening, torch.ones_like(state.opening))
  speed = gather_rows(state.speed, state.target)
  spin = gather_rows(state.spin, state.target)
  static = 1.0 - torch.tanh(10.0 * speed + spin)
  settled = (opening + static + _retreat(state.ee, cube)) / 3.0
  zero = torch.zeros_like(reach)

  reaching = _reaching(reach)
  hovering = _kernel(hover_error, HOVER_STD)
  placing = _kernel(place_error, PLACE_STD)
  def band(index: int, fraction: torch.Tensor, gate: torch.Tensor | None = None):
    low, high = BANDS[index]
    value = low + (high - low) * fraction
    return value if gate is None else torch.where(gate, value, zero)

  bands = torch.stack(
    (
      band(0, _reach_and_grasp(reaching, state.closure, held)),
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


def _tanh_squared(error: torch.Tensor, margin: float) -> torch.Tensor:
  """DeepMind's shaping loss, ``1 - tanh(w d)^2`` with ``loss_at_margin`` 0.95.

  ``reward_functions.tanh_squared``, verbatim: ``w`` is chosen so the loss is
  0.95 exactly at ``margin``. It is flat near zero and falls off hard past the
  margin, where ``1 - tanh(d / std)`` is steep near zero and has a long tail.
  """
  weight = math.atanh(math.sqrt(0.95)) / margin
  return 1.0 - torch.tanh(weight * error) ** 2


def _reaching(reach: torch.Tensor) -> torch.Tensor:
  """Flat 1.0 inside the tolerance, ManiSkill's ``1 - tanh(5 d)`` outside.

  The two meet where they should: ``1 - tanh(d / 0.2)`` is 0.90 at exactly
  20.1 mm, so the flat top begins at the tolerance rather than jumping to it.
  """
  return torch.where(
    reach <= REACH_TOLERANCE, torch.ones_like(reach), _kernel(reach, REACH_STD)
  )


def _reach_and_grasp(
  reaching: torch.Tensor, closure: torch.Tensor, held: torch.Tensor
) -> torch.Tensor:
  """DeepMind's ``ConditionalAnd(reach_red, grasp, 0.9)``.

  Their first stage fuses reaching with grasping and caps reaching alone at
  **half** the band; the other half unlocks only once the hand is essentially
  on the cube, and is then ``Max((close_fingers, 0.5), (grasp, 1.0))`` -- a
  continuous half for squeezing and a full one for a grasp that holds.

  This band used to be reaching alone at full value, and that is the measured
  reason three state runs stalled. Hovering over a cube with an open hand
  scored 0.162 of the scalar against 0.280 for a grasp, so 58% of everything a
  course could earn was available without touching anything, immediately and
  with no risk -- closing on an off-centre cube pushes it away and *loses*
  reach. All three runs converged on hovering: ``reward_stage`` flat at 0.94 of
  5 for 1,000+ iterations, policy entropy down to 0.013, peak fingertip force
  climbing to 90 N against cubes that never moved. Under DeepMind's structure
  the same hover pays 0.09, squeezing pays 0.135, a grasp pays 0.18 and lifting
  0.28 -- a monotone path where there was a cliff.
  """
  grasp = torch.maximum(0.5 * closure, held.float())
  return torch.where(
    reaching > GRASP_GATE, (0.5 + 0.5 * grasp) * reaching, 0.5 * reaching
  )


def _retreat(ee: torch.Tensor, cube: torch.Tensor) -> torch.Tensor:
  """DeepMind's ``_get_reward_above_red``: get the hand off the cube you placed.

  Their last stage is ``Product(stack, above_red)``, so a course only pays in
  full once the pinch point is ``RETREAT_HEIGHT`` above it. Horizontal position
  is free -- their ``position_tolerance`` is ``(1, 1, 0.03)``, metres -- so this
  asks for height, not a particular withdrawal path.

  Without it the only reward for getting out of the way came from
  ``home_pose``, which fires on a *finished* tower, so courses 1 to 3 had none
  at all: the hand could sit on the cube it had just released while the next
  approach started from inside the tower.
  """
  delta = cube - ee
  delta = torch.stack(
    (delta[:, 0], delta[:, 1], delta[:, 2] + RETREAT_HEIGHT), dim=-1
  )
  tolerance = delta.new_tensor(RETREAT_TOLERANCE)
  inside = torch.linalg.vector_norm(delta / tolerance, dim=-1) <= 1.0
  error = torch.linalg.vector_norm(delta, dim=-1)
  return torch.where(
    inside, torch.ones_like(error), _tanh_squared(error, RETREAT_MARGIN)
  )


def _kernel(error: torch.Tensor, std: float) -> torch.Tensor:
  return 1.0 - torch.tanh(error / std)


__all__ = [
  "BANDS",
  "GRASP_GATE",
  "REACH_STD",
  "REACH_TOLERANCE",
  "RETREAT_HEIGHT",
  "HOVER_CLEARANCE",
  "HOVER_TOL",
  "LIFT_CLEARANCE",
  "STAGES",
  "TASK_SHARE",
  "home_pose",
  "stack_progress",
  "stage_scalar",
]
