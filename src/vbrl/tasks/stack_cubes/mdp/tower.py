"""Where the tower is, how tall it is, and which cube the robot owes it next.

The one piece of genuinely task-specific state Stack-Cubes needs, and the only
module that knows what "stacked" means. Everything here is permutation
independent: a level belongs to whichever cube happens to occupy it, so a tower
built ``cube_3, cube_0, cube_2`` is the same tower as ``cube_0, cube_1,
cube_2``. The height is the *contiguous* run of occupied levels from the table
upward, which is what stops two separate pairs from reading as a four-stack.

All four cubes are always in play. There is no active mask and no parking: the
task is to get every cube on the table into the tower, and the only thing that
changes during an episode is how much of it is built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from mjlab.entity import Entity
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import matrix_from_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


MAX_CUBES = 4
CUBE_NAMES = tuple(f"cube_{index}" for index in range(MAX_CUBES))

# ``asset_zoo/objects/cube.xml`` is a 40 mm box. Every height in this task is
# derived from it rather than written out, so a different cube moves the whole
# geometry with it.
CUBE_SIZE = 0.04
CUBE_HALF = CUBE_SIZE / 2

# The fixed tower position, in the robot's base frame. It is the centre of the
# region MJLab's Lift-Cube samples objects from (x 0.20-0.40, y +-0.20), which
# is the part of the workspace this arm is known to grasp in, and y = 0 puts the
# column in the arm's own sagittal plane where a vertical stack is best
# conditioned. Damped-least-squares IK reaches every spawn slot at table height
# and the top of a four-cube tower from here to under 0.1 mm.
STACK_XY = (0.34, 0.0)

# The loose-cube workspace: uniform, not a slot grid. Its size is a three-way
# compromise, measured rather than chosen. Four cubes plus the tower and
# gripper keep-outs exclude 62% of this rectangle, so the fourth cube accepts a
# uniform draw 38% of the time -- shrink it and rejection sampling runs into
# the random-close-packing wall (the first attempt, 0.25 x 0.37 m, excluded 81%
# and could not be sampled), widen it and the wrist camera stops covering it
# from the observation pose (8.5 px of margin here, 3.7 px at 0.28 x 0.40).
# Damped-least-squares IK reaches every corner at grasp height.
WORKSPACE_X = (0.210, 0.470)
WORKSPACE_Y = (-0.190, 0.190)
# ...intersected with the arm's own reach, because the rectangle's far corners
# are not. Measured by damped-least-squares IK from 25 seeds under the soft
# joint limits: the gripper can be brought straight down onto a cube at
# r = 0.479 but not at r = 0.484, in every direction tried, so the graspable
# region is a disc rather than a box. 0.47 leaves a centimetre for an IK that
# knows nothing about self-collision. Without this cap 5.1% of the rectangle --
# one spawned cube in twenty -- would be somewhere the policy could never pick
# it up, which caps the task reward for the whole episode and looks exactly
# like a policy failure.
# A 40 mm cube has a 56.6 mm face diagonal, so two centres closer than that can
# interpenetrate at some yaw. These leave 3.4 mm and 5.4 mm of clearance.
MIN_CUBE_SEPARATION = 0.060
STACK_KEEPOUT = 0.062
# A cube never spawns anywhere the arm cannot pick it up from. The measured
# graspable boundary is r = 0.479 (see the comment on WORKSPACE_X), and this
# stops 29 mm inside it: that IK knew nothing about self-collision, only
# checked the gripper axis at the cube's centre, and found the optimal solution
# rather than the one a learned policy will find. It also keeps loose cubes
# well away from the recovery boundary -- measured over 40 s episodes of
# untrained-policy flailing, the 90th-percentile peak cube radius is 0.468, so
# cubes are lost by starting near the edge far more than by being pushed there.
REACH_MAX = 0.45
# A cube is never dropped into the hand. The gripper only blocks ground while
# it is low enough to touch a cube lying on the table; above that it is out of
# the way and stops blocking by itself.
GRIPPER_KEEPOUT = 0.070
GRIPPER_KEEPOUT_Z = 0.12
# The tower column cannot be moved out of the gripper's way, so a mid-episode
# redraw skips an env whose hand is standing in it and picks the next firing
# up instead. A four-cube tower's top face is at 0.16; the robot's own start
# pose sits at 0.21, so an episode reset never trips this.
TOWER_KEEPOUT_Z = 0.18
# Redraws per cube. The reach cap takes acceptance to 29% for the fourth cube,
# which leaves a 1.5e-6 chance of placing one on a blocked spot -- measured at
# zero over a million draws in `tests/test_stack_cubes.py`. Each pass is one
# small vectorized draw and resets are per-episode, so the cost is noise.
RESAMPLE_PASSES = 40

# A level counts when the cube is within 20 mm of the stack axis and 12 mm of
# the level's height. ManiSkill3's Stack-Cube accepts a placement inside the
# cube's own half-width; this is tighter, because a tower has to survive the
# next cube landing on it.
STACK_XY_TOL = 0.02
STACK_Z_TOL = 0.012
# One body axis within 18 degrees of world z. A cube is symmetric, so "upright"
# is "a face is down", not "the yaw is right".
UPRIGHT_MIN = 0.95
# Measured on cubes that are geometrically seated and out of the gripper: the
# 99th percentile of their residual motion is 0.015 m/s and 0.52 rad/s, because
# a stiff contact rings rather than going exactly to zero. The first thresholds
# tried, 0.02 and 0.5, cut the tower's own height on 1.5% of steps -- a 0.3
# dip in a reward that only spans 1.0, for nothing. These keep 99.75% of seated
# cubes. They cannot let a cube in transit count: to be judged at all it must
# already be upright, within 20 mm of the stack axis and within 12 mm of an
# exact level height, and a cube falling through that band from one cube-height
# up crosses it in a single substep at 30x this speed.
STATIC_SPEED = 0.03
STATIC_SPIN = 1.0

# The carriage position that counts as a closed hand, derived rather than
# guessed -- getting this wrong is silent and total.
#
# The pad *faces* are 0.003 + 2*carriage apart (centre-to-centre is
# 0.013 + 2*carriage and each pad is 5 mm thick, which is the distinction the
# first attempt got wrong). A hand squeezing a 40 mm cube therefore cannot
# close past (0.040 - 0.003)/2 = 0.0185: the cube is in the way. Measured, it
# settles at 0.01857 against a commanded 0.0001.
#
# The first threshold was 0.018 -- five millimetres too tight, and by bad luck
# just *below* that stall. `held` was then false whenever the gripper was
# actually holding something, so the policy reached, closed, squeezed at up to
# 113 N and never left the reaching band for 2,300 iterations. The threshold
# has to sit *above* the stall and below an open hand.
GRIPPER_PAD_FACE_OFFSET = 0.003
GRIPPER_GRIP_M = (CUBE_SIZE - GRIPPER_PAD_FACE_OFFSET) / 2
GRIPPER_CLOSED_M = GRIPPER_GRIP_M + 0.002
GRIPPER_OPEN_M = 0.022
# One contact sensor per cube, named after it. The primary set is the six
# fingertip pad geoms, so the sensor reports which pads are touching that cube
# and how hard -- which is what turns "the hand is near a cube" into "the hand
# is holding it".
CONTACT_SENSORS = tuple(f"{name}_grasp" for name in CUBE_NAMES)


class TowerState(NamedTuple):
  """One step of tower bookkeeping, shared by rewards, observations and events."""

  position: torch.Tensor  # (B, N, 3) cube positions in the env frame
  reach: torch.Tensor  # (B, N) end-effector to cube distance
  held: torch.Tensor  # (B, N) inside a closed gripper
  still: torch.Tensor  # (B, N) not moving
  at_level: torch.Tensor  # (B, N) seated on its level, grasp and motion aside
  level: torch.Tensor  # (B, N) which level its height puts it on
  stacked: torch.Tensor  # (B, N) part of the contiguous tower
  height: torch.Tensor  # (B,) contiguous stable cubes, 0..MAX_CUBES
  complete: torch.Tensor  # (B,) every cube is in the tower
  target: torch.Tensor  # (B,) the cube to place next


def level_height(level: torch.Tensor | int) -> torch.Tensor | float:
  """Centre height of a cube resting on ``level`` others."""
  return CUBE_HALF + level * CUBE_SIZE


def stack_point(level: torch.Tensor) -> torch.Tensor:
  """The env-frame point a cube must reach to claim ``level``. Shape (B, 3)."""
  point = torch.zeros(len(level), 3, device=level.device)
  point[:, 0] = STACK_XY[0]
  point[:, 1] = STACK_XY[1]
  point[:, 2] = level_height(level.float())
  return point


def _pinched(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Whether a left *and* a right fingertip touch each cube. Shape (B, N)."""
  sides = []
  for name in CONTACT_SENSORS:
    sensor = env.scene[name]
    found = sensor.data.found
    assert found is not None, f"Contact sensor {name!r} reports no `found`."
    pads = sensor.primary_names
    left = torch.tensor(
      [index for index, pad in enumerate(pads) if "left" in pad],
      device=found.device,
    )
    right = torch.tensor(
      [index for index, pad in enumerate(pads) if "right" in pad],
      device=found.device,
    )
    touching = found > 0
    sides.append(touching[:, left].any(-1) & touching[:, right].any(-1))
  return torch.stack(sides, dim=1)


def grasp_force(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Largest fingertip contact force on each cube, in newtons. Shape (B, N)."""
  peaks = []
  for name in CONTACT_SENSORS:
    force = env.scene[name].data.force
    assert force is not None, f"Contact sensor {name!r} reports no `force`."
    peaks.append(torch.linalg.vector_norm(force, dim=-1).amax(dim=-1))
  return torch.stack(peaks, dim=1)


def tower_state(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> TowerState:
  """Read the whole task state off the simulator.

  ``asset_cfg`` names the robot, its end-effector site, and the gripper carriage
  joint -- everything about the robot this task's logic depends on.
  """
  robot: Entity = env.scene[asset_cfg.name]
  origins = env.scene.env_origins
  cubes = [env.scene[name] for name in CUBE_NAMES]

  position = (
    torch.stack([cube.data.root_link_pos_w for cube in cubes], dim=1)
    - origins.unsqueeze(1)
  )
  quat = torch.stack([cube.data.root_link_quat_w for cube in cubes], dim=1)
  speed = torch.stack(
    [cube.data.root_link_lin_vel_w.norm(dim=-1) for cube in cubes], dim=1
  )
  spin = torch.stack(
    [cube.data.root_link_ang_vel_w.norm(dim=-1) for cube in cubes], dim=1
  )

  ee = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1) - origins
  carriage = robot.data.joint_pos[:, asset_cfg.joint_ids].squeeze(1)

  reach = torch.linalg.vector_norm(position - ee.unsqueeze(1), dim=-1)
  # A cube is held when both fingers are touching it and the hand is closed.
  # This is ManiSkill3's two-finger contact test rather than a distance
  # threshold, and the difference matters in one specific place: a closed but
  # empty gripper withdrawing past the tower used to read as "holding" the cube
  # it passed, which struck that course out of the count and dropped the reward
  # for no reason. Contact cannot be fooled that way.
  closed = (carriage < GRIPPER_CLOSED_M).unsqueeze(1)
  held = closed & _pinched(env)
  still = (speed < STATIC_SPEED) & (spin < STATIC_SPIN)

  # Row 2 of the rotation matrix is the world-z component of each body axis, so
  # its largest magnitude says how close the nearest face is to being down.
  world_z = matrix_from_quat(quat.reshape(-1, 4))[:, 2, :].abs().amax(dim=-1)
  upright = world_z.reshape(quat.shape[:2]) > UPRIGHT_MIN

  height_error = position[..., 2] - CUBE_HALF
  level = torch.round(height_error / CUBE_SIZE)
  at_level = (
    upright
    & (
      torch.linalg.vector_norm(
        position[..., :2] - position.new_tensor(STACK_XY), dim=-1
      )
      < STACK_XY_TOL
    )
    & ((height_error - level * CUBE_SIZE).abs() < STACK_Z_TOL)
    & (level >= 0)
    & (level < MAX_CUBES)
  )
  level = level.clamp(0, MAX_CUBES - 1).long()

  seated = at_level & still & ~held
  occupied = torch.stack(
    [(seated & (level == index)).any(dim=1) for index in range(MAX_CUBES)], dim=1
  )
  height = torch.cumprod(occupied.long(), dim=1).sum(dim=1)
  stacked = seated & (level < height.unsqueeze(1))

  complete = height >= MAX_CUBES

  # The cube already in hand stays the target; otherwise the nearest loose one.
  # Identity plays no part, which is what makes the choice permutation
  # independent.
  loose = ~stacked
  score = torch.where(held & loose, torch.full_like(reach, -1.0), reach)
  score = torch.where(loose, score, torch.full_like(reach, float("inf")))

  return TowerState(
    position=position,
    reach=reach,
    held=held,
    still=still,
    at_level=at_level,
    level=level,
    stacked=stacked,
    height=height,
    complete=complete,
    target=score.argmin(dim=1),
  )


def _uniform_xy(envs: int, cubes: int, device) -> torch.Tensor:
  lower = torch.tensor([WORKSPACE_X[0], WORKSPACE_Y[0]], device=device)
  upper = torch.tensor([WORKSPACE_X[1], WORKSPACE_Y[1]], device=device)
  return lower + (upper - lower) * torch.rand(envs, cubes, 2, device=device)


def sample_loose_xy(envs: int, cubes: int, blocked: torch.Tensor) -> torch.Tensor:
  """Uniform table positions, far enough apart and clear of every blocker.

  ``blocked`` is ``(B, M, 3)`` -- an xy centre and an exclusion radius per
  blocker, which is how the tower column and the gripper keep cubes out of
  themselves. A radius of zero disables one, so the gripper stops blocking
  simply by lifting.

  Uniform over the whole rectangle rather than over a grid of slots, because
  where a cube can be is most of what a visual policy has to generalize over.
  Cubes are placed one at a time and each redraw only moves the cube being
  placed, so a rejection cannot cascade into the ones already down and the
  failure rate falls geometrically with ``RESAMPLE_PASSES`` -- which resampling
  all four together does not do, because every repair breaks its neighbours.
  Returns ``(B, cubes, 2)``.
  """
  device = blocked.device
  placed = blocked
  drawn = []
  for _ in range(cubes):
    xy = _uniform_xy(envs, 1, device).squeeze(1)
    for _ in range(RESAMPLE_PASSES):
      gap = torch.linalg.vector_norm(xy.unsqueeze(1) - placed[..., :2], dim=-1)
      invalid = (gap < placed[..., 2]).any(dim=-1)
      invalid |= torch.linalg.vector_norm(xy, dim=-1) > REACH_MAX
      if not invalid.any():
        break
      fresh = _uniform_xy(envs, 1, device).squeeze(1)
      xy = torch.where(invalid.unsqueeze(-1), fresh, xy)
    drawn.append(xy)
    radius = xy.new_full((envs, 1), MIN_CUBE_SEPARATION)
    placed = torch.cat((placed, torch.cat((xy, radius), dim=-1).unsqueeze(1)), dim=1)
  return torch.stack(drawn, dim=1)


def blockers(
  env: ManagerBasedRlEnv,
  ids: torch.Tensor,
  asset_cfg: SceneEntityCfg | None,
) -> torch.Tensor:
  """The tower column, plus the gripper when it is low enough to matter.

  ``asset_cfg`` of ``None`` drops the gripper blocker, which is what an episode
  reset wants: the arm is on its way back to the start pose, so its current
  position says nothing about where it will be.
  """
  device = env.device
  blocked = torch.zeros(len(ids), 2, 3, device=device)
  blocked[:, 0, :2] = torch.tensor(STACK_XY, device=device)
  blocked[:, 0, 2] = STACK_KEEPOUT
  if asset_cfg is not None:
    ee = gripper_position(env, ids, asset_cfg)
    blocked[:, 1, :2] = ee[:, :2]
    blocked[:, 1, 2] = torch.where(
      ee[:, 2] < GRIPPER_KEEPOUT_Z, GRIPPER_KEEPOUT, 0.0
    )
  return blocked


def gripper_position(
  env: ManagerBasedRlEnv, ids: torch.Tensor, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """End-effector position in the env frame, for the selected envs. (B, 3)."""
  robot: Entity = env.scene[asset_cfg.name]
  return (
    robot.data.site_pos_w[ids][:, asset_cfg.site_ids].squeeze(1)
    - env.scene.env_origins[ids]
  )


def gripper_blocks_the_tower(
  env: ManagerBasedRlEnv, ids: torch.Tensor, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Whether the hand is standing where the tower would be rebuilt. (B,)."""
  ee = gripper_position(env, ids, asset_cfg)
  reach = torch.linalg.vector_norm(
    ee[:, :2] - ee.new_tensor(STACK_XY), dim=-1
  )
  return (reach < GRIPPER_KEEPOUT) & (ee[:, 2] < TOWER_KEEPOUT_Z)


def gather_rows(values: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
  """Pick one column per row: ``values[b, index[b]]``."""
  return values[torch.arange(len(index), device=values.device), index]


__all__ = [
  "CUBE_HALF",
  "CUBE_NAMES",
  "CUBE_SIZE",
  "CONTACT_SENSORS",
  "GRIPPER_CLOSED_M",
  "GRIPPER_GRIP_M",
  "GRIPPER_PAD_FACE_OFFSET",
  "GRIPPER_OPEN_M",
  "MAX_CUBES",
  "MIN_CUBE_SEPARATION",
  "REACH_MAX",
  "RESAMPLE_PASSES",
  "STACK_KEEPOUT",
  "STACK_XY",
  "STACK_XY_TOL",
  "STACK_Z_TOL",
  "STATIC_SPEED",
  "STATIC_SPIN",
  "UPRIGHT_MIN",
  "WORKSPACE_X",
  "WORKSPACE_Y",
  "TowerState",
  "blockers",
  "gather_rows",
  "grasp_force",
  "gripper_blocks_the_tower",
  "gripper_position",
  "level_height",
  "sample_loose_xy",
  "stack_point",
  "tower_state",
]
