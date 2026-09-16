"""Stack-Cubes terminations.

Deliberately short. A finished tower is not a termination and neither is a
tower that falls over -- both are states the policy is supposed to keep working
from. Two things do end an episode: a cube that has left the part of the table
the arm can work in, and a world whose physics has diverged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers import SceneEntityCfg

from .tower import CUBE_NAMES, tower_state

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# Where a cube is still worth recovering. The arm can grasp out to r = 0.479
# (measured -- see `tower.REACH_MAX`), so this is that radius with slack for a
# cube the policy can still push back into range, and a floor for one that has
# left the tabletop. It is deliberately larger than the spawn region: a cube
# nudged just outside the workspace is a state to recover from, not a failure.
RECOVERABLE_RADIUS = 0.60
FLOOR_Z = -0.02

# Divergence guards, at Push-T's velocity thresholds. An untrained policy drives
# this arm at 25+ rad/s, and MuJoCo Warp's near-rigid contact answers a fast
# gripper arriving inside a cube with an impulse large enough to take the whole
# world to NaN in a single step -- measured, every cube and every joint at once.
# Catching the state one step earlier, while it is merely implausible rather
# than non-finite, is what keeps the run alive: the world resets and training
# continues instead of RSL-RL's `check_nan` killing the rank and hanging the
# other three on an all-reduce that never arrives.
MAX_CUBE_HEIGHT = 0.50
MAX_CUBE_SPEED = 5.0
MAX_CUBE_SPIN = 50.0


def cube_out_of_reach(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Whether any cube has ended up somewhere unrecoverable.

  The tabletop is 1.0 m deep and the arm reaches 0.48, so a cube shoved far
  enough forward is genuinely gone -- that is a property of the real rig, not a
  modelling choice, and it is the one thing worth cutting an episode for. A
  tower that falls over is not: every cube stays in reach and rebuilding it is
  the behaviour being trained.
  """
  position = tower_state(env, asset_cfg).position
  radius = torch.linalg.vector_norm(position[..., :2], dim=-1)
  lost = (radius > RECOVERABLE_RADIUS) | (position[..., 2] < FLOOR_Z)
  return lost.any(dim=1)


def diverged(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Whether a cube has reached a speed or height physics cannot have produced."""
  cubes = [env.scene[name] for name in CUBE_NAMES]
  height = tower_state(env, asset_cfg).position[..., 2]
  speed = torch.stack(
    [cube.data.root_link_lin_vel_w.norm(dim=-1) for cube in cubes], dim=1
  )
  spin = torch.stack(
    [cube.data.root_link_ang_vel_w.norm(dim=-1) for cube in cubes], dim=1
  )
  implausible = (
    (height > MAX_CUBE_HEIGHT) | (speed > MAX_CUBE_SPEED) | (spin > MAX_CUBE_SPIN)
  )
  # `~isfinite` rather than `isnan`: a diverging solve reaches +-inf first.
  return (implausible | ~torch.isfinite(height)).any(dim=1)


__all__ = [
  "FLOOR_Z",
  "MAX_CUBE_HEIGHT",
  "MAX_CUBE_SPEED",
  "MAX_CUBE_SPIN",
  "RECOVERABLE_RADIUS",
  "cube_out_of_reach",
  "diverged",
]
