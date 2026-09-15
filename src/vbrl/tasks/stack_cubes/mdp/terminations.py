"""Stack-Cubes terminations.

Deliberately short. A finished tower is not a termination and neither is a
tower that falls over -- both are states the policy is supposed to keep working
from. Only a cube that has left the part of the table the arm can work in is
unrecoverable, and that is the one thing worth cutting an episode for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers import SceneEntityCfg

from .tower import tower_state

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# Where a cube is still worth recovering. The arm can grasp out to r = 0.479
# (measured -- see `tower.REACH_MAX`), so this is that radius with slack for a
# cube the policy can still push back into range, and a floor for one that has
# left the tabletop. It is deliberately larger than the spawn region: a cube
# nudged just outside the workspace is a state to recover from, not a failure.
RECOVERABLE_RADIUS = 0.60
FLOOR_Z = -0.02


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


__all__ = ["FLOOR_Z", "RECOVERABLE_RADIUS", "cube_out_of_reach"]
