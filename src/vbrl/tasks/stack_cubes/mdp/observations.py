"""Stack-Cubes observation terms and the W&B progress metrics.

The visual actor gets none of this: it sees proprioception and one image, so
that the deployed policy has to read "a cube is on the table and not in the
tower" off pixels, which is the behaviour the real rig needs. The state actor
and every critic are privileged and take the lot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers import SceneEntityCfg

from .rewards import stage_scalar
from .tower import MAX_CUBES, gather_rows, grasp_force, tower_state

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# Close enough to the observation pose to count as arrived, for the metric
# only: 0.15 rad summed over six joints is a pose a human would call the same.
HOME_TOLERANCE_RAD = 0.15


def cube_positions(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Every cube slot's position in the env frame, flattened. Shape (B, 4*3).

  Position only. Cubes are geometrically symmetric, so their yaw decides
  nothing about whether they stack, and carrying it would be four dead inputs.
  """
  state = tower_state(env, asset_cfg)
  return state.position.flatten(start_dim=1)


def ee_to_target_cube(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Vector from the end effector to the cube due next. Shape (B, 3)."""
  state = tower_state(env, asset_cfg)
  robot = env.scene[asset_cfg.name]
  ee = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1) - env.scene.env_origins
  return gather_rows(state.position, state.target) - ee


def tower_progress(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """How much of the tower stands, over MAX_CUBES. Shape (B, 1)."""
  height = tower_state(env, asset_cfg).height
  return (height.float() / MAX_CUBES).unsqueeze(-1)


# --- metrics ----------------------------------------------------------------


def tower_height(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  return tower_state(env, asset_cfg).height.float()


def stack_fraction(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  return tower_state(env, asset_cfg).height.float() / MAX_CUBES


def tower_complete(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  return tower_state(env, asset_cfg).complete.float()


def peak_grasp_force(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Hardest the fingertips press any cube this step, in newtons."""
  return grasp_force(env).amax(dim=1)


def reward_stage(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """How far the course in progress has got, in [0, 1].

  Landmarks, measured in simulation on a course that adds cube two to a
  one-cube tower: **0.08** with the arm still at the observation pose, **0.57**
  with the hand on the cube before it has moved, **0.78** with the cube carried
  over its level, and **1.0** once it is set down, released and the hand is
  clear. It used to be reported on a 0-5 band scale, which the band ladder gave
  meaning and the Lift-Cube product form does not.
  """
  state = tower_state(env, asset_cfg)
  return torch.where(
    state.complete,
    torch.ones_like(state.height, dtype=torch.float),
    stage_scalar(state),
  )


def home_reached(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  arm_cfg: SceneEntityCfg,
  gripper_cfg: SceneEntityCfg,
  home_joint_pos: tuple[float, ...],
) -> torch.Tensor:
  """A complete tower *and* the arm withdrawn -- the deployed resting state."""
  robot = env.scene[arm_cfg.name]
  arm = robot.data.joint_pos[:, arm_cfg.joint_ids]
  error = torch.linalg.vector_norm(arm - arm.new_tensor(home_joint_pos), dim=-1)
  return (tower_state(env, asset_cfg).complete & (error < HOME_TOLERANCE_RAD)).float()


__all__ = [
  "HOME_TOLERANCE_RAD",
  "cube_positions",
  "ee_to_target_cube",
  "home_reached",
  "peak_grasp_force",
  "reward_stage",
  "stack_fraction",
  "tower_complete",
  "tower_height",
  "tower_progress",
]
