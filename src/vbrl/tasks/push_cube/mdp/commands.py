"""Separated planar command sampling for Push-Cube."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.manipulation.mdp import LiftingCommand, LiftingCommandCfg
from mjlab.utils.lab_api.math import (
  quat_from_euler_xyz,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_TARGET_CANDIDATES = 64


class PushingCommand(LiftingCommand):
  """MJLab lifting command with a separated planar object and goal."""

  cfg: PushingCommandCfg

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self.episode_success[env_ids] = 0.0
    object_range = self.cfg.object_pose_range
    assert object_range is not None

    def sample_xyz(ranges, shape) -> torch.Tensor:
      lower = torch.tensor(
        [ranges.x[0], ranges.y[0], ranges.z[0]], device=self.device
      )
      upper = torch.tensor(
        [ranges.x[1], ranges.y[1], ranges.z[1]], device=self.device
      )
      return sample_uniform(lower, upper, shape, device=self.device)

    object_pos = sample_xyz(object_range, (n, 3))
    target_range = self.cfg.target_position_range
    candidates = sample_xyz(target_range, (n, _TARGET_CANDIDATES, 3))
    valid = (
      torch.linalg.vector_norm(
        candidates[..., :2] - object_pos[:, None, :2], dim=-1
      )
      >= self.cfg.min_xy_separation
    )
    batch = torch.arange(n, device=self.device)
    target_pos = candidates[batch, valid.to(torch.int64).argmax(dim=-1)]

    corners = target_pos.new_tensor(
      [
        [target_range.x[0], target_range.y[0]],
        [target_range.x[0], target_range.y[1]],
        [target_range.x[1], target_range.y[0]],
        [target_range.x[1], target_range.y[1]],
      ]
    )
    farthest = torch.linalg.vector_norm(
      corners[None] - object_pos[:, None, :2], dim=-1
    ).argmax(dim=-1)
    target_pos[:, :2] = torch.where(
      valid.any(dim=-1, keepdim=True), target_pos[:, :2], corners[farthest]
    )

    origins = self._env.scene.env_origins[env_ids]
    self.target_pos[env_ids] = target_pos + origins
    yaw = sample_uniform(
      object_range.yaw[0],
      object_range.yaw[1],
      (n,),
      device=self.device,
    )
    zeros = torch.zeros(n, device=self.device)
    pose = torch.cat(
      (object_pos + origins, quat_from_euler_xyz(zeros, zeros, yaw)), dim=-1
    )
    self.object.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    self.object.write_root_link_velocity_to_sim(
      torch.zeros(n, 6, device=self.device), env_ids=env_ids
    )


@dataclass(kw_only=True)
class PushingCommandCfg(LiftingCommandCfg):
  min_xy_separation: float = 0.15

  def build(self, env: ManagerBasedRlEnv) -> PushingCommand:
    return PushingCommand(self, env)


def pushing_command(env, name: str) -> "PushingCommand":
  """Resolve one command term as the Push-Cube goal sampler."""
  return cast(PushingCommand, env.command_manager.get_term(name))


__all__ = [
  "pushing_command","PushingCommand", "PushingCommandCfg"]
