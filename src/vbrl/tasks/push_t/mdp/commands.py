from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.tasks.manipulation.mdp import LiftingCommand, LiftingCommandCfg
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_from_euler_xyz,
  sample_uniform,
  wrap_to_pi,
)

from ..geometry import (
  FOOTPRINT_PARTS,
  HALF_HEIGHT,
  MASK_HALF_WIDTH,
  FootprintPart,
  FootprintRasterizer,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


_MAX_GOAL_DRAWS = 512


class PushTCommand(LiftingCommand):
  cfg: PushTCommandCfg
  episode_success: torch.Tensor

  def __init__(self, cfg: PushTCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.target_yaw = torch.zeros(self.num_envs, device=self.device)
    self.observation_offset = torch.zeros(self.num_envs, 3, device=self.device)
    self.observation_yaw_offset = torch.zeros(self.num_envs, device=self.device)
    self._rasterizer = FootprintRasterizer(
      cfg.footprint_parts,
      device=self.device,
      dtype=self.object.data.root_link_pos_w.dtype,
      resolution=cfg.mask_resolution,
      half_width=cfg.mask_half_width,
    )
    self._goal_marker = (
      env.scene[cfg.goal_marker_name] if cfg.goal_marker_name else None
    )
    self._overlap_cache = torch.zeros(self.num_envs, device=self.device)
    self._overlap_cache_step: int | None = None
    self.metrics["yaw_error"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["overlap"] = torch.zeros(
      self.num_envs, device=self.device
    )

  @property
  def command(self) -> torch.Tensor:
    return torch.cat((self.target_pos, self.target_yaw[:, None]), dim=-1)

  def _compute_overlap(self) -> torch.Tensor:
    object_yaw = euler_xyz_from_quat(
      self.object.data.root_link_quat_w
    )[2]
    return self._rasterizer.overlap(
      object_xy=self.object.data.root_link_pos_w[:, :2],
      object_yaw=object_yaw,
      target_xy=self.target_pos[:, :2],
      target_yaw=self.target_yaw,
    )

  def get_overlap(self, *, force_refresh: bool = False) -> torch.Tensor:
    step = int(getattr(self._env, "common_step_counter", -1))
    if force_refresh or self._overlap_cache_step != step:
      self._overlap_cache = self._compute_overlap()
      self._overlap_cache_step = step
    return self._overlap_cache

  def _record_success(self, overlap: torch.Tensor) -> torch.Tensor:
    object_pos = self.object.data.root_link_pos_w
    object_yaw = euler_xyz_from_quat(
      self.object.data.root_link_quat_w
    )[2]
    at_goal = overlap >= self.cfg.success_threshold
    self.episode_success = torch.maximum(
      self.episode_success, at_goal.to(self.episode_success.dtype)
    )
    self.metrics["position_error"] = torch.linalg.vector_norm(
      self.target_pos[:, :2] - object_pos[:, :2], dim=-1
    )
    self.metrics["yaw_error"] = torch.abs(
      wrap_to_pi(self.target_yaw - object_yaw)
    )
    self.metrics["overlap"] = overlap
    self.metrics["at_goal"] = at_goal.to(overlap.dtype)
    self.metrics["episode_success"] = self.episode_success
    return at_goal

  def get_at_goal(self) -> torch.Tensor:
    return self._record_success(self.get_overlap())

  def _update_metrics(self) -> None:
    self.metrics["object_height"] = self.object.data.root_link_pos_w[:, 2]
    self._record_success(self.get_overlap())

  def compute_success(self) -> torch.Tensor:
    return self.get_at_goal()

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self.episode_success[env_ids] = 0.0
    self._overlap_cache_step = None
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
    target_pos = sample_xyz(target_range, (n, 3))
    lower = target_pos.new_tensor([target_range.x[0], target_range.y[0]])
    upper = target_pos.new_tensor([target_range.x[1], target_range.y[1]])

    floor = target_pos.new_full((n,), self.cfg.min_xy_separation)
    ceiling = target_pos.new_full(
      (n,), float(torch.linalg.vector_norm(upper - lower))
    )

    pending = torch.ones(n, dtype=torch.bool, device=self.device)
    for _ in range(_MAX_GOAL_DRAWS):
      if not bool(pending.any()):
        break
      angle = sample_uniform(0.0, 2.0 * math.pi, (n,), device=self.device)
      radius = floor + (ceiling - floor) * torch.rand(n, device=self.device)
      drawn = object_pos[:, :2] + radius[:, None] * torch.stack(
        (angle.cos(), angle.sin()), dim=-1
      )
      target_pos[:, :2] = torch.where(pending[:, None], drawn, target_pos[:, :2])
      pending = pending & ((drawn < lower) | (drawn > upper)).any(dim=-1)
    else:
      # Unreachable for any sane band, but a floor wider than the rectangle can
      # hold would otherwise spin forever rather than say so.
      raise RuntimeError(
        f"Goal sampling did not converge in {_MAX_GOAL_DRAWS} draws: no goal "
        f"satisfies min_xy_separation={self.cfg.min_xy_separation} inside "
        f"{target_range.x} x {target_range.y}."
      )

    origins = self._env.scene.env_origins[env_ids]
    if self.cfg.fixed_target is not None:
      target_pos[:, 0] = self.cfg.fixed_target[0]
      target_pos[:, 1] = self.cfg.fixed_target[1]
    self.target_pos[env_ids] = target_pos + origins
    if self.cfg.observation_position_noise > 0.0:
      bound = self.cfg.observation_position_noise
      self.observation_offset[env_ids] = sample_uniform(
        -bound, bound, (n, 3), device=self.device
      )
      self.observation_offset[env_ids, 2] = 0.0
    if self.cfg.observation_yaw_noise > 0.0:
      bound = self.cfg.observation_yaw_noise
      self.observation_yaw_offset[env_ids] = sample_uniform(
        -bound, bound, (n,), device=self.device
      )
    object_yaw = sample_uniform(
      object_range.yaw[0],
      object_range.yaw[1],
      (n,),
      device=self.device,
    )
    target_yaw = sample_uniform(
      self.cfg.target_yaw_range[0],
      self.cfg.target_yaw_range[1],
      (n,),
      device=self.device,
    )
    if self.cfg.fixed_target is not None:
      target_yaw = torch.full_like(target_yaw, self.cfg.fixed_target[2])
    self.target_yaw[env_ids] = target_yaw
    zeros = torch.zeros(n, device=self.device)
    pose = torch.cat(
      (
        object_pos + origins,
        quat_from_euler_xyz(zeros, zeros, object_yaw),
      ),
      dim=-1,
    )
    self.object.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    self.object.write_root_link_velocity_to_sim(
      torch.zeros(n, 6, device=self.device), env_ids=env_ids
    )
    if self._goal_marker is not None:
      marker_pos = self.target_pos[env_ids].clone()
      marker_pos[:, 2] = origins[:, 2]
      marker_pose = torch.cat(
        (
          marker_pos,
          quat_from_euler_xyz(zeros, zeros, self.target_yaw[env_ids]),
        ),
        dim=-1,
      )
      self._goal_marker.write_mocap_pose_to_sim(marker_pose, env_ids=env_ids)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    del env_ids

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    for batch in visualizer.get_env_indices(self.num_envs):
      for index in torch.as_tensor(batch).flatten().tolist():
        target_pos = self.target_pos[index].detach().cpu().numpy()
        yaw = float(self.target_yaw[index].detach().cpu())
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        rotation = np.array(
          (
            (cos_yaw, -sin_yaw, 0.0),
            (sin_yaw, cos_yaw, 0.0),
            (0.0, 0.0, 1.0),
          )
        )
        for part_index, part in enumerate(self.cfg.footprint_parts):
          x, y = part.center_xy
          center = target_pos.copy()
          center[:2] += (
            cos_yaw * x - sin_yaw * y,
            sin_yaw * x + cos_yaw * y,
          )
          visualizer.add_box(
            center=center,
            size=np.array((*part.half_extents_xy, self.cfg.goal_half_height)),
            mat=rotation,
            color=self.cfg.viz.target_color,
            label=f"push_t_goal_part_{part_index}_{index}",
          )


@dataclass(kw_only=True)
class PushTCommandCfg(LiftingCommandCfg):
  fixed_target: tuple[float, float, float] | None = None
  min_xy_separation: float = 0.15
  # A calibration error is a fixed bias for a whole deployment session, not
  # per-step jitter, so these are drawn once per episode. Measured chain on the
  # rig: 1.33 mm extrinsic repeatability, ~0.4 mm tag detection, +/-1 mm
  # hand-measured margin, ~1.5 mm plane and placement -> ~2.3 mm and ~1 deg.
  # They perturb only what the actor observes; the reward keeps the true goal.
  observation_position_noise: float = 0.0
  observation_yaw_noise: float = 0.0
  target_yaw_range: tuple[float, float] = (-math.pi, math.pi)
  orientation_weight: float = 0.5
  footprint_parts: tuple[FootprintPart, ...] = FOOTPRINT_PARTS
  mask_resolution: int = 64
  mask_half_width: float = MASK_HALF_WIDTH
  goal_half_height: float = HALF_HEIGHT
  goal_marker_name: str | None = None

  def build(self, env: ManagerBasedRlEnv) -> PushTCommand:
    return PushTCommand(self, env)


def push_t_command(env, name: str) -> PushTCommand:
  command = env.command_manager.get_term(name)
  if not isinstance(command, PushTCommand):
    raise TypeError(f"Command {name!r} must be a PushTCommand.")
  return command


__all__ = ["PushTCommand", "PushTCommandCfg", "push_t_command"]
