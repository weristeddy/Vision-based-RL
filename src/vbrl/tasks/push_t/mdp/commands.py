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
  footprint_signed_distance,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer




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
    for name in ("object_height", "at_goal"):
      self.metrics.pop(name)
    self.metrics["yaw_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["overlap"] = torch.zeros(self.num_envs, device=self.device)
    if cfg.near_object_start:
      self._init_start_bank(env)

  def _init_start_bank(self, env: ManagerBasedRlEnv) -> None:
    from ..arm_placement import gripper_rotation, placement_table

    self._robot = env.scene["robot"]
    yaws = np.linspace(0.0, np.pi, self.cfg.start_yaws, endpoint=False)
    tilt = self.cfg.start_tilt
    rotations = [gripper_rotation(y, t) for y in yaws for t in (0.0, tilt, -tilt)]
    seed = self._robot.data.default_joint_pos[0, :6].cpu().numpy()
    limits = self._robot.data.soft_joint_pos_limits[0, :6].cpu().numpy()
    xs, ys = np.arange(0.13, 0.58, 0.01), np.arange(-0.27, 0.28, 0.01)
    tables = [
      placement_table(env.sim.mj_model, seed, limits, xs, ys, rotations, height)
      for height, _, _ in self.cfg.start_levels
    ]
    grid = tables[0][0]
    placements = np.stack([np.nan_to_num(t[1]) for t in tables])
    self._pad_xy = torch.tensor(grid, dtype=torch.float, device=self.device)
    self._placements = torch.tensor(placements, dtype=torch.float, device=self.device)
    self._placement_valid = torch.tensor(
      np.stack([t[2] for t in tables]), device=self.device
    )
    self._start_mode = torch.full((self.num_envs,), -1, device=self.device)
    self._mode_success = torch.full(
      (len(self.cfg.start_levels) + 1,), 0.5, device=self.device
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

  # An env ended by nan_detection resets with a NaN object state, which would
  # turn the whole step's metric mean into NaN.
  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    extras = {
      name: value[env_ids].nanmean().item() for name, value in self.metrics.items()
    }
    super().reset(env_ids)
    return extras

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
    self.metrics["episode_success"] = self.episode_success
    return at_goal

  def get_at_goal(self) -> torch.Tensor:
    return self._record_success(self.get_overlap())

  def _update_metrics(self) -> None:
    self._record_success(self.get_overlap())

  def compute_success(self) -> torch.Tensor:
    return self.get_at_goal()

  def _update_mode_success(self, env_ids: torch.Tensor) -> None:
    modes = self._start_mode[env_ids]
    done = modes >= 0
    success = self.episode_success[env_ids][done].float()
    levels = len(self._mode_success)
    count = torch.bincount(modes[done], minlength=levels)
    mean = torch.bincount(modes[done], weights=success, minlength=levels)
    mean = mean / count.clamp_min(1)
    keep = (1.0 - self.cfg.start_success_rate) ** count
    self._mode_success = keep * self._mode_success + (1.0 - keep) * mean

  def _place_arm(
    self, env_ids: torch.Tensor, object_pos: torch.Tensor, object_yaw: torch.Tensor
  ) -> None:
    p = self._mode_success
    weights = self.cfg.start_sampling_floor + torch.sqrt(p * (1.0 - p))
    mode = torch.multinomial(weights, len(env_ids), replacement=True)
    base = self._robot.data.default_root_state[env_ids, :2]
    dx, dy = (self._pad_xy - (object_pos[:, :2] - base)[:, None]).unbind(-1)
    cos, sin = torch.cos(object_yaw)[:, None], torch.sin(object_yaw)[:, None]
    local = torch.stack((cos * dx + sin * dy, cos * dy - sin * dx), dim=-1)
    gap = footprint_signed_distance(local, self.cfg.footprint_parts)
    joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
    for level, (_, low, high) in enumerate(self.cfg.start_levels):
      valid = self._placement_valid[level]
      candidates = (gap >= low) & (gap <= high) & valid.any(1)
      chosen = (mode == level + 1) & candidates.any(1)
      mode[(mode == level + 1) & ~chosen] = 0
      if not chosen.any():
        continue
      point = torch.multinomial(candidates[chosen].float(), 1).squeeze(1)
      rotation = torch.multinomial(valid[point].float(), 1).squeeze(1)
      joint_pos[chosen, :6] = self._placements[level, point, rotation]
    self._start_mode[env_ids] = mode
    ids = env_ids[mode > 0]
    if len(ids) == 0:
      return
    joint_pos = joint_pos[mode > 0]
    velocity = torch.zeros_like(joint_pos)
    self._robot.write_joint_state_to_sim(joint_pos, velocity, env_ids=ids)
    self._env.action_manager.get_term("joint_pos").reset(ids)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    if self.cfg.near_object_start:
      self._update_mode_success(env_ids)
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
    target_pos = sample_xyz(self.cfg.target_position_range, (n, 3))
    origins = self._env.scene.env_origins[env_ids]
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
    if self.cfg.near_object_start:
      self._place_arm(env_ids, object_pos, object_yaw)
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
  near_object_start: bool = False
  # (pad height over the table, signed distance to the outline low, high), one
  # start type each besides home: pads beside the T at mid-height, 3 mm over the
  # top face, 3 cm over it. None starts in contact: at 2 cm beside the T a 25 deg
  # tilt leaned the hand onto the top face in 15 of 131 starts.
  start_levels: tuple[tuple[float, float, float], ...] = (
    (HALF_HEIGHT, 0.03, 0.05),
    (2 * HALF_HEIGHT + 0.008 + 0.003, -1.0, 0.03),
    (2 * HALF_HEIGHT + 0.008 + 0.03, -1.0, 0.05),
  )
  start_yaws: int = 4
  start_tilt: float = math.radians(25.0)
  start_sampling_floor: float = 0.05
  start_success_rate: float = 0.01

  def build(self, env: ManagerBasedRlEnv) -> PushTCommand:
    return PushTCommand(self, env)


def push_t_command(env, name: str) -> PushTCommand:
  command = env.command_manager.get_term(name)
  if not isinstance(command, PushTCommand):
    raise TypeError(f"Command {name!r} must be a PushTCommand.")
  return command


__all__ = ["PushTCommand", "PushTCommandCfg", "push_t_command"]
