"""Episode-level object and goal sampling for Push-T."""

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
  """MJLab lifting command extended with a planar pose and overlap goal."""

  cfg: PushTCommandCfg
  episode_success: torch.Tensor

  def __init__(self, cfg: PushTCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.target_yaw = torch.zeros(self.num_envs, device=self.device)
    self._rasterizer = FootprintRasterizer(
      cfg.footprint_parts,
      device=self.device,
      dtype=self.object.data.root_link_pos_w.dtype,
      resolution=cfg.mask_resolution,
      half_width=cfg.mask_half_width,
    )
    # The drawn target, when the task registers one. Held here rather than in an
    # event so it cannot be posed from a stale command: the marker is written in
    # the same call that samples the pose it represents.
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
    """Evaluate the GPU mask at most once per environment step."""
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

    # The goal is drawn on a radius about the object: a bearing, and a separation
    # inside the band this variant allows. The band is the *only* thing the three
    # start-state variants differ by -- a floor alone leaves the goal unbounded,
    # a ceiling grown over training is a reverse curriculum, and a tight band
    # drawn for a fraction of episodes is a near-goal mixture. An unbounded
    # ceiling means the rectangle's diagonal, the widest separation it holds.
    floor = target_pos.new_full((n,), self.cfg.min_xy_separation)
    ceiling = target_pos.new_full(
      (n,),
      float(torch.linalg.vector_norm(upper - lower))
      if self.cfg.max_xy_separation is None
      else self.cfg.max_xy_separation,
    )
    near = torch.zeros(n, dtype=torch.bool, device=self.device)
    if self.cfg.near_goal_probability > 0.0:
      near = torch.rand(n, device=self.device) < self.cfg.near_goal_probability
      low, high = self.cfg.near_goal_separation_range
      floor = torch.where(near, floor.new_full((n,), low), floor)
      ceiling = torch.where(near, ceiling.new_full((n,), high), ceiling)

    # Draw, and redraw whatever landed off the table. Every object position has
    # admissible bearings -- the rectangle is 20 cm across its short side and the
    # floor is 1 cm, so aiming inward always fits -- which is why this terminates
    # and needs no fallback. Only the environments that missed are redrawn, and
    # the acceptance rate is high enough that it is a handful of passes.
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
    self.target_pos[env_ids] = target_pos + origins
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
    levels = self.cfg.target_yaw_levels
    if levels is not None and levels > 0:
      # Round onto `levels` evenly spaced angles spanning the full circle. The
      # support stays the whole range; only its cardinality shrinks.
      spacing = 2.0 * math.pi / levels
      target_yaw = wrap_to_pi(torch.round(target_yaw / spacing) * spacing)
    self.target_yaw[env_ids] = target_yaw
    if bool(near.any()):
      # Near episodes get a bounded yaw error instead of a uniform one: the
      # bonus needs orientation as well as position, and a uniform draw leaves
      # it out of reach even at 6 mm.
      low, high = self.cfg.near_goal_yaw_range
      offset = sample_uniform(low, high, (n,), device=self.device)
      sign = torch.where(
        torch.rand(n, device=self.device) < 0.5, -1.0, 1.0
      )
      object_yaw = torch.where(
        near, wrap_to_pi(self.target_yaw[env_ids] + sign * offset), object_yaw
      )
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
      # Flat on the table at the goal pose. `target_pos` already carries
      # `env_origins`, so the marker lands in its own env like everything else.
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
    # The command is a pure function of the state written by
    # `_resample_command`, so there is no per-step state to advance and
    # nothing to scope to `env_ids`.
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
  """Configuration for the planar Push-T pose command."""

  min_xy_separation: float = 0.15
  max_xy_separation: float | None = None
  """Upper bound on object-goal separation, applied to every episode.

  ``None`` leaves the goal unbounded. A reverse curriculum writes this field,
  growing it until it exceeds the largest separation the target range allows, at
  which point the distribution is exactly the unbounded one.
  """
  near_goal_probability: float = 0.0
  """Fraction of episodes started just short of the success threshold.

  A stationary mixture rather than a schedule: the state distribution never
  changes, so there is no handover for a converged policy to fall off.
  """
  near_goal_separation_range: tuple[float, float] = (0.006, 0.015)
  """Separation band for those episodes, in metres.

  Overlap reaches the 0.90 threshold only inside roughly 5 mm and 5 degrees, so a
  wider band leaves the sparse at-goal bonus -- which replaces the whole reward
  with 3.0 -- just as unreachable as it is from across the table. At 6 mm and
  perfect alignment overlap is 0.891, immediately below the threshold, which is
  the point of the lower bound: close enough that one correction earns the bonus,
  never so close that the episode begins already solved.
  """
  near_goal_yaw_range: tuple[float, float] = (0.087, 0.349)
  """Absolute yaw error for those episodes, in radians (5 to 20 degrees).

  Bounded away from zero deliberately. A relative-yaw curriculum that started the
  object *at* the goal orientation was tried and removed: with every reward term
  decreasing in yaw error, leaving the object alone was optimal. The floor keeps
  these episodes clear of that, and the ceiling keeps them inside the range a
  single correction can close.
  """
  target_yaw_range: tuple[float, float] = (-math.pi, math.pi)
  target_yaw_levels: int | None = None
  """Quantize the goal yaw to this many evenly spaced angles, or ``None``.

  A curriculum on the goal's *resolution* rather than its range. The goal still
  covers the full circle from the first episode and the object's yaw stays
  uniform, so nothing here makes the starting state easier -- which is what
  every distance-based curriculum did before collapsing into a do-nothing
  policy. What it reduces is how many distinct goal orientations the policy has
  to hold at once, which is the axis the goal-yaw curriculum also moves along,
  by pinning to one.
  """
  orientation_weight: float = 0.5
  """Share of the dense reward's shaped half that scores orientation.

  ``0.5`` is ManiSkill's split and reproduces it exactly. Raising it makes
  rotation the thing worth doing while the object still starts far from the
  goal, so contact still pays and no do-nothing attractor appears.
  """
  footprint_parts: tuple[FootprintPart, ...] = FOOTPRINT_PARTS
  mask_resolution: int = 64
  mask_half_width: float = MASK_HALF_WIDTH
  goal_half_height: float = HALF_HEIGHT
  goal_marker_name: str | None = None
  """Scene entity to pose at the goal, drawing the target into the camera.

  ``None`` keeps VBRL's original setup, where the goal reaches the policy only
  as numbers. Naming an entity restores what every published Push-T does.
  """

  def build(self, env: ManagerBasedRlEnv) -> PushTCommand:
    return PushTCommand(self, env)


def push_t_command(env, name: str) -> "PushTCommand":
  """Resolve one command term, asserting it is the Push-T sampler."""
  command = env.command_manager.get_term(name)
  if not isinstance(command, PushTCommand):
    raise TypeError(f"Command {name!r} must be a PushTCommand.")
  return command


__all__ = [
  "push_t_command","PushTCommand", "PushTCommandCfg"]
