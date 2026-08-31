"""Deterministic planar geometry used by the Push-T task."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


def yaw_from_quat(quat: "torch.Tensor") -> "torch.Tensor":
  """Return the Z-axis rotation of a world-frame quaternion."""
  from mjlab.utils.lab_api.math import euler_xyz_from_quat

  return euler_xyz_from_quat(quat)[2]


@dataclass(frozen=True)
class FootprintPart:
  center_xy: tuple[float, float]
  half_extents_xy: tuple[float, float]


HALF_HEIGHT = 0.012
REST_HEIGHT = 0.013
MASK_HALF_WIDTH = 0.09
FOOTPRINT_PARTS = (
  FootprintPart(
    center_xy=(0.0, -0.0225),
    half_extents_xy=(0.06, 0.015),
  ),
  FootprintPart(
    center_xy=(0.0, 0.0375),
    half_extents_xy=(0.015, 0.045),
  ),
)


def _part_tensors(
  footprint_parts: Sequence[FootprintPart],
  *,
  device: torch.device | str,
  dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
  if not footprint_parts:
    raise ValueError("Push-T requires at least one footprint part.")

  centers = torch.tensor(
    [part.center_xy for part in footprint_parts],
    device=device,
    dtype=dtype,
  )
  half_extents = torch.tensor(
    [part.half_extents_xy for part in footprint_parts],
    device=device,
    dtype=dtype,
  )
  if centers.shape != half_extents.shape or centers.shape[1:] != (2,):
    raise ValueError(
      "Each footprint part must have two-dimensional center and half extents."
    )
  if bool(torch.any(half_extents <= 0.0).item()):
    raise ValueError("Footprint half extents must be positive.")
  return centers, half_extents


class FootprintRasterizer:
  """GPU-vectorized 64x64 target-frame footprint overlap calculator.

  Pixel centers are defined deterministically in the goal frame. The same
  registered rectangle metadata creates the goal mask and tests membership in
  the transformed object mask, so geometry, visualization, and success cannot
  drift apart.
  """

  def __init__(
    self,
    footprint_parts: Sequence[FootprintPart],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    resolution: int = 64,
    half_width: float = MASK_HALF_WIDTH,
  ) -> None:
    if resolution <= 0:
      raise ValueError("Footprint mask resolution must be positive.")
    if half_width <= 0.0:
      raise ValueError("Footprint mask half-width must be positive.")

    self.resolution = resolution
    self.half_width = half_width
    self.centers, self.half_extents = _part_tensors(
      footprint_parts,
      device=device,
      dtype=dtype,
    )
    footprint_bound = torch.max(
      torch.abs(self.centers) + self.half_extents
    ).item()
    if footprint_bound > half_width:
      raise ValueError(
        "Footprint mask does not contain the registered object footprint: "
        f"{footprint_bound:.6g} > {half_width:.6g}."
      )

    pixel_width = 2.0 * half_width / resolution
    axis = torch.arange(resolution, device=device, dtype=dtype)
    axis = -half_width + (axis + 0.5) * pixel_width
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    all_goal_points = torch.stack((grid_x, grid_y), dim=-1).reshape(-1, 2)
    goal_mask = self._inside_footprint(all_goal_points)
    # Only goal-occupied pixels can contribute to intersection-over-goal-area.
    # Prefiltering retains the exact deterministic 64x64 mask while avoiding
    # per-environment transforms for the large empty background.
    self.goal_points = all_goal_points[goal_mask]
    self.goal_area = self.goal_points.shape[0]
    if self.goal_area == 0:
      raise ValueError("Registered footprint occupies no pixels in the mask.")

  def _inside_footprint(self, points: torch.Tensor) -> torch.Tensor:
    offsets = torch.abs(points.unsqueeze(-2) - self.centers)
    return torch.all(offsets <= self.half_extents, dim=-1).any(dim=-1)

  def overlap(
    self,
    *,
    object_xy: torch.Tensor,
    object_yaw: torch.Tensor,
    target_xy: torch.Tensor,
    target_yaw: torch.Tensor,
  ) -> torch.Tensor:
    """Return intersection-over-goal-area for each batched planar pose."""
    batch_size = object_xy.shape[0]
    if object_xy.shape != target_xy.shape or object_xy.shape[-1] != 2:
      raise ValueError("Object and target XY tensors must both have shape (N, 2).")
    if object_yaw.shape != (batch_size,) or target_yaw.shape != (batch_size,):
      raise ValueError("Object and target yaw tensors must both have shape (N,).")

    goal_x = self.goal_points[:, 0].unsqueeze(0)
    goal_y = self.goal_points[:, 1].unsqueeze(0)
    target_cos = torch.cos(target_yaw).unsqueeze(-1)
    target_sin = torch.sin(target_yaw).unsqueeze(-1)

    # Transform goal-frame pixel centers into world-space displacements from
    # the current object center.
    delta_x = (
      target_cos * goal_x
      - target_sin * goal_y
      + (target_xy[:, 0] - object_xy[:, 0]).unsqueeze(-1)
    )
    delta_y = (
      target_sin * goal_x
      + target_cos * goal_y
      + (target_xy[:, 1] - object_xy[:, 1]).unsqueeze(-1)
    )

    # Apply the inverse object rotation to obtain local object coordinates.
    object_cos = torch.cos(object_yaw).unsqueeze(-1)
    object_sin = torch.sin(object_yaw).unsqueeze(-1)
    local_x = object_cos * delta_x + object_sin * delta_y
    local_y = -object_sin * delta_x + object_cos * delta_y
    local_points = torch.stack((local_x, local_y), dim=-1)

    object_mask = self._inside_footprint(local_points)
    intersection = object_mask.sum(dim=-1)
    return intersection.to(dtype=object_xy.dtype) / self.goal_area


def footprint_overlap_from_pose(
  *,
  object_xy: torch.Tensor,
  object_yaw: torch.Tensor,
  target_xy: torch.Tensor,
  target_yaw: torch.Tensor,
  footprint_parts: Sequence[FootprintPart],
  resolution: int = 64,
  half_width: float = MASK_HALF_WIDTH,
) -> torch.Tensor:
  """Convenience wrapper for testing or one-off batched overlap evaluation."""
  rasterizer = FootprintRasterizer(
    footprint_parts,
    device=object_xy.device,
    dtype=object_xy.dtype,
    resolution=resolution,
    half_width=half_width,
  )
  return rasterizer.overlap(
    object_xy=object_xy,
    object_yaw=object_yaw,
    target_xy=target_xy,
    target_yaw=target_yaw,
  )


__all__ = [
  "yaw_from_quat",
  "FOOTPRINT_PARTS",
  "FootprintPart",
  "FootprintRasterizer",
  "HALF_HEIGHT",
  "MASK_HALF_WIDTH",
  "REST_HEIGHT",
  "footprint_overlap_from_pose",
]
