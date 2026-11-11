from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from vbrl.tasks.push_t.geometry import FOOTPRINT_PARTS, HALF_HEIGHT

# Green chromaticity G / (R + G + B) relative to the table's median: on the rig the
# table spans 0.30-0.33 around 0.316, the T 0.18-0.25 and the marker sits above 0.34.
CHROMATICITY_MARGIN = 0.04
# Below this R + G + B the chromaticity of a pixel is noise. The rig's T stays above 98;
# a side face away from the light renders at 34-40 and is still clearly red.
MIN_INTENSITY = 20.0
# The rig's first 1280 x 720 frame: T 0.936, taped hole 0.968; the arm over the T scored
# 0.21-0.61 on the 224 crops.
MIN_T_IOU = 0.8
MIN_GOAL_IOU = 0.8
WORKSPACE_CENTER = (0.35, 0.0)
WORKSPACE_HALF_EXTENTS = (0.25, 0.3)
YAW_STEPS = 36


@dataclass(frozen=True)
class Fit:
  pose: tuple[float, float, float]
  iou: float
  reliable: bool


def _rectangle(center: Any, half_extents: Any, samples: int = 16) -> Any:
  hx, hy = half_extents
  t = np.linspace(-1.0, 1.0, samples, endpoint=False)
  one = np.ones(samples)
  edges = (
    np.c_[t * hx, -hy * one],
    np.c_[hx * one, t * hy],
    np.c_[-t * hx, hy * one],
    np.c_[-hx * one, -t * hy],
  )
  return np.concatenate(edges) + center


def _part(part: Any, grow: float = 0.0) -> Any:
  return _rectangle(part.center_xy, np.add(part.half_extents_xy, grow))


def _placed(view: Any, local: Any, pose: Any, height: float) -> Any:
  x, y, yaw = pose
  c, s = math.cos(yaw), math.sin(yaw)
  xy = local @ np.array([[c, s], [-s, c]]) + (x, y)
  return np.c_[xy, view.table_z(xy[:, 0], xy[:, 1]) + height]


def _fill(mask: Any, pixels: Any, value: int) -> None:
  import cv2

  cv2.fillPoly(mask, [np.round(pixels * 16).astype(np.int32)], value, shift=4)


def render_t(view: Any, pose: Any) -> Any:
  import cv2

  mask = np.zeros((view.height, view.width), np.uint8)
  for part in FOOTPRINT_PARTS:
    corners = np.concatenate(
      [
        view.project(_placed(view, _part(part), pose, height))
        for height in (0.0, 2.0 * HALF_HEIGHT)
      ]
    )
    _fill(mask, cv2.convexHull(corners.astype(np.float32))[:, 0], 1)
  return mask


def render_footprint(view: Any, pose: Any, grow: float = 0.0) -> Any:
  mask = np.zeros((view.height, view.width), np.uint8)
  for part in FOOTPRINT_PARTS:
    _fill(mask, view.project(_placed(view, _part(part, grow), pose, 0.0)), 1)
  return mask


def _footprint_with_slack(view: Any, pose_and_slack: Any) -> Any:
  return render_footprint(view, pose_and_slack[:3], pose_and_slack[3])


def _largest_region(mask: Any) -> Any:
  import cv2

  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  region = np.zeros_like(mask)
  if contours:
    cv2.drawContours(region, [max(contours, key=cv2.contourArea)], -1, 1, cv2.FILLED)
  return region


def _relative_green(view: Any, image: Any) -> tuple[Any, Any]:
  rgb = image.astype(np.float32)
  intensity = rgb.sum(axis=-1)
  table = _rectangle(WORKSPACE_CENTER, WORKSPACE_HALF_EXTENTS)
  workspace = np.zeros((view.height, view.width), np.uint8)
  _fill(workspace, view.project(_placed(view, table, (0.0, 0.0, 0.0), 0.0)), 1)
  green = rgb[..., 1] / np.maximum(intensity, 1.0)
  valid = (workspace > 0) & (intensity > MIN_INTENSITY)
  return green - np.median(green[valid]), valid


def _fit(
  view: Any,
  observed: Any,
  render: Any,
  height: float,
  min_iou: float,
  extra: tuple = (),
) -> Fit:
  from scipy.optimize import minimize

  if not observed.any():
    return Fit((math.nan, math.nan, math.nan), 0.0, False)
  rows, columns = np.nonzero(observed)
  x, y, _ = view.on_table((columns.mean(), rows.mean()), height)
  observed = observed.astype(bool)

  def cost(pose: Any) -> float:
    rendered = render(view, pose).astype(bool)
    return 1.0 - (rendered & observed).sum() / (rendered | observed).sum()

  yaws = np.linspace(-math.pi, math.pi, YAW_STEPS, endpoint=False)
  yaw = min(yaws, key=lambda yaw: cost((x, y, yaw, *extra)))
  pose = np.array([x, y, yaw, *extra])
  for step in (0.01, 0.002):
    steps = np.full(len(pose), step)
    steps[2] *= 10.0
    simplex = np.vstack([pose, pose + np.diag(steps)])
    options = {"initial_simplex": simplex, "xatol": 1e-5, "fatol": 1e-6}
    pose = minimize(cost, pose, method="Nelder-Mead", options=options).x
  yaw = math.remainder(pose[2], 2.0 * math.pi)
  iou = 1.0 - cost(pose)
  return Fit((float(pose[0]), float(pose[1]), yaw), iou, iou >= min_iou)


def measure_t(view: Any, image: Any) -> Fit:
  green, valid = _relative_green(view, image)
  mask = _largest_region(((green < -CHROMATICITY_MARGIN) & valid).astype(np.uint8))
  return _fit(view, mask, render_t, HALF_HEIGHT, MIN_T_IOU)


def measure_goal(view: Any, image: Any) -> Fit:
  green, valid = _relative_green(view, image)
  mask = ((green > CHROMATICITY_MARGIN) & valid).astype(np.uint8)
  enclosed = _largest_region(mask)
  hole = enclosed & (1 - mask)
  # The fourth parameter is how far the tape sits outside the T's edge; without it a
  # hole taped too narrow or too wide leaves the T free to slide inside it.
  return _fit(
    view, hole, _footprint_with_slack, 0.0, MIN_GOAL_IOU, extra=(0.0,)
  )


def overlap(t: Any, goal: Any) -> float:
  import torch

  from vbrl.tasks.push_t.geometry import footprint_overlap_from_pose

  t, goal = torch.tensor([t]), torch.tensor([goal])
  return float(
    footprint_overlap_from_pose(
      object_xy=t[:, :2],
      object_yaw=t[:, 2],
      target_xy=goal[:, :2],
      target_yaw=goal[:, 2],
      footprint_parts=FOOTPRINT_PARTS,
    )[0]
  )


__all__ = [
  "MIN_GOAL_IOU",
  "MIN_T_IOU",
  "Fit",
  "measure_goal",
  "measure_t",
  "overlap",
  "render_footprint",
  "render_t",
]
