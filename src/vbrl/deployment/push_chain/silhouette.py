from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from vbrl.deployment.push_t.silhouette import (
  CHROMATICITY_MARGIN,
  _fill,
  _largest_region,
  _placed,
  _rectangle,
  _relative_green,
)
from vbrl.tasks.push_chain.geometry import (
  HINGE_LIMIT,
  LINK_HALF_EXTENTS,
  LINK_HALF_HEIGHT,
  LINK_HALF_WIDTH,
  PIN_RADIUS,
)

MIN_CHAIN_IOU = 0.8
MIN_GOAL_IOU = 0.8
YAW_STEPS = 36
HINGE_STEPS = 7
STARTS = 3
FREE_ENDS = ((-1.0,), (), (1.0,))
_CIRCLE = np.c_[
  np.cos(np.linspace(0, 2 * np.pi, 32, False)),
  np.sin(np.linspace(0, 2 * np.pi, 32, False)),
]


@dataclass(frozen=True)
class Fit:
  pose: tuple[float, ...]
  iou: float
  reliable: bool


def links(pose: Any) -> Any:
  import torch

  from vbrl.tasks.push_chain.geometry import link_poses

  x, y, yaw, *hinges = (float(v) for v in pose[:5])
  return link_poses(
    torch.tensor([[x, y]]), torch.tensor([yaw]), torch.tensor([hinges])
  )[0].numpy()


def _pins(chain: Any) -> Any:
  ends = chain[::2]
  half = np.array((LINK_HALF_EXTENTS[0][0], -LINK_HALF_EXTENTS[2][0]))
  return ends[:, :2] + half[:, None] * np.c_[np.cos(ends[:, 2]), np.sin(ends[:, 2])]


def _prism(view: Any, local: Any, pose: Any) -> Any:
  import cv2

  corners = np.concatenate(
    [view.project(_placed(view, local, pose, h)) for h in (0.0, 2 * LINK_HALF_HEIGHT)]
  )
  return cv2.convexHull(corners.astype(np.float32))[:, 0]


def render_chain(view: Any, pose: Any) -> Any:
  mask = np.zeros((view.height, view.width), np.uint8)
  chain = links(pose)
  for link, half in zip(chain, LINK_HALF_EXTENTS, strict=True):
    _fill(mask, _prism(view, _rectangle((0.0, 0.0), half), link), 1)
  for x, y in _pins(chain):
    _fill(mask, _prism(view, PIN_RADIUS * _CIRCLE, (x, y, 0.0)), 1)
  return mask


def _mitres(chain: Any, hinges: Any, width: float) -> list[Any]:
  quads = []
  for i, (pin, hinge) in enumerate(zip(_pins(chain), hinges, strict=True)):
    side = -1.0 if hinge >= 0.0 else 1.0
    normals = [
      side * np.array((-math.sin(yaw), math.cos(yaw))) for yaw in chain[i : i + 2, 2]
    ]
    mitre = width * (normals[0] + normals[1]) / (1.0 + normals[0] @ normals[1])
    quads.append(
      pin + np.array((0.0 * mitre, width * normals[0], mitre, width * normals[1]))
    )
  return quads


def render_goal(view: Any, pose: Any) -> Any:
  slack = float(pose[5])
  width = LINK_HALF_WIDTH + slack
  mask = np.zeros((view.height, view.width), np.uint8)
  chain = links(pose)
  for link, (half_length, _), free in zip(
    chain, LINK_HALF_EXTENTS, FREE_ENDS, strict=True
  ):
    back = half_length + (slack if -1.0 in free else 0.0)
    front = half_length + (slack if 1.0 in free else 0.0)
    local = _rectangle(((front - back) / 2, 0.0), ((front + back) / 2, width))
    _fill(mask, view.project(_placed(view, local, link, 0.0)), 1)
  table = (0.0, 0.0, 0.0)
  for x, y in _pins(chain):
    _fill(mask, view.project(_placed(view, (x, y) + width * _CIRCLE, table, 0.0)), 1)
  for quad in _mitres(chain, pose[3:5], width):
    _fill(mask, view.project(_placed(view, quad, table, 0.0)), 1)
  return mask


def _iou(rendered: Any, observed: Any) -> float:
  rendered = rendered.astype(bool)
  return float((rendered & observed).sum() / max((rendered | observed).sum(), 1))


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
    return Fit((math.nan,) * (5 + len(extra)), 0.0, False)
  rows, columns = np.nonzero(observed)
  x, y, _ = view.on_table((columns.mean(), rows.mean()), height)
  observed = observed.astype(bool)

  def cost(pose: Any) -> float:
    return 1.0 - _iou(render(view, pose), observed)

  def refine(pose: Any) -> Any:
    for step in (0.01, 0.002):
      steps = np.full(len(pose), step)
      steps[2:5] *= 10.0
      simplex = np.vstack([pose, pose + np.diag(steps)])
      options = {"initial_simplex": simplex, "xatol": 1e-5, "fatol": 1e-6}
      pose = minimize(cost, pose, method="Nelder-Mead", options=options).x
    return pose

  yaws = np.linspace(-math.pi, math.pi, YAW_STEPS, endpoint=False)
  hinges = np.linspace(-HINGE_LIMIT, HINGE_LIMIT, HINGE_STEPS)
  candidates = [
    np.array((*np.subtract((x, y), _centre((yaw, a, b))), yaw, a, b, *extra))
    for yaw, a, b in itertools.product(yaws, hinges, hinges)
  ]
  starts = sorted(candidates, key=cost)[:STARTS]
  pose = min((refine(start) for start in starts), key=cost)
  pose[2] = math.remainder(pose[2], 2.0 * math.pi)
  iou = 1.0 - cost(pose)
  return Fit(tuple(float(v) for v in pose), iou, iou >= min_iou)


def _centre(shape: Any) -> Any:
  chain = links((0.0, 0.0, *shape))
  lengths = np.array([half for half, _ in LINK_HALF_EXTENTS])
  return lengths @ chain[:, :2] / lengths.sum()


def measure_chain(view: Any, image: Any) -> Fit:
  green, valid = _relative_green(view, image)
  mask = _largest_region(((green < -CHROMATICITY_MARGIN) & valid).astype(np.uint8))
  return _fit(view, mask, render_chain, LINK_HALF_HEIGHT, MIN_CHAIN_IOU)


def measure_goal(view: Any, image: Any) -> Fit:
  green, valid = _relative_green(view, image)
  tape = ((green > CHROMATICITY_MARGIN) & valid).astype(np.uint8)
  hole = _largest_region(tape) & (1 - tape)
  # The sixth parameter is how far the tape sits outside the chain's edge.
  return _fit(view, hole, render_goal, 0.0, MIN_GOAL_IOU, extra=(0.0,))


def overlap(chain: Any, goal: Any) -> tuple[float, float]:
  import torch

  from vbrl.tasks.push_chain.geometry import ChainOverlap, LinkOverlap

  current = torch.tensor(links(chain), dtype=torch.float)[None]
  target = torch.tensor(links(goal), dtype=torch.float)[None]
  coverage = ChainOverlap(device="cpu", dtype=torch.float)(current, target)
  per_link = LinkOverlap(device="cpu", dtype=torch.float).best(current, target)
  return float(coverage[0]), float(per_link[0].min())


__all__ = [
  "MIN_CHAIN_IOU",
  "MIN_GOAL_IOU",
  "Fit",
  "links",
  "measure_chain",
  "measure_goal",
  "overlap",
  "render_chain",
  "render_goal",
]
