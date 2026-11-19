from __future__ import annotations

import functools
import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from vbrl.asset_zoo.objects import push_chain_xml
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
  LINK_HALF_HEIGHT,
  LINK_HALF_WIDTH,
  half_lengths,
)

# Push-T's window cut 11 % of sampled chain outlines: the chain and its tape reach
# 13 cm past the centre link's sampling range, towards the robot too.
WORKSPACE_CENTER = (0.35, 0.0)
WORKSPACE_HALF_EXTENTS = (0.30, 0.35)
MIN_CHAIN_IOU = 0.8
MIN_GOAL_IOU = 0.8
YAW_STEPS = 36
# 36 yaws x 7 x 7 hinges for three links; longer chains get fewer steps per hinge.
CANDIDATES = 1764
STARTS = 3
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

  from vbrl.tasks.push_chain.geometry import chain_from_pose

  return chain_from_pose(torch.tensor(np.asarray(pose, dtype=np.float64))).numpy()


def _pins(chain: Any) -> Any:
  half = np.array(half_lengths(len(chain)))[:-1, None]
  ends = chain[:-1]
  return ends[:, :2] + half * np.c_[np.cos(ends[:, 2]), np.sin(ends[:, 2])]


@functools.cache
def _parts(num_links: int) -> tuple[tuple[int, Any, float, float], ...]:
  import mujoco

  model = mujoco.MjModel.from_xml_path(str(push_chain_xml(num_links)))
  parts = []
  for geom in range(model.ngeom):
    if not model.geom_contype[geom]:
      continue
    body = model.body(model.geom_bodyid[geom]).name
    centre, size = model.geom_pos[geom], model.geom_size[geom]
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
      outline, half_height = _rectangle(centre[:2], size[:2]), size[2]
    else:
      outline, half_height = centre[:2] + size[0] * _CIRCLE, size[1]
    # Body frames sit at the chain's mid-height above the table.
    low = LINK_HALF_HEIGHT + centre[2] - half_height
    link = int(body.removeprefix("push_chain_link"))
    parts.append((link, outline, low, low + 2.0 * half_height))
  return tuple(parts)


def _prism(view: Any, local: Any, pose: Any, low: float, high: float) -> Any:
  import cv2

  levels = (view.project(_placed(view, local, pose, h)) for h in (low, high))
  return cv2.convexHull(np.concatenate(list(levels)).astype(np.float32))[:, 0]


def render_chain(view: Any, pose: Any) -> Any:
  mask = np.zeros((view.height, view.width), np.uint8)
  chain = links(pose)
  for link, outline, low, high in _parts(len(chain)):
    _fill(mask, _prism(view, outline, chain[link], low, high), 1)
  return mask


def _mitres(chain: Any, hinges: Any, width: float) -> list[Any]:
  quads = []
  for i, (pin, hinge) in enumerate(zip(_pins(chain), hinges, strict=True)):
    side = -1.0 if hinge >= 0.0 else 1.0
    normals = [
      side * np.array((-math.sin(yaw), math.cos(yaw))) for yaw in chain[i : i + 2, 2]
    ]
    mitre = width * (normals[0] + normals[1]) / (1.0 + normals[0] @ normals[1])
    corners = (0.0 * mitre, width * normals[0], mitre, width * normals[1])
    quads.append(pin + np.array(corners))
  return quads


def render_goal(view: Any, pose: Any) -> Any:
  *chain_pose, slack = pose
  width = LINK_HALF_WIDTH + slack
  mask = np.zeros((view.height, view.width), np.uint8)
  chain = links(chain_pose)
  last = len(chain) - 1
  for i, (link, length) in enumerate(zip(chain, half_lengths(len(chain)), strict=True)):
    back = length + (slack if i == 0 else 0.0)
    front = length + (slack if i == last else 0.0)
    local = _rectangle(((front - back) / 2, 0.0), ((front + back) / 2, width))
    _fill(mask, view.project(_placed(view, local, link, 0.0)), 1)
  table = (0.0, 0.0, 0.0)
  for x, y in _pins(chain):
    _fill(mask, view.project(_placed(view, (x, y) + width * _CIRCLE, table, 0.0)), 1)
  for quad in _mitres(chain, chain_pose[3:], width):
    _fill(mask, view.project(_placed(view, quad, table, 0.0)), 1)
  return mask


def _iou(rendered: Any, observed: Any) -> float:
  rendered = rendered.astype(bool)
  return float((rendered & observed).sum() / max((rendered | observed).sum(), 1))


def _centre(shape: Any) -> Any:
  chain = links((0.0, 0.0, *shape))
  lengths = np.array(half_lengths(len(chain)))
  return lengths @ chain[:, :2] / lengths.sum()


def _fit(
  view: Any,
  observed: Any,
  render: Any,
  num_links: int,
  height: float,
  min_iou: float,
  extra: tuple = (),
) -> Fit:
  from scipy.optimize import minimize

  if not observed.any():
    return Fit((math.nan,) * (2 + num_links + len(extra)), 0.0, False)
  rows, columns = np.nonzero(observed)
  x, y, _ = view.on_table((columns.mean(), rows.mean()), height)
  observed = observed.astype(bool)

  def cost(pose: Any) -> float:
    return 1.0 - _iou(render(view, pose), observed)

  def refine(pose: Any) -> Any:
    for step in (0.01, 0.002):
      steps = np.full(len(pose), step)
      steps[2 : 2 + num_links] *= 10.0
      simplex = np.vstack([pose, pose + np.diag(steps)])
      options = {"initial_simplex": simplex, "xatol": 1e-5, "fatol": 1e-6}
      pose = minimize(cost, pose, method="Nelder-Mead", options=options).x
    return pose

  hinge_steps = max(2, round((CANDIDATES / YAW_STEPS) ** (1 / (num_links - 1))))
  yaws = np.linspace(-math.pi, math.pi, YAW_STEPS, endpoint=False)
  hinges = np.linspace(-HINGE_LIMIT, HINGE_LIMIT, hinge_steps)
  candidates = [
    np.array((*np.subtract((x, y), _centre(shape)), *shape, *extra))
    for shape in itertools.product(yaws, *[hinges] * (num_links - 1))
  ]
  starts = sorted(candidates, key=cost)[:STARTS]
  pose = min((refine(start) for start in starts), key=cost)
  pose[2] = math.remainder(pose[2], 2.0 * math.pi)
  iou = 1.0 - cost(pose)
  return Fit(tuple(float(v) for v in pose), iou, iou >= min_iou)


def _colour(view: Any, image: Any) -> tuple[Any, Any]:
  return _relative_green(view, image, WORKSPACE_CENTER, WORKSPACE_HALF_EXTENTS)


def measure_chain(view: Any, image: Any, num_links: int = 3) -> Fit:
  green, valid = _colour(view, image)
  mask = _largest_region(((green < -CHROMATICITY_MARGIN) & valid).astype(np.uint8))
  return _fit(view, mask, render_chain, num_links, LINK_HALF_HEIGHT, MIN_CHAIN_IOU)


def measure_goal(view: Any, image: Any, num_links: int = 3) -> Fit:
  green, valid = _colour(view, image)
  tape = ((green > CHROMATICITY_MARGIN) & valid).astype(np.uint8)
  hole = _largest_region(tape) & (1 - tape)
  # The last parameter is how far the tape sits outside the chain's edge.
  return _fit(view, hole, render_goal, num_links, 0.0, MIN_GOAL_IOU, extra=(0.0,))


def overlap(chain: Any, goal: Any) -> tuple[float, float]:
  import torch

  from vbrl.tasks.push_chain.geometry import ChainOverlap, LinkOverlap

  current = torch.tensor(links(chain), dtype=torch.float)[None]
  target = torch.tensor(links(goal), dtype=torch.float)[None]
  count = current.shape[1]
  coverage = ChainOverlap(count, device="cpu", dtype=torch.float)(current, target)
  per_link = LinkOverlap(count, device="cpu", dtype=torch.float).best(current, target)
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
