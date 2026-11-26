from __future__ import annotations

import argparse
import functools
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TASKS = ("lift_cube", "push_t", "push_chain")
# Green chromaticity G / (R + G + B) against the table's median: on the rig the table
# spans 0.30-0.33 around 0.316, the red objects 0.18-0.25, the tape above 0.34.
CHROMATICITY_MARGIN = 0.04
# Below this R + G + B a pixel's chromaticity is noise; a red side face away from the
# light renders at 34-40.
MIN_INTENSITY = 20.0
# The arm over the object scored 0.21-0.61 on 224 crops, clean frames above 0.93.
MIN_IOU = 0.8
WORKSPACE = ((0.35, 0.0), (0.30, 0.35))
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


def object_xml(task: str, num_links: int = 3) -> Path:
  from vbrl.asset_zoo.objects import CUBE_XML, PUSH_T_XML, push_chain_xml

  return {"lift_cube": CUBE_XML, "push_t": PUSH_T_XML}.get(task) or push_chain_xml(
    num_links
  )


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


@functools.cache
def parts(xml: Path) -> tuple[tuple[int, str, Any, Any, float, float], ...]:
  import mujoco

  model = mujoco.MjModel.from_xml_path(str(xml))
  geoms = [g for g in range(model.ngeom) if model.geom_contype[g]]
  floor = min(
    model.geom_pos[g][2]
    - model.geom_size[g][2 if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else 1]
    for g in geoms
  )
  found = []
  for g in geoms:
    box = model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
    centre, size = model.geom_pos[g], model.geom_size[g]
    half_height = size[2] if box else size[1]
    low = centre[2] - half_height - floor
    name = model.body(model.geom_bodyid[g]).name
    link = int(name.removeprefix("push_chain_link")) if "link" in name else 0
    extent = size[:2] if box else size[:1]
    found.append(
      (
        link,
        "box" if box else "cylinder",
        centre[:2],
        extent,
        low,
        low + 2 * half_height,
      )
    )
  return tuple(found)


def _outline(kind: str, centre: Any, extent: Any, grow: float = 0.0) -> Any:
  if kind == "box":
    return _rectangle(centre, np.add(extent, grow))
  return centre + (extent[0] + grow) * _CIRCLE


def links(pose: Any) -> Any:
  import torch

  from vbrl.tasks.push_chain.geometry import chain_from_pose

  return chain_from_pose(torch.tensor(np.asarray(pose, dtype=np.float64))).numpy()


def bodies(task: str, pose: Any) -> Any:
  return links(pose) if task == "push_chain" else np.asarray([pose[:3]])


def _placed(view: Any, local: Any, pose: Any, height: float) -> Any:
  x, y, yaw = pose
  c, s = math.cos(yaw), math.sin(yaw)
  xy = local @ np.array([[c, s], [-s, c]]) + (x, y)
  return np.c_[xy, view.table_z(xy[:, 0], xy[:, 1]) + height]


def _fill(mask: Any, pixels: Any) -> None:
  import cv2

  cv2.fillPoly(mask, [np.round(pixels * 16).astype(np.int32)], 1, shift=4)


def render_object(view: Any, task: str, pose: Any, num_links: int = 3) -> Any:
  import cv2

  mask = np.zeros((view.height, view.width), np.uint8)
  placed = bodies(task, pose)
  for link, kind, centre, extent, low, high in parts(object_xml(task, num_links)):
    local = _outline(kind, centre, extent)
    levels = [view.project(_placed(view, local, placed[link], h)) for h in (low, high)]
    _fill(mask, cv2.convexHull(np.concatenate(levels).astype(np.float32))[:, 0])
  return mask


def _chain_tape(view: Any, chain_pose: Any, slack: float, mask: Any) -> None:
  from vbrl.tasks.push_chain.geometry import LINK_HALF_WIDTH, half_lengths

  width = LINK_HALF_WIDTH + slack
  chain = links(chain_pose)
  halves = half_lengths(len(chain))
  for i, (link, length) in enumerate(zip(chain, halves, strict=True)):
    back = length + (slack if i == 0 else 0.0)
    front = length + (slack if i == len(chain) - 1 else 0.0)
    local = _rectangle(((front - back) / 2, 0.0), ((front + back) / 2, width))
    _fill(mask, view.project(_placed(view, local, link, 0.0)))
  ends = chain[:-1]
  pins = (
    ends[:, :2]
    + np.array(halves[:-1])[:, None] * np.c_[np.cos(ends[:, 2]), np.sin(ends[:, 2])]
  )
  table = (0.0, 0.0, 0.0)
  for i, (pin, hinge) in enumerate(zip(pins, chain_pose[3:], strict=True)):
    _fill(mask, view.project(_placed(view, pin + width * _CIRCLE, table, 0.0)))
    side = -1.0 if hinge >= 0.0 else 1.0
    normals = [
      side * np.array((-math.sin(a), math.cos(a))) for a in chain[i : i + 2, 2]
    ]
    mitre = width * (normals[0] + normals[1]) / (1.0 + normals[0] @ normals[1])
    quad = pin + np.array((0.0 * mitre, width * normals[0], mitre, width * normals[1]))
    _fill(mask, view.project(_placed(view, quad, table, 0.0)))


def render_goal(view: Any, task: str, pose: Any, num_links: int = 3) -> Any:
  *pose, slack = pose
  mask = np.zeros((view.height, view.width), np.uint8)
  if task == "push_chain":
    _chain_tape(view, pose, slack, mask)
    return mask
  for _, kind, centre, extent, _, _ in parts(object_xml(task, num_links)):
    _fill(
      mask,
      view.project(_placed(view, _outline(kind, centre, extent, slack), pose, 0.0)),
    )
  return mask


def _largest_region(mask: Any) -> Any:
  import cv2

  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  region = np.zeros_like(mask)
  if contours:
    cv2.drawContours(region, [max(contours, key=cv2.contourArea)], -1, 1, cv2.FILLED)
  return region


def colour_masks(view: Any, image: Any) -> tuple[Any, Any]:
  rgb = image.astype(np.float32)
  intensity = rgb.sum(axis=-1)
  workspace = np.zeros((view.height, view.width), np.uint8)
  _fill(workspace, view.project(_placed(view, _rectangle(*WORKSPACE), (0, 0, 0), 0.0)))
  green = rgb[..., 1] / np.maximum(intensity, 1.0)
  valid = (workspace > 0) & (intensity > MIN_INTENSITY)
  green -= np.median(green[valid])
  red = _largest_region(((green < -CHROMATICITY_MARGIN) & valid).astype(np.uint8))
  tape = ((green > CHROMATICITY_MARGIN) & valid).astype(np.uint8)
  return red, _largest_region(tape) & (1 - tape)


def _shapes(task: str, num_links: int) -> list[tuple[float, ...]]:
  from vbrl.tasks.push_chain.geometry import HINGE_LIMIT

  if task == "lift_cube":
    return [(yaw,) for yaw in np.linspace(0, math.pi / 2, YAW_STEPS // 4, False)]
  yaws = np.linspace(-math.pi, math.pi, YAW_STEPS, endpoint=False)
  if task == "push_t":
    return [(yaw,) for yaw in yaws]
  steps = max(2, round((CANDIDATES / YAW_STEPS) ** (1 / (num_links - 1))))
  hinges = np.linspace(-HINGE_LIMIT, HINGE_LIMIT, steps)
  return list(itertools.product(yaws, *[hinges] * (num_links - 1)))


def _offset(task: str, shape: Any) -> Any:
  if task != "push_chain":
    return np.zeros(2)
  from vbrl.tasks.push_chain.geometry import half_lengths

  chain = links((0.0, 0.0, *shape))
  lengths = np.array(half_lengths(len(chain)))
  return lengths @ chain[:, :2] / lengths.sum()


def _fit(
  view: Any,
  task: str,
  observed: Any,
  render: Any,
  height: float,
  extra: tuple,
  num_links: int,
) -> Fit:
  from scipy.optimize import minimize

  shapes = _shapes(task, num_links)
  size = 2 + len(shapes[0]) + len(extra)
  if not observed.any():
    return Fit((math.nan,) * size, 0.0, False)
  rows, columns = np.nonzero(observed)
  centre = view.on_table((columns.mean(), rows.mean()), height)[:2]
  observed = observed.astype(bool)

  def cost(pose: Any) -> float:
    rendered = render(view, task, pose, num_links).astype(bool)
    return 1.0 - (rendered & observed).sum() / max((rendered | observed).sum(), 1)

  def refine(pose: Any) -> Any:
    for step in (0.01, 0.002):
      steps = np.full(size, step)
      steps[2 : 2 + len(shapes[0])] *= 10.0
      simplex = np.vstack([pose, pose + np.diag(steps)])
      options = {"initial_simplex": simplex, "xatol": 1e-5, "fatol": 1e-6}
      pose = minimize(cost, pose, method="Nelder-Mead", options=options).x
    return pose

  candidates = [np.array((*(centre - _offset(task, s)), *s, *extra)) for s in shapes]
  pose = min((refine(p) for p in sorted(candidates, key=cost)[:STARTS]), key=cost)
  period = math.pi / 2 if task == "lift_cube" else 2 * math.pi
  pose[2] = math.remainder(pose[2], period)
  iou = 1.0 - cost(pose)
  return Fit(tuple(float(v) for v in pose), iou, iou >= MIN_IOU)


def measure_object(view: Any, image: Any, task: str, num_links: int = 3) -> Fit:
  height = max(p[5] for p in parts(object_xml(task, num_links))) / 2
  red, _ = colour_masks(view, image)
  return _fit(view, task, red, render_object, height, (), num_links)


def measure_goal(view: Any, image: Any, task: str, num_links: int = 3) -> Fit:
  _, hole = colour_masks(view, image)
  # The last parameter is how far the tape sits outside the object's edge.
  return _fit(view, task, hole, render_goal, 0.0, (0.0,), num_links)


def overlap(task: str, pose: Any, goal: Any) -> float:
  import torch

  if task == "push_chain":
    from vbrl.tasks.push_chain.geometry import LinkOverlap

    current = torch.tensor(links(pose), dtype=torch.float)[None]
    target = torch.tensor(links(goal[: len(pose)]), dtype=torch.float)[None]
    per_link = LinkOverlap(current.shape[1], device="cpu", dtype=torch.float)
    return float(per_link.best(current, target)[0].min())
  from vbrl.tasks.push_t.geometry import FOOTPRINT_PARTS, footprint_overlap_from_pose

  t, g = torch.tensor([pose[:3]]), torch.tensor([goal[:3]])
  return float(
    footprint_overlap_from_pose(
      object_xy=t[:, :2],
      object_yaw=t[:, 2],
      target_xy=g[:, :2],
      target_yaw=g[:, 2],
      footprint_parts=FOOTPRINT_PARTS,
    )[0]
  )


def view_for(image: Any) -> Any:
  from vbrl.rig.view import ExternalView

  height, width = image.shape[:2]
  if (width, height) == (224, 224):
    return ExternalView.load(424, 240).cropped(224)
  return ExternalView.load(width, height)


def read(path: Path) -> Any:
  import cv2

  image = cv2.imread(str(path))
  if image is None:
    raise SystemExit(f"Could not read {path}.")
  return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write(path: Path, image: Any) -> None:
  import cv2

  path.parent.mkdir(parents=True, exist_ok=True)
  cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def annotate(
  view: Any, image: Any, task: str, fit: Fit, goal: bool, num_links: int = 3
) -> Any:
  import cv2

  render = render_goal if goal else render_object
  out = image.copy()
  mask = render(view, task, fit.pose, num_links)
  contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
  cv2.drawContours(out, contours, -1, (255, 0, 255), 1)
  return out


def load_goal(value: str) -> list[float]:
  if Path(value).is_file():
    return json.loads(Path(value).read_text())["goal"]
  return [float(v) for v in value.split(",")]


def main(argv: Any = None) -> None:
  parser = argparse.ArgumentParser(
    description="Measure the goal (frame without the object) or the object in one "
    "external-camera image; with --goal also its overlap."
  )
  parser.add_argument("task", choices=TASKS)
  parser.add_argument("image", type=Path, help="a frame, or configs/deployment/<task>.yaml to capture one")
  parser.add_argument(
    "--goal", help="goal.json or x,y,yaw[,hinges]: measure the object"
  )
  parser.add_argument("--save", type=Path, help="write the measured goal as goal.json")
  parser.add_argument("--links", type=int, default=3)
  parser.add_argument("--annotate", type=Path)
  args = parser.parse_args(argv)
  if args.image.suffix == ".yaml":
    from vbrl.deployment.camera import still
    from vbrl.deployment.session import load_session

    image = still(load_session(args.image).config(camera="external"), 1280, 720)
  else:
    image = read(args.image)
  view = view_for(image)
  goal = not args.goal and args.task != "lift_cube"
  measure = measure_goal if goal else measure_object
  fit = measure(view, image, args.task, args.links)
  values = " ".join(f"{v:+.4f}" for v in fit.pose)
  print(f"{values}  IoU {fit.iou:.3f}{'' if fit.reliable else '  UNRELIABLE'}")
  if args.goal:
    print(f"overlap {overlap(args.task, fit.pose, load_goal(args.goal)):.3f}")
  elif args.save and args.task != "lift_cube":
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(
      json.dumps({"goal": list(fit.pose[:-1]), "iou": fit.iou}, indent=2)
    )
  if args.annotate:
    write(args.annotate, annotate(view, image, args.task, fit, goal, args.links))


if __name__ == "__main__":
  main()


__all__ = [
  "TASKS",
  "Fit",
  "annotate",
  "colour_masks",
  "load_goal",
  "measure_goal",
  "measure_object",
  "object_xml",
  "overlap",
  "read",
  "render_goal",
  "render_object",
  "view_for",
  "write",
]
