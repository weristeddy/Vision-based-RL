from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING

import torch

from vbrl.tasks.push_chain.geometry import (
  HINGE_LIMIT,
  LINK_HALF_EXTENTS,
  LINK_HALF_WIDTH,
)
from vbrl.tasks.push_t.goal_marker import (
  GOAL_HALF_THICKNESS,
  GOAL_OUTLINE_THICKNESS,
  GOAL_REAL_RGBA_RANGE,
  GOAL_RGBA,
)

if TYPE_CHECKING:
  import mujoco

GOAL_ENTITY_NAMES = tuple(f"goal_marker_link{i}" for i in range(3))
GOAL_MATERIAL_NAME = "push_chain_goal_green"
SIDE_GEOM_NAMES = ("side0", "side1")
SIDES = (-1.0, 1.0)
_OUTER = LINK_HALF_WIDTH + GOAL_OUTLINE_THICKNESS
MAX_EXTENSION = _OUTER * math.tan(HINGE_LIMIT / 2.0)


def _extension(hinge: torch.Tensor, side: float) -> torch.Tensor:
  bend = hinge.abs()
  outer = _OUTER * torch.tan(bend / 2.0)
  inner = torch.maximum(
    LINK_HALF_WIDTH * torch.tan(bend / 2.0),
    (LINK_HALF_WIDTH - _OUTER * torch.cos(bend)) / torch.sin(bend).clamp_min(1e-6),
  )
  return torch.where(side * hinge > 0.0, -inner, outer)


def side_strips(hinges: torch.Tensor) -> torch.Tensor:
  half = hinges.new_tensor([length for length, _ in LINK_HALF_EXTENTS])
  free = torch.full_like(hinges[..., 0], GOAL_OUTLINE_THICKNESS)
  strips = []
  for side in SIDES:
    first, second = (_extension(hinges[..., i], side) for i in range(2))
    back = torch.stack((free, first, second), dim=-1) + half
    front = torch.stack((first, second, free), dim=-1) + half
    strips.append(torch.stack(((front - back) / 2.0, (front + back) / 2.0), dim=-1))
  return torch.stack(strips, dim=-2)


def _boxes(link: int):
  half_length, half_width = LINK_HALF_EXTENTS[link]
  t = GOAL_OUTLINE_THICKNESS
  longest = side_strips(torch.full((2,), HINGE_LIMIT))[link, 0].tolist()
  sides = [
    (name, (longest[0], side * (half_width + t / 2.0)), (longest[1], t / 2.0))
    for name, side in zip(SIDE_GEOM_NAMES, SIDES, strict=True)
  ]
  caps = [
    ("cap", (side * (half_length + t / 2.0), 0.0), (t / 2.0, half_width))
    for side in ((-1.0,), (), (1.0,))[link]
  ]
  return sides + caps


def goal_marker_spec(link: int) -> mujoco.MjSpec:
  import mujoco

  spec = mujoco.MjSpec()
  spec.add_material(name=GOAL_MATERIAL_NAME, rgba=GOAL_RGBA)
  body = spec.worldbody.add_body(name=GOAL_ENTITY_NAMES[link], mocap=True)
  for name, centre, half in _boxes(link):
    body.add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(*half, GOAL_HALF_THICKNESS),
      pos=(*centre, GOAL_HALF_THICKNESS),
      material=GOAL_MATERIAL_NAME,
      contype=0,
      conaffinity=0,
      mass=0.0,
    )
  return spec


def add_goal_marker(cfg) -> None:
  from mjlab.entity import EntityCfg
  from mjlab.managers.event_manager import EventTermCfg

  from vbrl.tasks.push_chain.mdp.events import shared_material_rgb

  for link, name in enumerate(GOAL_ENTITY_NAMES):
    cfg.scene.entities[name] = EntityCfg(spec_fn=partial(goal_marker_spec, link))
  cfg.events["goal_marker_color"] = EventTermCfg(
    func=shared_material_rgb,
    mode="reset",
    params={"entity_names": GOAL_ENTITY_NAMES, "ranges": GOAL_REAL_RGBA_RANGE},
  )


__all__ = [
  "GOAL_ENTITY_NAMES",
  "GOAL_MATERIAL_NAME",
  "MAX_EXTENSION",
  "SIDE_GEOM_NAMES",
  "add_goal_marker",
  "goal_marker_spec",
  "side_strips",
]
