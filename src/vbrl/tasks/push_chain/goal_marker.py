from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from vbrl.tasks.push_chain.geometry import LINK_HALF_EXTENTS
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


def _frame(half_length: float, half_width: float, t: float):
  return (
    ((0.0, half_width + t / 2), (half_length + t, t / 2)),
    ((0.0, -half_width - t / 2), (half_length + t, t / 2)),
    ((half_length + t / 2, 0.0), (t / 2, half_width)),
    ((-half_length - t / 2, 0.0), (t / 2, half_width)),
  )


def goal_marker_spec(link: int) -> mujoco.MjSpec:
  import mujoco

  spec = mujoco.MjSpec()
  spec.add_material(name=GOAL_MATERIAL_NAME, rgba=GOAL_RGBA)
  name = GOAL_ENTITY_NAMES[link]
  body = spec.worldbody.add_body(name=name, mocap=True)
  frame = _frame(*LINK_HALF_EXTENTS[link], GOAL_OUTLINE_THICKNESS)
  for index, (centre, half) in enumerate(frame):
    body.add_geom(
      name=f"{name}_{index}",
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
  from mjlab.envs.mdp import dr
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  for link, name in enumerate(GOAL_ENTITY_NAMES):
    cfg.scene.entities[name] = EntityCfg(spec_fn=partial(goal_marker_spec, link))
    cfg.events[f"{name}_color"] = EventTermCfg(
      func=dr.mat_rgba,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg(name, material_names=(GOAL_MATERIAL_NAME,)),
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0, 1, 2],
        "ranges": dict(enumerate(GOAL_REAL_RGBA_RANGE)),
        "shared_random": True,
      },
    )


__all__ = [
  "GOAL_ENTITY_NAMES",
  "GOAL_MATERIAL_NAME",
  "add_goal_marker",
  "goal_marker_spec",
]
