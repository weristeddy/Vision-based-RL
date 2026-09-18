from __future__ import annotations

from typing import TYPE_CHECKING

from vbrl.tasks.push_t.geometry import FOOTPRINT_PARTS

if TYPE_CHECKING:
  import mujoco

GOAL_ENTITY_NAME = "goal_marker"
GOAL_COLOUR_EVENT = "goal_color"
GOAL_MATERIAL_NAME = "push_t_goal_green"
GOAL_RGBA = (0.16, 0.62, 0.29, 1.0)
# Thin enough to read as drawn on the surface rather than as a second block, and
# sunk so its top face sits just above the table at z=0.
GOAL_HALF_THICKNESS = 0.0012


def goal_colour_event():
  from mjlab.envs.mdp import dr
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  return EventTermCfg(
    func=dr.mat_rgba,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg(
        GOAL_ENTITY_NAME, material_names=(GOAL_MATERIAL_NAME,)
      ),
      "operation": "abs",
      "distribution": "uniform",
      "axes": [0, 1, 2],
      "ranges": (0.0, 1.0),
      "shared_random": True,
    },
  )


def goal_marker_spec() -> mujoco.MjSpec:
  import mujoco

  spec = mujoco.MjSpec()
  spec.add_material(name=GOAL_MATERIAL_NAME, rgba=GOAL_RGBA)
  body = spec.worldbody.add_body(name=GOAL_ENTITY_NAME, mocap=True)
  for index, part in enumerate(FOOTPRINT_PARTS):
    body.add_geom(
      name=f"{GOAL_ENTITY_NAME}_{index}",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(*part.half_extents_xy, GOAL_HALF_THICKNESS),
      pos=(*part.center_xy, GOAL_HALF_THICKNESS),
      material=GOAL_MATERIAL_NAME,
      contype=0,
      conaffinity=0,
      mass=0.0,
    )
  return spec


__all__ = [
  "GOAL_COLOUR_EVENT",
  "GOAL_ENTITY_NAME",
  "GOAL_HALF_THICKNESS",
  "GOAL_MATERIAL_NAME",
  "GOAL_RGBA",
  "goal_colour_event",
  "goal_marker_spec",
]
