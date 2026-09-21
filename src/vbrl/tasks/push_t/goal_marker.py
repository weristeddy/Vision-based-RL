from __future__ import annotations

from typing import TYPE_CHECKING

from vbrl.tasks.push_t.geometry import FOOTPRINT_PARTS

if TYPE_CHECKING:
  import mujoco

GOAL_ENTITY_NAME = "goal_marker"
GOAL_COLOUR_EVENT = "goal_color"
GOAL_MATERIAL_NAME = "push_t_goal_green"
GOAL_RGBA = (0.16, 0.62, 0.29, 1.0)
# Measured off the rig's printed marker in
# artifacts/deployment/t_apriltag_1280x720_Color.png, 4948 px: mean
# 0.021/0.331/0.227, per-channel p10-p90 0.000-0.059, 0.314-0.349, 0.208-0.247.
# Widened about threefold on each spread for lighting headroom, then scaled by
# 0.68 because these are albedos and the scene lighting amplifies them: at the
# unscaled values the render observed 0.092/0.496/0.337 against the marker's
# 0.021/0.331/0.227. The same correction the table's rgba carries.
GOAL_REAL_RGBA_RANGE = ((0.0, 0.061), (0.190, 0.265), (0.115, 0.190))
GOAL_OUTLINE_THICKNESS = 0.015
# Thin enough to read as drawn on the surface rather than as a second block, and
# sunk so its top face sits just above the table at z=0.
GOAL_HALF_THICKNESS = 0.0012


def goal_colour_event(ranges=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))):
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
      "ranges": dict(enumerate(ranges)),
      "shared_random": True,
    },
  )


# Eight boxes forming a closed frame whose hole is exactly the T footprint.
def _outline_boxes(thickness: float):
  (cx, cy), (chx, chy) = FOOTPRINT_PARTS[0].center_xy, FOOTPRINT_PARTS[0].half_extents_xy
  (sx, sy), (shx, shy) = FOOTPRINT_PARTS[1].center_xy, FOOTPRINT_PARTS[1].half_extents_xy
  cx0, cx1, cy0, cy1 = cx - chx, cx + chx, cy - chy, cy + chy
  sx0, sx1, sy1 = sx - shx, sx + shx, sy + shy
  t = thickness
  spans = (
    ((cx0 - t, cx1 + t), (cy0 - t, cy0)),      # under the crossbar
    ((cx0 - t, cx0), (cy0, cy1)),              # crossbar left
    ((cx1, cx1 + t), (cy0, cy1)),              # crossbar right
    ((cx0 - t, sx0), (cy1, cy1 + t)),          # left shoulder
    ((sx1, cx1 + t), (cy1, cy1 + t)),          # right shoulder
    ((sx0 - t, sx0), (cy1 + t, sy1)),          # stem left
    ((sx1, sx1 + t), (cy1 + t, sy1)),          # stem right
    ((sx0 - t, sx1 + t), (sy1, sy1 + t)),      # over the stem
  )
  return [
    (((x0 + x1) / 2.0, (y0 + y1) / 2.0), ((x1 - x0) / 2.0, (y1 - y0) / 2.0))
    for (x0, x1), (y0, y1) in spans
  ]


def goal_marker_spec(outline: bool = False) -> mujoco.MjSpec:
  import mujoco

  spec = mujoco.MjSpec()
  spec.add_material(name=GOAL_MATERIAL_NAME, rgba=GOAL_RGBA)
  body = spec.worldbody.add_body(name=GOAL_ENTITY_NAME, mocap=True)
  parts = (
    _outline_boxes(GOAL_OUTLINE_THICKNESS)
    if outline
    else [(p.center_xy, p.half_extents_xy) for p in FOOTPRINT_PARTS]
  )
  for index, (centre, half) in enumerate(parts):
    body.add_geom(
      name=f"{GOAL_ENTITY_NAME}_{index}",
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(*half, GOAL_HALF_THICKNESS),
      pos=(*centre, GOAL_HALF_THICKNESS),
      material=GOAL_MATERIAL_NAME,
      contype=0,
      conaffinity=0,
      mass=0.0,
    )
  return spec


__all__ = [
  "GOAL_COLOUR_EVENT",
  "GOAL_OUTLINE_THICKNESS",
  "GOAL_REAL_RGBA_RANGE",
  "GOAL_ENTITY_NAME",
  "GOAL_HALF_THICKNESS",
  "GOAL_MATERIAL_NAME",
  "GOAL_RGBA",
  "goal_colour_event",
  "goal_marker_spec",
]
