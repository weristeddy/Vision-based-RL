"""A visual-only T drawn on the table at the commanded goal pose.

Every published Push-T renders its target into the observation: Implicit
Behavioral Cloning and Diffusion Policy draw a green outline on the plane, and
ManiSkill3 builds a grey kinematic ``goal_Tee`` that its RGB cameras see. VBRL
omitted it, so the policy has had to estimate the object's absolute yaw from
pixels and compose that with a goal yaw arriving as two numbers in a separate
proprioceptive stream. This entity restores the benchmark's own design.

The footprint comes from :data:`FOOTPRINT_PARTS`, the same constant the overlap
rasteriser scores against, so the drawn target cannot drift from the shape the
reward measures. The body is declared mocap, which is what lets a fixed base be
posed per reset (``write_root_link_pose_to_sim`` refuses one); the geoms are
massless and collisionless, so the marker is scenery for the camera and
invisible to physics.

:func:`goal_colour_event` resamples its colour every reset, so :data:`GOAL_RGBA`
is only the value the spec is built with.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vbrl.tasks.push_t.geometry import FOOTPRINT_PARTS

if TYPE_CHECKING:
  import mujoco

GOAL_ENTITY_NAME = "goal_marker"
GOAL_COLOUR_EVENT = "goal_color"
GOAL_MATERIAL_NAME = "push_t_goal_green"
# Only the value the spec is built with: `goal_colour_event` resamples it every
# reset, so nothing should depend on this being green.
GOAL_RGBA = (0.16, 0.62, 0.29, 1.0)
# Thin enough to read as drawn on the surface rather than as a second block, and
# sunk so its top face sits just above the table at z=0.
GOAL_HALF_THICKNESS = 0.0012


def goal_colour_event():
  """Resample the marker's colour every reset, uniform over the RGB cube.

  Independent of the object's colour event, and unguarded. Sampling both
  uniformly lands them within an RGB distance of 0.1 on about 0.4% of resets
  and within a 1.2:1 luminance ratio on 22% -- episodes that ask a policy to
  align two shapes it can barely separate, and a filled marker shares the
  object's silhouette exactly. Accepted deliberately: the point is a policy
  that finds the target whatever colour it is.
  """
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


def goal_marker_spec() -> "mujoco.MjSpec":
  """Build the target T as a jointless, massless, collisionless body."""
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
