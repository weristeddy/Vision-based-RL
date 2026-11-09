from __future__ import annotations

from vbrl.tasks.push_t.mdp.events import (
  OBJECT_TABLE_FRICTION_MEAN,
  OBJECT_TABLE_FRICTION_STD,
  TABLE_COLLISION_GEOM,
  randomize_object_table_friction,
)

OBJECT_COLLISION_GEOMS = r"push_chain_link\d+_.*_collision"
HINGE_JOINTS = r"push_chain_hinge\d"
HINGE_SCALE_RANGE = (0.5, 2.0)


__all__ = [
  "HINGE_JOINTS",
  "HINGE_SCALE_RANGE",
  "OBJECT_COLLISION_GEOMS",
  "OBJECT_TABLE_FRICTION_MEAN",
  "OBJECT_TABLE_FRICTION_STD",
  "TABLE_COLLISION_GEOM",
  "randomize_object_table_friction",
]
