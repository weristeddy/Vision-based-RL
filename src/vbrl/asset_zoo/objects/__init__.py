"""Packaged task objects."""

from pathlib import Path


OBJECTS_DIR = Path(__file__).resolve().parent
MESHES_DIR = OBJECTS_DIR / "meshes"

CUBE_XML = OBJECTS_DIR / "cube.xml"
PUSH_T_XML = OBJECTS_DIR / "push_t.xml"
SYRINGE_XML = OBJECTS_DIR / "syringe.xml"
# A receptacle rather than a manipulated object: its body is mocap, so it is
# posed per reset and never moved by contact.
KIDNEY_DISH_XML = OBJECTS_DIR / "kidney_dish.xml"


__all__ = [
  "CUBE_XML",
  "KIDNEY_DISH_XML",
  "MESHES_DIR",
  "OBJECTS_DIR",
  "PUSH_T_XML",
  "SYRINGE_XML",
]
