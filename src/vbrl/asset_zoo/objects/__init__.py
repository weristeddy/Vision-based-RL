"""Packaged task objects."""

from pathlib import Path


OBJECTS_DIR = Path(__file__).resolve().parent
CUBE_XML = OBJECTS_DIR / "cube.xml"
PUSH_T_XML = OBJECTS_DIR / "push_t.xml"


__all__ = ["CUBE_XML", "OBJECTS_DIR", "PUSH_T_XML"]
