"""Packaged texture assets and where they live.

Owns the texture paths for the whole tree, ``asset_zoo`` being the layer below
``scenes``: :mod:`vbrl.scenes.presets` imports :data:`TEXTURES_DIR` and
:data:`AMBIENTCG_DIR` from here rather than recomputing them, so the fetcher,
the material banks and the object assets cannot end up pointing at different
directories. Kept to :mod:`pathlib` alone, because ``presets`` is imported by
CLIs and task configs that must not pay for MuJoCo.

Object MJCFs reach their own textures by relative path rather than through
constants here -- ``kidney_dish.xml`` takes brushed steel out of the AmbientCG
bank below, and ``syringe.xml`` takes its printed scale from ``syringe/`` --
which keeps each asset readable on its own.
"""

from __future__ import annotations

from pathlib import Path


TEXTURES_DIR = Path(__file__).resolve().parent
AMBIENTCG_DIR = TEXTURES_DIR / "ambientcg" / "basecolor_256"
OOD_DIR = TEXTURES_DIR / "ood"
UNIT_BOX_UV_MESH = TEXTURES_DIR / "unit_box_uv.obj"


__all__ = [
  "AMBIENTCG_DIR",
  "OOD_DIR",
  "TEXTURES_DIR",
  "UNIT_BOX_UV_MESH",
]
