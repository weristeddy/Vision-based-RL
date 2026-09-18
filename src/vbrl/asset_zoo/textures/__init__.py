from __future__ import annotations

from pathlib import Path

TEXTURES_DIR = Path(__file__).resolve().parent
AMBIENTCG_DIR = TEXTURES_DIR / "ambientcg" / "basecolor_256"
UNIT_BOX_UV_MESH = TEXTURES_DIR / "unit_box_uv.obj"


__all__ = [
  "AMBIENTCG_DIR",
  "TEXTURES_DIR",
  "UNIT_BOX_UV_MESH",
]
