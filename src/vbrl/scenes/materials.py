"""MuJoCo material construction for every scene material bank."""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

import mujoco

from .presets import (
  PROCEDURAL_KINDS,
  PROCEDURAL_VARIANTS_PER_KIND,
  MaterialBank,
  TexturePreset,
  ambientcg_pool_paths,
  ambientcg_texture_names,
  procedural_material_names,
)


def texture_slots(color_texture: str) -> list[str]:
  """Return MuJoCo's role-indexed texture list with the RGB texture assigned."""
  return [""] + [color_texture] + [""] * 8


def add_image_material(
  spec: mujoco.MjSpec,
  *,
  material_name: str,
  texture_name: str,
  image: Path,
  rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
  texrepeat: tuple[float, float] = (1.0, 1.0),
  roughness: float | None = None,
  specular: float | None = None,
) -> str:
  """Add one file-backed RGB texture and its material to ``spec``."""
  spec.add_texture(
    name=texture_name,
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    file=str(image),
  )
  kwargs: dict[str, object] = {
    "name": material_name,
    "rgba": rgba,
    "texrepeat": texrepeat,
    "texuniform": False,
  }
  if roughness is not None:
    kwargs["roughness"] = roughness
  if specular is not None:
    kwargs["specular"] = specular
  material = spec.add_material(**kwargs)
  material.textures = texture_slots(texture_name)
  return material_name


def add_preset_material(
  spec: mujoco.MjSpec,
  *,
  prefix: str,
  preset: TexturePreset,
) -> str:
  """Add a catalogued image texture with its rendering metadata."""
  return add_image_material(
    spec,
    material_name=f"{prefix}_material",
    texture_name=f"{prefix}_texture",
    image=preset.require(),
    rgba=preset.rgba,
    texrepeat=preset.texrepeat,
    roughness=preset.roughness,
    specular=preset.specular,
  )


def _rgb_texture_data(
  width: int,
  height: int,
  pixel: Callable[[int, int], tuple[float, float, float]],
) -> bytes:
  data = bytearray()
  for y in range(height):
    for x in range(width):
      data.extend(
        max(0, min(255, round(channel * 255.0))) for channel in pixel(x, y)
      )
  return bytes(data)


def _add_procedural_material(
  spec: mujoco.MjSpec,
  *,
  name: str,
  color_a: tuple[float, float, float],
  color_b: tuple[float, float, float],
  kind: str,
  checker_repeat: float,
) -> None:
  if kind == "solid":
    spec.add_material(name=name, rgba=(*color_a, 1.0))
    return

  width, height = (64, 2) if kind == "gradient" else (64, 64)
  pixel: Callable[[int, int], tuple[float, float, float]]
  if kind == "gradient":

    def gradient_pixel(x: int, _y: int) -> tuple[float, float, float]:
      alpha = x / (width - 1)
      return (
        (1.0 - alpha) * color_a[0] + alpha * color_b[0],
        (1.0 - alpha) * color_a[1] + alpha * color_b[1],
        (1.0 - alpha) * color_a[2] + alpha * color_b[2],
      )

    pixel = gradient_pixel

  elif kind == "checker":

    def checker_pixel(x: int, y: int) -> tuple[float, float, float]:
      return (
        color_a
        if ((2 * x) // width + (2 * y) // height) % 2 == 0
        else color_b
      )

    pixel = checker_pixel

  else:
    raise ValueError(f"Unsupported procedural material kind: {kind}")

  texture_name = f"{name}_texture"
  texture = spec.add_texture(
    name=texture_name,
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    width=width,
    height=height,
  )
  texture.data = _rgb_texture_data(width, height, pixel)
  repeat = (checker_repeat, checker_repeat) if kind == "checker" else (1.0, 1.0)
  material = spec.add_material(
    name=name,
    rgba=(1.0, 1.0, 1.0, 1.0),
    texrepeat=repeat,
    texuniform=False,
  )
  material.textures = texture_slots(texture_name)


def _add_procedural_pool(spec: mujoco.MjSpec, bank: MaterialBank) -> tuple[str, ...]:
  """Bake 32 seeded, full-range two-color variants of every material kind."""
  assert bank.seed is not None
  rng = random.Random(bank.seed)
  for kind in PROCEDURAL_KINDS:
    for index in range(PROCEDURAL_VARIANTS_PER_KIND):
      color_a = (rng.random(), rng.random(), rng.random())
      color_b = (rng.random(), rng.random(), rng.random())
      _add_procedural_material(
        spec,
        name=f"{bank.prefix}_{kind}_{index:02d}",
        color_a=color_a,
        color_b=color_b,
        kind=kind,
        checker_repeat=bank.checker_repeat,
      )
  return procedural_material_names(bank.prefix)


def _add_ambientcg_pool(spec: mujoco.MjSpec, bank: MaterialBank) -> tuple[str, ...]:
  """Bake the pooled AmbientCG textures behind a single material.

  Every pooled image becomes a texture; one material points its RGB role slot
  at the first of them, and ``dr.mat_texid`` repoints that slot per
  environment. The bank costs one material, but textures are still capped --
  see :func:`ambientcg_pool_paths`.
  """
  names = ambientcg_texture_names()
  for path, texture_name in zip(ambientcg_pool_paths(), names, strict=True):
    spec.add_texture(
      name=texture_name,
      type=mujoco.mjtTexture.mjTEXTURE_2D,
      file=str(path),
    )
  material = spec.add_material(
    name=bank.material_name,
    rgba=(1.0, 1.0, 1.0, 1.0),
    texrepeat=(1.0, 1.0),
    texuniform=False,
  )
  material.textures = texture_slots(names[0])
  return (bank.material_name,)


def _add_image_bank(spec: mujoco.MjSpec, bank: MaterialBank) -> tuple[str, ...]:
  assert bank.image is not None
  return (add_preset_material(spec, prefix=bank.prefix, preset=bank.image),)


_BUILDERS = {
  "procedural": _add_procedural_pool,
  "ambientcg": _add_ambientcg_pool,
  "image": _add_image_bank,
}


def add_bank(spec: mujoco.MjSpec, bank: MaterialBank) -> tuple[str, ...]:
  """Bake one material bank into ``spec`` and return its material names."""
  return _BUILDERS[bank.kind](spec, bank)


__all__ = [
  "add_bank",
  "add_image_material",
  "add_preset_material",
  "texture_slots",
]
