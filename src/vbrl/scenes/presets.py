"""Scene appearance declared as data.

Deliberately free of MuJoCo and MJLab imports: CLIs read ``--scene`` choices
from here and task terminations read the tabletop extents.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

# Imported rather than recomputed: `asset_zoo.textures` owns these paths, and it
# stays pathlib-only so this module keeps its freedom from MuJoCo.
from vbrl.asset_zoo.textures import (
  AMBIENTCG_DIR,
  TEXTURES_DIR,
  UNIT_BOX_UV_MESH,
)

# --- Tabletop geometry, shared by every scene -------------------------------
TABLE_GEOM_NAME = "table_top"
TABLE_VISUAL_GEOM_NAME = "table_top_visual"
TABLE_LIGHT_NAME = "sun"
FILL_LIGHT_NAME = "sim2sim_fill"
TABLE_VISUAL_MESH_NAME = "table_top_visual_mesh"

# 1.0 x 1.0 m, matching the plywood the real rig is built on. Was
# (0.45, 0.35): 300 mm narrower in y, which is why the sim table's side edges
# cut into the tilted camera's view well before the real table's do.
#
# push_t's out-of-bounds termination reads these too, so the boundary moves --
# outwards only. Object and goal sampling stay far inside it, so this can
# retire a termination that would have fired but cannot introduce one.
TABLE_HALF_EXTENTS = (0.5, 0.5, 0.02)
TABLE_CENTER = (0.3, 0.0, -0.02)
CAMERA_POSITION_DR_RANGE_M = 0.025
CAMERA_ROTATION_DR_RANGE_RAD = 0.03

# --- Material bank sizing ---------------------------------------------------
PROCEDURAL_VARIANTS_PER_KIND = 32
PROCEDURAL_KINDS = ("solid", "gradient", "checker")
PROCEDURAL_TABLE_SEED = 170306907
PROCEDURAL_OBJECT_SEED = 170306908
# MuJoCo's classic renderer aborts on a model with more than 1,000 textures,
# and ``--video`` builds one. The AmbientCG catalog is larger than that, so a
# deterministic sample is baked in. Size and seed reproduce the bank the
# retained real-texture checkpoints were trained against.
AMBIENTCG_POOL_SIZE = 768
AMBIENTCG_POOL_SEED = 170306909
MUJOCO_MAX_TEXTURES = 1000

BankKind = Literal["procedural", "ambientcg", "image"]
LightSet = Literal["training", "realistic"]
EvaluationDr = Literal["fixed", "matched"]
# Which MuJoCo slot a bank's per-reset event samples. ``matid`` repoints a geom
# at another material and is the only option for banks whose variants differ in
# more than their image (the procedural ``solid`` kind carries its colour on
# ``mat_rgba`` and has no texture at all). ``texid`` keeps one material and
# swaps the texture in its RGB role slot, which is what lets the photographic
# bank grow past MuJoCo's 1,000-material renderer limit.
RandomizedSlot = Literal["matid", "texid"]


@dataclass(frozen=True)
class TexturePreset:
  """A named PBR-style RGB texture and its MuJoCo material settings."""

  name: str
  image: Path
  texrepeat: tuple[float, float]
  rgba: tuple[float, float, float, float]
  roughness: float
  specular: float

  def require(self) -> Path:
    if not self.image.is_file():
      raise FileNotFoundError(f"Missing texture asset for {self.name!r}: {self.image}")
    return self.image


WOOD_TABLE = TexturePreset(
  name="wood",
  image=TEXTURES_DIR / "ood" / "wood" / "wood_table_worn_diffuse_1k.png",
  texrepeat=(3.0, 2.2),
  rgba=(1.0, 1.0, 1.0, 1.0),
  roughness=0.88,
  specular=0.08,
)
PEACOCK_TABLE = TexturePreset(
  name="peacock",
  image=TEXTURES_DIR / "ood" / "peacock" / "peacock_feathers_color_1k.png",
  texrepeat=(1.0, 1.0),
  rgba=(1.0, 1.0, 1.0, 1.0),
  roughness=0.32,
  specular=0.30,
)
PLASTER_TABLE = TexturePreset(
  name="plaster",
  image=TEXTURES_DIR / "ood" / "plaster" / "white_plaster_02_diffuse_1k.png",
  texrepeat=(2.4, 1.9),
  rgba=(0.72, 0.72, 0.72, 1.0),
  roughness=0.95,
  specular=0.02,
)
RED_PLASTIC_OBJECT = TexturePreset(
  name="red_plastic",
  image=TEXTURES_DIR / "ood" / "red_plastic" / "plastic007_color_1k.png",
  texrepeat=(1.0, 1.0),
  rgba=(1.0, 1.0, 1.0, 1.0),
  roughness=0.62,
  specular=0.16,
)


@dataclass(frozen=True)
class MaterialBank:
  """One baked material bank plus how a scene randomizes over it.

  ``pattern`` is ``None`` for a single fixed material, which is what makes an
  OOD evaluation table fixed rather than randomized: with no pattern there is
  nothing for the per-reset event to sample from. When it is set, ``slot``
  decides what the pattern selects -- materials for ``matid``, textures for
  ``texid``.
  """

  kind: BankKind
  prefix: str
  pattern: str | None = None
  slot: RandomizedSlot = "matid"
  tint: bool = False
  shared_random: bool = False
  # Objects opt into a bank by declaring it in a ``<text name="appearances">``
  # element, so one scene can dress the cube and leave the T untouched.
  appearance_tag: str | None = None
  image: TexturePreset | None = None
  seed: int | None = None
  checker_repeat: float = 6.0
  # Physics proxy colour for the table box hidden behind a textured mesh.
  # AmbientCG uses a transparent proxy to avoid z-fighting with its coplanar
  # visual mesh; the others keep the historical opaque grey.
  proxy_rgba: tuple[float, float, float, float] = (0.45, 0.45, 0.45, 1.0)

  @property
  def material_name(self) -> str:
    """The single material a ``texid`` bank spends on its whole catalog."""
    return f"{self.prefix}material"

  @property
  def material_selector(self) -> str:
    """The regex selecting whichever materials this bank owns."""
    if self.slot == "texid":
      return self.material_name
    assert self.pattern is not None
    return self.pattern


@dataclass(frozen=True)
class ScenePreset:
  """One public scene selector, expressed entirely as data."""

  name: str
  table: MaterialBank | None = None
  obj: MaterialBank | None = None
  lights: LightSet = "training"
  wide_lighting: bool = False
  # Whether the flat table and object colours are resampled every reset. False
  # keeps the colours their MJCF declares, leaving lighting and camera pose as
  # the only visual randomization.
  colour_dr: bool = True
  ood: bool = False


def _procedural_pattern(prefix: str) -> str:
  return rf"{prefix}_(solid|gradient|checker)_[0-9]{{2}}"


def procedural_material_names(prefix: str) -> tuple[str, ...]:
  return tuple(
    f"{prefix}_{kind}_{index:02d}"
    for kind in PROCEDURAL_KINDS
    for index in range(PROCEDURAL_VARIANTS_PER_KIND)
  )


AMBIENTCG_PREFIX = "ambientcg_table_"
# The photographic bank spends one material -- ``MaterialBank.material_name``
# names it -- and keeps its catalog in textures that ``dr.mat_texid`` swaps
# into that material's RGB role slot.
AMBIENTCG_TEXTURE_PATTERN = rf"{AMBIENTCG_PREFIX}texture_[0-9]{{4}}"

_PROCEDURAL_TABLE = MaterialBank(
  kind="procedural",
  prefix="proc_table",
  pattern=_procedural_pattern("proc_table"),
  tint=True,
  seed=PROCEDURAL_TABLE_SEED,
  checker_repeat=6.0,
)
_PROCEDURAL_OBJECT = MaterialBank(
  kind="procedural",
  prefix="proc_object",
  pattern=_procedural_pattern("proc_object"),
  tint=True,
  shared_random=True,
  appearance_tag="procedural",
  seed=PROCEDURAL_OBJECT_SEED,
  checker_repeat=2.0,
)
_AMBIENTCG_TABLE = MaterialBank(
  kind="ambientcg",
  prefix=AMBIENTCG_PREFIX,
  pattern=AMBIENTCG_TEXTURE_PATTERN,
  slot="texid",
  proxy_rgba=(0.0, 0.0, 0.0, 0.0),
)
_RED_PLASTIC_OBJECT = MaterialBank(
  kind="image",
  prefix="ood_red_plastic_object",
  appearance_tag="red_plastic",
  image=RED_PLASTIC_OBJECT,
)


def _ood_table(preset: TexturePreset) -> MaterialBank:
  return MaterialBank(kind="image", prefix=f"ood_{preset.name}_table", image=preset)


_PRESETS: dict[str, ScenePreset] = {
  # Fixed grey tabletop and the object's own colour, with the same lighting and
  # camera randomization every training scene gets. It is the baseline the
  # textured scenes are compared against: identical apart from appearance.
  "default": ScenePreset("default", colour_dr=False, wide_lighting=True),
  # Baked two-colour procedural banks on both table and object.
  "procedural": ScenePreset(
    "procedural",
    table=_PROCEDURAL_TABLE,
    obj=_PROCEDURAL_OBJECT,
    wide_lighting=True,
  ),
  # Photographic AmbientCG bank on the table; the object keeps colour DR.
  "real_texture": ScenePreset(
    "real_texture",
    table=_AMBIENTCG_TABLE,
    wide_lighting=True,
  ),
  # Same photographic table, but the object keeps the red its MJCF declares.
  # Object colour DR samples uniformly over the whole RGB cube independently of
  # the table, which leaves a quarter of resets within a 1.2:1 luminance ratio of
  # their tabletop -- a silhouette no encoder can read an orientation off. The
  # table bank is applied whatever `colour_dr` says, so clearing the flag drops
  # only the object's flat-colour event and keeps all 1203 table textures.
  "real_texture_red": ScenePreset(
    "real_texture_red",
    table=_AMBIENTCG_TABLE,
    colour_dr=False,
    wide_lighting=True,
  ),
  # Fixed out-of-distribution evaluation textures under realistic lighting.
  "wood": ScenePreset(
    "wood", table=_ood_table(WOOD_TABLE), obj=_RED_PLASTIC_OBJECT,
    lights="realistic", ood=True,
  ),
  "plaster": ScenePreset(
    "plaster", table=_ood_table(PLASTER_TABLE), obj=_RED_PLASTIC_OBJECT,
    lights="realistic", ood=True,
  ),
  "peacock": ScenePreset(
    "peacock", table=_ood_table(PEACOCK_TABLE), obj=_RED_PLASTIC_OBJECT,
    lights="realistic", ood=True,
  ),
}


def list_scenes() -> tuple[str, ...]:
  return tuple(_PRESETS)


def ood_scenes() -> tuple[str, ...]:
  return tuple(name for name, preset in _PRESETS.items() if preset.ood)


def get_preset(
  name: str,
  *,
  eval_dr: EvaluationDr = "fixed",
  require_ood: bool = False,
) -> ScenePreset:
  """Resolve a scene selector, validating its ``eval_dr`` combination."""
  try:
    preset = _PRESETS[name]
  except KeyError as exc:
    choices = ", ".join(list_scenes())
    raise ValueError(f"Unknown scene {name!r}. Choose one of: {choices}.") from exc
  if eval_dr not in {"fixed", "matched"}:
    raise ValueError("eval_dr must be 'fixed' or 'matched'.")
  if eval_dr == "matched" and not preset.ood:
    supported = ", ".join(ood_scenes())
    raise ValueError(
      f"eval_dr={eval_dr!r} is only meaningful for {supported} scenes."
    )
  if require_ood and not preset.ood:
    raise ValueError(
      f"Evaluation scene {name!r} is not an OOD texture replacement."
    )
  return preset


@lru_cache(maxsize=1)
def ambientcg_texture_paths() -> tuple[Path, ...]:
  """Return every prepared AmbientCG base-color asset."""
  if not AMBIENTCG_DIR.is_dir():
    raise FileNotFoundError(
      f"AmbientCG directory does not exist: {AMBIENTCG_DIR}. "
      "Prepare the texture assets before launching real-texture training."
    )
  paths = tuple(
    sorted(
      path
      for path in AMBIENTCG_DIR.iterdir()
      if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
  )
  if not paths:
    raise FileNotFoundError(f"No RGB texture files found in {AMBIENTCG_DIR}.")
  return paths


@lru_cache(maxsize=1)
def ambientcg_pool_paths() -> tuple[Path, ...]:
  """Return the deterministic sub-sample baked into each MuJoCo model.

  ``dr.mat_texid`` costs only one material, but MuJoCo's classic renderer --
  which ``--video`` builds through ``MjrContext`` -- rejects a model with more
  than 1,000 *textures*, and it aborts the run rather than degrading. The whole
  catalog is 1,203 files, so a fixed seeded sample is baked instead. Keep this
  cap: raising it past the limit breaks video recording, not just rendering
  quality.
  """
  paths = ambientcg_texture_paths()
  if len(paths) <= AMBIENTCG_POOL_SIZE:
    return paths
  indices = sorted(
    random.Random(AMBIENTCG_POOL_SEED).sample(range(len(paths)), AMBIENTCG_POOL_SIZE)
  )
  return tuple(paths[index] for index in indices)


@lru_cache(maxsize=1)
def ambientcg_texture_names() -> tuple[str, ...]:
  """Name every pooled entry, in the order ``mat_texid`` samples it."""
  return tuple(
    f"{AMBIENTCG_PREFIX}texture_{index:04d}"
    for index, _ in enumerate(ambientcg_pool_paths())
  )


__all__ = [
  "AMBIENTCG_PREFIX",
  "AMBIENTCG_TEXTURE_PATTERN",
  "CAMERA_POSITION_DR_RANGE_M",
  "CAMERA_ROTATION_DR_RANGE_RAD",
  "FILL_LIGHT_NAME",
  "PEACOCK_TABLE",
  "PLASTER_TABLE",
  "RED_PLASTIC_OBJECT",
  "TABLE_CENTER",
  "TABLE_GEOM_NAME",
  "TABLE_HALF_EXTENTS",
  "TABLE_LIGHT_NAME",
  "TABLE_VISUAL_GEOM_NAME",
  "TEXTURES_DIR",
  "UNIT_BOX_UV_MESH",
  "WOOD_TABLE",
  "EvaluationDr",
  "MaterialBank",
  "RandomizedSlot",
  "ScenePreset",
  "TexturePreset",
  "ambientcg_pool_paths",
  "ambientcg_texture_names",
  "ambientcg_texture_paths",
  "get_preset",
  "list_scenes",
  "ood_scenes",
  "procedural_material_names",
]
