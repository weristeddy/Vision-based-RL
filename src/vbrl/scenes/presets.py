from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from vbrl.asset_zoo.textures import (
  AMBIENTCG_DIR,
  TEXTURES_DIR,
  UNIT_BOX_UV_MESH,
)

TABLE_GEOM_NAME = "table_top"
TABLE_VISUAL_GEOM_NAME = "table_top_visual"
TABLE_LIGHT_NAME = "sun"
FILL_LIGHT_NAME = "sim2sim_fill"
TABLE_VISUAL_MESH_NAME = "table_top_visual_mesh"

# Measured off the real tabletop: 1.015 m along the robot's reach by 0.975 m across.
TABLE_HALF_EXTENTS = (0.5075, 0.4875, 0.02)
TABLE_CENTER = (0.4125, 0.0, -0.02)
CAMERA_POSITION_DR_RANGE_M = 0.025
CAMERA_ROTATION_DR_RANGE_RAD = 0.03

PROCEDURAL_VARIANTS_PER_KIND = 32
PROCEDURAL_KINDS = ("solid", "gradient", "checker")
PROCEDURAL_TABLE_SEED = 170306907
PROCEDURAL_OBJECT_SEED = 170306908
# MuJoCo's classic renderer aborts past 1,000 textures and `--video` builds one,
# so the 1,203-file catalog is sampled down. Seed reproduces the retained bank.
AMBIENTCG_POOL_SIZE = 768
AMBIENTCG_POOL_SEED = 170306909
MUJOCO_MAX_TEXTURES = 1000

BankKind = Literal["procedural", "ambientcg", "image"]
LightSet = Literal["training", "realistic"]
EvaluationDr = Literal["fixed", "matched"]
RandomizedSlot = Literal["matid", "texid"]


@dataclass(frozen=True)
class TexturePreset:
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
# The rig's own tabletop, pale unfinished spruce. Rotated 90 degrees from the
# supplied square file because texture u maps to the table's x -- verified with
# a striped test pattern, not inferred -- while the real planks run along y.
# Then centre-cropped 1305x1206 -> 1255x1206 for the table's 1.015 x 0.975
# aspect, so texrepeat (1, 1) lays it down once, centred, no tiling, no resample.
# rgba carries two measurements. The ratios 0.80 : 0.898 : 1.0 map the source's
# chromaticity (0.378/0.331/0.290) onto the real table's (0.340/0.335/0.325).
# The overall 0.55 lands the rendered median near 138 against the rig camera's
# 116-161; at 1.0 the pale texture blew out to 239-255.
REAL_TABLE = TexturePreset(
  name="real_table",
  image=TEXTURES_DIR / "real_table" / "real_table_color.png",
  texrepeat=(1.0, 1.0),
  rgba=(0.593, 0.488, 0.363, 1.0),
  roughness=0.93,
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
  kind: BankKind
  prefix: str
  pattern: str | None = None
  slot: RandomizedSlot = "matid"
  tint: bool = False
  shared_random: bool = False
  appearance_tag: str | None = None
  image: TexturePreset | None = None
  seed: int | None = None
  checker_repeat: float = 6.0
  proxy_rgba: tuple[float, float, float, float] = (0.45, 0.45, 0.45, 1.0)

  @property
  def material_name(self) -> str:
    return f"{self.prefix}material"

  @property
  def material_selector(self) -> str:
    if self.slot == "texid":
      return self.material_name
    assert self.pattern is not None
    return self.pattern


@dataclass(frozen=True)
class ScenePreset:
  name: str
  table: MaterialBank | None = None
  obj: MaterialBank | None = None
  lights: LightSet = "training"
  wide_lighting: bool = False
  colour_dr: bool = True
  # Per-channel (low, high) for the object's flat colour. The default spans the
  # whole RGB cube; the sim2real preset narrows it to the rig's measured
  # bordeaux (#68392c) so the sampled object stays a plausible real object.
  object_colour_range: tuple = ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))
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


_REAL_TABLE_BANK = MaterialBank(kind="image", prefix="real_table", image=REAL_TABLE)
# Measured off the rig: the object's bordeaux (#68392c) and the printed marker's
# green (#046147). Both keep colour DR, narrowed to what the camera really sees.
# Albedo, not observed colour. Tuned until the rendered object matched the rig
# at 0.426/0.192/0.090; the previous range rendered 0.452/0.247/0.192, with
# blue 2.1x too high.
REAL_TABLE_OBJECT_COLOUR_RANGE = ((0.22, 0.44), (0.055, 0.215), (0.0, 0.115))


def _ood_table(preset: TexturePreset) -> MaterialBank:
  return MaterialBank(kind="image", prefix=f"ood_{preset.name}_table", image=preset)


_PRESETS: dict[str, ScenePreset] = {
  "default": ScenePreset("default", colour_dr=False, wide_lighting=True),
  "procedural": ScenePreset(
    "procedural",
    table=_PROCEDURAL_TABLE,
    obj=_PROCEDURAL_OBJECT,
    wide_lighting=True,
  ),
  "real_texture": ScenePreset(
    "real_texture",
    table=_AMBIENTCG_TABLE,
    wide_lighting=True,
  ),
  "real_texture_red": ScenePreset(
    "real_texture_red",
    table=_AMBIENTCG_TABLE,
    colour_dr=False,
    wide_lighting=True,
  ),
  # The object stays a real-looking bordeaux instead of any colour in the cube,
  # measured off the rig at #68392c under matched exposure.
  "real_texture_bordeaux": ScenePreset(
    "real_texture_bordeaux",
    table=_AMBIENTCG_TABLE,
    wide_lighting=True,
    object_colour_range=((0.30, 0.52), (0.14, 0.30), (0.10, 0.26)),
  ),
  # One fixed table texture and no texture DR at all -- the rig's tabletop, laid
  # down once. Object and goal keep colour DR inside their measured ranges, and
  # the lighting stays randomized.
  "real_table": ScenePreset(
    "real_table",
    table=_REAL_TABLE_BANK,
    wide_lighting=True,
    object_colour_range=REAL_TABLE_OBJECT_COLOUR_RANGE,
  ),
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
  paths = ambientcg_texture_paths()
  if len(paths) <= AMBIENTCG_POOL_SIZE:
    return paths
  indices = sorted(
    random.Random(AMBIENTCG_POOL_SEED).sample(range(len(paths)), AMBIENTCG_POOL_SIZE)
  )
  return tuple(paths[index] for index in indices)


@lru_cache(maxsize=1)
def ambientcg_texture_names() -> tuple[str, ...]:
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
