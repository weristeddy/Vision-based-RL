from __future__ import annotations

import colorsys
import random
import re
from dataclasses import dataclass, replace
from functools import cache, lru_cache
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
# Per camera model: the external camera's calibration error against a wrist mount's
# tolerance. At 25 mm the wrist camera lands inside the D405 housing.
CAMERA_POSITION_DR_RANGE_M = {"external_cam": 0.025, "cam": 0.005}
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
AMBIENTCG_WOOD_FAMILIES = ("Wood", "WoodFloor", "Planks", "WoodSiding")

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
  roughness: float | None
  specular: float | None

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
# The fixed evaluation tables render like the training bank's AmbientCG material:
# laid once, MuJoCo's default roughness and specular. Wood049 is a training wood
# near the bank's median colour, contrast and rendered brightness (142 / 157 on the
# external / wrist camera against a training median of 131 / 138). Marble019 is an
# unseen family, scaled to 0.37 so it renders at that median (123 / 139) and a
# drop measures the material, not the exposure.
WOOD049_TABLE = TexturePreset(
  name="wood049",
  image=AMBIENTCG_DIR / "Wood049_Color.png",
  texrepeat=(1.0, 1.0),
  rgba=(1.0, 1.0, 1.0, 1.0),
  roughness=None,
  specular=None,
)
MARBLE019_TABLE = TexturePreset(
  name="marble019",
  image=AMBIENTCG_DIR / "Marble019_Color.png",
  texrepeat=(1.0, 1.0),
  rgba=(0.37, 0.37, 0.37, 1.0),
  roughness=None,
  specular=None,
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
  families: tuple[str, ...] = ()

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
  # Set on every evaluation scene so a scene swap never inherits another's goal;
  # None leaves the task's goal colour event alone.
  goal_colour_range: tuple | None = None
  # A fixed diffuse colour for the table light in place of the training draw.
  light_colour: tuple[float, float, float] | None = None
  # Evaluation only: every env gets its own colour of the sweep path (see
  # sweep_colour), set after the usual colour draw so the episodes stay paired.
  object_colour_sweep: bool = False
  goal_colour_sweep: bool = False
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
_AMBIENTCG_WOOD_TABLE = MaterialBank(
  kind="ambientcg",
  prefix=AMBIENTCG_PREFIX,
  pattern=AMBIENTCG_TEXTURE_PATTERN,
  slot="texid",
  proxy_rgba=(0.0, 0.0, 0.0, 0.0),
  families=AMBIENTCG_WOOD_FAMILIES,
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
# Albedo, not observed colour: the 5-95 % range that made the rendered T match
# the rig's T pixel for pixel across 120 renders on 2026-10-07 (median
# 0.20/0.068/0.011), with margin. The previous 0.22-0.44/0.055-0.215/0-0.115
# left 63 % of those matches outside it.
REAL_TABLE_OBJECT_COLOUR_RANGE = ((0.14, 0.36), (0.04, 0.11), (0.0, 0.03))
# The goal tape's measured green (see push_t.goal_marker), shared by every goal marker,
# and the one event every task colours its goal marker with.
GOAL_COLOUR_EVENT = "goal_color"
REAL_TABLE_GOAL_COLOUR_RANGE = ((0.085, 0.16), (0.115, 0.21), (0.035, 0.07))


def _fixed(rgb: tuple[float, float, float]) -> tuple:
  return tuple((value, value) for value in rgb)


# The swap gives each the other's training median.
SWAP_OBJECT = (0.12, 0.16, 0.05)
SWAP_GOAL = (0.25, 0.075, 0.015)
# The colour sweeps test a 16 x 16 colour chart: 15 hues of the rainbow, each in 16
# shades from dark to light (HSL, full saturation, lightness 0.1 to 0.9), plus 16
# greys from black to white. 256 colours, 2 of the 512 evaluation episodes each.
COLOUR_SWEEP_HUES = 15
COLOUR_SWEEP_SHADES = 16
COLOUR_SWEEP_COLOURS = (COLOUR_SWEEP_HUES + 1) * COLOUR_SWEEP_SHADES


def colour_sweep_rgb(index: int) -> tuple[float, float, float]:
  column, shade = divmod(index, COLOUR_SWEEP_SHADES)
  if column == COLOUR_SWEEP_HUES:
    grey = shade / (COLOUR_SWEEP_SHADES - 1)
    return (grey, grey, grey)
  lightness = 0.1 + 0.8 * shade / (COLOUR_SWEEP_SHADES - 1)
  return colorsys.hls_to_rgb(column / COLOUR_SWEEP_HUES, lightness, 1.0)


def sweep_colour(env_index: int, num_envs: int) -> int:
  return env_index * COLOUR_SWEEP_COLOURS // num_envs


# Colosseum's Light_Color draws each channel from [0, 0.5]; this one is a hue the
# training tint (warm to cold only) never produces.
OOD_LIGHT_COLOUR = (0.08, 0.40, 0.45)


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
  # One fixed table texture and no texture DR at all -- the rig's tabletop, laid
  # down once. Object and goal keep colour DR inside their measured ranges, and
  # the lighting stays randomized.
  "real_table": ScenePreset(
    "real_table",
    table=_REAL_TABLE_BANK,
    wide_lighting=True,
    object_colour_range=REAL_TABLE_OBJECT_COLOUR_RANGE,
    goal_colour_range=REAL_TABLE_GOAL_COLOUR_RANGE,
  ),
  # real_table with the tabletop drawn from AmbientCG's wood families instead of
  # the rig's one photo.
  "real_texture_wood": ScenePreset(
    "real_texture_wood",
    table=_AMBIENTCG_WOOD_TABLE,
    wide_lighting=True,
    object_colour_range=REAL_TABLE_OBJECT_COLOUR_RANGE,
  ),
  # Fixed tables for evaluation: every episode of a seed is the same in all three,
  # only the tabletop differs.
  "wood049": ScenePreset(
    "wood049",
    table=MaterialBank(kind="image", prefix="wood049_table", image=WOOD049_TABLE),
    wide_lighting=True,
    object_colour_range=REAL_TABLE_OBJECT_COLOUR_RANGE,
    goal_colour_range=REAL_TABLE_GOAL_COLOUR_RANGE,
  ),
  "marble019": ScenePreset(
    "marble019",
    table=MaterialBank(kind="image", prefix="marble019_table", image=MARBLE019_TABLE),
    wide_lighting=True,
    object_colour_range=REAL_TABLE_OBJECT_COLOUR_RANGE,
    goal_colour_range=REAL_TABLE_GOAL_COLOUR_RANGE,
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
# wood049 with exactly one factor fixed out of distribution; everything else draws
# as in training, so a seed gives the same episodes as wood049.
for _name, _change in {
  "ood_light": {"light_colour": OOD_LIGHT_COLOUR},
  "colour_swap": {
    "object_colour_range": _fixed(SWAP_OBJECT),
    "goal_colour_range": _fixed(SWAP_GOAL),
  },
  "object_color_sweep": {"object_colour_sweep": True},
  "goal_color_sweep": {"goal_colour_sweep": True},
}.items():
  _PRESETS[_name] = replace(_PRESETS["wood049"], name=_name, **_change)

# Every student trains on real_texture_wood. The tables: in distribution (a
# training wood), near OOD (the rig's own tabletop), far OOD (an unseen material);
# then wood049 with the light colour out of distribution, object and goal colours
# swapped, and the object or goal colour swept over a colour chart.
EVALUATION_SCENES = (
  "wood049",
  "real_table",
  "marble019",
  "ood_light",
  "colour_swap",
  "object_color_sweep",
  "goal_color_sweep",
)
GOAL_COLOUR_SCENES = ("goal_color_sweep",)
COLOUR_SWEEP_SCENES = ("object_color_sweep", "goal_color_sweep")


def evaluation_scenes(*, goal_visible: bool) -> tuple[str, ...]:
  if goal_visible:
    return EVALUATION_SCENES
  return tuple(scene for scene in EVALUATION_SCENES if scene not in GOAL_COLOUR_SCENES)


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


def ambientcg_family(path: Path) -> str:
  match = re.match(r"[A-Za-z]+", path.name)
  return match.group() if match else ""


@cache
def ambientcg_pool_paths(families: tuple[str, ...] = ()) -> tuple[Path, ...]:
  paths = ambientcg_texture_paths()
  if families:
    paths = tuple(path for path in paths if ambientcg_family(path) in families)
    if not paths:
      raise ValueError(f"No AmbientCG texture belongs to the families {families}.")
  if len(paths) <= AMBIENTCG_POOL_SIZE:
    return paths
  indices = sorted(
    random.Random(AMBIENTCG_POOL_SEED).sample(range(len(paths)), AMBIENTCG_POOL_SIZE)
  )
  return tuple(paths[index] for index in indices)


@cache
def ambientcg_texture_names(families: tuple[str, ...] = ()) -> tuple[str, ...]:
  return tuple(
    f"{AMBIENTCG_PREFIX}texture_{index:04d}"
    for index, _ in enumerate(ambientcg_pool_paths(families))
  )


__all__ = [
  "COLOUR_SWEEP_COLOURS",
  "COLOUR_SWEEP_HUES",
  "COLOUR_SWEEP_SHADES",
  "COLOUR_SWEEP_SCENES",
  "EVALUATION_SCENES",
  "GOAL_COLOUR_EVENT",
  "GOAL_COLOUR_SCENES",
  "OOD_LIGHT_COLOUR",
  "REAL_TABLE_GOAL_COLOUR_RANGE",
  "SWAP_GOAL",
  "SWAP_OBJECT",
  "MARBLE019_TABLE",
  "WOOD049_TABLE",
  "AMBIENTCG_PREFIX",
  "AMBIENTCG_TEXTURE_PATTERN",
  "AMBIENTCG_WOOD_FAMILIES",
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
  "ambientcg_family",
  "ambientcg_pool_paths",
  "ambientcg_texture_names",
  "ambientcg_texture_paths",
  "evaluation_scenes",
  "colour_sweep_rgb",
  "get_preset",
  "sweep_colour",
  "list_scenes",
  "ood_scenes",
  "procedural_material_names",
]
