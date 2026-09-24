from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .materials import add_bank
from .presets import (
  CAMERA_POSITION_DR_RANGE_M,
  CAMERA_ROTATION_DR_RANGE_RAD,
  FILL_LIGHT_NAME,
  TABLE_CENTER,
  TABLE_GEOM_NAME,
  TABLE_HALF_EXTENTS,
  TABLE_LIGHT_NAME,
  TABLE_VISUAL_GEOM_NAME,
  TABLE_VISUAL_MESH_NAME,
  UNIT_BOX_UV_MESH,
  EvaluationDr,
  MaterialBank,
  ScenePreset,
  get_preset,
)

if TYPE_CHECKING:
  from vbrl.asset_zoo.robots.definition import CameraView, RobotDefinition


SpecSource = Callable[[], mujoco.MjSpec]

# Reapplying a scene clears all of these, so a replacement cannot inherit a stale term
# from the preset it replaces.
LIGHT_COLOUR_EVENTS = ("light_intensity",)

SCENE_EVENTS = (
  "table_color",
  "object_color",
  "table_material",
  "object_material",
  "table_material_tint",
  "object_material_tint",
  "light_position",
  "light_direction",
  "light_intensity",
  "fill_light_direction",
  "background",
  "camera_position",
  "camera_orientation",
)

_LIGHT_RANGES = {
  False: {  # standard
    "position": {0: (-0.4, 0.4), 1: (-0.4, 0.4), 2: (1.0, 2.0)},
    "direction": {0: (-0.4, 0.4), 1: (-0.4, 0.4), 2: (-1.0, -0.4)},
  },
  True: {  # wide, retained from the visual-training scenes
    "position": {0: (-0.75, 0.75), 1: (-0.75, 0.75), 2: (0.75, 2.35)},
    "direction": {0: (-0.45, 0.45), 1: (-0.45, 0.45), 2: (-1.0, -0.55)},
  },
}
_MATCHED_RANGES = {
  "position": {0: (-0.18, 0.18), 1: (-0.18, 0.18), 2: (-0.25, 0.25)},
  "direction": {0: (-0.12, 0.12), 1: (-0.12, 0.12), 2: (-0.08, 0.08)},
  "fill": {0: (-0.10, 0.10), 1: (-0.10, 0.10), 2: (-0.06, 0.06)},
}
_LIGHT_INTENSITY_RANGES = {
  "diffuse": (0.34, 1.30),
  "specular": (0.02, 0.18),
  "ambient": (0.04, 0.16),
  "warmth": (-0.25, 0.25),
  "shadow_probability": 0.5,
}


BACKDROP_COUNT = 32
BACKDROP_PREFIX = "backdrop_"
_BACKDROP_FACE_WIDTH = 64
_BACKDROP_LUMINANCE = (0.004, 0.35)
_BACKDROP_WARMTH = (-0.12, 0.30)


def _backdrop_colour(rng) -> tuple[float, float, float]:
  low, high = _BACKDROP_LUMINANCE
  level = low + (high - low) * rng.random() ** 2
  tilt = rng.uniform(*_BACKDROP_WARMTH)
  return tuple(
    min(1.0, max(0.0, level * scale)) for scale in (1.0 + tilt, 1.0, 1.0 - tilt)
  )


def _add_backdrops(spec: mujoco.MjSpec) -> None:
  import random

  rng = random.Random(0)
  for index in range(BACKDROP_COUNT):
    spec.add_texture(
      name=f"{BACKDROP_PREFIX}{index}",
      type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
      builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
      width=_BACKDROP_FACE_WIDTH,
      height=_BACKDROP_FACE_WIDTH * 6,
      rgb1=_backdrop_colour(rng),
      rgb2=_backdrop_colour(rng),
    )


def _widen_skybox_to_one_per_world() -> None:
  """Give every world its own skybox slot before the render graph is captured.

  MuJoCo Warp indexes the skybox per world already, but builds a length-1 array,
  and mjlab renders inside a captured CUDA graph -- so the array has to be the
  right length before capture and can only be written in place afterwards.
  """
  import warp as wp
  from mjlab.sensor import sensor_context

  create = sensor_context.mjwarp.create_render_context
  if getattr(create, "_vbrl_per_world_skybox", False):
    return

  def create_render_context(*args, **kwargs):
    context = create(*args, **kwargs)
    worlds = int(kwargs.get("nworld", 1))
    for field in ("skybox_tex_id", "skybox_face_width"):
      values = getattr(context, field).numpy()
      if worlds > 1 and values.shape[0] == 1:
        setattr(context, field, wp.array(values.repeat(worlds), dtype=int))
    return context

  create_render_context._vbrl_per_world_skybox = True
  sensor_context.mjwarp.create_render_context = create_render_context


def randomize_background(env, env_ids, count: int) -> None:
  import torch
  import warp as wp

  context = env.scene.sensor_context
  if context is None:
    return
  slots = wp.to_torch(context.render_context.skybox_tex_id)
  if slots.shape[0] != env.num_envs:
    return
  choices = getattr(context, "_vbrl_backdrop_ids", None)
  if choices is None:
    model = env.sim.mj_model
    skybox = int(mujoco.mjtTexture.mjTEXTURE_SKYBOX)
    choices = torch.tensor(
      [i for i in range(model.ntex) if model.tex_type[i] == skybox],
      device=slots.device,
      dtype=slots.dtype,
    )
    context._vbrl_backdrop_ids = choices
  if choices.numel() == 0:
    return
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=slots.device)
  env_ids = env_ids.to(slots.device).long()
  picks = torch.randint(min(count, choices.numel()), env_ids.shape, device=slots.device)
  slots[env_ids] = choices[picks]


def _load_mjcf(path: str) -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(path)


def _add_lights(spec: mujoco.MjSpec, preset: ScenePreset) -> None:
  if preset.lights == "training":
    spec.worldbody.add_light(
      name=TABLE_LIGHT_NAME,
      pos=(0.12, -0.35, 1.35),
      dir=(0.18, 0.28, -1.0),
      type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
      diffuse=(0.88, 0.84, 0.78),
      ambient=(0.10, 0.10, 0.10),
      specular=(0.08, 0.08, 0.08),
      intensity=0.65,
      castshadow=True,
    )
    return
  spec.worldbody.add_light(
    name=TABLE_LIGHT_NAME,
    pos=(0.12, -0.35, 1.25),
    dir=(0.15, 0.30, -1.0),
    type=mujoco.mjtLightType.mjLIGHT_SPOT,
    diffuse=(0.78, 0.74, 0.68),
    ambient=(0.04, 0.04, 0.04),
    specular=(0.08, 0.08, 0.08),
    intensity=0.34,
    cutoff=70.0,
    exponent=8.0,
    castshadow=True,
  )
  spec.worldbody.add_light(
    name=FILL_LIGHT_NAME,
    pos=(-0.45, 0.35, 0.95),
    dir=(0.35, -0.25, -1.0),
    type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
    diffuse=(0.12, 0.13, 0.15),
    ambient=(0.01, 0.01, 0.012),
    specular=(0.02, 0.02, 0.02),
    intensity=0.04,
    castshadow=False,
  )


def table_spec(preset: ScenePreset) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  _add_lights(spec, preset)
  _add_backdrops(spec)
  bank = preset.table
  if bank is not None:
    names = add_bank(spec, bank)
    spec.add_mesh(
      name=TABLE_VISUAL_MESH_NAME,
      file=str(UNIT_BOX_UV_MESH),
      scale=TABLE_HALF_EXTENTS,
    )
    body_kwargs = {"group": 5, "rgba": bank.proxy_rgba}
  else:
    body_kwargs = {"rgba": (0.45, 0.45, 0.45, 1.0)}

  body = spec.worldbody.add_body(name="table")
  body.add_geom(
    name=TABLE_GEOM_NAME,
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=TABLE_HALF_EXTENTS,
    pos=TABLE_CENTER,
    # MuJoCo mixes solref/solimp across a contact pair, so stiffening only the object
    # does almost nothing -- the T still sank 13.3 mm of its 24 mm.
    solref=(0.01, 1.0),
    solimp=(0.95, 0.99, 0.0002, 0.5, 2.0),
    **body_kwargs,
  )
  if bank is not None:
    body.add_geom(
      name=TABLE_VISUAL_GEOM_NAME,
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname=TABLE_VISUAL_MESH_NAME,
      pos=TABLE_CENTER,
      material=names[0],
      contype=0,
      conaffinity=0,
      mass=0.0,
    )
  return spec


def _declares(spec: mujoco.MjSpec, tag: str) -> bool:
  return tag in next(
    (text.data.split() for text in spec.texts if text.name == "appearances"),
    (),
  )


def object_spec(preset: ScenePreset, source: SpecSource) -> mujoco.MjSpec:
  spec = source()
  bank = preset.obj
  if bank is None or not _declares(spec, bank.appearance_tag or ""):
    return spec

  material = add_bank(spec, bank)[0]
  for geom in spec.geoms:
    if geom.contype or geom.conaffinity:
      geom.group = 5
    else:
      geom.group = 0
      geom.material = material
  return spec


def _colour_event(entity: str, ranges, *, materials=(), shared_random=False):
  if materials:
    asset_cfg = SceneEntityCfg(entity, material_names=tuple(materials))
    func = dr.mat_rgba
  else:
    asset_cfg = SceneEntityCfg(entity)
    func = dr.geom_rgba
  params: dict[str, Any] = {
    "asset_cfg": asset_cfg,
    "operation": "abs",
    "distribution": "uniform",
    "axes": [0, 1, 2],
    "ranges": dict(enumerate(ranges)),
  }
  if shared_random:
    params["shared_random"] = True
  return EventTermCfg(func=func, mode="reset", params=params)


def _appearance_event(entity: str, bank: MaterialBank, geom_names=None):
  if bank.slot == "texid":
    return EventTermCfg(
      func=dr.mat_texid,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg(
          entity,
          material_names=(bank.material_name,),
          texture_names=(bank.pattern,),
        ),
        "shared_random": bank.shared_random,
      },
    )
  asset_cfg = SceneEntityCfg(entity, material_names=(bank.pattern,))
  if geom_names is not None:
    asset_cfg.geom_names = geom_names
  return EventTermCfg(
    func=dr.geom_matid,
    mode="reset",
    params={"asset_cfg": asset_cfg, "shared_random": bank.shared_random},
  )


def _tint_event(entity: str, bank: MaterialBank):
  return EventTermCfg(
    func=dr.mat_rgba,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg(
        entity, material_names=(bank.material_selector,)
      ),
      "operation": "scale",
      "distribution": "uniform",
      "axes": [0, 1, 2],
      "ranges": (0.65, 1.20),
    },
  )


def _light_event(func, light: str, ranges, *, operation: str):
  return EventTermCfg(
    func=func,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("table", light_names=(light,)),
      "operation": operation,
      "distribution": "uniform",
      "ranges": ranges,
    },
  )


@requires_model_fields(
  "light_diffuse", "light_specular", "light_ambient", "light_castshadow"
)
def randomize_light_intensity(
  env,
  env_ids,
  asset_cfg: SceneEntityCfg,
  diffuse: tuple[float, float],
  specular: tuple[float, float],
  ambient: tuple[float, float],
  warmth: tuple[float, float],
  shadow_probability: float,
) -> None:
  import torch

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)
  light = env.scene[asset_cfg.name].indexing.light_ids[asset_cfg.light_ids]
  grid = torch.meshgrid(env_ids, light, indexing="ij")

  def scalar(lo: float, hi: float):
    return torch.rand(env_ids.numel(), 1, 1, device=env.device) * (hi - lo) + lo

  tilt = scalar(*warmth)
  ones = torch.ones_like(tilt)
  tint = torch.cat([ones + tilt, ones, ones - tilt], dim=2)

  env.sim.model.light_diffuse[grid] = scalar(*diffuse) * tint
  env.sim.model.light_specular[grid] = scalar(*specular).expand_as(tint)
  env.sim.model.light_ambient[grid] = scalar(*ambient).expand_as(tint)
  env.sim.model.light_castshadow[grid] = (
    torch.rand(env_ids.numel(), 1, device=env.device) < shadow_probability
  )


def _camera_events(camera_model: str) -> dict[str, EventTermCfg]:
  asset_cfg = SceneEntityCfg("robot", camera_names=(camera_model,))
  return {
    "camera_position": EventTermCfg(
      func=dr.cam_pos,
      mode="startup",
      params={
        "asset_cfg": asset_cfg,
        "operation": "add",
        "distribution": "uniform",
        "ranges": {
          axis: (-CAMERA_POSITION_DR_RANGE_M, CAMERA_POSITION_DR_RANGE_M)
          for axis in range(3)
        },
      },
    ),
    "camera_orientation": EventTermCfg(
      func=dr.cam_quat,
      mode="startup",
      params={
        "asset_cfg": asset_cfg,
        "roll_range": (-CAMERA_ROTATION_DR_RANGE_RAD, CAMERA_ROTATION_DR_RANGE_RAD),
        "pitch_range": (-CAMERA_ROTATION_DR_RANGE_RAD, CAMERA_ROTATION_DR_RANGE_RAD),
        "yaw_range": (-CAMERA_ROTATION_DR_RANGE_RAD, CAMERA_ROTATION_DR_RANGE_RAD),
        "distribution": "uniform",
      },
    ),
  }


def _bank_events(
  key: str, entity: str, bank: MaterialBank, geoms
) -> dict[str, EventTermCfg]:
  events: dict[str, EventTermCfg] = {}
  if bank.pattern is not None:
    events[f"{key}_material"] = _appearance_event(entity, bank, geoms)
  if bank.tint:
    events[f"{key}_material_tint"] = _tint_event(entity, bank)
  return events


def _events(
  preset: ScenePreset,
  *,
  object_name: str,
  object_materials: tuple[str, ...],
  object_dressed: bool,
  camera_model: str | None,
  eval_dr: EvaluationDr,
) -> dict[str, EventTermCfg]:
  events: dict[str, EventTermCfg] = {}
  if camera_model is None:
    return events
  if preset.ood and eval_dr == "fixed":
    return events

  matched = preset.ood and eval_dr == "matched"
  if not matched:
    if preset.colour_dr and preset.table is None:
      events["table_color"] = _colour_event("table", ((0.15, 0.85),) * 3)
    if preset.table is not None:
      events.update(
        _bank_events("table", "table", preset.table, (TABLE_VISUAL_GEOM_NAME,))
      )
    if preset.colour_dr and not object_dressed:
      events["object_color"] = _colour_event(
        object_name,
        preset.object_colour_range,
        materials=object_materials,
        shared_random=True,
      )
    if object_dressed:
      assert preset.obj is not None
      events.update(_bank_events("object", object_name, preset.obj, None))

  ranges = _MATCHED_RANGES if matched else _LIGHT_RANGES[preset.wide_lighting]
  operation = "add" if matched else "abs"
  events["light_position"] = _light_event(
    dr.light_pos, TABLE_LIGHT_NAME, ranges["position"], operation=operation
  )
  events["light_direction"] = _light_event(
    dr.light_dir, TABLE_LIGHT_NAME, ranges["direction"], operation=operation
  )
  if matched:
    events["fill_light_direction"] = _light_event(
      dr.light_dir, FILL_LIGHT_NAME, ranges["fill"], operation="add"
    )
  else:
    # Held out of the matched branch so a sim2sim evaluation measures exactly
    # the lighting it was calibrated against.
    events["light_intensity"] = EventTermCfg(
      func=randomize_light_intensity,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("table", light_names=(TABLE_LIGHT_NAME,)),
        **_LIGHT_INTENSITY_RANGES,
      },
    )
  if camera_model is not None:
    events.update(_camera_events(camera_model))
    _widen_skybox_to_one_per_world()
    events["background"] = EventTermCfg(
      func=randomize_background, mode="reset", params={"count": BACKDROP_COUNT}
    )
  return events


def _apply(
  cfg: ManagerBasedRlEnvCfg,
  preset: ScenePreset,
  *,
  object_name: str,
  object_source: SpecSource,
  camera_model: str | None,
  eval_dr: EvaluationDr,
) -> ManagerBasedRlEnvCfg:
  probe = object_source()
  object_materials = tuple(material.name for material in probe.materials)
  object_dressed = preset.obj is not None and _declares(
    probe, preset.obj.appearance_tag or ""
  )

  cfg.scene.entities["table"] = EntityCfg(spec_fn=partial(table_spec, preset))
  cfg.scene.entities[object_name] = EntityCfg(
    spec_fn=partial(object_spec, preset, object_source)
  )
  for name in SCENE_EVENTS:
    cfg.events.pop(name, None)
  cfg.events.update(
    _events(
      preset,
      object_name=object_name,
      object_materials=object_materials,
      object_dressed=object_dressed,
      camera_model=camera_model,
      eval_dr=eval_dr,
    )
  )
  return cfg


def hold_lighting_colour_fixed(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  for name in LIGHT_COLOUR_EVENTS:
    cfg.events.pop(name, None)
  return cfg


def apply_scene(
  cfg: ManagerBasedRlEnvCfg,
  *,
  scene: str,
  robot: RobotDefinition,
  camera_view: CameraView | None = None,
  object_xml: Path,
  object_name: str,
  eval_dr: EvaluationDr = "fixed",
) -> ManagerBasedRlEnvCfg:
  preset = get_preset(scene, eval_dr=eval_dr)
  # The terrain the tabletop base installs is the env-origin grid, not scenery,
  # so a scene must not clear it: see tasks.utils.lay_out_envs_on_a_grid.
  return _apply(
    cfg,
    preset,
    object_name=object_name,
    object_source=partial(_load_mjcf, str(object_xml)),
    camera_model=(
      None if camera_view is None else robot.resolve_camera(camera_view).model_name
    ),
    eval_dr=eval_dr,
  )


_NON_OBJECT_ENTITIES = frozenset({"robot", "table", "goal_marker"})


def _task_object(entities) -> str:
  names = [
    name
    for name, entity in entities.items()
    if name not in _NON_OBJECT_ENTITIES and entity.spec_fn is not None
  ]
  if not names:
    raise ValueError("Scene replacement requires a task object.")
  return names[0]


def _camera_model_from(cfg: ManagerBasedRlEnvCfg) -> str | None:
  event = cfg.events.get("camera_position")
  if event is None:
    return None
  cameras = event.params["asset_cfg"].camera_names
  return cameras[0] if cameras else None


def replace_scene(
  cfg: ManagerBasedRlEnvCfg,
  *,
  scene: str,
  eval_dr: EvaluationDr = "fixed",
) -> ManagerBasedRlEnvCfg:
  preset = get_preset(scene, eval_dr=eval_dr, require_ood=True)
  object_name = _task_object(cfg.scene.entities)
  source = cfg.scene.entities[object_name].spec_fn
  assert source is not None
  return _apply(
    cfg,
    preset,
    object_name=object_name,
    object_source=source,
    camera_model=_camera_model_from(cfg),
    eval_dr=eval_dr,
  )


__all__ = [
  "LIGHT_COLOUR_EVENTS",
  "SCENE_EVENTS",
  "apply_scene",
  "hold_lighting_colour_fixed",
  "BACKDROP_COUNT",
  "object_spec",
  "randomize_background",
  "randomize_light_intensity",
  "replace_scene",
  "table_spec",
]
