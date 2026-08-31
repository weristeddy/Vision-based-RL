"""Compose one scene preset onto a task configuration.

:func:`apply_scene` (registration time) and :func:`replace_scene` (runtime OOD
swap) share :func:`_apply`, so a scene cannot behave differently depending on
when it was built.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
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

# Every event this module owns. Reapplying a scene clears all of them first so
# a replacement can never inherit a stale term from the preset it replaces.
# The three appearance-only light terms: pose jitter is deliberately not here.
LIGHT_COLOUR_EVENTS = ("light_diffuse", "light_specular", "light_ambient")

SCENE_EVENTS = (
  "table_color",
  "object_color",
  "table_material",
  "object_material",
  "table_material_tint",
  "object_material_tint",
  "light_position",
  "light_direction",
  "light_diffuse",
  "light_specular",
  "light_ambient",
  "fill_light_direction",
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
# Absolute colour ranges for the directional training sun, spanning the static
# values it is built with (diffuse 0.88/0.84/0.78, ambient 0.10, specular
# 0.08). Randomizing per channel varies colour temperature as well as
# brightness. MJLab 1.6 also ships dr.light_attenuation, dr.light_cutoff, and
# dr.light_exponent; MuJoCo ignores all three for directional lights, so they
# have nothing to act on here and are deliberately not wired up.
_LIGHT_COLOR_RANGES = {
  "diffuse": {axis: (0.70, 1.00) for axis in range(3)},
  "specular": {axis: (0.02, 0.18) for axis in range(3)},
  "ambient": {axis: (0.04, 0.16) for axis in range(3)},
}


def _load_mjcf(path: str) -> mujoco.MjSpec:
  """Load MJCF through a Python function that TorchrunX can serialize."""
  return mujoco.MjSpec.from_file(path)


# --- world construction -----------------------------------------------------


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
  """Build the support table, its lights, and its material bank."""
  spec = mujoco.MjSpec()
  _add_lights(spec, preset)
  bank = preset.table
  if bank is not None:
    names = add_bank(spec, bank)
    spec.add_mesh(
      name=TABLE_VISUAL_MESH_NAME,
      file=str(UNIT_BOX_UV_MESH),
      scale=TABLE_HALF_EXTENTS,
    )
    # The box stays as the physics proxy; the textured mesh is render-only.
    body_kwargs = {"group": 5, "rgba": bank.proxy_rgba}
  else:
    body_kwargs = {"rgba": (0.45, 0.45, 0.45, 1.0)}

  body = spec.worldbody.add_body(name="table")
  body.add_geom(
    name=TABLE_GEOM_NAME,
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=TABLE_HALF_EXTENTS,
    pos=TABLE_CENTER,
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
  """Whether an object MJCF opts into an appearance through ``<text>``."""
  return tag in next(
    (text.data.split() for text in spec.texts if text.name == "appearances"),
    (),
  )


def object_spec(preset: ScenePreset, source: SpecSource) -> mujoco.MjSpec:
  """Load the task object and dress it if it opts into this scene's bank."""
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


# --- events -----------------------------------------------------------------


def _colour_event(entity: str, ranges, *, materials=(), shared_random=False):
  """Randomize an entity's colour through whichever slot it actually has."""
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
    "ranges": ranges,
  }
  if shared_random:
    params["shared_random"] = True
  return EventTermCfg(func=func, mode="reset", params=params)


def _appearance_event(entity: str, bank: MaterialBank, geom_names=None):
  """Sample this bank's appearance through whichever slot it varies."""
  if bank.slot == "texid":
    # One material, many textures: repoint its RGB role slot. The geoms bound
    # to the material never change, so `geom_names` has nothing to scope.
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


def _events(
  preset: ScenePreset,
  *,
  object_name: str,
  object_materials: tuple[str, ...],
  object_dressed: bool,
  camera_model: str | None,
  eval_dr: EvaluationDr,
) -> dict[str, EventTermCfg]:
  """Derive the complete visual-randomization event set from one preset."""
  events: dict[str, EventTermCfg] = {}
  # Nothing renders the scene, so there is no appearance to randomize. This is
  # what keeps the state tasks free of dead lighting and camera terms.
  if camera_model is None:
    return events
  # A fixed OOD scene is the evaluation baseline: one appearance, no variation.
  if preset.ood and eval_dr == "fixed":
    return events

  matched = preset.ood and eval_dr == "matched"
  if not matched:
    # A randomized material bank replaces flat colour randomization.
    if preset.colour_dr and preset.table is None:
      events["table_color"] = _colour_event("table", (0.15, 0.85))
    if preset.colour_dr and not object_dressed:
      events["object_color"] = _colour_event(
        object_name, (0.0, 1.0), materials=object_materials, shared_random=True
      )
    for entity, bank, geoms in (
      ("table", preset.table, (TABLE_VISUAL_GEOM_NAME,)),
      (object_name, preset.obj if object_dressed else None, None),
    ):
      if bank is None:
        continue
      key = "table" if entity == "table" else "object"
      if bank.pattern is not None:
        events[f"{key}_material"] = _appearance_event(entity, bank, geoms)
      if bank.tint:
        events[f"{key}_material_tint"] = _tint_event(entity, bank)

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
    # Colour and intensity of the training sun. Held out of the matched branch
    # so a sim2sim evaluation keeps measuring exactly the lighting it was
    # calibrated against.
    for field, func in (
      ("diffuse", dr.light_diffuse),
      ("specular", dr.light_specular),
      ("ambient", dr.light_ambient),
    ):
      events[f"light_{field}"] = _light_event(
        func, TABLE_LIGHT_NAME, _LIGHT_COLOR_RANGES[field], operation="abs"
      )
  if camera_model is not None:
    events.update(_camera_events(camera_model))
  return events


# --- entry points -----------------------------------------------------------


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
  """Drop the sun's colour randomization, keeping its pose jitter.

  Each channel of `diffuse` is sampled independently over a range wide enough
  to tint the whole shot, so a recording's floor -- one plane shared by every
  env, lit by whichever env the recorder makes primary -- comes out a different
  colour every run. Clearing these three leaves the sun at the colour the scene
  declares, which is inside the range a policy trained under this DR has seen,
  so a recording still shows the policy on an input distribution it knows.
  """
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
  """Compose one scene around a freshly built task configuration."""
  preset = get_preset(scene, eval_dr=eval_dr)
  # The terrain the tabletop base installs is the env-origin grid, not scenery,
  # so a scene must not clear it: see tasks.utils.lay_out_envs_on_a_grid.
  return _apply(
    cfg,
    preset,
    object_name=object_name,
    object_source=partial(_load_mjcf, str(object_xml)),
    # None means the task has no camera, so it gets no visual randomization.
    camera_model=(
      None if camera_view is None else robot.resolve_camera(camera_view).model_name
    ),
    eval_dr=eval_dr,
  )


# Scenery and fixtures, none of which is the manipulated object. `goal_marker`
# is the drawn target the visual-goal variants install.
_NON_OBJECT_ENTITIES = frozenset({"robot", "table", "goal_marker"})


def _sole_object(entities) -> str:
  names = tuple(
    name
    for name, entity in entities.items()
    if name not in _NON_OBJECT_ENTITIES and entity.spec_fn is not None
  )
  if len(names) != 1:
    raise ValueError(
      f"Scene replacement requires exactly one task object; found {names}."
    )
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
  """Replace only the visual scene of an already-registered configuration."""
  preset = get_preset(scene, eval_dr=eval_dr, require_ood=True)
  object_name = _sole_object(cfg.scene.entities)
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
  "object_spec",
  "replace_scene",
  "table_spec",
]
