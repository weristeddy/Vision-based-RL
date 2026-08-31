"""Behavioural contracts for the scene preset table and its builder."""

from __future__ import annotations


import pytest


pytest.importorskip("mjlab")
mujoco = pytest.importorskip("mujoco")

from vbrl.scenes.presets import (  # noqa: E402
  TABLE_GEOM_NAME,
  TABLE_VISUAL_GEOM_NAME,
  get_preset,
  list_scenes,
  ood_scenes,
)


ALL_SCENES = (
  "default", "procedural", "real_texture", "real_texture_red",
  "wood", "plaster", "peacock",
)
# Every training preset randomizes the sun's colour on top of its pose. The
# matched-evaluation branch deliberately does not; see `_events` in the builder.
LIGHT_COLOR_EVENTS = frozenset({"light_diffuse", "light_specular", "light_ambient"})


def _object_xml(name: str):
  from vbrl.asset_zoo.objects import OBJECTS_DIR

  return OBJECTS_DIR / f"{name}.xml"


def _apply(scene: str, *, object_name="cube", eval_dr="fixed"):
  """Apply one scene to a bare cfg stand-in and return it."""
  from types import SimpleNamespace

  from vbrl.asset_zoo.robots import get_robot
  from vbrl.scenes.builder import apply_scene

  cfg = SimpleNamespace(
    scene=SimpleNamespace(terrain=object(), entities={"robot": object()}),
    events={},
  )
  return apply_scene(
    cfg,
    scene=scene,
    robot=get_robot("trossen"),
    camera_view="wrist",
    object_xml=_object_xml(object_name),
    object_name=object_name,
    eval_dr=eval_dr,
  )


def test_scene_table_lists_every_public_selector() -> None:
  assert list_scenes() == ALL_SCENES
  assert ood_scenes() == ("wood", "plaster", "peacock")


def test_unknown_scene_reports_the_valid_choices() -> None:
  with pytest.raises(ValueError, match="Unknown scene 'nope'"):
    get_preset("nope")


@pytest.mark.parametrize("scene", ("default", "procedural", "real_texture"))
def test_matched_dr_is_rejected_for_non_ood_scenes(scene: str) -> None:
  with pytest.raises(ValueError, match="only meaningful for"):
    get_preset(scene, eval_dr="matched")


def test_replace_scene_rejects_a_non_ood_target() -> None:
  from vbrl.scenes.builder import replace_scene

  cfg = _apply("real_texture")
  with pytest.raises(ValueError, match="not an OOD texture replacement"):
    replace_scene(cfg, scene="procedural")


@pytest.mark.parametrize(
  ("scene", "expected"),
  [
    (
      "default",
      frozenset(
        {
          "light_position",
          "light_direction",
          "camera_position",
          "camera_orientation",
        }
      ),
    ),
    (
      "procedural",
      frozenset(
        {
          "table_material",
          "table_material_tint",
          "object_material",
          "object_material_tint",
          "light_position",
          "light_direction",
          "camera_position",
          "camera_orientation",
        }
      ),
    ),
    (
      "real_texture",
      frozenset(
        {
          "table_material",
          "object_color",
          "light_position",
          "light_direction",
          "camera_position",
          "camera_orientation",
        }
      ),
    ),
    ("wood", frozenset()),
  ],
)
def test_each_preset_derives_its_own_event_set(scene: str, expected) -> None:
  """Colour DR exists only where a material bank does not replace it."""
  if expected:
    expected |= LIGHT_COLOR_EVENTS
  assert frozenset(_apply(scene).events) == expected


def test_a_task_without_a_camera_gets_no_visual_randomization() -> None:
  """State tasks render nothing, so lighting and camera terms would be dead."""
  from types import SimpleNamespace

  from vbrl.asset_zoo.robots import get_robot
  from vbrl.scenes.builder import apply_scene

  cfg = SimpleNamespace(
    scene=SimpleNamespace(terrain=object(), entities={"robot": object()}),
    events={},
  )
  apply_scene(
    cfg,
    scene="real_texture",
    robot=get_robot("trossen"),
    camera_view=None,
    object_xml=_object_xml("cube"),
    object_name="cube",
  )

  assert cfg.events == {}


def test_default_differs_from_real_texture_only_in_appearance() -> None:
  """The controlled pair: same lighting and camera DR, different tabletop."""
  default = frozenset(_apply("default").events)
  real_texture = frozenset(_apply("real_texture").events)

  assert real_texture - default == {"table_material", "object_color"}
  assert default - real_texture == frozenset()
  assert "table_color" not in default


def test_matched_evaluation_adds_the_fill_light_and_shifts_lighting() -> None:
  cfg = _apply("wood", eval_dr="matched")
  assert frozenset(cfg.events) == frozenset(
    {
      "light_position",
      "light_direction",
      "fill_light_direction",
      "camera_position",
      "camera_orientation",
    }
  )
  assert cfg.events["light_position"].params["operation"] == "add"


@pytest.mark.parametrize("scene", ALL_SCENES)
def test_table_geometry_matches_the_bank(scene: str) -> None:
  """A textured table keeps the box as a group-5 physics proxy."""
  from vbrl.scenes.builder import table_spec

  preset = get_preset(scene)
  model = table_spec(preset).compile()
  box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_GEOM_NAME)
  visual = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, TABLE_VISUAL_GEOM_NAME)
  assert box >= 0
  if preset.table is None:
    assert visual < 0
    assert model.geom_group[box] == 0
  else:
    assert visual >= 0
    assert model.geom_group[box] == 5
    assert model.geom_contype[visual] == 0
    assert model.geom_matid[visual] >= 0


def test_real_texture_stays_under_the_renderer_texture_limit() -> None:
  """MuJoCo's classic renderer aborts above 1,000 textures, and --video builds one.

  This is a hard failure at env construction, not degraded rendering, so the
  pool size is pinned here rather than discovered on a cluster node.
  """
  from vbrl.scenes.builder import table_spec
  from vbrl.scenes.presets import MUJOCO_MAX_TEXTURES, ambientcg_texture_paths

  model = table_spec(get_preset("real_texture")).compile()

  assert model.ntex < MUJOCO_MAX_TEXTURES
  # The cap must actually bind: the catalog on disk is larger than the limit.
  assert len(ambientcg_texture_paths()) > MUJOCO_MAX_TEXTURES


def test_real_texture_spends_one_material_on_its_texture_pool() -> None:
  """The photographic bank randomizes a texture slot, not a material id."""
  import re

  from mjlab.envs.mdp import dr

  from vbrl.scenes.builder import table_spec
  from vbrl.scenes.presets import ambientcg_texture_names

  preset = get_preset("real_texture")
  bank = preset.table
  assert bank is not None and bank.pattern is not None

  event = _apply("real_texture").events["table_material"]
  assert event.func is dr.mat_texid
  asset_cfg = event.params["asset_cfg"]
  assert asset_cfg.material_names == (bank.material_name,)
  assert asset_cfg.texture_names == (bank.pattern,)

  names = ambientcg_texture_names()
  assert names, "The AmbientCG catalog must not be empty."
  assert all(re.fullmatch(bank.pattern, name) for name in names)

  model = table_spec(preset).compile()
  assert model.nmat == 1
  assert model.ntex == len(names)


def test_realistic_scenes_add_a_fill_light() -> None:
  from vbrl.scenes.builder import table_spec
  from vbrl.scenes.presets import FILL_LIGHT_NAME

  for scene, has_fill in (("real_texture", False), ("wood", True)):
    model = table_spec(get_preset(scene)).compile()
    found = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, FILL_LIGHT_NAME)
    assert (found >= 0) is has_fill, scene


@pytest.mark.parametrize("scene", ood_scenes())
def test_apply_scene_and_replace_scene_agree(scene: str) -> None:
  """The registration-time and runtime paths must produce the same scene.

  ``replace_scene`` recovers the object source and camera from an already
  built configuration instead of receiving them directly. Everything after
  that is shared, and this pins it.
  """
  from vbrl.scenes.builder import replace_scene

  direct = _apply(scene)
  replaced = replace_scene(_apply("real_texture"), scene=scene)

  assert frozenset(direct.events) == frozenset(replaced.events)
  for name, term in direct.events.items():
    other = replaced.events[name]
    assert term.func is other.func, name
    assert term.mode == other.mode, name
    assert term.params == other.params, name
  assert frozenset(direct.scene.entities) == frozenset(replaced.scene.entities)
  for entity in ("table", "cube"):
    assert (
      direct.scene.entities[entity].spec_fn().to_xml()
      == replaced.scene.entities[entity].spec_fn().to_xml()
    ), entity


def test_a_recording_holds_the_sun_colour_but_keeps_its_pose_jitter() -> None:
  """One plane is shared by every env and lit by whichever is primary, so a
  randomized sun colour tints the whole shot differently every run. Only the
  colour is dropped: the pose jitter costs nothing and the range that remains
  is one the policy trained across."""
  from mjlab.tasks.registry import load_env_cfg

  import vbrl.tasks  # noqa: F401
  from vbrl.scenes.builder import LIGHT_COLOUR_EVENTS, hold_lighting_colour_fixed

  cfg = load_env_cfg("Mjlab-PushT-SlowGoal-NatureCnn-Flatten-TrossenRealistic")
  assert frozenset(LIGHT_COLOUR_EVENTS) <= frozenset(cfg.events)

  hold_lighting_colour_fixed(cfg)

  assert not frozenset(LIGHT_COLOUR_EVENTS) & frozenset(cfg.events)
  for kept in ("light_position", "light_direction", "table_material"):
    assert kept in cfg.events
