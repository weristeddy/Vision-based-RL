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
  "default", "procedural", "real_texture", "real_table", "real_texture_wood",
  "wood049", "marble019", "wood", "plaster", "peacock", "ood_light", "object_blue",
  "goal_blue", "object_blue_goal_purple", "colour_swap", "object_color_sweep",
  "goal_color_sweep",
)
# Every training preset randomizes the sun's colour on top of its pose. The
# matched-evaluation branch deliberately does not; see `_events` in the builder.
LIGHT_COLOR_EVENTS = frozenset({"light_intensity"})
# Every scene with a camera also randomizes the room behind the table.
CAMERA_EVENTS = frozenset({"background"})


def _object_xml(name: str):
  from vbrl.asset_zoo.objects import OBJECTS_DIR

  return OBJECTS_DIR / f"{name}.xml"


def _apply(scene: str, *, object_name="cube", eval_dr="fixed"):
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
    robot=get_robot("trossen_measured"),
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


def test_replace_scene_rejects_an_unknown_scene() -> None:
  from vbrl.scenes.builder import replace_scene

  with pytest.raises(ValueError, match="Unknown scene"):
    replace_scene(_apply("real_texture"), scene="marble")


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
  if expected:
    expected |= LIGHT_COLOR_EVENTS | CAMERA_EVENTS
  assert frozenset(_apply(scene).events) == expected


def test_a_task_without_a_camera_gets_no_visual_randomization() -> None:
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
    robot=get_robot("trossen_measured"),
    camera_view=None,
    object_xml=_object_xml("cube"),
    object_name="cube",
  )

  assert cfg.events == {}


def test_default_differs_from_real_texture_only_in_appearance() -> None:
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
      "background",
    }
  )
  assert cfg.events["light_position"].params["operation"] == "add"


@pytest.mark.parametrize("scene", ALL_SCENES)
def test_table_geometry_matches_the_bank(scene: str) -> None:
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
  from vbrl.scenes.builder import table_spec
  from vbrl.scenes.presets import MUJOCO_MAX_TEXTURES, ambientcg_texture_paths

  model = table_spec(get_preset("real_texture")).compile()

  assert model.ntex < MUJOCO_MAX_TEXTURES
  assert len(ambientcg_texture_paths()) > MUJOCO_MAX_TEXTURES


def test_real_texture_spends_one_material_on_its_texture_pool() -> None:
  import re

  from mjlab.envs.mdp import dr

  from vbrl.scenes.builder import BACKDROP_COUNT, table_spec
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
  assert model.ntex == len(names) + BACKDROP_COUNT


def test_realistic_scenes_add_a_fill_light() -> None:
  from vbrl.scenes.builder import table_spec
  from vbrl.scenes.presets import FILL_LIGHT_NAME

  for scene, has_fill in (("real_texture", False), ("wood", True)):
    model = table_spec(get_preset(scene)).compile()
    found = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, FILL_LIGHT_NAME)
    assert (found >= 0) is has_fill, scene


@pytest.mark.parametrize("scene", ("real_table", "real_texture_wood", *ood_scenes()))
def test_apply_scene_and_replace_scene_agree(scene: str) -> None:
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
    specs = [cfg.scene.entities[entity].spec_fn() for cfg in (direct, replaced)]
    for spec in specs:
      spec.compile()
    assert specs[0].to_xml() == specs[1].to_xml(), entity


def test_a_recording_holds_the_sun_colour_but_keeps_its_pose_jitter() -> None:
  from mjlab.tasks.registry import load_env_cfg

  import vbrl.tasks  # noqa: F401
  from vbrl.scenes.builder import LIGHT_COLOUR_EVENTS, hold_lighting_colour_fixed

  cfg = load_env_cfg("Mjlab-PushT-WoodTable-DinoV2ViTS14-AfaFull6-Trossen")
  assert frozenset(LIGHT_COLOUR_EVENTS) <= frozenset(cfg.events)

  hold_lighting_colour_fixed(cfg)

  assert not frozenset(LIGHT_COLOUR_EVENTS) & frozenset(cfg.events)
  for kept in ("light_position", "light_direction", "table_material"):
    assert kept in cfg.events


def test_real_texture_wood_draws_only_wood_textures() -> None:
  from vbrl.scenes.builder import BACKDROP_COUNT, table_spec
  from vbrl.scenes.presets import (
    AMBIENTCG_WOOD_FAMILIES,
    ambientcg_family,
    ambientcg_pool_paths,
  )

  preset = get_preset("real_texture_wood")
  assert preset.table is not None
  paths = ambientcg_pool_paths(preset.table.families)

  assert paths
  assert all(ambientcg_family(path) in AMBIENTCG_WOOD_FAMILIES for path in paths)
  assert table_spec(preset).compile().ntex == len(paths) + BACKDROP_COUNT


def test_every_student_is_evaluated_in_its_tasks_scenes() -> None:
  from mjlab.tasks.registry import load_rl_cfg

  import vbrl.tasks  # noqa: F401
  from vbrl.scenes.presets import EVALUATION_SCENES, GOAL_COLOUR_SCENES
  from vbrl.tasks import vbrl_task_ids

  assert EVALUATION_SCENES == (
    "wood049",
    "real_table",
    "marble019",
    "ood_light",
    "object_blue",
    "goal_blue",
    "object_blue_goal_purple",
    "colour_swap",
    "object_color_sweep",
    "goal_color_sweep",
  )
  students = [t for t in vbrl_task_ids() if "-WoodTable-" in t]
  assert len(students) == 21
  for task_id in students:
    expected = EVALUATION_SCENES
    if "LiftCube" in task_id:
      expected = tuple(s for s in EVALUATION_SCENES if s not in GOAL_COLOUR_SCENES)
    assert load_rl_cfg(task_id).evaluation_scenes == expected, task_id


def test_the_evaluation_scenes_draw_the_same_events() -> None:
  from vbrl.scenes.presets import EVALUATION_SCENES

  configs = [_apply(scene) for scene in EVALUATION_SCENES]
  for scene, cfg in zip(EVALUATION_SCENES[1:], configs[1:], strict=True):
    # A sweep adds one deterministic override after the draws, so it stays paired.
    sweep = [name for name in cfg.events if name.endswith("_color_sweep")]
    assert sweep == ([scene] if scene == "object_color_sweep" else []), scene
    assert [name for name in cfg.events if name not in sweep] == list(configs[0].events)
    for name, term in cfg.events.items():
      if name in sweep:
        continue
      reference = configs[0].events[name]
      assert term.func is reference.func, name
      assert term.mode == reference.mode, name
      assert set(term.params) == set(reference.params), name


def test_each_ood_scene_moves_exactly_its_factor() -> None:
  from vbrl.scenes.presets import (
    BLUE_GOAL,
    BLUE_OBJECT,
    OOD_LIGHT_COLOUR,
    PURPLE_GOAL,
    REAL_TABLE_GOAL_COLOUR_RANGE,
    REAL_TABLE_OBJECT_COLOUR_RANGE,
    SWAP_GOAL,
    SWAP_OBJECT,
  )

  def fixed(rgb):
    return tuple((value, value) for value in rgb)

  expected = {
    "wood049": (REAL_TABLE_OBJECT_COLOUR_RANGE, REAL_TABLE_GOAL_COLOUR_RANGE, None),
    "ood_light": (
      REAL_TABLE_OBJECT_COLOUR_RANGE, REAL_TABLE_GOAL_COLOUR_RANGE, OOD_LIGHT_COLOUR
    ),
    "object_blue": (fixed(BLUE_OBJECT), REAL_TABLE_GOAL_COLOUR_RANGE, None),
    "goal_blue": (REAL_TABLE_OBJECT_COLOUR_RANGE, fixed(BLUE_GOAL), None),
    "object_blue_goal_purple": (fixed(BLUE_OBJECT), fixed(PURPLE_GOAL), None),
    "colour_swap": (fixed(SWAP_OBJECT), fixed(SWAP_GOAL), None),
  }
  for scene, (obj, goal, light) in expected.items():
    preset = get_preset(scene)
    assert preset.table == get_preset("wood049").table, scene
    assert preset.object_colour_range == obj, scene
    assert preset.goal_colour_range == goal, scene
    assert preset.light_colour == light, scene
    events = _apply(scene).events
    assert events["object_color"].params["ranges"] == dict(enumerate(obj)), scene
    assert events["light_intensity"].params["colour"] == light, scene


@pytest.mark.parametrize(
  ("task_id", "event"),
  [
    ("Mjlab-PushT-WoodTable-NatureCnn-FlattenRelu-Trossen", "goal_color"),
    ("Mjlab-PushChain3-WoodTable-NatureCnn-FlattenRelu-Trossen", "goal_color"),
  ],
)
def test_replace_scene_sets_each_tasks_goal_colour(task_id: str, event: str) -> None:
  from mjlab.tasks.registry import load_env_cfg

  import vbrl.tasks  # noqa: F401
  from vbrl.scenes.builder import replace_scene
  from vbrl.scenes.presets import BLUE_GOAL, REAL_TABLE_GOAL_COLOUR_RANGE

  def ranges(cfg):
    return tuple(cfg.events[event].params["ranges"])

  assert ranges(load_env_cfg(task_id)) == REAL_TABLE_GOAL_COLOUR_RANGE
  blue = replace_scene(load_env_cfg(task_id, play=True), scene="goal_blue")
  assert ranges(blue) == tuple((value, value) for value in BLUE_GOAL)
  back = replace_scene(blue, scene="wood049")
  assert ranges(back) == REAL_TABLE_GOAL_COLOUR_RANGE


def test_the_colour_sweep_is_a_colour_chart_of_hues_shades_and_greys() -> None:
  from vbrl.scenes.presets import (
    COLOUR_SWEEP_COLOURS,
    COLOUR_SWEEP_HUES,
    COLOUR_SWEEP_SHADES,
    colour_sweep_rgb,
    sweep_colour,
  )

  assert (COLOUR_SWEEP_HUES, COLOUR_SWEEP_SHADES, COLOUR_SWEEP_COLOURS) == (31, 16, 512)
  colours = [colour_sweep_rgb(k) for k in range(COLOUR_SWEEP_COLOURS)]
  assert len(set(colours)) == 512
  assert colours[0] == pytest.approx((0.2, 0.0, 0.0))
  assert colours[COLOUR_SWEEP_HUES * COLOUR_SWEEP_SHADES] == (0.0, 0.0, 0.0)
  assert colours[-1] == (1.0, 1.0, 1.0)
  assert [sweep_colour(i, 512) for i in (0, 1, 511)] == [0, 1, 511]
