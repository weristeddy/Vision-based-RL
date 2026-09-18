from __future__ import annotations

import pytest

pytest.importorskip("mjlab")


LIFT_ACTOR = ("joint_pos", "joint_vel", "actions", "goal_position")
LIFT_CRITIC = (
  "joint_pos",
  "joint_vel",
  "ee_to_cube",
  "cube_to_goal",
  "actions",
)
PUSH_T_STATE = (
  "joint_pos",
  "joint_vel",
  "ee_to_object",
  "object_to_goal",
  "object_heading",
  "relative_yaw",
  "actions",
)
PUSH_T_RGB_ACTOR = (
  "joint_pos",
  "joint_vel",
  "actions",
  "target_pose",
)


def _camera(cfg, name: str):
  return next(sensor for sensor in cfg.scene.sensors if sensor.name == name)


def test_lift_collision_and_visual_versions_preserve_distinct_cameras() -> None:
  from vbrl.tasks.lift_cube.config.trossen.env_cfgs import (
    trossen_lift_cube_env_cfg,
  )

  collision = trossen_lift_cube_env_cfg(camera_geometry="collision")
  visual = trossen_lift_cube_env_cfg(camera_geometry="visual")

  for cfg in (collision, visual):
    assert set(cfg.scene.entities) == {"robot", "table", "cube"}
    assert tuple(cfg.observations["actor"].terms) == LIFT_ACTOR
    assert tuple(cfg.observations["critic"].terms) == LIFT_CRITIC
    assert tuple(cfg.observations["camera"].terms) == ("cam_rgb",)
    camera = _camera(cfg, "cam")
    assert camera.camera_name == "robot/cam"
    assert (camera.width, camera.height) == (224, 224)

  assert _camera(collision, "cam").enabled_geom_groups == (0, 3)
  assert _camera(visual, "cam").enabled_geom_groups == (0, 2)
  # Recorded video must draw the same geometry generation the policy is fed, plus the
  # floor in group 1, which is scenery for whoever is watching -- and now the D405 body.
  assert collision.viewer.geom_group == (1, 1, 0, 1, 0, 0)
  assert visual.viewer.geom_group == (1, 1, 1, 0, 0, 0)


def test_every_task_lays_its_envs_out_on_a_grid() -> None:
  from mjlab.sensor import CameraSensorCfg
  from mjlab.tasks.registry import load_env_cfg
  from mjlab.terrains import TerrainEntityCfg

  import vbrl.tasks  # noqa: F401
  from vbrl.tasks import vbrl_task_ids
  from vbrl.tasks.utils.tabletop_env_cfg import (
    ENV_SPACING_M,
    ORIGIN_PLANE_GROUP,
    _floor_for_the_human_views_only,
  )

  for task_id in vbrl_task_ids():
    cfg = load_env_cfg(task_id)
    terrain = cfg.scene.terrain
    assert terrain is not None, task_id
    assert terrain.terrain_type == "plane", task_id
    # MJLab's own checker groundplane, kept; its sun dropped, because the
    # scene builds its own lighting and a second one would reach the camera.
    assert terrain.textures == TerrainEntityCfg().textures, task_id
    assert terrain.materials == TerrainEntityCfg().materials, task_id
    assert terrain.lights == (), task_id
    assert cfg.scene.env_spacing == ENV_SPACING_M, task_id
    assert cfg.scene.spec_fn is _floor_for_the_human_views_only, task_id
    assert "reset_table_base" in cfg.events, task_id
    assert cfg.viewer.geom_group[ORIGIN_PLANE_GROUP] == 1, task_id
    for sensor in cfg.scene.sensors or ():
      if isinstance(sensor, CameraSensorCfg):
        assert ORIGIN_PLANE_GROUP not in sensor.enabled_geom_groups, task_id


def test_the_visual_camera_is_the_default() -> None:
  from vbrl.tasks.lift_cube.config.trossen.env_cfgs import (
    trossen_lift_cube_env_cfg,
  )

  default = trossen_lift_cube_env_cfg()

  assert _camera(default, "cam").enabled_geom_groups == (0, 2)
  assert default.viewer.geom_group == (1, 1, 1, 0, 0, 0)


def test_any_scene_can_be_registered_against_a_task() -> None:
  from vbrl.tasks.lift_cube.config.trossen.env_cfgs import (
    trossen_lift_cube_env_cfg,
  )

  real = trossen_lift_cube_env_cfg(scene="real_texture")
  procedural = trossen_lift_cube_env_cfg(scene="procedural")

  assert "table_material" in real.events
  assert "table_material" in procedural.events
  assert real.events["table_material"].func is not procedural.events[
    "table_material"
  ].func
  assert real.actions == procedural.actions
  assert real.rewards == procedural.rewards


def test_push_t_state_and_rgb_share_physics_but_not_actor_observations() -> None:
  from vbrl.tasks.push_t.config.trossen_realistic.env_cfgs import (
    trossen_realistic_push_t_rgb_env_cfg,
    trossen_realistic_push_t_state_env_cfg,
  )

  state = trossen_realistic_push_t_state_env_cfg()
  rgb = trossen_realistic_push_t_rgb_env_cfg(action_delta=0.03)

  assert tuple(state.observations["actor"].terms) == PUSH_T_STATE
  assert tuple(state.observations["critic"].terms) == PUSH_T_STATE
  assert "camera" not in state.observations
  assert tuple(rgb.observations["actor"].terms) == PUSH_T_RGB_ACTOR
  assert tuple(rgb.observations["critic"].terms) == PUSH_T_STATE + ("target_pose",)
  assert tuple(rgb.observations["camera"].terms) == ("external_cam_rgb",)
  # The two differ by the per-step delta cap alone: the RGB tasks train at the
  # deployable 0.03 so the raw policy output needs no clamp on the arm.
  state_action, rgb_action = state.actions["joint_pos"], rgb.actions["joint_pos"]
  assert state_action.scale == 0.1 and rgb_action.scale == 0.03
  assert state_action.actuator_names == rgb_action.actuator_names
  state_goal, rgb_goal = state.commands["push_t_goal"], rgb.commands["push_t_goal"]
  assert state_goal.goal_marker_name is None
  assert rgb_goal.goal_marker_name == "goal_marker"
  assert state_goal.object_pose_range == rgb_goal.object_pose_range
  assert state_goal.min_xy_separation == rgb_goal.min_xy_separation
  assert state_goal.success_threshold == rgb_goal.success_threshold
  assert state.curriculum == {}
  assert tuple(rgb.curriculum) == ("goal_yaw_range",)
  assert state.rewards == rgb.rewards
  assert state.terminations == rgb.terminations
  assert state.episode_length_s == rgb.episode_length_s == 5.0

  camera = _camera(rgb, "external_cam")
  assert camera.camera_name == "robot/external_cam"
  assert camera.enabled_geom_groups == (0, 2)
  assert camera.fovy is None
  assert (camera.width, camera.height) == (224, 224)
  assert rgb.viewer.geom_group == (1, 1, 1, 0, 0, 0)

  target_pose = rgb.observations["actor"].terms["target_pose"]
  assert target_pose.params["command_name"] == "push_t_goal"
  assert "object_name" not in target_pose.params
  for privileged in ("ee_to_object", "object_to_goal", "object_heading"):
    assert privileged not in rgb.observations["actor"].terms
    assert privileged in rgb.observations["critic"].terms


def test_rgb_camera_term_preserves_native_uint8_bchw() -> None:
  from types import SimpleNamespace

  import torch

  from vbrl.tasks.utils.camera import camera_rgb_uint8

  rgb = torch.arange(2 * 5 * 7 * 3, dtype=torch.uint8).reshape(2, 5, 7, 3)
  env = SimpleNamespace(
    scene={"camera": SimpleNamespace(data=SimpleNamespace(rgb=rgb))}
  )

  observation = camera_rgb_uint8(env, "camera")
  assert observation.dtype is torch.uint8
  assert observation.shape == (2, 3, 5, 7)
  assert torch.equal(observation, rgb.permute(0, 3, 1, 2))


@pytest.mark.parametrize(
  ("factory", "kwargs"),
  (
    ("trossen_lift_cube_env_cfg", {"camera_geometry": "collision"}),
    ("trossen_lift_cube_env_cfg", {"camera_geometry": "visual"}),
    ("trossen_realistic_push_t_state_env_cfg", {}),
    ("trossen_realistic_push_t_rgb_env_cfg", {"action_delta": 0.03}),
  ),
  ids=("lift-collision", "lift-visual", "push-t-state", "push-t-rgb"),
)
def test_task_local_play_factories_are_small_clean_copies(
  factory: str, kwargs: dict
) -> None:
  if factory.startswith("trossen_lift_cube"):
    from vbrl.tasks.lift_cube.config.trossen import env_cfgs
  else:
    from vbrl.tasks.push_t.config.trossen_realistic import env_cfgs

  build = getattr(env_cfgs, factory)
  train = build(**kwargs)
  play = build(play=True, **kwargs)
  assert train is not play
  assert train.scene.num_envs == 1024
  assert play.scene.num_envs == 1
  assert play.observations["actor"].enable_corruption is False
  assert play.curriculum == {}


def test_native_registry_play_configs_match_task_local_factories() -> None:
  from mjlab.tasks.registry import load_env_cfg

  import vbrl.tasks  # noqa: F401

  for task_id in (
    "Mjlab-LiftCube-CollisionCam-DinoV2ViTS14-LocalGrid7-Trossen",
    "Mjlab-LiftCube-RealTexture-DinoV2ViTS14-LocalGrid7-Trossen",
    "Mjlab-PushT-State-TrossenRealistic",
    "Mjlab-PushT-VisualSlowStep-DinoV2ViTS14-Afa6-TrossenRealistic",
  ):
    train = load_env_cfg(task_id)
    play = load_env_cfg(task_id, play=True)
    assert train.scene.num_envs == 1024
    assert play.scene.num_envs == 1
    assert play.observations["actor"].enable_corruption is False
