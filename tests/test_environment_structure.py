from __future__ import annotations

import pytest

pytest.importorskip("mjlab")


LIFT_STATE = ("qpos", "qvel", "target_qpos", "tcp_pose", "goal_position", "obj_pose")
LIFT_CRITIC = (*LIFT_STATE, "ee_to_cube", "cube_to_goal")
PUSH_T_STATE = (
  "qpos",
  "qvel",
  "target_qpos",
  "tcp_pose",
  "target_pose",
  "obj_pose",
)
PUSH_T_CRITIC = (*PUSH_T_STATE, "relative_yaw", "object_to_goal", "ee_to_object")
PUSH_T_RGB_ACTOR = (
  "qpos",
  "qvel",
  "target_qpos",
  "tcp_pose",
  "target_pose",
)


def _camera(cfg, name: str):
  return next(sensor for sensor in cfg.scene.sensors if sensor.name == name)


def test_lift_state_and_rgb_share_control_and_the_student_sees_the_wrist() -> None:
  from vbrl.tasks.lift_cube.config.trossen_realistic.env_cfgs import (
    trossen_realistic_lift_cube_rgb_env_cfg,
    trossen_realistic_lift_cube_state_env_cfg,
  )

  state = trossen_realistic_lift_cube_state_env_cfg()
  rgb = trossen_realistic_lift_cube_rgb_env_cfg(scene="real_texture_wood")

  for cfg in (state, rgb):
    assert set(cfg.scene.entities) == {"robot", "table", "cube"}
    assert tuple(cfg.observations["critic"].terms) == LIFT_CRITIC
    assert cfg.decimation == 10
  assert tuple(state.observations["actor"].terms) == LIFT_STATE
  assert tuple(rgb.observations["actor"].terms) == LIFT_STATE[:-1]
  assert tuple(rgb.observations["teacher"].terms) == LIFT_STATE
  assert "camera" not in state.observations
  assert tuple(rgb.observations["camera"].terms) == ("cam_rgb",)
  assert state.actions == rgb.actions
  action = state.actions["joint_pos"]
  assert action.actuator_names[-1] == "left_carriage_joint"
  assert action.absolute_names == ("left_carriage_joint",)
  assert "left_carriage_joint" not in action.scale
  assert (action.scale["joint_0"], action.wrist_roll_scale, action.ramp_s) == (
    0.025,
    0.02,
    0.05,
  )
  assert state.rewards == rgb.rewards
  assert state.scale_rewards_by_dt is rgb.scale_rewards_by_dt is True
  assert state.rewards["joint_vel_l2"].weight == -0.05
  assert state.rewards["action_rate_l2"].weight == -0.01
  assert state.rewards["ee_vertical"].weight == 0.0

  camera = _camera(rgb, "cam")
  assert camera.camera_name == "robot/cam"
  assert camera.enabled_geom_groups == (0, 2)
  assert (camera.width, camera.height) == (224, 224)
  assert rgb.viewer.geom_group == (1, 1, 1, 0, 0, 0)
  wrist = rgb.events["camera_position"].params["ranges"]
  assert wrist == dict.fromkeys(range(3), (-0.005, 0.005))


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


def test_any_scene_can_be_registered_against_a_task() -> None:
  from vbrl.tasks.lift_cube.config.trossen_realistic.env_cfgs import (
    trossen_realistic_lift_cube_rgb_env_cfg,
  )

  real = trossen_realistic_lift_cube_rgb_env_cfg(scene="real_texture")
  procedural = trossen_realistic_lift_cube_rgb_env_cfg(scene="procedural")

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
  rgb = trossen_realistic_push_t_rgb_env_cfg(scene="real_table")

  assert tuple(state.observations["actor"].terms) == PUSH_T_STATE
  assert tuple(state.observations["critic"].terms) == PUSH_T_CRITIC
  assert "camera" not in state.observations
  assert tuple(rgb.observations["actor"].terms) == PUSH_T_RGB_ACTOR
  assert tuple(rgb.observations["critic"].terms) == PUSH_T_CRITIC
  assert tuple(rgb.observations["camera"].terms) == ("external_cam_rgb",)
  state_action, rgb_action = state.actions["joint_pos"], rgb.actions["joint_pos"]
  assert state_action == rgb_action
  assert (state_action.scale, state_action.wrist_roll_scale) == (0.025, 0.02)
  assert state_action.ramp_s == 0.05
  assert state_action.actuator_names == rgb_action.actuator_names
  state_goal, rgb_goal = state.commands["push_t_goal"], rgb.commands["push_t_goal"]
  assert state_goal.goal_marker_name is None
  assert rgb_goal.goal_marker_name == "goal_marker"
  assert state_goal.object_pose_range == rgb_goal.object_pose_range
  assert state_goal.success_threshold == rgb_goal.success_threshold
  assert state_goal.near_object_start and rgb_goal.near_object_start
  assert state_goal.start_types == rgb_goal.start_types
  assert state.curriculum == rgb.curriculum == {}
  assert state.rewards == rgb.rewards
  assert set(state.terminations) - set(rgb.terminations) == {"top_face_contact"}
  assert state.episode_length_s == rgb.episode_length_s == 10.0
  assert state.decimation == rgb.decimation == 10

  camera = _camera(rgb, "external_cam")
  assert camera.camera_name == "robot/external_cam"
  assert camera.enabled_geom_groups == (0, 2)
  assert camera.fovy is None
  assert (camera.width, camera.height) == (224, 224)
  assert rgb.viewer.geom_group == (1, 1, 1, 0, 0, 0)

  target_pose = rgb.observations["actor"].terms["target_pose"]
  assert target_pose.params["command_name"] == "push_t_goal"
  assert "object_name" not in target_pose.params
  assert "obj_pose" not in rgb.observations["actor"].terms
  assert "obj_pose" in rgb.observations["critic"].terms


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
    ("trossen_realistic_lift_cube_state_env_cfg", {}),
    ("trossen_realistic_lift_cube_rgb_env_cfg", {"scene": "real_texture_wood"}),
    ("trossen_realistic_push_t_state_env_cfg", {}),
    ("trossen_realistic_push_t_rgb_env_cfg", {"scene": "real_table"}),
  ),
  ids=("lift-state", "lift-rgb", "push-t-state", "push-t-rgb"),
)
def test_task_local_play_factories_are_small_clean_copies(
  factory: str, kwargs: dict
) -> None:
  if "lift_cube" in factory:
    from vbrl.tasks.lift_cube.config.trossen_realistic import env_cfgs
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
    "Mjlab-LiftCube-State-Trossen",
    "Mjlab-LiftCube-WoodTable-DinoV2ViTS14-AfaFull6-Trossen",
    "Mjlab-PushT-State-Trossen",
    "Mjlab-PushT-RealTable-DinoV2ViTS14-AfaFull6-Trossen",
  ):
    train = load_env_cfg(task_id)
    play = load_env_cfg(task_id, play=True)
    assert train.scene.num_envs == 1024
    assert play.scene.num_envs == 1
    assert play.observations["actor"].enable_corruption is False
