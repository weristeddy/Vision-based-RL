from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")


PUSH_T_STATE_TERMS = (
  "qpos",
  "qvel",
  "target_qpos",
  "tcp_pose",
  "target_pose",
  "obj_pose",
)
APPEARANCE_EVENTS = {
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
  "camera_position",
  "camera_orientation",
}


def _push_t(*, play: bool = False):
  from vbrl.tasks.push_t.config.trossen_realistic.env_cfgs import (
    trossen_realistic_push_t_state_env_cfg,
  )

  return trossen_realistic_push_t_state_env_cfg(play=play)


def test_every_registered_task_freezes_actor_and_privileged_critic_groups() -> None:
  from mjlab.tasks.registry import load_rl_cfg

  from vbrl.tasks import vbrl_task_ids

  task_ids = vbrl_task_ids()
  assert len(task_ids) == 41
  for task_id in task_ids:
    agent = load_rl_cfg(task_id)
    visual = agent.actor.cnn_cfg is not None
    assert tuple(agent.obs_groups) == ("actor", "critic")
    assert agent.obs_groups["actor"] == (
      ("actor", "camera") if visual else ("actor",)
    )
    assert agent.obs_groups["critic"] == ("critic",)
    expected_actor = "vbrl.vision.model:VisionModel" if visual else "MLPModel"
    assert agent.actor.class_name == expected_actor, task_id
    assert agent.critic.class_name == "MLPModel"
    assert agent.critic.cnn_cfg is None


def test_push_t_overlap_is_deterministic_and_pose_aware() -> None:
  from vbrl.tasks.push_t.geometry import (
    FOOTPRINT_PARTS,
    footprint_overlap_from_pose,
  )

  object_xy = torch.tensor(
    [[0.0, 0.0], [0.4, -0.2], [0.0, 0.0], [0.0, 0.0]], dtype=torch.float32
  )
  object_yaw = torch.tensor([0.0, 1.2, 0.0, math.pi / 2])
  target_xy = torch.tensor(
    [[0.0, 0.0], [0.4, -0.2], [0.4, 0.0], [0.0, 0.0]], dtype=torch.float32
  )
  target_yaw = torch.tensor([0.0, 1.2, 0.0, 0.0])
  overlap = dict(
    object_xy=object_xy,
    object_yaw=object_yaw,
    target_xy=target_xy,
    target_yaw=target_yaw,
    footprint_parts=FOOTPRINT_PARTS,
    resolution=64,
    half_width=0.09,
  )

  first = footprint_overlap_from_pose(**overlap)
  second = footprint_overlap_from_pose(**overlap)

  assert torch.equal(first, second)
  assert first[:2].tolist() == pytest.approx([1.0, 1.0])
  assert first[2] == pytest.approx(0.0)
  assert 0.0 < first[3] < 0.9


def test_push_t_success_threshold_latches_metrics() -> None:
  from vbrl.tasks.push_t.mdp.commands import PushTCommand

  command = object.__new__(PushTCommand)
  command.cfg = SimpleNamespace(success_threshold=0.90)
  command.target_pos = torch.tensor([[0.40, 0.0, 0.02], [0.40, 0.0, 0.02]])
  command.target_yaw = torch.zeros(2)
  command.object = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=command.target_pos.clone(),
      root_link_quat_w=torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
      ),
    )
  )
  command.episode_success = torch.zeros(2)
  command.metrics = {
    name: torch.zeros(2)
    for name in (
      "position_error",
      "yaw_error",
      "overlap",
      "at_goal",
      "episode_success",
    )
  }
  overlaps = iter((torch.tensor([0.90, 0.899]), torch.tensor([0.0, 1.0])))
  command.get_overlap = lambda: next(overlaps)

  first = PushTCommand.get_at_goal(command)
  second = PushTCommand.get_at_goal(command)

  assert torch.equal(first, torch.tensor([True, False]))
  assert torch.equal(second, torch.tensor([False, True]))
  assert torch.equal(command.episode_success, torch.ones(2))
  assert torch.equal(command.metrics["episode_success"], torch.ones(2))


def test_push_t_command_sampling_uses_episode_ranges_and_zero_velocity() -> None:
  from vbrl.tasks.push_t.geometry import FOOTPRINT_PARTS
  from vbrl.tasks.push_t.mdp.commands import PushTCommand, PushTCommandCfg

  cfg = PushTCommandCfg(
    entity_name="object",
    resampling_time_range=(5.0, 5.0),
    footprint_parts=FOOTPRINT_PARTS,
  )
  # Real draws: the sampler redraws whatever lands off the table, so the call
  # count is not fixed and a scripted sequence would run out.
  cfg.min_xy_separation = 0.02
  cfg.target_position_range = PushTCommandCfg.TargetPositionRangeCfg(
    x=(0.25, 0.45), y=(-0.2, 0.2), z=(0.012, 0.012)
  )
  cfg.object_pose_range = PushTCommandCfg.ObjectPoseRangeCfg(
    x=(0.25, 0.45), y=(-0.2, 0.2), z=(0.013, 0.013), yaw=(-math.pi, math.pi)
  )
  worlds = 512

  written: dict[str, torch.Tensor] = {}
  pushed_object = SimpleNamespace(
    write_root_link_pose_to_sim=lambda pose, env_ids: written.update(
      pose=pose.clone()
    ),
    write_root_link_velocity_to_sim=lambda velocity, env_ids: written.update(
      velocity=velocity.clone()
    ),
  )
  marker: dict[str, torch.Tensor] = {}
  fake_marker = SimpleNamespace(
    write_mocap_pose_to_sim=lambda pose, env_ids: marker.update(pose=pose.clone())
  )
  fake_command = SimpleNamespace(
    _goal_marker=fake_marker,
    cfg=cfg,
    device="cpu",
    episode_success=torch.ones(worlds),
    target_pos=torch.zeros(worlds, 3),
    target_yaw=torch.zeros(worlds),
    _overlap_cache_step=8,
    _env=SimpleNamespace(
      scene=SimpleNamespace(env_origins=torch.zeros(worlds, 3))
    ),
    object=pushed_object,
  )

  PushTCommand._resample_command(fake_command, torch.arange(worlds))

  separation = torch.linalg.vector_norm(
    fake_command.target_pos[:, :2] - written["pose"][:, :2], dim=-1
  )
  assert torch.all(separation >= cfg.min_xy_separation - 1e-6)
  goals = fake_command.target_pos[:, :2]
  lower = torch.tensor([cfg.target_position_range.x[0], cfg.target_position_range.y[0]])
  upper = torch.tensor([cfg.target_position_range.x[1], cfg.target_position_range.y[1]])
  assert torch.all(goals >= lower - 1e-6) and torch.all(goals <= upper + 1e-6)
  assert torch.allclose(written["pose"][:, 2], torch.full((worlds,), 0.013))
  assert torch.count_nonzero(written["velocity"]) == 0
  assert torch.count_nonzero(fake_command.episode_success) == 0
  assert fake_command._overlap_cache_step is None

  assert torch.allclose(marker["pose"][:, :2], fake_command.target_pos[:, :2])
  assert torch.count_nonzero(marker["pose"][:, 2]) == 0
  half = fake_command.target_yaw / 2.0
  assert torch.allclose(marker["pose"][:, 3], torch.cos(half), atol=1e-6)
  assert torch.allclose(marker["pose"][:, 6], torch.sin(half), atol=1e-6)
  assert torch.count_nonzero(marker["pose"][:, 4:6]) == 0

  command = PushTCommand.command.fget(fake_command)
  assert command.shape == (worlds, 4)
  assert torch.equal(command[:, :3], fake_command.target_pos)
  assert torch.equal(command[:, 3], fake_command.target_yaw)


def test_push_t_goal_sampling_converges_for_a_wide_separation_floor() -> None:
  from vbrl.tasks.push_t.geometry import FOOTPRINT_PARTS
  from vbrl.tasks.push_t.mdp.commands import PushTCommand, PushTCommandCfg

  worlds = 2048
  cfg = PushTCommandCfg(
    entity_name="object",
    resampling_time_range=(5.0, 5.0),
    footprint_parts=FOOTPRINT_PARTS,
    min_xy_separation=0.15,
  )
  cfg.object_pose_range = PushTCommandCfg.ObjectPoseRangeCfg(
    x=(0.20, 0.40), y=(-0.2, 0.2), z=(0.013, 0.013), yaw=(-math.pi, math.pi)
  )
  cfg.target_position_range = PushTCommandCfg.TargetPositionRangeCfg(
    x=(0.30, 0.50), y=(-0.2, 0.2), z=(0.012, 0.012)
  )
  written: dict[str, torch.Tensor] = {}
  fake = SimpleNamespace(
    _goal_marker=None,
    cfg=cfg,
    device="cpu",
    episode_success=torch.ones(worlds),
    target_pos=torch.zeros(worlds, 3),
    target_yaw=torch.zeros(worlds),
    _overlap_cache_step=1,
    _env=SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(worlds, 3))),
    object=SimpleNamespace(
      write_root_link_pose_to_sim=lambda pose, env_ids: written.update(
        pose=pose.clone()
      ),
      write_root_link_velocity_to_sim=lambda velocity, env_ids: None,
    ),
  )
  for _ in range(5):
    PushTCommand._resample_command(fake, torch.arange(worlds))
    separation = torch.linalg.vector_norm(
      fake.target_pos[:, :2] - written["pose"][:, :2], dim=-1
    )
    assert torch.all(separation >= cfg.min_xy_separation - 1e-6)
    goals = fake.target_pos[:, :2]
    lower = torch.tensor(
      [cfg.target_position_range.x[0], cfg.target_position_range.y[0]]
    )
    upper = torch.tensor(
      [cfg.target_position_range.x[1], cfg.target_position_range.y[1]]
    )
    assert torch.all(goals >= lower - 1e-6) and torch.all(goals <= upper + 1e-6)


def test_push_t_reward_is_maniskill_dense_with_a_linear_orientation_summand() -> None:
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  from vbrl.tasks.push_t.mdp import maniskill_dense_reward
  from vbrl.tasks.push_t.mdp.commands import PushTCommand

  command = object.__new__(PushTCommand)
  command.cfg = SimpleNamespace(orientation_weight=0.5)
  command.target_pos = torch.tensor([[0.20, 0.0, 0.02], [0.10, 0.0, 0.02]])
  command.target_yaw = torch.zeros(2)
  command.get_at_goal = lambda: torch.zeros(2, dtype=torch.bool)
  pushed_object = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[0.0, 0.0, 0.02], [0.0, 0.0, 0.02]]),
      root_link_quat_w=torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
      ),
    ),
  )
  robot = SimpleNamespace(
    data=SimpleNamespace(
      site_pos_w=torch.tensor([[[0.10, 0.0, 0.02]], [[0.20, 0.0, 0.02]]]),
      root_link_quat_w=torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
      ),
    )
  )
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda name: command),
    scene={"robot": robot, "object": pushed_object},
  )
  asset_cfg = SceneEntityCfg("robot")
  asset_cfg.site_ids = [0]

  distances = torch.tensor([0.20, 0.10])
  ee_distances = torch.tensor([0.10, 0.20])
  yaw_errors = torch.tensor([0.0, math.pi])
  shaped = (
    (1.0 - yaw_errors.abs() / math.pi) / 2.0
    + (1.0 - torch.tanh(5.0 * distances)).square() / 2.0
    + torch.sqrt(1.0 - torch.tanh(5.0 * ee_distances)) / 20.0
  )
  assert torch.allclose(
    maniskill_dense_reward(env, "push_t_goal", "object", asset_cfg),
    shaped / 3.0,
    atol=1.0e-6,
  )

  command.get_at_goal = lambda: torch.ones(2, dtype=torch.bool)
  assert torch.equal(
    maniskill_dense_reward(env, "push_t_goal", "object", asset_cfg),
    torch.ones(2),
  )


def test_push_t_object_origin_sits_on_its_centre_of_mass() -> None:
  import mujoco

  from vbrl.tasks.push_t.geometry import FOOTPRINT_PARTS

  model = mujoco.MjModel.from_xml_path("src/vbrl/asset_zoo/objects/push_t.xml")
  body = model.body("push_t")

  assert max(abs(float(v)) for v in body.ipos) < 1.0e-9
  # Uniform density, as ManiSkill does it. 50.7 g, weighed on the printed
  # object; equalising the two boxes would move the centre of mass to 7.5 mm.
  assert float(body.mass[0]) == pytest.approx(0.0507, abs=1.0e-4)

  for part, suffix in zip(FOOTPRINT_PARTS, ("crossbar", "stem"), strict=True):
    for kind in ("collision", "visual"):
      geom = model.geom(f"push_t_{suffix}_{kind}")
      assert float(geom.pos[1]) == pytest.approx(part.center_xy[1], abs=1.0e-9)
      assert float(geom.size[0]) == pytest.approx(part.half_extents_xy[0])
      assert float(geom.size[1]) == pytest.approx(part.half_extents_xy[1])

  areas = [4.0 * p.half_extents_xy[0] * p.half_extents_xy[1] for p in FOOTPRINT_PARTS]
  centroid = sum(a * p.center_xy[1] for a, p in zip(areas, FOOTPRINT_PARTS, strict=True))
  assert centroid / sum(areas) == pytest.approx(0.0, abs=1.0e-12)


def _push_t_reward_env(target_pos, target_yaw, object_pos, object_yaw, weight=0.5):
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  from vbrl.tasks.push_t.mdp.commands import PushTCommand

  command = object.__new__(PushTCommand)
  command.cfg = SimpleNamespace(orientation_weight=weight)
  command.target_pos = torch.as_tensor(target_pos, dtype=torch.float32)
  command.target_yaw = torch.as_tensor(target_yaw, dtype=torch.float32)
  command.get_at_goal = lambda: torch.zeros(len(command.target_yaw), dtype=torch.bool)
  yaw = torch.as_tensor(object_yaw, dtype=torch.float32)
  half = yaw / 2.0
  pushed = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.as_tensor(object_pos, dtype=torch.float32),
      root_link_quat_w=torch.stack(
        (torch.cos(half), torch.zeros_like(half), torch.zeros_like(half),
         torch.sin(half)),
        dim=-1,
      ),
    ),
  )
  robot = SimpleNamespace(
    data=SimpleNamespace(
      site_pos_w=pushed.data.root_link_pos_w[:, None, :].clone(),
      root_link_quat_w=torch.zeros(len(half), 4).index_fill_(
        1, torch.tensor([0]), 1.0
      ),
    )
  )
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda name: command),
    scene={"robot": robot, "object": pushed},
  )
  asset_cfg = SceneEntityCfg("robot")
  asset_cfg.site_ids = [0]
  return env, asset_cfg


def test_push_t_height_ceiling_is_linear_above_the_object() -> None:
  from mjlab.managers import SceneEntityCfg

  from vbrl.tasks.push_t.mdp import fingertip_height_excess

  robot = SimpleNamespace(
    data=SimpleNamespace(
      geom_pos_w=torch.tensor([[[0.0, 0.0, 0.048]], [[0.0, 0.0, 0.012]]])
    )
  )
  env = SimpleNamespace(scene={"robot": robot})
  cfg = SceneEntityCfg("robot")
  cfg.geom_ids = slice(None)

  assert torch.allclose(
    fingertip_height_excess(env, cfg, 0.024), torch.tensor([1.0, 0.0])
  )
  with pytest.raises(ValueError, match="ceiling > 0"):
    fingertip_height_excess(env, cfg, 0.0)

def test_push_t_table_touch_is_binary_over_the_whole_robot() -> None:
  from vbrl.tasks.push_t.mdp import max_contact_force, table_touch

  force = torch.tensor([[[[n, 0.0, 0.0]]] for n in (0.0, 0.5, 200.0)])
  sensor = SimpleNamespace(
    data=SimpleNamespace(
      found=torch.tensor([[0.0], [1.0], [1.0]]), force=None, force_history=force
    )
  )
  env = SimpleNamespace(scene={"robot_table_contact": sensor})
  assert torch.equal(
    table_touch(env, "robot_table_contact"), torch.tensor([0.0, 1.0, 1.0])
  )
  assert torch.allclose(
    max_contact_force(env, "robot_table_contact"), torch.tensor([0.0, 0.5, 200.0])
  )


def test_push_t_at_goal_action_penalty_is_zero_until_the_object_is_placed() -> None:
  from vbrl.tasks.push_t.mdp import at_goal_action_l1
  from vbrl.tasks.push_t.mdp.commands import PushTCommand

  command = object.__new__(PushTCommand)
  command.get_at_goal = lambda: torch.tensor([True, False, True])
  env = SimpleNamespace(
    action_manager=SimpleNamespace(
      action=torch.tensor([[1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, 0.0]])
    ),
    command_manager=SimpleNamespace(get_term=lambda name: command),
  )
  assert torch.allclose(
    at_goal_action_l1(env, "push_t_goal"), torch.tensor([2.0, 0.0, 0.0])
  )


def test_push_t_forceful_top_contact_terminates_only_on_hard_vertical_press() -> None:
  from vbrl.tasks.push_t.mdp import forceful_top_contact

  sensor = SimpleNamespace(
    data=SimpleNamespace(
      found=torch.tensor([[1.0], [1.0], [1.0], [0.0]]),
      force=torch.tensor(
        [[[40.0, 0.0, 0.0]], [[40.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]], [[40.0, 0.0, 0.0]]]
      ),
      normal=torch.tensor(
        [[[1.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]]
      ),
    )
  )
  env = SimpleNamespace(scene={"ee_object_contact": sensor})
  out = forceful_top_contact(env, "ee_object_contact", force_threshold=5.0)
  assert out.tolist() == [False, True, False, False]
  with pytest.raises(ValueError, match="force_threshold > 0"):
    forceful_top_contact(env, "ee_object_contact", force_threshold=0.0)


def test_push_t_object_table_press_is_zero_for_a_pure_lateral_push() -> None:
  from vbrl.tasks.push_t.mdp import object_table_press, peak_object_press

  W = 0.497
  sensor = SimpleNamespace(
    data=SimpleNamespace(
      force=torch.tensor(
        [
          [[0.0, 0.0, -W]],
          [[200.0, -150.0, -W]],
          [[3.0, 0.0, -(W + 2.0)]],
          [[8.0, 0.0, -(W + 30.0)]],
        ]
      ),
    )
  )
  env = SimpleNamespace(scene={"object_table_contact": sensor})
  out = object_table_press(env, "object_table_contact", onset=W + 2.0, scale=5.0)

  assert float(out[0]) == 0.0
  assert float(out[1]) == 0.0
  assert float(out[2]) == pytest.approx(0.0)
  assert float(out[3]) == pytest.approx(28.0 / 5.0)

  press = peak_object_press(env, "object_table_contact", weight_n=W)
  assert press.tolist() == pytest.approx([0.0, 0.0, 2.0, 30.0])

  with pytest.raises(ValueError, match="scale > 0"):
    object_table_press(env, "object_table_contact", onset=W, scale=0.0)


def test_push_t_max_contact_force_splits_top_from_side() -> None:
  from vbrl.tasks.push_t.mdp import max_contact_force_on_face

  sensor = SimpleNamespace(
    data=SimpleNamespace(
      found=torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]),
      force=torch.tensor(
        [
          [[0.0, 0.0, 30.0], [4.0, 0.0, 0.0]],
          [[9.0, 0.0, 0.0], [50.0, 0.0, 0.0]],
          [[0.0, 0.0, 80.0], [0.0, 0.0, 0.0]],
        ]
      ),
      normal=torch.tensor(
        [
          [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
          [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
          [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        ]
      ),
    )
  )
  env = SimpleNamespace(scene={"ee_object_contact": sensor})

  assert max_contact_force_on_face(
    env, "ee_object_contact", vertical=True
  ).tolist() == [30.0, 0.0, 0.0]
  assert max_contact_force_on_face(
    env, "ee_object_contact", vertical=False
  ).tolist() == [4.0, 9.0, 0.0]


def test_push_t_side_contact_align_rewards_pushing_a_vertical_face() -> None:
  from vbrl.tasks.push_t.mdp import side_contact_align, top_contact_share

  sensor = SimpleNamespace(
    data=SimpleNamespace(
      found=torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
      normal=torch.tensor(
        [
          [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
          [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
          [[0.0, 0.6, 0.8], [0.0, 0.0, 1.0]],
          [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        ]
      ),
    )
  )
  env = SimpleNamespace(scene={"ee_object_contact": sensor})
  out = side_contact_align(env, "ee_object_contact")

  assert torch.allclose(out, torch.tensor([1.0, 0.0, 1.0 - 0.8, 0.0]))
  assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
  assert float(out[0]) == pytest.approx(1.0)

  share = top_contact_share(env, "ee_object_contact")
  assert share.tolist() == [0.0, 1.0, 1.0, 0.0]

def test_push_t_observations_are_maniskill_poses_in_the_base_frame() -> None:
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  from vbrl.tasks.push_t.mdp import obj_pose, target_pose, tcp_pose
  from vbrl.tasks.push_t.mdp.commands import PushTCommand

  command = object.__new__(PushTCommand)
  command.target_pos = torch.tensor([[0.40, 0.10, 0.02], [0.50, -0.10, 0.02]])
  command.target_yaw = torch.tensor([math.pi / 2, -math.pi])
  command.observation_offset = torch.zeros(2, 3)
  command.observation_yaw_offset = torch.zeros(2)
  yaw90 = [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
  robot = SimpleNamespace(
    data=SimpleNamespace(
      site_pos_w=torch.tensor([[[0.20, 0.00, 0.10]], [[0.20, 0.00, 0.10]]]),
      site_quat_w=torch.tensor([[yaw90], [yaw90]]),
      root_link_quat_w=torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
      ),
      root_link_pos_w=torch.tensor([[0.10, 0.00, 0.00], [0.10, 0.00, 0.00]]),
    ),
  )
  pushed_object = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[0.30, 0.05, 0.02], [0.40, -0.05, 0.02]]),
      root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0], yaw90]),
    ),
  )
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda name: command),
    scene={"robot": robot, "object": pushed_object},
  )
  asset_cfg = SceneEntityCfg("robot")
  asset_cfg.site_ids = [0]

  assert torch.allclose(
    tcp_pose(env, asset_cfg),
    torch.tensor([[0.10, 0.0, 0.10, *yaw90], [0.10, 0.0, 0.10, *yaw90]]),
    atol=1.0e-6,
  )
  assert torch.allclose(
    obj_pose(env, "object"),
    torch.tensor([[0.20, 0.05, 0.02, 1.0, 0.0, 0.0, 0.0], [0.30, -0.05, 0.02, *yaw90]]),
    atol=1.0e-6,
  )
  assert torch.allclose(
    target_pose(env, "push_t_goal", asset_cfg),
    torch.tensor(
      [[0.30, 0.10, 0.02, 1.0, 0.0], [0.40, -0.10, 0.02, 0.0, -1.0]]
    ),
    atol=1.0e-6,
  )


def test_push_t_invalid_state_terminations_cover_table_and_velocity() -> None:
  from vbrl.scenes.presets import TABLE_CENTER, TABLE_HALF_EXTENTS
  from vbrl.tasks.push_t.mdp import invalid_object_state, object_off_table

  beyond_x = TABLE_CENTER[0] + TABLE_HALF_EXTENTS[0] + 0.05

  obj = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor(
        [
          [0.30, 0.00, 0.013],
          [beyond_x, 0.00, 0.013],
          [0.30, 0.00, -0.06],
          [0.30, 0.00, 0.30],
          [0.30, 0.00, 0.013],
          [float("nan"), 0.00, 0.013],
        ]
      ),
      root_link_vel_w=torch.tensor(
        [
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [6.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
      ),
    )
  )

  class _Scene(dict):
    pass

  scene = _Scene(object=obj)
  scene.env_origins = torch.zeros(6, 3)
  env = SimpleNamespace(scene=scene)

  assert torch.equal(
    object_off_table(env, "object"),
    torch.tensor([False, True, True, False, False, False]),
  )
  assert torch.equal(
    invalid_object_state(env, "object"),
    torch.tensor([False, False, False, True, True, False]),
  )


def test_push_t_gaussian_joint_reset_preserves_selection_and_clamping(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  from vbrl.tasks.push_t.mdp import events

  written: dict[str, object] = {}
  asset = SimpleNamespace(
    data=SimpleNamespace(
      default_joint_pos=torch.tensor(
        [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]]
      ),
      default_joint_vel=torch.tensor(
        [[0.0, 0.01, 0.02], [0.1, 0.11, 0.12], [0.2, 0.21, 0.22]]
      ),
      soft_joint_pos_limits=torch.tensor(
        [
          [[-3.0, 3.0], [0.0, 0.15], [0.0, 0.25]],
          [[-3.0, 3.0], [1.0, 1.15], [1.0, 1.25]],
          [[-3.0, 3.0], [2.0, 2.15], [2.0, 2.25]],
        ]
      ),
    ),
    write_joint_state_to_sim=lambda position, velocity, **kwargs: written.update(
      position=position.clone(), velocity=velocity.clone(), **kwargs
    ),
  )
  env = SimpleNamespace(num_envs=3, device="cpu", scene={"robot": asset})
  asset_cfg = SimpleNamespace(name="robot", joint_ids=[1, 2])
  calls: list[tuple[object, ...]] = []

  def sample_gaussian(mean, std, shape, *, device):
    calls.append((mean, std, shape, device))
    return torch.tensor([[0.10, -0.30], [-0.50, 0.10]])

  monkeypatch.setattr(events, "sample_gaussian", sample_gaussian)
  events.reset_joints_with_gaussian_offset(
    env, torch.tensor([2.0, 0.0]), position_std=0.02, asset_cfg=asset_cfg
  )

  assert calls == [(0.0, 0.02, (2, 2), "cpu")]
  assert torch.equal(written["env_ids"], torch.tensor([2, 0], dtype=torch.int))
  assert torch.equal(written["joint_ids"], torch.tensor([1, 2]))
  assert torch.equal(
    written["position"], torch.tensor([[2.15, 2.0], [0.0, 0.25]])
  )
  assert torch.equal(
    written["velocity"], torch.tensor([[0.21, 0.22], [0.01, 0.02]])
  )


def test_push_t_friction_sample_is_coupled_across_object_and_table(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  from vbrl.tasks.push_t.mdp import events

  friction = torch.full((3, 8, 3), -1.0)
  env = SimpleNamespace(
    num_envs=3,
    device="cpu",
    scene={
      "object": SimpleNamespace(
        indexing=SimpleNamespace(geom_ids=torch.tensor([4, 6], dtype=torch.int))
      ),
      "table": SimpleNamespace(
        indexing=SimpleNamespace(geom_ids=torch.tensor([2], dtype=torch.int))
      ),
    },
    sim=SimpleNamespace(model=SimpleNamespace(geom_friction=friction)),
  )

  monkeypatch.setattr(
    events,
    "sample_gaussian",
    lambda *args, **kwargs: torch.tensor([[0.35], [-0.1]]),
  )
  events.randomize_object_table_friction(
    env,
    torch.tensor([2, 0]),
    mean=0.3,
    std=0.025,
    object_asset_cfg=SimpleNamespace(name="object", geom_ids=[0, 1]),
    table_asset_cfg=SimpleNamespace(name="table", geom_ids=[0]),
  )

  assert torch.equal(friction[2, (2, 4, 6), 0], torch.full((3,), 0.35))
  assert torch.equal(friction[0, (2, 4, 6), 0], torch.zeros(3))
  assert torch.equal(friction[1], torch.full((8, 3), -1.0))
  assert torch.equal(friction[:, :, 1:], torch.full((3, 8, 2), -1.0))


def test_push_t_config_pins_the_trained_contract() -> None:
  from vbrl.asset_zoo.robots import get_robot
  from vbrl.tasks.push_t.mdp import TargetRelativeJointPositionActionCfg
  from vbrl.tasks.push_t.push_t_env_cfg import (
    ACTION_PATH_LENGTH_WEIGHT,
    ACTION_RATE_WEIGHT,
    ACTION_SCALE,
    AT_GOAL_ACTION_WEIGHT,
    EE_HEIGHT_CEILING_M,
    EE_HEIGHT_WEIGHT,
    OBJECT_PRESS_ONSET_N,
    OBJECT_PRESS_SCALE_N,
    OBJECT_WEIGHT_N,
    SIDE_CONTACT_ALIGN_WEIGHT,
    START_JOINT_POS,
    TABLE_TOUCH_WEIGHT,
  )

  cfg = _push_t()
  definition = get_robot("trossen_realistic")

  sensors = {sensor.name: sensor for sensor in cfg.scene.sensors}
  assert sensors["object_table_contact"].reduce == "netforce"
  assert "force" in sensors["object_table_contact"].fields

  assert set(cfg.scene.entities) == {"robot", "table", "object"}
  assert cfg.episode_length_s == 16.0
  assert cfg.sim.mujoco.timestep == 0.005
  assert cfg.decimation == 4
  assert cfg.scale_rewards_by_dt is False
  assert set(cfg.metrics) == {
    "peak_table_force",
    "peak_top_face_force",
    "peak_side_face_force",
    "peak_object_press",
    "top_contact_share",
    "at_goal_share",
  }
  assert cfg.metrics["peak_table_force"].reduce == "max"
  assert cfg.metrics["top_contact_share"].reduce == "mean"
  # episode_success latches; this reports how long the goal was held.
  assert cfg.metrics["at_goal_share"].reduce == "mean"

  action = cfg.actions["joint_pos"]
  assert isinstance(action, TargetRelativeJointPositionActionCfg)
  assert action.actuator_names == definition.arm_actuator_names
  assert len(action.actuator_names) == 6
  assert action.scale == pytest.approx(ACTION_SCALE) == pytest.approx(0.03)
  assert action.clip is None
  assert {
    name: cfg.scene.entities["robot"].init_state.joint_pos[name]
    for name in definition.closed_gripper_joint_pos
  } == dict(definition.closed_gripper_joint_pos)
  joint_pos = cfg.scene.entities["robot"].init_state.joint_pos
  assert {name: joint_pos[name] for name in START_JOINT_POS} == START_JOINT_POS

  command = cfg.commands["push_t_goal"]
  assert command.success_threshold == pytest.approx(0.90)
  assert command.min_xy_separation == pytest.approx(0.15)
  assert command.resampling_time_range == (1.0e9, 1.0e9)
  assert command.mask_resolution == 64

  assert tuple(cfg.rewards) == (
    "maniskill_dense",
    "side_contact_align",
    "top_contact",
    "action_path_length",
    "action_rate_l2",
    "at_goal_action",
    "table_touch",
    "object_table_press",
    "ee_height_ceiling",
  )
  assert cfg.rewards["maniskill_dense"].weight == pytest.approx(1.0)
  assert cfg.rewards["side_contact_align"].weight == pytest.approx(0.05)
  assert SIDE_CONTACT_ALIGN_WEIGHT == pytest.approx(0.05)
  assert cfg.rewards["top_contact"].weight == pytest.approx(-0.05)
  assert cfg.rewards["top_contact"].params["sensor_name"] == "ee_object_contact"
  assert cfg.rewards["side_contact_align"].params["sensor_name"] == "ee_object_contact"
  assert cfg.rewards["action_path_length"].weight == pytest.approx(-0.002)
  assert ACTION_PATH_LENGTH_WEIGHT == pytest.approx(-0.002)
  assert cfg.rewards["action_rate_l2"].weight == pytest.approx(-0.002)
  assert ACTION_RATE_WEIGHT == pytest.approx(-0.002)
  assert "action_acc_l2" not in cfg.rewards
  assert "joint_vel_hinge" not in cfg.rewards
  assert cfg.rewards["at_goal_action"].weight == pytest.approx(-0.05)
  assert AT_GOAL_ACTION_WEIGHT == pytest.approx(-0.05)

  from vbrl.tasks.push_t.geometry import HALF_HEIGHT

  assert EE_HEIGHT_CEILING_M == pytest.approx(2.0 * HALF_HEIGHT) == pytest.approx(0.024)
  assert cfg.rewards["ee_height_ceiling"].weight == pytest.approx(-0.02)
  assert EE_HEIGHT_WEIGHT == pytest.approx(-0.02)
  assert "sensor_name" not in cfg.rewards["ee_height_ceiling"].params
  assert cfg.rewards["ee_height_ceiling"].params["ceiling"] == EE_HEIGHT_CEILING_M
  assert "vertical_contact_force" not in cfg.rewards
  assert "object_contact_force" not in cfg.rewards
  assert cfg.rewards["table_touch"].weight == TABLE_TOUCH_WEIGHT
  assert pytest.approx(-2 / 3) == TABLE_TOUCH_WEIGHT
  press = cfg.rewards["object_table_press"].params
  assert press["sensor_name"] == "object_table_contact"
  assert OBJECT_WEIGHT_N == pytest.approx(0.497)
  assert press["onset"] == pytest.approx(1.497) == OBJECT_PRESS_ONSET_N
  assert press["scale"] == pytest.approx(5.0) == OBJECT_PRESS_SCALE_N
  assert cfg.rewards["object_table_press"].weight == pytest.approx(-0.01)

  assert cfg.rewards["table_touch"].params["sensor_name"] == "robot_table_contact"
  table = sensors["robot_table_contact"]
  assert (table.primary.mode, table.primary.pattern) == ("subtree", "base_link")
  assert table.secondary.entity == "table"
  assert cfg.metrics["peak_table_force"].params["sensor_name"] == "robot_table_contact"

  assert cfg.curriculum == {}


  assert tuple(cfg.terminations) == (
    "time_out",
    "object_off_table",
    "invalid_object_state",
    "nan_detection",
  )
  assert "forceful_top_contact" not in cfg.terminations
  assert cfg.terminations["time_out"].time_out is True

  actor, critic = cfg.observations["actor"], cfg.observations["critic"]
  assert tuple(actor.terms) == tuple(critic.terms) == PUSH_T_STATE_TERMS
  assert actor.enable_corruption is True
  assert critic.enable_corruption is False
  assert actor.nan_policy == critic.nan_policy == "sanitize"
  assert "camera" not in cfg.observations

  assert set(cfg.events) == {
    "reset_base",
    "reset_table_base",
    "reset_robot_joints",
    "arm_joint_position_noise",
    "fingertip_friction_slide",
    "object_friction",
  }
  assert APPEARANCE_EVENTS.isdisjoint(cfg.events)


def test_push_t_play_only_disables_actor_noise_and_curriculum() -> None:
  cfg = _push_t(play=True)

  assert cfg.observations["actor"].enable_corruption is False
  assert cfg.curriculum == {}
  assert cfg.episode_length_s == 16.0
  assert cfg.commands["push_t_goal"].resampling_time_range == (1.0e9, 1.0e9)
