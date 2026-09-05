"""Behavioural contracts for the Push-T and Push-Cube MDPs.

These pin what the retained checkpoints were trained against: the ManiSkill
dense-reward formula, the 0.98 overlap success threshold, coupled object/table
friction, the Gaussian joint reset, and the 3200-step curriculum boundary.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest


pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")


PUSH_T_STATE_TERMS = (
  "joint_pos",
  "joint_vel",
  "ee_to_object",
  "object_to_goal",
  "object_heading",
  "relative_yaw",
  "actions",
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


def _push_cube(*, play: bool = False):
  from vbrl.tasks.push_cube.config.trossen.env_cfgs import (
    trossen_push_cube_env_cfg,
  )

  return trossen_push_cube_env_cfg(play=play)


# --- registered policy contract ---------------------------------------------


def test_every_registered_task_freezes_actor_and_privileged_critic_groups() -> None:
  from mjlab.tasks.registry import load_rl_cfg

  from vbrl.tasks import vbrl_task_ids

  task_ids = vbrl_task_ids()
  assert len(task_ids) == 258
  for task_id in task_ids:
    agent = load_rl_cfg(task_id)
    visual = agent.actor.cnn_cfg is not None
    assert tuple(agent.obs_groups) == ("actor", "critic")
    assert agent.obs_groups["actor"] == (
      ("actor", "camera") if visual else ("actor",)
    )
    assert agent.obs_groups["critic"] == ("critic",)
    # `Balanced` is the one visual arm with a different actor: it projects
    # proprioception to the visual width before the concat, so its class differs
    # while everything else about the model is shared.
    if not visual:
      expected_actor = "MLPModel"
    elif "-Balanced-" in task_id:
      expected_actor = "vbrl.vision.model:BalancedVisionModel"
    else:
      expected_actor = "vbrl.vision.model:VisionModel"
    assert agent.actor.class_name == expected_actor, task_id
    assert agent.critic.class_name == "MLPModel"
    assert agent.critic.cnn_cfg is None


# --- Push-T geometry and goal -----------------------------------------------


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


def test_push_t_success_threshold_is_98_percent_and_latches_metrics() -> None:
  from vbrl.tasks.push_t.mdp.commands import PushTCommand

  command = object.__new__(PushTCommand)
  command.cfg = SimpleNamespace(success_threshold=0.98)
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
  overlaps = iter((torch.tensor([0.98, 0.979]), torch.tensor([0.0, 1.0])))
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
  # Real draws rather than a scripted sequence: the sampler redraws whatever
  # lands off the table, so the number of calls it makes is not fixed and a
  # fixed script would run out. What matters is the invariant it maintains over
  # a batch -- every goal at least the floor away, and every goal on the table.
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
  # The drawn target, for the visual-goal variants. Standing it up here turns
  # what would be a missing attribute into coverage of the pose it is given.
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
  # Every goal lands inside the declared target range -- the redraw is what
  # guarantees this, and a goal placed off the rectangle would slip through any
  # separation-only assertion.
  goals = fake_command.target_pos[:, :2]
  lower = torch.tensor([cfg.target_position_range.x[0], cfg.target_position_range.y[0]])
  upper = torch.tensor([cfg.target_position_range.x[1], cfg.target_position_range.y[1]])
  assert torch.all(goals >= lower - 1e-6) and torch.all(goals <= upper + 1e-6)
  assert torch.allclose(written["pose"][:, 2], torch.full((worlds,), 0.013))
  assert torch.count_nonzero(written["velocity"]) == 0
  assert torch.count_nonzero(fake_command.episode_success) == 0
  assert fake_command._overlap_cache_step is None

  # The marker is posed at the goal it represents: same xy, flat on the table,
  # yaw as a rotation about z.
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
  """The 15 cm floor variants must reset, not exhaust the redraw budget.

  `Uniform`, `SlowGoal` and `VisualGoal` draw the goal from a window offset from
  the object's, with a 15 cm floor. Only about 12% of polar draws land inside
  that window, against 28% for the 1 cm floor `FreeStart` uses, so a redraw
  budget tuned on `FreeStart` silently fails here -- and it fails at reset, on
  every one of a thousand-plus worlds, which is a dead sweep rather than a
  degraded one. Sized like a training batch because the budget has to cover the
  unluckiest world in it, not the average one.
  """
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


def test_separation_curriculum_follows_the_rollout_length_and_honours_a_pin() -> None:
  """The ramp counts environment steps, so it must know the real rollout length.

  Its `steps_per_iteration` is a literal in the registration -- the task config
  cannot see the agent config -- so a run at any other `num_steps_per_env` would
  burn through the ramp at the wrong rate, silently. At 100 steps against the
  registered 16 the cap would reach full range at iteration 640 instead of 4000,
  six times too fast, with nothing in the logs to say so.
  """
  import torch

  from vbrl.scripts.train import TrainConfig, _retime_separation_curriculum
  from vbrl.tasks.push_t.mdp.curriculums import separation_curriculum
  from vbrl.tasks.push_t.push_t_env_cfg import (
    FREE_START_MAX_SEPARATION,
    SEPARATION_CURRICULUM_ITERATIONS,
  )

  def cap_at(iteration: int, *, steps: int, pin: int | None = None) -> float:
    cfg = TrainConfig.from_task(
      "Mjlab-PushT-GrowStart-DinoV2ViTS14-LocalGrid16-TrossenRealistic"
    )
    cfg.agent.num_steps_per_env = steps
    if pin is not None:
      object.__setattr__(cfg, "separation_pin_iterations", pin)
    _retime_separation_curriculum(cfg)
    term = cfg.env.curriculum["separation_range"]
    command_cfg = SimpleNamespace(max_xy_separation=None)
    env = SimpleNamespace(
      device="cpu",
      common_step_counter=iteration * steps,
      command_manager=SimpleNamespace(
        get_term=lambda name: SimpleNamespace(cfg=command_cfg)
      ),
    )
    term_obj = separation_curriculum(term, env)
    result = term_obj(env, torch.tensor([0]), **dict(term.params))
    return float(result["max_separation"])

  ramp = SEPARATION_CURRICULUM_ITERATIONS
  # The ramp completes at the same *iteration* whatever the rollout length.
  for steps in (16, 100):
    assert cap_at(0, steps=steps) == pytest.approx(0.05)
    assert cap_at(ramp // 2, steps=steps) == pytest.approx(
      0.05 + 0.5 * (FREE_START_MAX_SEPARATION - 0.05), abs=1e-6
    )
    assert cap_at(ramp, steps=steps) == pytest.approx(FREE_START_MAX_SEPARATION)

  # A pin holds the floor, then the ramp runs its full length from there.
  assert cap_at(1000, steps=16, pin=2000) == pytest.approx(0.05)
  assert cap_at(2000, steps=16, pin=2000) == pytest.approx(0.05)
  assert cap_at(2000 + ramp, steps=16, pin=2000) == pytest.approx(
    FREE_START_MAX_SEPARATION
  )


# --- Push-T rewards, observations, terminations ------------------------------


def test_push_t_reward_exactly_matches_maniskill_normalized_dense_formula() -> None:
  from mjlab.managers.scene_entity_config import SceneEntityCfg

  from vbrl.tasks.push_t.mdp import maniskill_dense_reward
  from vbrl.tasks.push_t.mdp.commands import PushTCommand

  command = object.__new__(PushTCommand)
  # The reward reads its orientation share off the command config, so a bare
  # command needs one. 0.5 is the registered value and ManiSkill's split.
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
  expected = (
    ((torch.cos(yaw_errors) + 1.0) / 2.0).square() / 2.0
    + (1.0 - torch.tanh(5.0 * distances)).square() / 2.0
    + torch.sqrt(1.0 - torch.tanh(5.0 * ee_distances)) / 20.0
  ) / 3.0

  assert torch.allclose(
    maniskill_dense_reward(env, "push_t_goal", "object", asset_cfg),
    expected,
    atol=1.0e-6,
  )

  # Reaching the goal replaces the shaped value with exactly 1.
  command.get_at_goal = lambda: torch.ones(2, dtype=torch.bool)
  assert torch.equal(
    maniskill_dense_reward(env, "push_t_goal", "object", asset_cfg),
    torch.ones(2),
  )


def test_push_t_vertical_contact_force_penalizes_forceful_top_contact() -> None:
  from vbrl.tasks.push_t.mdp import vertical_contact_force

  sensor = SimpleNamespace(
    data=SimpleNamespace(
      found=torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
      force=torch.tensor(
        [
          [[100.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          [[10.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          [[100.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
      ),
      normal=torch.tensor(
        [
          [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
          [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
          [[0.0, 0.6, 0.8], [0.0, 0.0, 0.0]],
          [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        ]
      ),
    )
  )
  env = SimpleNamespace(scene={"ee_object_contact": sensor})

  assert torch.allclose(
    vertical_contact_force(env, "ee_object_contact", force_scale=10.0),
    torch.tensor([0.0, math.tanh(0.1), 0.8 * math.tanh(1.0), 0.0]),
  )
  with pytest.raises(ValueError, match="force_scale must be positive"):
    vertical_contact_force(env, "ee_object_contact", force_scale=0.0)


def test_push_t_observations_use_mjlab_translation_and_task_yaw_terms() -> None:
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  from mjlab.tasks.manipulation.mdp import (
    ee_to_object_distance,
    object_to_goal_distance,
  )

  from vbrl.tasks.push_t.mdp import object_heading, relative_yaw, target_pose
  from vbrl.tasks.push_t.mdp.commands import PushTCommand

  command = object.__new__(PushTCommand)
  command.target_pos = torch.tensor([[0.40, 0.10, 0.02], [0.50, -0.10, 0.02]])
  command.target_yaw = torch.tensor([math.pi / 2, -math.pi])
  robot = SimpleNamespace(
    data=SimpleNamespace(
      site_pos_w=torch.tensor([[[0.20, 0.00, 0.10]], [[0.20, 0.00, 0.10]]]),
      root_link_quat_w=torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
      ),
      root_link_pos_w=torch.tensor([[0.10, 0.00, 0.00], [0.10, 0.00, 0.00]]),
    ),
  )
  pushed_object = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[0.30, 0.05, 0.02], [0.40, -0.05, 0.02]]),
      root_link_quat_w=torch.tensor(
        [
          [1.0, 0.0, 0.0, 0.0],
          [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)],
        ],
      ),
    ),
  )
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda name: command),
    scene={"robot": robot, "object": pushed_object},
  )
  asset_cfg = SceneEntityCfg("robot")
  asset_cfg.site_ids = [0]

  assert torch.allclose(
    ee_to_object_distance(env, "object", asset_cfg),
    torch.tensor([[0.10, 0.05, -0.08], [0.20, -0.05, -0.08]]),
  )
  assert torch.allclose(
    object_to_goal_distance(env, "object", "push_t_goal", asset_cfg),
    torch.tensor([[0.10, 0.05, 0.0], [0.10, -0.05, 0.0]]),
  )
  assert torch.allclose(
    object_heading(env, "object"),
    torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    atol=1.0e-6,
  )
  assert torch.allclose(
    relative_yaw(env, "push_t_goal", "object"),
    torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
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

  # Derived, not hardcoded: this probe used to be a literal 0.80, which stopped
  # being off the table the moment it was widened to 1 x 1 m.
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


# --- Push-T domain randomization --------------------------------------------


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

  # One draw per env is shared by the object and table geoms, and clamped at 0.
  assert torch.equal(friction[2, (2, 4, 6), 0], torch.full((3,), 0.35))
  assert torch.equal(friction[0, (2, 4, 6), 0], torch.zeros(3))
  assert torch.equal(friction[1], torch.full((8, 3), -1.0))
  assert torch.equal(friction[:, :, 1:], torch.full((3, 8, 2), -1.0))


# --- Push-T registered configuration ----------------------------------------


def test_push_t_config_pins_the_trained_contract() -> None:
  from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg

  from vbrl.asset_zoo.robots import get_robot
  from vbrl.tasks.push_t.push_t_env_cfg import (
    VERTICAL_CONTACT_FORCE_CURRICULUM_STEP,
  )

  cfg = _push_t()
  definition = get_robot("trossen_realistic")

  assert set(cfg.scene.entities) == {"robot", "table", "object"}
  assert cfg.episode_length_s == 5.0
  assert cfg.sim.mujoco.timestep == 0.005
  assert cfg.decimation == 4
  assert cfg.scale_rewards_by_dt is False
  assert cfg.metrics == {}

  action = cfg.actions["joint_pos"]
  assert isinstance(action, RelativeJointPositionActionCfg)
  assert action.actuator_names == definition.arm_actuator_names
  assert len(action.actuator_names) == 6
  assert action.scale == pytest.approx(0.1)
  assert action.clip == {
    name: pytest.approx((-0.1, 0.1)) for name in definition.arm_actuator_names
  }
  assert {
    name: cfg.scene.entities["robot"].init_state.joint_pos[name]
    for name in definition.closed_gripper_joint_pos
  } == dict(definition.closed_gripper_joint_pos)

  command = cfg.commands["push_t_goal"]
  assert command.success_threshold == pytest.approx(0.98)
  assert command.min_xy_separation == pytest.approx(0.15)
  assert command.resampling_time_range == (1.0e9, 1.0e9)
  assert command.mask_resolution == 64

  assert tuple(cfg.rewards) == (
    "maniskill_dense",
    "ee_table_contact",
    "vertical_contact_force",
  )
  assert cfg.rewards["maniskill_dense"].weight == pytest.approx(1.0)
  assert cfg.rewards["ee_table_contact"].weight == pytest.approx(-0.02)
  assert cfg.rewards["vertical_contact_force"].weight == pytest.approx(0.0)

  assert VERTICAL_CONTACT_FORCE_CURRICULUM_STEP == 3200
  assert cfg.curriculum["vertical_contact_force_weight"].params["stages"] == [
    {"step": 0, "weight": 0.0},
    {"step": VERTICAL_CONTACT_FORCE_CURRICULUM_STEP, "weight": -0.025},
  ]

  assert tuple(cfg.terminations) == (
    "time_out",
    "object_off_table",
    "invalid_object_state",
    "nan_detection",
  )
  assert cfg.terminations["time_out"].time_out is True

  actor, critic = cfg.observations["actor"], cfg.observations["critic"]
  assert tuple(actor.terms) == tuple(critic.terms) == PUSH_T_STATE_TERMS
  assert actor.enable_corruption is True
  assert critic.enable_corruption is False
  assert actor.nan_policy == critic.nan_policy == "sanitize"
  assert "camera" not in cfg.observations

  # The plain scene is the state task's baseline: task DR only, no appearance.
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
  assert cfg.episode_length_s == 5.0
  assert cfg.commands["push_t_goal"].resampling_time_range == (1.0e9, 1.0e9)


# --- Push-Cube ---------------------------------------------------------------


def test_push_command_rejects_targets_inside_minimum_separation(
  monkeypatch,
) -> None:
  from vbrl.tasks.push_cube.mdp import commands as task_module

  cfg = _push_cube().commands["push_goal"]
  # Every candidate in world 0 is invalid, so it uses the deterministic
  # farthest-corner fallback. World 1 is already valid.
  sampled = iter(
    (
      torch.tensor([[0.30, 0.00, 0.03], [0.20, -0.20, 0.02]]),
      torch.tensor([[[0.31, 0.00, 0.02]] * 64, [[0.50, 0.20, 0.02]] * 64]),
      torch.zeros(2),
    )
  )

  def sample_uniform(lower, upper, shape, *, device):
    del lower, upper, shape
    return next(sampled).to(device)

  written: dict[str, torch.Tensor] = {}
  cube = SimpleNamespace(
    write_root_link_pose_to_sim=lambda pose, env_ids: written.update(
      pose=pose.clone()
    ),
    write_root_link_velocity_to_sim=lambda velocity, env_ids: written.update(
      velocity=velocity.clone()
    ),
  )
  # The drawn target, for the visual-goal variants. Standing it up here turns
  # what would be a missing attribute into coverage of the pose it is given.
  marker: dict[str, torch.Tensor] = {}
  fake_marker = SimpleNamespace(
    write_mocap_pose_to_sim=lambda pose, env_ids: marker.update(pose=pose.clone())
  )
  fake_command = SimpleNamespace(
    _goal_marker=fake_marker,
    cfg=cfg,
    device="cpu",
    episode_success=torch.ones(2),
    target_pos=torch.zeros(2, 3),
    _env=SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(2, 3))),
    object=cube,
  )
  monkeypatch.setattr(task_module, "sample_uniform", sample_uniform)

  task_module.PushingCommand._resample_command(fake_command, torch.arange(2))

  separation = torch.linalg.vector_norm(
    fake_command.target_pos[:, :2] - written["pose"][:, :2], dim=-1
  )
  assert torch.all(separation >= cfg.min_xy_separation)
  assert torch.all(fake_command.target_pos[:, 2] == 0.02)
  assert torch.count_nonzero(written["velocity"]) == 0
  assert torch.count_nonzero(fake_command.episode_success) == 0


def test_push_reward_helpers_use_goal_conditioned_push_point() -> None:
  from vbrl.tasks.push_cube.mdp import (
    ee_push_point_distance,
    ee_to_push_point,
    object_goal_distance,
    push_point_position,
  )
  from vbrl.tasks.push_cube.mdp.commands import PushingCommand

  command = object.__new__(PushingCommand)
  command.target_pos = torch.tensor([[1.0, 0.0, 0.02], [0.0, 1.0, 0.02]])
  cube = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.tensor([[0.0, 0.0, 0.02], [0.0, 0.0, 0.02]])
    )
  )
  robot = SimpleNamespace(
    data=SimpleNamespace(
      site_pos_w=torch.tensor([[[-0.025, 0.0, 0.02]], [[0.0, -0.025, 1.02]]]),
      root_link_quat_w=torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
      ),
    )
  )
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda name: command),
    scene={"robot": robot, "cube": cube},
  )
  asset_cfg = _push_cube().rewards["ee_push_point_distance"].params["asset_cfg"]

  assert torch.equal(
    object_goal_distance(env, "push_goal", "cube"), torch.tensor([1.0, 1.0])
  )
  assert torch.allclose(
    push_point_position(env, "push_goal", "cube"),
    torch.tensor([[-0.025, 0.0, 0.02], [0.0, -0.025, 0.02]]),
  )
  assert torch.allclose(
    ee_to_push_point(env, "push_goal", "cube", asset_cfg),
    torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
  )
  assert torch.allclose(
    ee_push_point_distance(env, "push_goal", "cube", asset_cfg),
    torch.tensor([0.0, 1.0]),
  )


def test_push_point_is_finite_at_goal() -> None:
  from vbrl.tasks.push_cube.mdp import push_point_position
  from vbrl.tasks.push_cube.mdp.commands import PushingCommand

  command = object.__new__(PushingCommand)
  command.target_pos = torch.tensor([[0.4, 0.0, 0.02]])
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda name: command),
    scene={
      "cube": SimpleNamespace(
        data=SimpleNamespace(root_link_pos_w=command.target_pos.clone())
      )
    },
  )

  point = push_point_position(env, "push_goal", "cube")
  assert torch.equal(point, command.target_pos)
  assert torch.isfinite(point).all()


def test_push_cube_config_pins_its_state_contract() -> None:
  cfg = _push_cube()

  assert set(cfg.scene.entities) == {"robot", "table", "cube"}
  assert "camera" not in cfg.observations
  assert len(cfg.actions["joint_pos"].actuator_names) == 6
  assert APPEARANCE_EVENTS.isdisjoint(cfg.events)
  assert _push_cube(play=True).observations["actor"].enable_corruption is False
