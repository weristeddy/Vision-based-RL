"""Behavioural contracts for the Stack-Cubes MDP.

The task's own logic -- which cube counts as stacked, how tall the tower is,
what the reward pays for -- is checked against hand-built states rather than a
simulator, so a mistake surfaces in seconds. The three registered IDs and their
observation split are pinned here too, because the task ID is the contract.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest


pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

from vbrl.tasks.stack_cubes.mdp import tower as T  # noqa: E402


STACK_CUBES_IDS = (
  "Mjlab-StackCubes-State-TrossenRealistic",
  "Mjlab-StackCubes-Ext-NatureCnn-SpatialSoftmax-TrossenRealistic",
  "Mjlab-StackCubes-Wrist-NatureCnn-SpatialSoftmax-TrossenRealistic",
)
PRIVILEGED = ("cube_positions", "ee_to_target_cube", "tower_progress")
PROPRIOCEPTION = ("joint_pos", "joint_vel", "actions")


class _Scene(dict):
  """A scene stand-in: entity lookup by name plus the env-origin grid."""

  env_origins: torch.Tensor


def _cube(position: torch.Tensor, yaw: torch.Tensor, moving: bool = False):
  half = yaw / 2
  quat = torch.stack(
    [torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)],
    dim=-1,
  )
  velocity = torch.full_like(position, 1.0 if moving else 0.0)
  return SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=position,
      root_link_quat_w=quat,
      root_link_lin_vel_w=velocity,
      root_link_ang_vel_w=velocity,
    )
  )


# The pad order a real ContactSensor reports for the Trossen fingertips.
_PADS = [f"{side}_finger_pad_{index}_collision" for side in ("right", "left")
         for index in range(3)]


def _grasp_sensor(count: int, pinched: bool):
  """A contact-sensor stand-in: both fingers touching, or neither."""
  found = torch.full((count, len(_PADS)), 1.0 if pinched else 0.0)
  force = torch.zeros(count, len(_PADS), 3)
  force[..., 0] = 4.0 if pinched else 0.0
  return SimpleNamespace(
    primary_names=list(_PADS),
    data=SimpleNamespace(found=found, force=force),
  )


def _env(
  positions,
  *,
  yaws=None,
  ee=(1.0, 1.0, 1.0),
  held=None,
  arm=(0.0,) * 6,
  moving=None,
):
  """Build an env stand-in from explicit cube positions, one row per env.

  ``held`` names the cube the gripper has hold of, which closes the hand and
  makes that cube's contact sensor report both fingers touching -- the same
  two-finger test the real sensors drive.
  """
  from mjlab.managers import SceneEntityCfg

  cubes = torch.as_tensor(positions, dtype=torch.float32)
  count = cubes.shape[0]
  yaws = torch.zeros(count, T.MAX_CUBES) if yaws is None else torch.as_tensor(yaws)
  moving = [False] * T.MAX_CUBES if moving is None else moving
  carriage = T.GRIPPER_CLOSED_M - 1e-3 if held is not None else T.GRIPPER_OPEN_M
  scene = _Scene(
    {
      name: _cube(cubes[:, index], yaws[:, index], moving[index])
      for index, name in enumerate(T.CUBE_NAMES)
    }
  )
  for index, sensor in enumerate(T.CONTACT_SENSORS):
    scene[sensor] = _grasp_sensor(count, index == held)
  scene["robot"] = SimpleNamespace(
    data=SimpleNamespace(
      site_pos_w=torch.tensor([ee], dtype=torch.float32).expand(count, 1, 3),
      joint_pos=torch.tensor(
        [[*arm, carriage]], dtype=torch.float32
      ).expand(count, 7),
    )
  )
  scene.env_origins = torch.zeros(count, 3)
  asset_cfg = SceneEntityCfg("robot")
  asset_cfg.site_ids = [0]
  asset_cfg.joint_ids = [6]
  return SimpleNamespace(scene=scene), asset_cfg


def _level(index: int, offset=(0.0, 0.0)):
  return [T.STACK_XY[0] + offset[0], T.STACK_XY[1] + offset[1], T.level_height(index)]


def _loose(index: int):
  """A cube lying on the table well clear of the stack column."""
  spread = ((0.24, -0.16), (0.44, -0.16), (0.24, 0.16), (0.44, 0.16))
  return [*spread[index], T.CUBE_HALF]


# --- registration -----------------------------------------------------------


def test_stack_cubes_registers_exactly_three_contracts() -> None:
  """One state baseline and one policy per camera; no other architecture."""
  from vbrl.tasks import vbrl_task_ids

  registered = [t for t in vbrl_task_ids() if "StackCubes" in t]
  assert sorted(registered) == sorted(STACK_CUBES_IDS)


def test_stack_cubes_keeps_lift_cube_control_timing_at_a_finer_substep() -> None:
  """50 Hz control, unchanged; the substep halves so a tower can stand up."""
  from mjlab.tasks.registry import load_env_cfg

  for task_id in STACK_CUBES_IDS:
    cfg = load_env_cfg(task_id)
    assert cfg.sim.mujoco.timestep * cfg.decimation == pytest.approx(0.02)
    assert cfg.episode_length_s == pytest.approx(40.0)


# --- tower state ------------------------------------------------------------


def test_tower_height_is_independent_of_which_cube_is_where() -> None:
  """cube_3/cube_0/cube_2 is the same three-stack as cube_0/cube_1/cube_2."""
  ordered = [[_level(0), _level(1), _level(2), _loose(3)]]
  shuffled = [[_level(1), _loose(0), _level(2), _level(0)]]

  for positions in (ordered, shuffled):
    state = T.tower_state(*_env(positions))
    assert state.height.tolist() == [3]
    # Three of four is not finished: every cube has to go in the tower.
    assert state.complete.tolist() == [False]


def test_tower_is_contiguous_from_the_table_up() -> None:
  """Two separate two-stacks are not a four-stack, and a gap stops the count."""
  # Levels 0 and 1 at the stack, plus a pair stacked 0.2 m to the side.
  aside = [T.STACK_XY[0], T.STACK_XY[1] + 0.2, T.level_height(0)]
  above = [T.STACK_XY[0], T.STACK_XY[1] + 0.2, T.level_height(1)]
  env, asset_cfg = _env([[_level(0), _level(1), aside, above]])
  assert T.tower_state(env, asset_cfg).height.tolist() == [2]

  # Level 0 missing: nothing above it counts, however well placed.
  env, asset_cfg = _env([[_level(1), _level(2), _loose(2), _loose(3)]])
  assert T.tower_state(env, asset_cfg).height.tolist() == [0]


def test_the_tower_is_only_complete_with_every_cube_in_it() -> None:
  """All four cubes are always in play; there is no smaller finished tower."""
  for built in range(T.MAX_CUBES + 1):
    rows = [_level(index) for index in range(built)]
    rows += [_loose(index) for index in range(built, T.MAX_CUBES)]
    state = T.tower_state(*_env([rows]))
    assert state.height.tolist() == [built]
    assert state.complete.tolist() == [built == T.MAX_CUBES]


def test_a_held_or_moving_cube_does_not_claim_its_level() -> None:
  """Holding a cube in exactly the right place is not the same as placing it."""
  rows = [[_level(0), _level(1), _loose(2), _loose(3)]]

  env, asset_cfg = _env(rows, ee=_level(1), held=1)
  assert T.tower_state(env, asset_cfg).height.tolist() == [1]

  env, asset_cfg = _env(rows, moving=[False, True, False, False])
  assert T.tower_state(env, asset_cfg).height.tolist() == [1]

  # Let go over the same cube and it counts again.
  env, asset_cfg = _env(rows, ee=_level(1))
  assert T.tower_state(env, asset_cfg).height.tolist() == [2]


def test_a_tipped_cube_does_not_claim_its_level() -> None:
  from mjlab.managers import SceneEntityCfg

  env, asset_cfg = _env([[_level(0), _level(1), _loose(2), _loose(3)]])
  tipped = math.radians(45.0)
  half = torch.tensor([tipped / 2])
  # Roll about x rather than yaw about z: no face is down any more.
  env.scene[T.CUBE_NAMES[1]].data.root_link_quat_w = torch.stack(
    [torch.cos(half), torch.sin(half), torch.zeros(1), torch.zeros(1)], dim=-1
  )
  assert isinstance(asset_cfg, SceneEntityCfg)
  assert T.tower_state(env, asset_cfg).height.tolist() == [1]


def test_yaw_never_decides_whether_a_cube_is_stacked() -> None:
  """Cube orientation about z is free; the task encodes no order at all."""
  yaws = torch.tensor([[0.0, 1.1, -2.4, 0.7]])
  env, asset_cfg = _env(
    [[_level(0), _level(1), _level(2), _level(3)]], yaws=yaws
  )
  assert T.tower_state(env, asset_cfg).height.tolist() == [4]


def test_the_target_cube_is_the_held_one_then_the_nearest() -> None:
  loose_near = [0.24, -0.16, T.CUBE_HALF]
  loose_far = [0.44, 0.16, T.CUBE_HALF]
  rows = [[_level(0), loose_near, loose_far, _loose(3)]]

  env, asset_cfg = _env(rows, ee=[0.25, -0.16, 0.05])
  assert T.tower_state(env, asset_cfg).target.tolist() == [1]

  # Gripper closed on the far cube: it stays the target from across the table.
  env, asset_cfg = _env(rows, ee=loose_far, held=2)
  assert T.tower_state(env, asset_cfg).target.tolist() == [2]


# --- reward -----------------------------------------------------------------


def _reward(env, asset_cfg, *, arm=(0.0,) * 6):
  from mjlab.managers import SceneEntityCfg

  from vbrl.tasks.stack_cubes.mdp.rewards import stack_progress

  arm_cfg = SceneEntityCfg("robot")
  arm_cfg.joint_ids = list(range(6))
  gripper_cfg = SceneEntityCfg("robot")
  gripper_cfg.joint_ids = [6]
  return stack_progress(env, asset_cfg, arm_cfg, gripper_cfg, arm)


def test_reward_is_batched_finite_and_roughly_normalized() -> None:
  positions = [
    [_level(0), [0.24, -0.16, T.CUBE_HALF], _loose(2), _loose(3)],
    [_level(0), _level(1), _level(2), _level(3)],
    [[0.24, 0.16, T.CUBE_HALF], [0.44, 0.16, T.CUBE_HALF], _loose(2), _loose(3)],
  ]
  env, asset_cfg = _env(positions)
  reward = _reward(env, asset_cfg)

  assert reward.shape == (3,)
  assert torch.isfinite(reward).all()
  assert float(reward.min()) >= 0.0
  assert float(reward.max()) <= 1.0


def test_each_completed_level_raises_the_reward() -> None:
  """Monotone in tower height, at a fixed active cube count."""
  loose = [0.44, 0.16, T.CUBE_HALF]
  rows = [
    [loose, [0.24, 0.16, T.CUBE_HALF], [0.24, -0.16, T.CUBE_HALF], _loose(3)],
    [_level(0), [0.24, 0.16, T.CUBE_HALF], [0.24, -0.16, T.CUBE_HALF], _loose(3)],
    [_level(0), _level(1), [0.24, -0.16, T.CUBE_HALF], _loose(3)],
    [_level(0), _level(1), _level(2), _loose(3)],
  ]
  env, asset_cfg = _env(rows, ee=[1.0, 1.0, 1.0])
  reward = _reward(env, asset_cfg)
  assert reward[0] < reward[1] < reward[2] < reward[3]


def test_releasing_a_placed_cube_beats_holding_it_there() -> None:
  """The last stage is release *and* settle, not arrival."""
  placed = [[_level(0), _level(1), _loose(2), _loose(3)]]

  holding, asset_cfg = _env(placed, ee=_level(1), held=1)
  released, _ = _env(placed, ee=_level(1))
  assert float(_reward(holding, asset_cfg)) < float(_reward(released, asset_cfg))


def test_a_complete_tower_pays_more_at_the_observation_pose() -> None:
  """0.9 for the tower; the last tenth buys withdrawing and opening up."""
  from vbrl.tasks.stack_cubes.stack_cubes_env_cfg import OBSERVATION_JOINT_POS

  home = tuple(OBSERVATION_JOINT_POS[f"joint_{index}"] for index in range(6))
  complete = [[_level(0), _level(1), _level(2), _level(3)]]

  at_home, asset_cfg = _env(complete, arm=home)
  away, _ = _env(complete, arm=(0.0,) * 6)
  assert float(_reward(at_home, asset_cfg, arm=home)) == pytest.approx(1.0, abs=1e-3)
  assert 0.9 <= float(_reward(away, asset_cfg, arm=home)) < 1.0


def test_breaking_the_tower_removes_the_home_bonus() -> None:
  """The home reward is gated on completeness, so it vanishes by itself."""
  from vbrl.tasks.stack_cubes.stack_cubes_env_cfg import OBSERVATION_JOINT_POS

  home = tuple(OBSERVATION_JOINT_POS[f"joint_{index}"] for index in range(6))
  complete, asset_cfg = _env(
    [[_level(0), _level(1), _level(2), _level(3)]], arm=home
  )
  broken, _ = _env(
    [[_level(0), _level(1), _level(2), _loose(3)]], arm=home
  )
  assert float(_reward(complete, asset_cfg, arm=home)) > 0.9
  assert float(_reward(broken, asset_cfg, arm=home)) < 0.9


def test_every_completed_course_is_worth_the_same() -> None:
  """0.9 split four ways, so no course is worth more than another."""
  rows = []
  for built in range(T.MAX_CUBES):
    row = [_level(index) for index in range(built)]
    row += [_loose(index) for index in range(built, T.MAX_CUBES)]
    rows.append(row)
  reward = _reward(*_env(rows, ee=(1.0, 1.0, 1.0)))
  steps = reward[1:] - reward[:-1]
  assert torch.allclose(steps, steps[0].expand(len(steps)), atol=1e-6)
  assert float(steps[0]) == pytest.approx(0.9 / T.MAX_CUBES, abs=1e-6)


# --- reset pool -------------------------------------------------------------


class _Recorder:
  """A cube stand-in that keeps whatever an event writes to it."""

  def __init__(self, count: int):
    self.pose = torch.zeros(count, 7)
    self.data = SimpleNamespace(
      root_link_pos_w=self.pose[:, :3],
      root_link_quat_w=self.pose[:, 3:],
      root_link_lin_vel_w=torch.zeros(count, 3),
      root_link_ang_vel_w=torch.zeros(count, 3),
    )

  def write_root_link_pose_to_sim(self, pose, env_ids=None):
    self.pose[env_ids] = pose
    self.data.root_link_pos_w = self.pose[:, :3]
    self.data.root_link_quat_w = self.pose[:, 3:]

  def write_root_link_velocity_to_sim(self, velocity, env_ids=None):
    del velocity, env_ids


def _drawn_scenarios(count: int, ee=None):
  """Run one scenario draw against a recording scene and return cube positions.

  ``ee`` installs a robot whose gripper is at that point, which is what a
  mid-episode redraw has to work around; ``None`` is the episode-reset case,
  where the arm is on its way back to the start pose and blocks nothing.
  """
  from mjlab.managers import SceneEntityCfg

  from vbrl.tasks.stack_cubes.mdp.events import reset_stack_scenario

  scene = _Scene({name: _Recorder(count) for name in T.CUBE_NAMES})
  scene.env_origins = torch.zeros(count, 3)
  asset_cfg = None
  if ee is not None:
    scene["robot"] = SimpleNamespace(
      data=SimpleNamespace(
        site_pos_w=torch.as_tensor(ee, dtype=torch.float32).reshape(-1, 1, 3),
        joint_pos=torch.zeros(count, 7),
      )
    )
    asset_cfg = SceneEntityCfg("robot")
    asset_cfg.site_ids = [0]
    asset_cfg.joint_ids = [6]
  env = SimpleNamespace(scene=scene, device="cpu", num_envs=count)
  reset_stack_scenario(env, torch.arange(count), asset_cfg=asset_cfg)
  return torch.stack([scene[name].pose[:, :3] for name in T.CUBE_NAMES], dim=1)


def test_the_reset_pool_starts_empty_most_often_and_tapers() -> None:
  """An empty table is the normal start; each further course is half as likely.

  All four cubes are always on the table -- what varies is how much of the
  tower is already built, from nothing to finished.
  """
  torch.manual_seed(0)
  positions = _drawn_scenarios(8192)
  stacked = _in_column(positions).sum(dim=1)

  share = [float((stacked == k).float().mean()) for k in range(T.MAX_CUBES + 1)]
  assert set(stacked.tolist()) == {0, 1, 2, 3, 4}
  # An empty table is the modal case, by a clear margin.
  assert share[0] > 0.45
  # And every further pre-built course is rarer than the one below it.
  assert all(share[k] > share[k + 1] for k in range(T.MAX_CUBES))
  # A finished tower stays the rare case it is configured to be.
  assert 0.01 < share[T.MAX_CUBES] < 0.08


def test_no_reset_ever_overlaps_two_cubes() -> None:
  """Uniform draws, repaired until every cube clears its neighbours."""
  torch.manual_seed(1)
  positions = _drawn_scenarios(4096)
  separation = torch.cdist(positions, positions)
  separation += torch.eye(T.MAX_CUBES) * 9.0
  # Cubes sharing the stack column are separated in z by construction.
  column = (
    torch.linalg.vector_norm(
      positions[..., :2] - positions.new_tensor(T.STACK_XY), dim=-1
    )
    < T.STACK_XY_TOL
  )
  tower_pair = column.unsqueeze(1) & column.unsqueeze(2)
  loose = torch.where(tower_pair, torch.full_like(separation, 9.0), separation)
  # The face diagonal of a 40 mm cube: closer than this and two boxes can touch.
  assert float(loose.min()) > T.CUBE_SIZE * math.sqrt(2.0)


def test_pre_stacked_towers_are_built_at_the_expected_levels() -> None:
  torch.manual_seed(2)
  positions = _drawn_scenarios(512)
  column = (
    torch.linalg.vector_norm(
      positions[..., :2] - positions.new_tensor(T.STACK_XY), dim=-1
    )
    < T.STACK_XY_TOL
  )
  heights = positions[..., 2][column]
  levels = (heights - T.CUBE_HALF) / T.CUBE_SIZE
  assert torch.allclose(levels, levels.round(), atol=1e-5)
  assert float(levels.max()) == pytest.approx(T.MAX_CUBES - 1, abs=1e-4)


def test_loose_cubes_are_spread_over_the_whole_workspace() -> None:
  """Uniform, not a handful of slots: where a cube can be is what a visual
  policy has to generalize over."""
  torch.manual_seed(7)
  positions = _drawn_scenarios(4096)
  on_table = positions[..., 2] < T.CUBE_HALF + 1e-4
  loose = positions[..., :2][on_table]

  assert float(loose[:, 0].min()) >= T.WORKSPACE_X[0]
  assert float(loose[:, 0].max()) <= T.WORKSPACE_X[1]
  assert float(loose[:, 1].min()) >= T.WORKSPACE_Y[0]
  assert float(loose[:, 1].max()) <= T.WORKSPACE_Y[1]
  # Both axes reach into the outer tenth of the rectangle at both ends, which
  # no fixed set of slot centres would.
  span_x = T.WORKSPACE_X[1] - T.WORKSPACE_X[0]
  span_y = T.WORKSPACE_Y[1] - T.WORKSPACE_Y[0]
  assert float(loose[:, 0].min()) < T.WORKSPACE_X[0] + 0.1 * span_x
  assert float(loose[:, 0].max()) > T.WORKSPACE_X[1] - 0.1 * span_x
  assert float(loose[:, 1].min()) < T.WORKSPACE_Y[0] + 0.1 * span_y
  assert float(loose[:, 1].max()) > T.WORKSPACE_Y[1] - 0.1 * span_y
  # ...but never past the arm's own reach, which clips the rectangle's corners.
  assert float(torch.linalg.vector_norm(loose, dim=-1).max()) <= T.REACH_MAX


def test_a_mid_episode_redraw_keeps_cubes_out_of_the_gripper() -> None:
  """The arm is not moved, so the cubes are drawn around it instead."""
  torch.manual_seed(8)
  count = 2048
  hand = torch.tensor([0.42, 0.12, 0.05]).expand(count, 3)
  positions = _drawn_scenarios(count, ee=hand)
  on_table = positions[..., 2] < T.CUBE_HALF + 1e-4
  distance = torch.linalg.vector_norm(
    positions[..., :2] - hand[:, None, :2], dim=-1
  )
  assert float(distance[on_table].min()) >= T.GRIPPER_KEEPOUT

  # Lift the same hand clear of the table and it stops blocking ground.
  torch.manual_seed(8)
  high = torch.tensor([0.42, 0.12, 0.30]).expand(count, 3)
  positions = _drawn_scenarios(count, ee=high)
  on_table = positions[..., 2] < T.CUBE_HALF + 1e-4
  distance = torch.linalg.vector_norm(
    positions[..., :2] - high[:, None, :2], dim=-1
  )
  assert float(distance[on_table].min()) < T.GRIPPER_KEEPOUT


def test_a_redraw_is_skipped_where_the_hand_is_standing_in_the_tower() -> None:
  """The column cannot dodge, so that env waits for the next firing."""
  torch.manual_seed(9)
  inside = torch.tensor([T.STACK_XY[0], T.STACK_XY[1], 0.10]).expand(64, 3)
  positions = _drawn_scenarios(64, ee=inside)
  # Nothing was written, so every cube is still at the recorder's zero pose.
  assert torch.count_nonzero(positions) == 0

  clear = torch.tensor([T.STACK_XY[0], T.STACK_XY[1], 0.30]).expand(64, 3)
  assert torch.count_nonzero(_drawn_scenarios(64, ee=clear)) > 0


def _in_column(positions: torch.Tensor) -> torch.Tensor:
  return (
    torch.linalg.vector_norm(
      positions[..., :2] - positions.new_tensor(T.STACK_XY), dim=-1
    )
    < T.STACK_XY_TOL
  )


def test_loose_cubes_never_spawn_inside_the_stack_target() -> None:
  torch.manual_seed(3)
  positions = _drawn_scenarios(2048)
  on_table = positions[..., 2] < T.CUBE_HALF + 1e-4
  distance = torch.linalg.vector_norm(
    positions[..., :2] - positions.new_tensor(T.STACK_XY), dim=-1
  )
  loose = on_table & (distance > T.STACK_XY_TOL)
  assert float(distance[loose].min()) >= T.STACK_KEEPOUT


# --- environment contracts --------------------------------------------------


def _cfg(task_id: str, play: bool = False):
  from mjlab.tasks.registry import load_env_cfg

  return load_env_cfg(task_id, play=play)


def test_the_state_task_renders_nothing_and_randomizes_no_appearance() -> None:
  """A state run must not pay for lighting, material or camera DR."""
  cfg = _cfg(STACK_CUBES_IDS[0])
  assert "object_color" not in cfg.events
  assert "camera" not in cfg.observations
  assert not [s for s in (cfg.scene.sensors or ()) if hasattr(s, "camera_name")]
  assert not [name for name in cfg.events if name.startswith(("light_", "camera_"))]
  assert not [name for name in cfg.events if "color" in name or "material" in name]
  # Physical randomization stays.
  assert "fingertip_friction_slide" in cfg.events


@pytest.mark.parametrize(
  ("task_id", "sensor"),
  [(STACK_CUBES_IDS[1], "external_cam"), (STACK_CUBES_IDS[2], "cam")],
)
def test_each_visual_task_carries_exactly_one_camera(task_id, sensor) -> None:
  from mjlab.sensor import CameraSensorCfg

  cfg = _cfg(task_id)
  cameras = [s for s in cfg.scene.sensors if isinstance(s, CameraSensorCfg)]
  assert [c.name for c in cameras] == [sensor]
  assert (cameras[0].height, cameras[0].width) == (224, 224)
  assert tuple(cfg.observations["camera"].terms) == (f"{sensor}_rgb",)


def test_the_visual_actor_gets_no_privileged_state_and_the_critic_does() -> None:
  for task_id in STACK_CUBES_IDS[1:]:
    cfg = _cfg(task_id)
    assert tuple(cfg.observations["actor"].terms) == PROPRIOCEPTION
    for term in PRIVILEGED:
      assert term in cfg.observations["critic"].terms, task_id


def test_the_state_actor_is_privileged_and_matches_its_critic() -> None:
  cfg = _cfg(STACK_CUBES_IDS[0])
  assert tuple(cfg.observations["actor"].terms) == tuple(
    cfg.observations["critic"].terms
  )
  for term in PRIVILEGED:
    assert term in cfg.observations["actor"].terms


def test_both_visual_tasks_use_the_one_registered_scratch_architecture() -> None:
  from mjlab.tasks.registry import load_rl_cfg

  from vbrl.vision.architectures import ARCHITECTURES

  expected = ARCHITECTURES["NatureCnn-SpatialSoftmax"].asdict()
  for task_id in STACK_CUBES_IDS[1:]:
    agent = load_rl_cfg(task_id)
    assert agent.actor.cnn_cfg is not None
    assert agent.actor.cnn_cfg["vision"] == expected
    assert agent.obs_groups["actor"] == ("actor", "camera")
  assert load_rl_cfg(STACK_CUBES_IDS[0]).actor.cnn_cfg is None


def test_every_cube_shares_one_saturated_colour() -> None:
  """Cube identity means nothing here, so the cubes look identical.

  Four distinguishable colours would hand the policy a feature it could use to
  encode an order the task does not have. The scene preset's own per-object
  draw is therefore off, and one event colours all four together.
  """
  cfg = _cfg(STACK_CUBES_IDS[1])
  assert set(cfg.scene.entities) == {"robot", "table", *T.CUBE_NAMES}
  assert [name for name in cfg.events if name.endswith("_color")] == ["object_color"]
  assert cfg.events["object_color"].func.__name__ == "randomize_cube_colour"
  # The table bank is untouched by turning the object draw off.
  assert "table_material" in cfg.events


def test_the_shared_cube_colour_is_always_saturated_and_bright() -> None:
  """Uniform RGB is what put a quarter of Push-T's objects at the tabletop's
  own luminance; with one shared colour that would hide every cube at once."""
  from vbrl.tasks.stack_cubes.mdp.events import (
    CUBE_COLOUR_CEILING,
    CUBE_COLOUR_FLOOR,
    randomize_cube_colour,
  )

  class _Model:
    geom_rgba = torch.ones(4096, 8, 4)

  scene = _Scene()
  for index, name in enumerate(T.CUBE_NAMES):
    scene[name] = SimpleNamespace(
      indexing=SimpleNamespace(geom_ids=torch.tensor([2 * index, 2 * index + 1]))
    )
  scene.env_origins = torch.zeros(4096, 3)
  env = SimpleNamespace(
    scene=scene, device="cpu", num_envs=4096, sim=SimpleNamespace(model=_Model())
  )
  torch.manual_seed(0)
  randomize_cube_colour(env, torch.arange(4096))

  rgb = _Model.geom_rgba[:, :, :3]
  # Every cube in an env carries the same colour...
  assert torch.allclose(rgb, rgb[:, :1].expand_as(rgb))
  # ...and every colour is fully saturated between the floor and the ceiling,
  # so none can come out as a table-coloured grey.
  assert rgb.amin(dim=(1, 2)).tolist() == pytest.approx([CUBE_COLOUR_FLOOR] * 4096)
  assert rgb.amax(dim=(1, 2)).tolist() == pytest.approx([CUBE_COLOUR_CEILING] * 4096)
  # Hue still covers the circle: all three channels take a turn being brightest.
  brightest = rgb[:, 0].argmax(dim=-1)
  assert set(brightest.tolist()) == {0, 1, 2}


def test_a_finished_tower_is_not_a_termination() -> None:
  """Only timeout, an unrecoverable cube, and a table collision end an episode."""
  for task_id in STACK_CUBES_IDS:
    assert set(_cfg(task_id).terminations) == {
      "time_out",
      "cube_out_of_reach",
      "ee_table_contact",
    }


def test_a_cube_inside_the_workspace_never_triggers_a_reset() -> None:
  from vbrl.tasks.stack_cubes.mdp.terminations import cube_out_of_reach

  env, asset_cfg = _env(
    [
      # a finished tower, and cubes at the edge of the spawn region
      [_level(0), _level(1), _level(2), _level(3)],
      [[0.24, -0.16, T.CUBE_HALF], [0.44, 0.16, T.CUBE_HALF], _loose(2), _loose(3)],
      # nudged just outside the workspace: recoverable, keep going
      [[0.52, 0.0, T.CUBE_HALF], _loose(1), _loose(2), _loose(3)],
      # shoved to the far end of a tabletop the arm cannot reach across
      [[0.9, 0.0, T.CUBE_HALF], _loose(1), _loose(2), _loose(3)],
    ]
  )
  assert cube_out_of_reach(env, asset_cfg).tolist() == [False, False, False, True]


def test_the_play_environments_drop_noise_and_the_curriculum() -> None:
  for task_id in STACK_CUBES_IDS:
    play = _cfg(task_id, play=True)
    assert play.observations["actor"].enable_corruption is False
    assert play.curriculum == {}
    assert play.scene.num_envs == 1
    assert _cfg(task_id).scene.num_envs == 1024


def test_stack_cubes_reuses_lift_cubes_safety_terms_at_upstream_weights() -> None:
  cfg = _cfg(STACK_CUBES_IDS[0])
  assert set(cfg.rewards) == {
    "stack_progress",
    "action_rate_l2",
    "joint_pos_limits",
    "joint_vel_hinge",
  }
  assert cfg.rewards["stack_progress"].weight == pytest.approx(1.0)
  assert cfg.rewards["action_rate_l2"].weight == pytest.approx(-0.01)
  assert cfg.rewards["joint_pos_limits"].weight == pytest.approx(-10.0)
  assert cfg.rewards["joint_vel_hinge"].weight == pytest.approx(-0.01)
  assert "joint_vel_hinge_weight" in cfg.curriculum


def test_the_task_does_not_import_the_lift_cube_package() -> None:
  """Stack-Cubes copies Lift-Cube's ideas, never its module tree."""
  from pathlib import Path

  root = Path("src/vbrl/tasks/stack_cubes")
  for path in root.rglob("*.py"):
    assert "tasks.lift_cube" not in path.read_text(encoding="utf-8"), path


def test_the_mid_episode_disturbance_runs_on_a_native_interval_event() -> None:
  from vbrl.tasks.stack_cubes import mdp

  cfg = _cfg(STACK_CUBES_IDS[0])
  assert cfg.events["stack_scenario"].mode == "reset"
  for name in ("collapse_tower",):
    assert cfg.events[name].mode == "interval"
    lower, upper = cfg.events[name].interval_range_s
    # Exactly one firing per episode. MJLab resamples the timer on every reset,
    # so the first lands in [lower, upper] and a second could only land at or
    # after 2*lower; this is the inequality that makes one certain and two
    # impossible. A policy disturbed twice never holds the observation pose
    # long enough for the home reward to be worth anything.
    assert upper <= cfg.episode_length_s < 2.0 * lower
  # The arm is never moved by a mid-episode disturbance -- the episode has not
  # ended. It is passed in so the draw can work around the hand instead.
  assert "asset_cfg" in cfg.events["collapse_tower"].params
  assert "asset_cfg" not in cfg.events["stack_scenario"].params
  for name in cfg.events:
    assert "reset_robot" not in cfg.events[name].params
  # The reset pool is an episode reset only: mid-episode nothing redraws the
  # whole scene, so a tower the policy built is never replaced wholesale.
  redraws = [
    name
    for name, event in cfg.events.items()
    if event.func is mdp.reset_stack_scenario
  ]
  assert redraws == ["stack_scenario"]
  assert cfg.events["stack_scenario"].mode == "reset"


# --- persistent-task disturbances -------------------------------------------


def _disturb(event, positions, *, ee, arm=(0.0,) * 6, **kwargs):
  """Run one interval disturbance against a recording scene and return poses."""
  from mjlab.managers import SceneEntityCfg

  cubes = torch.as_tensor(positions, dtype=torch.float32)
  count = cubes.shape[0]
  scene = _Scene({name: _Recorder(count) for name in T.CUBE_NAMES})
  for index, name in enumerate(T.CUBE_NAMES):
    scene[name].pose[:, :3] = cubes[:, index]
    scene[name].pose[:, 3] = 1.0
    scene[name].data.root_link_pos_w = scene[name].pose[:, :3]
    scene[name].data.root_link_quat_w = scene[name].pose[:, 3:]
  scene["robot"] = SimpleNamespace(
    data=SimpleNamespace(
      site_pos_w=torch.tensor([ee], dtype=torch.float32).expand(count, 1, 3),
      joint_pos=torch.tensor(
        [[*arm, T.GRIPPER_OPEN_M]], dtype=torch.float32
      ).expand(count, 7),
    )
  )
  for sensor in T.CONTACT_SENSORS:
    scene[sensor] = _grasp_sensor(count, False)
  scene.env_origins = torch.zeros(count, 3)
  env = SimpleNamespace(scene=scene, device="cpu", num_envs=count)
  asset_cfg = SceneEntityCfg("robot")
  asset_cfg.site_ids = [0]
  asset_cfg.joint_ids = [6]
  event(env, torch.arange(count), asset_cfg=asset_cfg, **kwargs)
  return torch.stack([scene[name].pose[:, :3] for name in T.CUBE_NAMES], dim=1)


def test_a_collapse_takes_the_top_off_and_leaves_the_rest_standing() -> None:
  """The disturbance the real rig presents: part of the stack comes down.

  What must *not* happen is the whole scene being redrawn -- the policy keeps
  everything below the break, so the episode stays continuous and the reward
  falls by exactly the courses that fell.
  """
  from vbrl.tasks.stack_cubes.mdp.events import collapse_tower

  torch.manual_seed(11)
  before = [[_level(0), _level(1), _level(2), _level(3)]] * 512
  after = _disturb(
    collapse_tower, before, ee=[0.05, 0.0, 0.4], max_falling=3
  )

  in_column = (
    torch.linalg.vector_norm(
      after[..., :2] - after.new_tensor(T.STACK_XY), dim=-1
    )
    < T.STACK_XY_TOL
  )
  fallen = (~in_column).sum(dim=1)

  # One to three courses come off a four-cube tower, never none and never all.
  assert set(fallen.tolist()) == {1, 2, 3}
  # Whatever stayed is still at its exact original level, untouched.
  for env in range(len(after)):
    kept = T.MAX_CUBES - int(fallen[env])
    standing = after[env][in_column[env]]
    heights = sorted(float(z) for z in standing[:, 2])
    assert heights == pytest.approx(
      [T.level_height(index) for index in range(kept)]
    ), env
  # And the cubes that fell are lying flat on the table, clear of the column.
  assert after[..., 2][~in_column].tolist() == pytest.approx(
    [T.CUBE_HALF] * int((~in_column).sum())
  )


def test_a_collapse_never_drops_a_cube_onto_another() -> None:
  """Falling courses land clear of each other and of what is already down."""
  from vbrl.tasks.stack_cubes.mdp.events import collapse_tower

  torch.manual_seed(12)
  loose = [0.24, 0.16, T.CUBE_HALF]
  before = [[_level(0), _level(1), _level(2), loose]] * 512
  after = _disturb(collapse_tower, before, ee=[0.05, 0.0, 0.4], max_falling=3)

  on_table = after[..., 2] < T.CUBE_HALF + 1e-4
  gap = torch.cdist(after[..., :2], after[..., :2]) + torch.eye(T.MAX_CUBES) * 9.0
  pair = on_table.unsqueeze(1) & on_table.unsqueeze(2)
  assert float(torch.where(pair, gap, torch.full_like(gap, 9.0)).min()) >= (
    T.MIN_CUBE_SEPARATION
  )


def test_a_tower_that_does_not_exist_cannot_collapse() -> None:
  from vbrl.tasks.stack_cubes.mdp.events import collapse_tower

  torch.manual_seed(13)
  before = [[[0.24, 0.16, T.CUBE_HALF], _loose(1), _loose(2), _loose(3)]]
  after = _disturb(collapse_tower, before, ee=[0.05, 0.0, 0.4], max_falling=3)
  assert torch.allclose(after, torch.tensor(before))


def test_a_disturbance_never_drops_a_cube_into_the_gripper() -> None:
  """The dislodged cubes are drawn from the same sampler a reset uses."""
  from vbrl.tasks.stack_cubes.mdp.events import collapse_tower

  torch.manual_seed(6)
  hand = [0.30, -0.10, 0.06]
  rows = [[_level(0), _level(1), _level(2), _loose(3)]] * 256
  after = _disturb(collapse_tower, rows, ee=hand, max_falling=1)
  moved = after[:, 2, :2]
  to_hand = torch.linalg.vector_norm(moved - torch.tensor(hand[:2]), dim=-1)
  to_stack = torch.linalg.vector_norm(moved - torch.tensor(T.STACK_XY), dim=-1)
  assert float(to_hand.min()) >= T.GRIPPER_KEEPOUT
  assert float(to_stack.min()) >= T.STACK_KEEPOUT


def test_an_ood_scene_replacement_re_dresses_every_cube() -> None:
  """Stack-Cubes is the first task with more than one manipulated object, and
  `replace_scene` used to refuse it outright. A replacement must also leave no
  per-cube appearance event behind from the preset it replaced."""
  from mjlab.tasks.registry import load_env_cfg

  from vbrl.scenes.builder import replace_scene

  cfg = load_env_cfg(STACK_CUBES_IDS[1], play=True)
  assert "object_color" in cfg.events

  replace_scene(cfg, scene="wood")
  assert set(cfg.scene.entities) == {"robot", "table", *T.CUBE_NAMES}
  stale = [
    name
    for name in cfg.events
    if "color" in name or "material" in name or name.startswith(("light_", "camera_"))
  ]
  assert stale == []


# --- the reward's goal is the task's goal -----------------------------------


def test_the_place_target_is_exactly_where_a_cube_claims_its_level() -> None:
  """The one consistency that silently ruins a task if it is wrong.

  The reward drives the current cube at ``stack_point(h)``; the tower counts a
  cube only when it satisfies ``at_level``. If those two disagreed by so much
  as a tolerance the policy would be pulled at a spot that never scores, and
  the failure would look like bad exploration rather than a bug. Put a cube
  exactly on the reward's target, release it, and the tower must grow.
  """
  from vbrl.tasks.stack_cubes.mdp.tower import stack_point

  for built in range(T.MAX_CUBES):
    rows = [_level(index) for index in range(built)]
    rows += [_loose(index) for index in range(built, T.MAX_CUBES)]
    before, asset_cfg = _env([rows])
    state = T.tower_state(before, asset_cfg)
    assert state.height.tolist() == [built]

    # Where the reward says the next cube belongs...
    target = stack_point(state.height)
    assert target[0].tolist() == pytest.approx(
      [*T.STACK_XY, T.level_height(built)]
    )

    # ...put it exactly there, hands off.
    placed = list(rows)
    placed[int(state.target)] = target[0].tolist()
    after, _ = _env([placed])

    # The tower grew by one, and the task paid for it.
    assert T.tower_state(after, asset_cfg).height.tolist() == [built + 1], built
    assert T.tower_state(after, asset_cfg).complete.tolist() == [
      built + 1 == T.MAX_CUBES
    ]
    assert float(_reward(after, asset_cfg)) > float(_reward(before, asset_cfg))


def test_arriving_at_the_target_still_held_is_not_a_completed_placement() -> None:
  """Reaching the spot is the fourth band; letting go and settling is the last.

  Holding a cube perfectly on target must not count, or the policy learns to
  hover on the tower forever instead of building it.
  """
  from vbrl.tasks.stack_cubes.mdp.rewards import stage_scalar
  from vbrl.tasks.stack_cubes.mdp.tower import stack_point

  rows = [_level(0), _loose(1), _loose(2), _loose(3)]
  target = stack_point(torch.tensor([1]))[0].tolist()
  arrived = list(rows)
  arrived[1] = target

  held, asset_cfg = _env([arrived], ee=target, held=1)
  released, _ = _env([arrived], ee=target)

  # On target but still in the hand: the fourth band, and no new course.
  assert 0.8 <= float(stage_scalar(T.tower_state(held, asset_cfg))) < 1.0
  assert T.tower_state(held, asset_cfg).height.tolist() == [1]
  # Opening the hand is what completes it, and it pays more.
  assert T.tower_state(released, asset_cfg).height.tolist() == [2]
  assert float(_reward(released, asset_cfg)) > float(_reward(held, asset_cfg))


def test_the_hover_point_sits_one_cube_above_the_place_point() -> None:
  """Transport clearance is derived from the cube, not chosen."""
  from vbrl.tasks.stack_cubes.mdp.rewards import HOVER_CLEARANCE
  from vbrl.tasks.stack_cubes.mdp.tower import stack_point

  for built in range(T.MAX_CUBES):
    place = stack_point(torch.tensor([built]))[0]
    assert HOVER_CLEARANCE == T.CUBE_SIZE
    # The hover pose clears the tower's own top face by a whole cube.
    assert float(place[2]) + HOVER_CLEARANCE > T.level_height(built) + T.CUBE_HALF


def test_the_task_reward_never_falls_along_a_successful_placement() -> None:
  """Walk an idealised pick-and-place and check the reward only goes up.

  The stage scalar alone is *not* monotone here and should not be: the moment a
  cube seats, the tower grows and the target moves to the next cube, so ``s``
  resets to the reaching band while ``h`` absorbs the finished course. What has
  to hold is that the sum never goes backwards -- otherwise there is a moment
  in every successful placement that the policy is paid to avoid.
  """
  from vbrl.tasks.stack_cubes.mdp.rewards import HOVER_CLEARANCE

  def lerp(a, b, t):
    return [x + (y - x) * t for x, y in zip(a, b, strict=True)]

  start = [0.40, -0.14, T.CUBE_HALF]
  place = _level(1)
  hover = [place[0], place[1], place[2] + HOVER_CLEARANCE]
  lifted = [start[0], start[1], T.CUBE_HALF + T.CUBE_SIZE]
  others = [_level(0), None, [0.24, 0.15, T.CUBE_HALF], [0.42, 0.12, T.CUBE_HALF]]

  # approach, grasp, lift, carry to the hover pose, lower onto the tower, let go
  legs = (
    ([1.0, 1.0, 0.6], start, start, start, False),
    (start, start, start, start, True),
    (start, lifted, start, lifted, True),
    (lifted, hover, lifted, hover, True),
    (hover, place, hover, place, True),
    (place, place, place, place, False),
  )
  rewards = []
  for ee_from, ee_to, cube_from, cube_to, closed in legs:
    for step in range(9):
      t = step / 8
      cube = lerp(cube_from, cube_to, t)
      rows = [others[0], cube, others[2], others[3]]
      env, asset_cfg = _env(
        [rows],
        ee=lerp(ee_from, ee_to, t),
        held=1 if closed else None,
      )
      rewards.append(float(_reward(env, asset_cfg)))

  trace = torch.tensor(rewards)
  worst = float((trace[1:] - trace[:-1]).min())
  assert worst >= -1e-3, f"reward falls by {-worst:.4f} mid-placement"
  # And the placement actually paid: one of four courses is worth 0.9/4.
  assert trace[-1] - trace[0] > 0.9 / T.MAX_CUBES


def test_an_empty_closed_gripper_does_not_strike_a_cube_off_the_tower() -> None:
  """Why the grasp test reads contact instead of distance.

  A closed but empty hand withdrawing past the tower sits well inside any
  sensible grasp radius of the course it passes. Under a distance test that
  course counted as "held", dropped out of the tower, and took a quarter of the
  reward with it for no reason at all. Both fingers have to actually touch.
  """
  tower = [[_level(0), _level(1), _level(2), _loose(3)]]
  # The hand is closed and 25 mm above the top course -- but touching nothing.
  passing = _env(tower, ee=[*T.STACK_XY, T.level_height(2) + 0.025])
  passing[0].scene["robot"].data.joint_pos = torch.tensor(
    [[0.0] * 6 + [T.GRIPPER_CLOSED_M - 1e-3]]
  )

  assert T.tower_state(*passing).height.tolist() == [3]
  assert T.tower_state(*passing).held.tolist() == [[False] * T.MAX_CUBES]

  # Close on it for real and it does stop counting, which is the behaviour the
  # held-test exists for in the first place.
  gripping, asset_cfg = _env(tower, ee=_level(2), held=2)
  assert T.tower_state(gripping, asset_cfg).height.tolist() == [2]


def test_grasp_force_is_reported_per_cube_for_the_safety_metric() -> None:
  """Without this a policy that slams cubes looks like one that places them."""
  from vbrl.tasks.stack_cubes.mdp.observations import peak_grasp_force

  env, _ = _env([[_level(0), _loose(1), _loose(2), _loose(3)]], held=1)
  forces = T.grasp_force(env)
  assert forces.shape == (1, T.MAX_CUBES)
  assert float(forces[0, 1]) > 0.0
  assert float(forces[0, 0]) == 0.0
  assert float(peak_grasp_force(env)) == pytest.approx(float(forces.max()))


# --- the command ------------------------------------------------------------


def test_the_command_publishes_the_contract_evaluation_needs() -> None:
  """`vbrl-evaluate` wants exactly one command with an `episode_success`."""
  from mjlab.tasks.registry import load_env_cfg

  from vbrl.tasks.stack_cubes.mdp.commands import StackCommandCfg

  for task_id in STACK_CUBES_IDS:
    cfg = load_env_cfg(task_id)
    assert list(cfg.commands) == ["stack_target"], task_id
    assert isinstance(cfg.commands["stack_target"], StackCommandCfg)
    # Fixed goal: the target follows the tower, never a resampling timer.
    assert min(cfg.commands["stack_target"].resampling_time_range) > 1.0e6


def test_the_command_target_climbs_with_the_tower() -> None:
  """It is a command precisely because it moves: one course per placement."""
  from vbrl.tasks.stack_cubes.mdp.tower import stack_point

  for built in range(T.MAX_CUBES + 1):
    rows = [_level(index) for index in range(built)]
    rows += [_loose(index) for index in range(built, T.MAX_CUBES)]
    state = T.tower_state(*_env([rows]))
    target = stack_point(state.height.clamp(max=T.MAX_CUBES - 1))
    assert float(target[0, 2]) == pytest.approx(
      T.level_height(min(built, T.MAX_CUBES - 1))
    )
    assert float(target[0, 0]) == pytest.approx(T.STACK_XY[0])


def test_the_command_does_not_duplicate_the_metric_terms() -> None:
  """Tower height and completeness are the command's; the rest are metrics."""
  cfg = _cfg(STACK_CUBES_IDS[0])
  assert set(cfg.metrics) == {
    "tower_height_peak",
    "home_reached",
    "peak_grasp_force",
    "reward_stage",
  }
