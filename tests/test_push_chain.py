from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

XML = "src/vbrl/asset_zoo/objects/push_chain.xml"
STATE_TASK_ID = "Mjlab-PushChain-State-Trossen"


def _model():
  import mujoco

  return mujoco.MjModel.from_xml_path(XML)


def test_geometry_constants_match_the_xml() -> None:
  from vbrl.tasks.push_chain import geometry as g

  model = _model()
  assert model.body("push_chain_link0").pos == pytest.approx((-g.LINK_SPACING, 0, 0))
  assert model.body("push_chain_link2").pos == pytest.approx((g.LINK_SPACING, 0, 0))
  assert model.joint("push_chain_hinge0").pos[0] == pytest.approx(
    g.END_LINK_HALF_LENGTH
  )
  assert model.joint("push_chain_hinge1").pos[0] == pytest.approx(
    -g.END_LINK_HALF_LENGTH
  )
  for hinge in ("push_chain_hinge0", "push_chain_hinge1"):
    assert model.joint(hinge).range == pytest.approx((-g.HINGE_LIMIT, g.HINGE_LIMIT))

  lower = model.geom("push_chain_link1_lower_collision")
  assert lower.size[:2] == pytest.approx((g.MIDDLE_LINK_HALF_LENGTH, g.LINK_HALF_WIDTH))
  ring = model.geom("push_chain_link1_ring_right_collision")
  assert ring.size[0] == pytest.approx(g.PIN_RADIUS)
  assert ring.pos[0] == pytest.approx(g.MIDDLE_LINK_HALF_LENGTH)
  for link, sign in ((0, 1.0), (2, -1.0)):
    body = model.geom(f"push_chain_link{link}_body_collision")
    assert body.size[1:] == pytest.approx((g.LINK_HALF_WIDTH, g.LINK_HALF_HEIGHT))
    free_end = body.pos[0] - sign * body.size[0]
    assert free_end == pytest.approx(-sign * g.END_LINK_HALF_LENGTH)
    tip = model.geom(f"push_chain_link{link}_plate_tip_collision")
    assert tip.size[0] == pytest.approx(g.PIN_RADIUS)
    assert tip.pos[0] == pytest.approx(sign * g.END_LINK_HALF_LENGTH)
  assert g.REST_HEIGHT - g.LINK_HALF_HEIGHT == pytest.approx(0.001)


def test_forward_kinematics_matches_mujoco() -> None:
  import mujoco

  from vbrl.tasks.push_chain.geometry import link_poses

  model = _model()
  data = mujoco.MjData(model)
  rng = np.random.default_rng(0)
  for _ in range(20):
    x, y, yaw = (
      rng.uniform(-0.3, 0.3),
      rng.uniform(-0.3, 0.3),
      rng.uniform(-np.pi, np.pi),
    )
    hinges = rng.uniform(-1.6, 1.6, 2)
    data.qpos[:7] = (x, y, 0.016, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
    data.qpos[7:] = hinges
    mujoco.mj_kinematics(model, data)
    links = link_poses(
      torch.tensor((x, y), dtype=torch.float64),
      torch.tensor(yaw, dtype=torch.float64),
      torch.tensor(hinges),
    ).numpy()
    for i in range(3):
      body = model.body(f"push_chain_link{i}").id
      w, _, _, z = data.xquat[body]
      assert links[i, :2] == pytest.approx(data.xpos[body, :2], abs=1e-6)
      error = math.remainder(links[i, 2] - 2 * math.atan2(z, w), 2 * math.pi)
      assert error == pytest.approx(0.0, abs=1e-6)


def test_validity_checks_include_the_pins() -> None:
  from vbrl.tasks.push_chain.geometry import (
    PIN_RADIUS,
    chain_signed_distance,
    end_links_gap,
    gap_to_boxes,
    link_poses,
    pin_positions,
  )

  straight = link_poses(torch.zeros(1, 2), torch.zeros(1), torch.zeros(1, 2))
  assert end_links_gap(straight).item() == pytest.approx(0.0875 - 2 * PIN_RADIUS)
  beyond_the_end = torch.tensor(((0.11375 + 0.015, 0.0, 0.005, 0.005),))
  assert gap_to_boxes(straight, beyond_the_end).item() == pytest.approx(0.01)

  bent = link_poses(
    torch.zeros(1, 2), torch.zeros(1), torch.tensor(((math.pi / 2, 0.0),))
  )
  outer_corner = pin_positions(bent)[:, 0] - 0.01
  assert chain_signed_distance(outer_corner, bent).item() == pytest.approx(0.01)
  with_pins = chain_signed_distance(outer_corner, bent, pins=True).item()
  assert with_pins == pytest.approx(math.hypot(0.01, 0.01) - PIN_RADIUS)


def _corners(x, y, yaw, half):
  c, s = math.cos(yaw), math.sin(yaw)
  hx, hy = half
  return [
    (x + c * px - s * py, y + s * px + c * py)
    for px, py in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))
  ]


def _side(p, a, b):
  return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _clip(subject, clip):
  for a, b in zip(clip, clip[1:] + clip[:1], strict=True):
    result = []
    for p, q in zip(subject, subject[1:] + subject[:1], strict=True):
      sp, sq = _side(p, a, b), _side(q, a, b)
      if (sp >= 0) != (sq >= 0):
        t = sp / (sp - sq)
        result.append((p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])))
      if sq >= 0:
        result.append(q)
    subject = result
  return subject


def _area(polygon):
  pairs = zip(polygon, polygon[1:] + polygon[:1], strict=True)
  return 0.5 * abs(sum(p[0] * q[1] - q[0] * p[1] for p, q in pairs))


def test_link_overlap_matches_exact_rectangle_intersection() -> None:
  from vbrl.tasks.push_chain.geometry import LINK_HALF_EXTENTS, LinkOverlap, link_poses

  rng = np.random.default_rng(1)
  n = 16
  goal = link_poses(
    torch.tensor(rng.uniform(0.2, 0.5, (n, 2)), dtype=torch.float),
    torch.tensor(rng.uniform(-np.pi, np.pi, n), dtype=torch.float),
    torch.tensor(rng.uniform(-1.4, 1.4, (n, 2)), dtype=torch.float),
  )
  chain = goal + torch.tensor(
    np.concatenate(
      (rng.uniform(-0.01, 0.01, (n, 3, 2)), rng.uniform(-0.2, 0.2, (n, 3, 1))), -1
    ),
    dtype=torch.float,
  )
  overlap = LinkOverlap(device="cpu", dtype=torch.float)(chain, goal)
  for env in range(n):
    for link, half in enumerate(LINK_HALF_EXTENTS):
      target = _corners(*goal[env, link].tolist(), half)
      piece = _corners(*chain[env, link].tolist(), half)
      exact = _area(_clip(target, piece)) / (4 * half[0] * half[1])
      assert overlap[env, link].item() == pytest.approx(exact, abs=0.02)


def _fake_command(goal_xy, goal_yaw, goal_hinges, chain_xy, chain_yaw, chain_hinges):
  from vbrl.tasks.push_chain.geometry import LinkOverlap, link_poses
  from vbrl.tasks.push_chain.mdp.commands import PushChainCommand, PushChainCommandCfg

  command = object.__new__(PushChainCommand)
  command.cfg = PushChainCommandCfg(
    entity_name="object",
    ee_site="ee_site",
    success_threshold=0.9,
    resampling_time_range=(1e9, 1e9),
  )
  command.target_pos = torch.cat((goal_xy, torch.full((len(goal_xy), 1), 0.015)), -1)
  command.target_yaw = goal_yaw
  command.target_hinges = goal_hinges
  command.target_links = link_poses(goal_xy, goal_yaw, goal_hinges)
  chain = link_poses(chain_xy, chain_yaw, chain_hinges)
  command.chain_links = lambda: chain
  command.chain_hinges = lambda: chain_hinges
  command.object = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.cat((chain_xy, torch.full((len(chain_xy), 1), 0.016)), -1)
    )
  )
  tcp = torch.tensor((0.6, 0.3, 0.2)).expand(len(goal_xy), 3)
  command._robot = SimpleNamespace(data=SimpleNamespace(site_pos_w=tcp[:, None]))
  command._ee_site = 0
  command._env = SimpleNamespace(
    common_step_counter=0, num_envs=len(goal_xy), device="cpu"
  )
  command._overlap = LinkOverlap(device="cpu", dtype=torch.float)
  command._overlap_cache_step = None
  command.metrics = {}
  command.episode_success = torch.zeros(len(goal_xy))
  return command


def _reward(command):
  from vbrl.tasks.push_chain.mdp.rewards import chain_dense_reward

  env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command))
  return chain_dense_reward(env, "goal", distance_scale=0.1)


def test_success_and_reward_accept_the_goal_in_either_order() -> None:
  xy = torch.tensor(((0.35, 0.05), (0.3, -0.1)))
  yaw = torch.tensor((0.4, -2.0))
  hinges = torch.tensor(((0.6, -1.2), (-0.3, 0.9)))

  forward = _fake_command(xy, yaw, hinges, xy, yaw, hinges)
  backward = _fake_command(xy, yaw + math.pi, -hinges.flip(-1), xy, yaw, hinges)
  for command in (forward, backward):
    assert command.get_overlap() == pytest.approx(torch.ones(2, 3))
    assert command.get_at_goal().all()
    assert command.landmark_distance() == pytest.approx(torch.zeros(2), abs=1e-6)
    assert command.goal_errors().abs().amax().item() == pytest.approx(0.0, abs=1e-5)
    assert ((_reward(command) > 2.0) & (_reward(command) < 2.05)).all()
  assert _reward(forward) == pytest.approx(_reward(backward))

  moved = torch.tensor(((0.02, 0.0), (0.0, 0.02)))
  near = _fake_command(xy, yaw, hinges, xy + moved, yaw, hinges)
  near_backward = _fake_command(
    xy, yaw + math.pi, -hinges.flip(-1), xy + moved, yaw, hinges
  )
  assert not near.get_at_goal().any()
  assert near.get_overlap() == pytest.approx(near_backward.get_overlap())
  assert _reward(near) == pytest.approx(_reward(near_backward))
  assert near.landmark_distance() == pytest.approx(torch.full((2,), 0.02))
  geometry = math.exp(-0.02 / 0.1)
  assert ((_reward(near) >= geometry) & (_reward(near) < geometry + 0.05)).all()
  assert near.goal_errors()[:, 0] == pytest.approx(torch.full((2,), 0.02))


def test_chain_overlap_covers_the_whole_goal_in_either_order() -> None:
  from vbrl.tasks.push_chain.geometry import (
    ChainOverlap,
    link_poses,
    reversed_links,
  )

  overlap = ChainOverlap(device="cpu", dtype=torch.float)
  bent = link_poses(
    torch.tensor(((0.3, 0.1),)), torch.tensor((0.7,)), torch.tensor(((1.2, -0.9),))
  )
  assert overlap(bent, bent).item() == pytest.approx(1.0)
  assert overlap(reversed_links(bent), bent).item() == pytest.approx(1.0)

  straight = link_poses(torch.zeros(1, 2), torch.zeros(1), torch.zeros(1, 2))
  shifted = link_poses(torch.tensor(((0.01, 0.0),)), torch.zeros(1), torch.zeros(1, 2))
  assert overlap(shifted, straight).item() == pytest.approx(1 - 0.01 / 0.2275, abs=1e-3)


def test_landmarks_run_tip_to_tip_and_reverse_with_the_chain() -> None:
  from vbrl.tasks.push_chain.geometry import (
    chain_landmarks,
    landmark_errors,
    link_poses,
    reversed_links,
  )

  straight = link_poses(torch.zeros(1, 2), torch.zeros(1), torch.zeros(1, 2))
  xs = (-0.11375, -0.07875, -0.04375, 0.0, 0.04375, 0.07875, 0.11375)
  expected = torch.tensor([[(x, 0.0) for x in xs]])
  assert chain_landmarks(straight) == pytest.approx(expected, abs=1e-6)

  bent = link_poses(
    torch.tensor(((0.3, 0.1),)), torch.tensor((0.7,)), torch.tensor(((0.9, -1.3),))
  )
  flipped = chain_landmarks(reversed_links(bent))
  assert flipped == pytest.approx(chain_landmarks(bent).flip(-2), abs=1e-6)
  errors = landmark_errors(bent, reversed_links(bent))
  assert errors[0, 1].item() == pytest.approx(0.0, abs=1e-6)
  assert errors[0, 0].item() > 0.05

  folded = link_poses(
    torch.zeros(1, 2), torch.zeros(1), torch.tensor(((math.pi / 2, 0.0),))
  )
  tip = 2.0 * 0.07 * math.sin(math.pi / 4)
  assert landmark_errors(folded, straight)[0, 0].item() == pytest.approx(tip, abs=1e-6)


def test_goal_tape_traces_the_chain_outline_at_every_bend() -> None:
  from vbrl.tasks.push_chain.geometry import (
    HINGE_LIMIT,
    chain_signed_distance,
    link_poses,
  )
  from vbrl.tasks.push_chain.goal_marker import _boxes, side_strips
  from vbrl.tasks.push_t.goal_marker import GOAL_OUTLINE_THICKNESS

  angles = torch.linspace(-HINGE_LIMIT, HINGE_LIMIT, 9)
  hinges = torch.cartesian_prod(angles, angles)
  links = link_poses(torch.zeros(len(hinges), 2), torch.zeros(len(hinges)), hinges)
  strips = side_strips(hinges)
  axis = torch.arange(-0.2, 0.2, 0.001)
  points = torch.cartesian_prod(axis, axis)
  distance = chain_signed_distance(points, links[:, None], pins=True)
  covered = torch.zeros_like(distance, dtype=torch.bool)
  for link in range(3):
    relative = points - links[:, link, None, :2]
    yaw = links[:, link, 2, None]
    u = torch.cos(yaw) * relative[..., 0] + torch.sin(yaw) * relative[..., 1]
    v = -torch.sin(yaw) * relative[..., 0] + torch.cos(yaw) * relative[..., 1]
    for index, (_, centre, half) in enumerate(_boxes(link)):
      centre, half = torch.tensor(centre).expand(len(hinges), 2), torch.tensor(half)
      length = half[0].expand(len(hinges))
      if index < 2:
        centre = torch.stack((strips[:, link, index, 0], centre[:, 1]), dim=-1)
        length = strips[:, link, index, 1]
      covered |= ((u - centre[:, None, 0]).abs() <= length[:, None] + 1e-9) & (
        (v - centre[:, None, 1]).abs() <= half[1] + 1e-9
      )
  assert not (covered & (distance < -1e-4)).any()
  band = (distance > 1e-4) & (distance < GOAL_OUTLINE_THICKNESS - 1e-4)
  assert covered[band].float().mean() > 0.99


@pytest.mark.sim
@pytest.mark.gpu
def test_state_env_steps_without_nan_and_reports_its_terms() -> None:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  import vbrl.tasks  # noqa: F401

  if not torch.cuda.is_available():
    pytest.skip("needs CUDA")
  cfg = load_env_cfg(STATE_TASK_ID)
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    env.reset()
    generator = torch.Generator(device="cuda:0").manual_seed(0)
    for _ in range(100):
      action = torch.rand(
        (4, env.action_manager.total_action_dim), device="cuda:0", generator=generator
      )
      obs, reward, *_ = env.step(2.0 * action - 1.0)
      assert all(torch.isfinite(value).all() for value in obs.values())
      assert torch.isfinite(reward).all()
    assert set(env.termination_manager.active_terms) == {
      "time_out",
      "object_off_table",
      "invalid_object_state",
      "nan_detection",
    }
    command = env.command_manager.get_term("push_chain_goal")
    assert {
      "link0_overlap",
      "link1_overlap",
      "link2_overlap",
      "min_link_overlap",
      "episode_success",
      "middle_position_error",
      "middle_yaw_error",
      "hinge0_error",
      "hinge1_error",
      "max_landmark_error_m",
      "tcp_distance",
      "chain_overlap",
      "chain_final_success",
      "chain_success",
    } == set(command.metrics)
    assert set(env.metrics_manager.active_terms) == {
      "peak_top_face_force",
      "peak_side_face_force",
    }
  finally:
    env.close()


def test_chain_colour_event_picks_up_the_chain_material() -> None:
  from mjlab.tasks.registry import load_env_cfg

  import vbrl.tasks  # noqa: F401

  cfg = load_env_cfg("Mjlab-PushChain-WoodTable-DinoV2ViTS14-AfaFull6-Trossen")
  materials = cfg.events["object_color"].params["asset_cfg"].material_names
  assert materials == ("push_chain_red",)
  assert {f"goal_marker_link{i}" for i in range(3)} <= set(cfg.scene.entities)


def test_students_see_the_state_actor_as_their_teacher() -> None:
  from mjlab.tasks.registry import load_env_cfg

  import vbrl.tasks  # noqa: F401

  state = load_env_cfg(STATE_TASK_ID)
  student = load_env_cfg("Mjlab-PushChain-WoodTable-NatureCnn-FlattenRelu-Trossen")
  teacher = student.observations["teacher"].terms
  assert list(teacher) == list(state.observations["actor"].terms)
  assert teacher["target_pose"].params["calibration_bias"] is False
  assert "calibration_bias" not in teacher["obj_pose"].params
  assert "obj_pose" not in student.observations["actor"].terms
  assert set(student.terminations) == set(state.terminations)


@pytest.mark.sim
@pytest.mark.gpu
def test_beside_starts_put_the_pads_near_the_chain_without_contact() -> None:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  import vbrl.tasks  # noqa: F401

  if not torch.cuda.is_available():
    pytest.skip("needs CUDA")
  cfg = load_env_cfg(STATE_TASK_ID)
  cfg.scene.num_envs = 64
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    env.reset()
    env.step(torch.zeros(64, env.action_manager.total_action_dim, device="cuda:0"))
    command = env.command_manager.get_term("push_chain_goal")
    beside = command._start_mode == 1
    distance = command.tcp_distance()[beside]
    chain_force = env.scene["ee_object_contact"].data.force.norm(dim=-1).amax(-1)
    table = env.scene["robot_table_contact"].data.found.amax(-1)
    assert beside.any() and (~beside).any()
    assert ((distance > 0.015) & (distance < 0.05)).all()
    assert (chain_force[beside] < 0.05).all()
    assert (table[beside] == 0).all()
  finally:
    env.close()


def test_command_metrics_skip_an_env_reset_with_a_nan_state() -> None:
  from vbrl.tasks.push_chain.mdp.commands import GOAL_ERRORS, PushChainCommand

  command = object.__new__(PushChainCommand)
  names = (*GOAL_ERRORS, "max_landmark_error_m")
  command.metrics = {name: torch.tensor((0.1, float("nan"))) for name in names}
  command._resample = lambda env_ids: None
  command.command_counter = torch.zeros(2)
  command.time_left = torch.zeros(2)
  command._pending_forward = False
  extras = PushChainCommand.reset(command, torch.arange(2))
  assert extras == pytest.approx(dict.fromkeys(names, 0.1))
