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
    entity_name="object", success_threshold=0.9, resampling_time_range=(1e9, 1e9)
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
  command._env = SimpleNamespace(common_step_counter=0)
  command._overlap = LinkOverlap(device="cpu", dtype=torch.float)
  command._overlap_cache_step = None
  command.metrics = {}
  command.episode_success = torch.zeros(len(goal_xy))
  return command


def _reward(command, tcp_xyz):
  from vbrl.tasks.push_chain.mdp.rewards import chain_dense_reward

  env = SimpleNamespace(
    scene={"robot": SimpleNamespace(data=SimpleNamespace(site_pos_w=tcp_xyz[:, None]))},
    command_manager=SimpleNamespace(get_term=lambda name: command),
  )
  return chain_dense_reward(env, "goal", SimpleNamespace(name="robot", site_ids=[0]))


def test_success_and_reward_accept_the_goal_in_either_order() -> None:
  from vbrl.tasks.push_chain.mdp.rewards import ordering_values

  xy = torch.tensor(((0.35, 0.05), (0.3, -0.1)))
  yaw = torch.tensor((0.4, -2.0))
  hinges = torch.tensor(((0.6, -1.2), (-0.3, 0.9)))
  tcp = torch.tensor(((0.6, 0.3, 0.2), (0.6, 0.3, 0.2)))

  forward = _fake_command(xy, yaw, hinges, xy, yaw, hinges)
  backward = _fake_command(xy, yaw + math.pi, -hinges.flip(-1), xy, yaw, hinges)
  for command in (forward, backward):
    assert command.get_overlap() == pytest.approx(torch.ones(2, 3))
    assert command.get_at_goal().all()
    assert ordering_values(command).amax(-1) == pytest.approx(torch.ones(2))
    assert _reward(command, tcp) == pytest.approx(torch.ones(2))

  moved = torch.tensor(((0.02, 0.0), (0.0, 0.02)))
  near = _fake_command(xy, yaw, hinges, xy + moved, yaw, hinges)
  near_backward = _fake_command(
    xy, yaw + math.pi, -hinges.flip(-1), xy + moved, yaw, hinges
  )
  assert not near.get_at_goal().any()
  assert near.get_overlap() == pytest.approx(near_backward.get_overlap())
  assert _reward(near, tcp) == pytest.approx(_reward(near_backward, tcp))
  assert (_reward(near, tcp) < 0.5).all()


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
