from __future__ import annotations

import contextlib
import functools
from pathlib import Path

import pytest

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

from vbrl.paths import model_root  # noqa: E402
from vbrl.vision.backbones import dinov2 as _dinov2  # noqa: E402
from vbrl.vision.backbones import r3m as _r3m  # noqa: E402
from vbrl.vision.config import VisionConfig  # noqa: E402
from vbrl.vision.registry import ENCODERS, build_encoder  # noqa: E402

_LOADERS = {"dinov2": _dinov2.load, "r3m": _r3m.load}


TASK_IDS = (
  "Mjlab-PushT-State-TrossenIdentified",
  "Mjlab-LiftCube-RealTexture-DinoV2ViTS14-LocalGrid7-Trossen",
  "Mjlab-PushT-GoalOutline-DinoV2ViTS14-Afa6-TrossenIdentified",
)
VISUAL_TASK_ID = "Mjlab-LiftCube-RealTexture-DinoV2ViTS14-LocalGrid7-Trossen"
STATE_TASK_ID = "Mjlab-PushT-State-TrossenIdentified"

NUM_ENVS = 8
IMAGE_SIZE = (224, 224)
DEVICE = "cuda:0"


def _weights_available() -> bool:
  from vbrl.vision.backbones.weights import huggingface_cache, r3m_files

  root = model_root()
  return huggingface_cache(root).is_dir() and all(
    f.is_file() for f in r3m_files(root)
  )


requires_weights = pytest.mark.skipif(
  not _weights_available(),
  reason=f"pretrained backbones absent from {model_root()}",
)
requires_cuda = pytest.mark.skipif(
  not torch.cuda.is_available(),
  reason="mujoco-warp steps on CUDA only",
)


@functools.cache
def _shared_backbone(encoder: str):
  return _LOADERS[encoder]()


@pytest.fixture
def _cached_pretrained_backbones(monkeypatch: pytest.MonkeyPatch) -> None:
  import copy

  monkeypatch.setattr(_dinov2, "load", lambda: copy.deepcopy(_shared_backbone("dinov2")))
  monkeypatch.setattr(_r3m, "load", lambda: copy.deepcopy(_shared_backbone("r3m")))


@functools.cache
def _registered_architectures() -> tuple[tuple[str, dict], ...]:
  from mjlab.tasks.registry import list_tasks, load_rl_cfg

  import vbrl.tasks  # noqa: F401

  found: dict[tuple, tuple[str, dict]] = {}
  for task_id in sorted(list_tasks()):
    actor = getattr(load_rl_cfg(task_id), "actor", None)
    if actor is None or actor.class_name != "vbrl.vision.model:VisionModel":
      continue
    vision = dict(actor.cnn_cfg["vision"])
    key = (vision["encoder"], vision["adapter"], vision.get("target_grid_size"))
    found.setdefault(key, (f"{key[0]}-{key[1]}-grid{key[2]}", vision))

  for encoder, spec in ENCODERS.items():
    if encoder == "none":
      continue
    key = (encoder, "global", None)
    found.setdefault(
      key,
      (
        f"{encoder}-global",
        {
          "encoder": encoder,
          "weights": spec.weights,
          "train_encoder": spec.trainable,
          "adapter": "global",
        },
      ),
    )
  return tuple(found.values())


def _architecture_cases() -> tuple[list[dict], list[str]]:
  cases = _registered_architectures()
  return [vision for _, vision in cases], [label for label, _ in cases]


_VISION_CONFIGS, _VISION_LABELS = _architecture_cases()


@requires_weights
@pytest.mark.usefixtures("_cached_pretrained_backbones")
@pytest.mark.parametrize("vision", _VISION_CONFIGS, ids=_VISION_LABELS)
def test_every_registered_architecture_trains_end_to_end(vision: dict) -> None:
  # R3M's model class moves itself to CUDA during construction, so a fresh
  # encoder can straddle two devices.
  device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
  encoder_module = build_encoder(
    VisionConfig.from_mapping(vision), input_dim=IMAGE_SIZE
  ).to(device)
  images = torch.randint(0, 256, (2, 3, *IMAGE_SIZE), dtype=torch.uint8, device=device)

  features = encoder_module(images)

  assert features.shape == (2, encoder_module.output_dim)
  assert torch.isfinite(features).all()
  if encoder_module.freeze_backbone:
    assert all(
      not p.requires_grad for p in encoder_module.backbone.parameters()
    ), "a frozen backbone exposed trainable parameters"

  trainable = [p for p in encoder_module.parameters() if p.requires_grad]
  if not trainable:
    assert features.grad_fn is None
    return

  before = [p.detach().clone() for p in trainable]
  optimizer = torch.optim.Adam(trainable, lr=1e-3)
  features.square().mean().backward()

  gradients = [p.grad for p in trainable if p.grad is not None]
  assert gradients, "no trainable tensor received a gradient"
  assert all(torch.isfinite(g).all() for g in gradients), "a gradient was non-finite"
  optimizer.step()
  assert any(
    not torch.equal(old, new) for old, new in zip(before, trainable, strict=True)
  ), "an optimizer step changed nothing"


@pytest.mark.sim
@pytest.mark.gpu
@requires_cuda
@requires_weights
@pytest.mark.parametrize("task_id", TASK_IDS)
def test_registered_task_resets_and_steps(task_id: str) -> None:
  from mjlab.tasks.registry import load_env_cfg

  from vbrl.runtime import build_env

  env = build_env(task_id, device=DEVICE, num_envs=NUM_ENVS, seed=0)
  try:
    declared = set(load_env_cfg(task_id, play=True).observations)
    observations, _ = env.reset()

    assert declared <= set(observations.keys()), (
      f"reset omitted declared observation groups: "
      f"{sorted(declared - set(observations.keys()))}"
    )
    for group in declared:
      tensor = observations[group]
      assert tensor.shape[0] == NUM_ENVS, f"{group} is not batched over envs"
      assert torch.isfinite(tensor.float()).all(), f"{group} contains non-finite values"

    for step in range(10):
      actions = torch.zeros((NUM_ENVS, env.action_space.shape[-1]), device=DEVICE)
      observations, rewards, dones, _ = env.step(actions)[:4]
      assert torch.isfinite(rewards).all(), f"non-finite reward at step {step}"
      assert dones.shape[0] == NUM_ENVS
      for group in declared:
        assert torch.isfinite(observations[group].float()).all(), (
          f"{group} went non-finite at step {step}"
        )
  finally:
    env.close()


@pytest.mark.sim
@pytest.mark.gpu
@requires_cuda
@requires_weights
def test_the_env_origin_grid_does_not_change_what_the_camera_sees() -> None:
  import mujoco

  from vbrl.runtime import build_env

  task_id = "Mjlab-PushT-GoalOutline-DinoV2ViTS14-Afa6-TrossenIdentified"
  env = build_env(task_id, device=DEVICE, num_envs=4, seed=0)
  try:
    model = env.sim.mj_model
    origins = env.scene.env_origins
    assert torch.unique(origins, dim=0).shape[0] == origins.shape[0], (
      "env origins must be distinct or the layout is not a grid"
    )

    def camera_image():
      env.scene.update(env.step_dt)
      return env.observation_manager.compute()["camera"].clone()

    env.reset()
    spread = camera_image()

    bodies = [
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
      for body in range(model.nbody)
    ]
    object_body = next(i for i, name in enumerate(bodies) if name == "object/push_t")
    root = int(model.jnt_qposadr[model.body_jntadr[object_body]])
    env.sim.data.mocap_pos[:, :, :] -= origins[:, None, :]
    env.sim.data.qpos[:, root : root + 3] -= origins
    env.sim.forward()

    assert torch.equal(spread, camera_image()), (
      "translating each env back to the world origin changed its camera image"
    )
  finally:
    env.close()


@contextlib.contextmanager
def _training_stack(task_id: str, log_dir: Path):
  from dataclasses import asdict

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import vbrl.tasks  # noqa: F401

  env_cfg = load_env_cfg(task_id)
  env_cfg.scene.num_envs = NUM_ENVS
  env_cfg.seed = 0
  agent_cfg = load_rl_cfg(task_id)
  agent_cfg.max_iterations = 1
  agent_cfg.num_steps_per_env = 8
  agent_cfg.save_interval = 1
  agent_cfg.logger = "tensorboard"

  environment = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE)
  environment = RslRlVecEnvWrapper(environment, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  try:
    yield (
      lambda: runner_cls(environment, asdict(agent_cfg), str(log_dir), DEVICE),
      environment,
    )
  finally:
    environment.close()


def _float_state(module) -> dict:
  return {
    name: tensor.detach().cpu().clone()
    for name, tensor in module.state_dict().items()
    if tensor.is_floating_point()
  }


@pytest.mark.sim
@pytest.mark.gpu
@requires_cuda
@requires_weights
@pytest.mark.parametrize("task_id", (STATE_TASK_ID, VISUAL_TASK_ID))
def test_one_learning_iteration_updates_the_policy(
  task_id: str, tmp_path: Path
) -> None:
  with _training_stack(task_id, tmp_path) as (make_runner, _):
    runner = make_runner()
    before = _float_state(runner.alg.actor)
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)

    after = _float_state(runner.alg.actor)
    assert any(
      not torch.equal(before[name], after[name]) for name in before
    ), "a full learning iteration left the policy unchanged"

    checkpoints = sorted(tmp_path.glob("model_*.pt"))
    assert checkpoints, f"no checkpoint written into {tmp_path}"
    saved = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
    assert {"actor_state_dict", "critic_state_dict", "optimizer_state_dict"} <= set(
      saved
    )

    for role, module in (("actor", runner.alg.actor), ("critic", runner.alg.critic)):
      expected = {name: tuple(v.shape) for name, v in module.state_dict().items()}
      stored = {name: tuple(v.shape) for name, v in saved[f"{role}_state_dict"].items()}
      assert stored == expected, (
        f"the checkpoint's {role} would not load with strict=True"
      )


@pytest.mark.sim
@pytest.mark.gpu
@requires_cuda
@requires_weights
def test_a_checkpoint_resumes_into_a_fresh_runner(tmp_path: Path) -> None:
  with _training_stack(VISUAL_TASK_ID, tmp_path) as (make_runner, _):
    trained = make_runner()
    trained.learn(num_learning_iterations=1, init_at_random_ep_len=True)
    checkpoint = sorted(tmp_path.glob("model_*.pt"))[-1]
    expected = _float_state(trained.alg.actor)

    resumed = make_runner()
    resumed.load(str(checkpoint))

    restored = _float_state(resumed.alg.actor)
    assert set(restored) == set(expected)
    assert all(
      torch.equal(expected[name], restored[name]) for name in expected
    ), "a resumed policy does not match the one that was saved"


@pytest.mark.sim
@pytest.mark.gpu
@requires_cuda
@requires_weights
def test_frozen_visual_features_are_cached_during_rollout(tmp_path: Path) -> None:
  with _training_stack(VISUAL_TASK_ID, tmp_path) as (make_runner, _):
    runner = make_runner()
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)
    algorithm = runner.alg

    assert getattr(algorithm, "cache_frozen_features", False), (
      "a frozen visual encoder did not enable feature caching"
    )
    stored = set(algorithm.storage.observations.keys())
    assert any(key.endswith("_features") for key in stored), (
      f"rollout storage holds no cached features: {sorted(stored)}"
    )


@pytest.mark.sim
@pytest.mark.gpu
@requires_cuda
def test_the_sampled_goal_yaw_reaches_the_command() -> None:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  import vbrl.tasks  # noqa: F401
  from vbrl.tasks.push_t.config.trossen_realistic.rl_cfg import STATE_TASK_ID
  from vbrl.tasks.push_t.push_t_env_cfg import GOAL_YAW_STAGES

  cfg = load_env_cfg(STATE_TASK_ID, play=False)
  cfg.scene.num_envs = 64
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    command = env.command_manager.get_term("push_t_goal")
    ids = torch.arange(env.num_envs, device=env.device)
    env.common_step_counter = GOAL_YAW_STAGES[-1]["step"]
    env.curriculum_manager.compute(env_ids=ids)
    assert float(command.cfg.target_yaw_range[1]) == pytest.approx(torch.pi, abs=1e-3)
    env.reset()
    yaw = command.target_yaw.detach()
    # Dropping `self.target_yaw[env_ids] = target_yaw` leaves this at its zero
    # initialization, which pins every goal to yaw 0 while target_yaw_range --
    # and the curriculum that narrows it -- still read as the full circle.
    assert float(yaw.std()) > 1.0
    assert float(yaw.abs().max()) > 2.5
    assert len(torch.unique(yaw)) > env.num_envs // 2
  finally:
    env.close()


@pytest.mark.sim
@pytest.mark.gpu
def test_the_arm_can_actually_push_the_object() -> None:
  from vbrl.runtime import build_env

  env = build_env(
    "Mjlab-PushT-State-TrossenIdentified", device=DEVICE, num_envs=16, seed=0
  )
  try:
    robot, obj = env.scene["robot"], env.scene["object"]
    env.reset()
    site = robot.data.site_pos_w[:, 0, :]
    start = obj.data.root_link_pos_w.clone()
    pose = torch.zeros(env.num_envs, 7, device=env.device)
    pose[:, :2] = site[:, :2]
    pose[:, 0] -= 0.03
    pose[:, 2] = start[:, 2]
    pose[:, 3] = 1.0
    obj.write_root_link_pose_to_sim(pose, torch.arange(env.num_envs, device=env.device))
    placed = obj.data.root_link_pos_w[:, :2].clone()

    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim)
    action = action.to(env.device)
    action[:, 1] = 1.0
    for _ in range(60):
      env.step(action)

    moved = (obj.data.root_link_pos_w[:, :2] - placed).norm(dim=-1).max()
    assert float(moved) > 0.002, f"pushed the object only {1000 * float(moved):.2f} mm"

    env.reset()
    held = robot.data.joint_pos[:, :6].clone()
    drift = torch.zeros((), device=env.device)
    for _ in range(200):
      env.step(torch.zeros_like(action))
      drift = torch.maximum(drift, (robot.data.joint_pos[:, :6] - held).abs().max())
    assert float(drift) < 0.005, (
      f"the arm drifted {float(drift):.4f} rad under zero action; the real arm "
      "holds home to 0.018 rad because its controller compensates gravity, and a "
      "zeroed target seeded into the delay buffer at reset jolts it for one substep"
    )
  finally:
    env.close()
