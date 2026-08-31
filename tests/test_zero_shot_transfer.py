"""What has to hold here before code is pushed to the cluster.

The rest of the suite checks *contracts*: registered IDs, observation groups,
module layouts, config wire formats. None of it constructs a simulator, so a
task whose configuration is well-formed but whose environment cannot reset, or
whose encoder cannot backpropagate, passes everything and then fails on a GPU
node. This module closes that gap by actually running things.

Three layers, cheapest first:

1. Every visual architecture any task registers forward- and
   backward-propagates and takes an optimizer step, with the real pretrained
   weights.
2. Every task family's environment builds, resets, and steps, with observations
   matching what the registered config declares.
3. The exact runner ``vbrl-train`` uses completes a learning iteration, moves
   the policy, and writes a checkpoint that reloads with ``strict=True``.

What this deliberately cannot cover: the four-rank TorchrunX launch, Slurm
submission, and W&B online logging. One short job on the ``testing`` profile
remains the final gate -- but it should be a formality, not a discovery.
"""

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


# One representative ID per axis that could plausibly differ between a
# workstation and a cluster node: both robots, both state tasks, a frozen
# pretrained encoder (which takes the cached-feature path) and a trainable
# scratch encoder (which does not).
TASK_IDS = (
  "Mjlab-PushCube-State-Trossen",
  "Mjlab-PushT-State-TrossenRealistic",
  "Mjlab-LiftCube-RealTexture-DinoV2ViTS14-LocalGrid7-Trossen",
  "Mjlab-PushT-RealTexture-NatureCnn-SpatialSoftmax-TrossenRealistic",
)
VISUAL_TASK_ID = "Mjlab-LiftCube-RealTexture-DinoV2ViTS14-LocalGrid7-Trossen"
STATE_TASK_ID = "Mjlab-PushCube-State-Trossen"

NUM_ENVS = 8
IMAGE_SIZE = (224, 224)
DEVICE = "cuda:0"


def _weights_available() -> bool:
  root = model_root()
  return (root / "dinov2-small" / "model.safetensors").is_file() and (
    root / "r3m" / "r3m_50" / "model.pt"
  ).is_file()


requires_weights = pytest.mark.skipif(
  not _weights_available(),
  reason=f"pretrained backbones absent from {model_root()}",
)
requires_cuda = pytest.mark.skipif(
  not torch.cuda.is_available(),
  reason="mujoco-warp steps on CUDA only",
)


# --- 1. every encoder x adapter, end to end ----------------------------------


@functools.cache
def _shared_backbone(encoder: str):
  """Load one pretrained backbone per encoder instead of once per combination.

  The loaders are captured at import time: the fixture below replaces
  ``dinov2.load``/``r3m.load``, and reading them through the module here would
  make this call itself.
  """
  return _LOADERS[encoder]()


@pytest.fixture
def _cached_pretrained_backbones(monkeypatch: pytest.MonkeyPatch) -> None:
  """Serve real weights from a per-encoder cache, deep-copied per build.

  The weights are what make this test meaningful, and loading DINOv2 and
  ResNet50 thirty times over is what would make it too slow to run. Copying
  keeps each combination's parameters independent.
  """
  import copy

  monkeypatch.setattr(_dinov2, "load", lambda: copy.deepcopy(_shared_backbone("dinov2")))
  monkeypatch.setattr(_r3m, "load", lambda: copy.deepcopy(_shared_backbone("r3m")))


@functools.cache
def _registered_architectures() -> tuple[tuple[str, dict], ...]:
  """Every distinct visual architecture any registered task actually uses.

  Sweeping the registry's encoders against its adapters would invent
  combinations no task registers and that cannot work: ``flatten`` reads the
  backbone's *native* grid, so its ``target_grid_size`` is 24 for NatureCNN and
  14 for CompactViT, not a free parameter. Taking the configurations from the
  task registry tests the architectures that exist, at the settings they run
  with, and picks up a new one automatically when a task is registered.

  ``global`` is appended per encoder because it is a registered adapter that no
  task currently selects, so nothing else would notice if it rotted.
  """
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
  """Builds, forwards, backpropagates, and takes an optimizer step.

  ``test_vision_checkpoint_layout`` pins what these modules *are*; this pins
  that they *work*. A frozen backbone must additionally stay frozen, which is
  the premise the whole cached-feature rollout path is built on.

  Not every parameter is expected to receive a gradient: the AFA pool carries a
  ``norm`` its forward never applies, kept only so the published PV-Robo
  state-dict layout matches. So the assertion is that learning happens, not
  that every tensor participates.
  """
  # R3M's upstream model class moves itself to CUDA while being constructed, so
  # a freshly built encoder can straddle two devices. Training always moves the
  # whole model to the run device, and so must this.
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
    # A frozen backbone behind a parameterless adapter is a pure feature
    # extractor. Nothing to optimize is correct, not a failure.
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


# --- 2. every task family's environment builds, resets, and steps ------------


@pytest.mark.sim
@pytest.mark.gpu
@requires_cuda
@requires_weights
@pytest.mark.parametrize("task_id", TASK_IDS)
def test_registered_task_resets_and_steps(task_id: str) -> None:
  """The gap between "the config is well-formed" and "the task runs"."""
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
  """Spreading the envs out must be a pure translation of each world.

  Every env is laid out at its own origin so a multi-env view or video does not
  stack robots on top of each other. That is only free if each env is the same
  scene moved: the camera rides the robot's base body and the lights ride the
  table's, so translating both mocap poses and the object's root back to the
  world origin has to reproduce the very same pixels. Anything left behind --
  a light, a prop, a world-frame camera -- shows up here as a non-zero delta.
  """
  import mujoco

  from vbrl.runtime import build_env

  task_id = "Mjlab-PushT-RealTexture-NatureCnn-SpatialSoftmax-TrossenRealistic"
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


# --- 3. the runner vbrl-train uses completes an iteration -------------------


@contextlib.contextmanager
def _training_stack(task_id: str, log_dir: Path):
  """Yield ``(make_runner, environment)`` off the same path ``train.py`` takes.

  The environment is the expensive part, so it is built once and handed back:
  a resume test needs a *second* runner over the same environment.
  """
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
  """A rollout, an update, and a checkpoint -- the whole training path."""
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

    # The strict=True contract a resume depends on, checked by comparison rather
    # than by loading: after a rollout the live normalizer buffers are inference
    # tensors, which cannot be updated in place. Loading into a fresh runner is
    # what a real resume does, and is covered below.
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
  """Jobs hit the three-day wall and requeue, so resume is a production path.

  Uses the visual task: its checkpoint carries the encoder and adapter subtrees,
  which is where a layout change would break a resume.
  """
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
  """The cached-feature path is what makes visual training affordable.

  A frozen encoder must land in rollout storage as features, with the raw images
  dropped; silently falling back to storing images still trains, just far slower
  and at many times the memory -- exactly the kind of regression that only shows
  up as a cluster job dying on memory.
  """
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
