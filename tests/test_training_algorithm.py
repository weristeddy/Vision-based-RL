from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from vbrl.training.ppo import VisualPPO
from vbrl.vision.config import VisionConfig
from vbrl.vision.model import VisionModel


def _build_test_ppo(
  algorithm_class: type[PPO],
  *,
  distribution_cfg: dict | None = None,
  **algorithm_kwargs,
) -> tuple[PPO, MLPModel, MLPModel, RolloutStorage]:
  observations = TensorDict({"actor": torch.zeros(2, 3)}, batch_size=[2])
  groups = {"actor": ["actor"], "critic": ["actor"]}
  actor = MLPModel(
    observations,
    groups,
    "actor",
    1,
    hidden_dims=(4,),
    distribution_cfg=distribution_cfg
    or {
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )
  critic = MLPModel(observations, groups, "critic", 1, hidden_dims=(4,))
  storage = RolloutStorage("rl", 2, 1, observations, [1], "cpu")
  if issubclass(algorithm_class, VisualPPO):
    algorithm_kwargs.setdefault("cache_frozen_features", False)
  algorithm = algorithm_class(
    actor,
    critic,
    storage,
    device="cpu",
    **algorithm_kwargs,
  )
  return algorithm, actor, critic, storage


def _matching_native_and_visual_ppo(
  accumulation_steps: int,
) -> tuple[PPO, VisualPPO]:
  """Build algorithms with identical deterministic models and rollouts."""
  kwargs = {
    "num_learning_epochs": 2,
    "num_mini_batches": 1,
    "schedule": "fixed",
    "desired_kl": None,
    "learning_rate": 7.0e-4,
    "max_grad_norm": 10.0,
  }
  torch.manual_seed(19)
  native, native_actor, native_critic, native_storage = _build_test_ppo(
    PPO, **kwargs
  )
  torch.manual_seed(19)
  visual, visual_actor, visual_critic, visual_storage = _build_test_ppo(
    VisualPPO,
    gradient_accumulation_steps=accumulation_steps,
    early_stop_kl=False,
    **kwargs,
  )
  visual_actor.load_state_dict(native_actor.state_dict())
  visual_critic.load_state_dict(native_critic.state_dict())

  observations = torch.tensor(
    [[[-0.5, 0.25, 0.75], [0.8, -0.4, 0.1]]]
  )
  actions = torch.tensor([[[-0.2], [0.35]]])
  for storage in (native_storage, visual_storage):
    storage.observations["actor"].copy_(observations)
    storage.actions.copy_(actions)
    storage.values.copy_(torch.tensor([[[0.1], [-0.15]]]))
    storage.returns.copy_(torch.tensor([[[0.4], [0.25]]]))
    storage.advantages.copy_(torch.tensor([[[-0.6], [0.8]]]))
    storage.actions_log_prob.copy_(torch.tensor([[[-0.9], [-1.05]]]))
    storage.distribution_params = (
      torch.zeros(1, 2, 1),
      torch.ones(1, 2, 1),
    )
  return native, visual


def _assert_ppo_parameters_close(
  native: PPO,
  visual: VisualPPO,
  *,
  rtol: float = 0.0,
  atol: float = 0.0,
) -> None:
  for native_model, visual_model in (
    (native.actor, visual.actor),
    (native.critic, visual.critic),
  ):
    for native_parameter, visual_parameter in zip(
      native_model.parameters(), visual_model.parameters(), strict=True
    ):
      torch.testing.assert_close(
        native_parameter,
        visual_parameter,
        rtol=rtol,
        atol=atol,
      )


# The pre-refactor implementation is the oracle for accumulated updates; see
# the data file's own comment before touching it.
_FROZEN = json.loads(
  (Path(__file__).parent / "data" / "visual_ppo_accumulated.json").read_text()
)


def test_early_kl_stop_requires_a_target() -> None:
  with pytest.raises(ValueError, match="early_stop_kl requires desired_kl"):
    _build_test_ppo(VisualPPO, early_stop_kl=True, desired_kl=None)


def test_opt_in_kl_stop_always_takes_one_update_first() -> None:
  """The KL stop trims an update; it must never skip one entirely.

  Aborting before any optimizer step is a trap, not a safeguard: the policy does
  not move, so the next iteration's rollout is equally off-policy and aborts the
  same way. That deadlock cost one 6,000-iteration cluster run its last 2,950
  iterations when a curriculum rung widened the goal distribution.
  """
  algorithm, actor, critic, storage = _build_test_ppo(
    VisualPPO,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 0.6065306597,
      "std_type": "log",
      "std_range": (0.15, 1.0),
    },
    num_learning_epochs=2,
    num_mini_batches=1,
    schedule="fixed",
    desired_kl=0.2,
    early_stop_kl=True,
    gradient_accumulation_steps=1,
  )
  storage.observations["actor"].zero_()
  storage.actions.zero_()
  storage.values.zero_()
  storage.returns.fill_(1.0)
  storage.advantages.fill_(1.0)
  storage.actions_log_prob.fill_(100.0)
  storage.distribution_params = (
    torch.zeros(1, 2, 1),
    torch.ones(1, 2, 1),
  )
  parameters = list(actor.parameters()) + list(critic.parameters())
  before = [parameter.detach().clone() for parameter in parameters]

  losses = algorithm.update()

  assert losses["kl_stopped_early"] == 1.0
  assert losses["approx_kl"] > algorithm.desired_kl
  # Two logical minibatches: the first steps unconditionally, the second is what
  # the KL stop rejects.
  assert losses["performed_updates"] == 1.0
  assert any(
    not torch.equal(previous, current)
    for previous, current in zip(before, parameters, strict=True)
  ), "an excessive KL left the policy completely unchanged, which is the deadlock"


def test_neutral_construction_delegates_to_native_ppo(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  observations = TensorDict({"actor": torch.zeros(2, 3)}, batch_size=[2])
  env = SimpleNamespace(num_envs=2, num_actions=1)
  config = {
    "algorithm": {"cache_frozen_features": False},
    "num_steps_per_env": 4,
    "torch_compile_mode": "default",
  }
  marker = object()
  calls: list[tuple] = []

  def construct(obs, native_env, cfg, device):
    calls.append((obs, native_env, cfg, device))
    return marker

  monkeypatch.setattr(PPO, "construct_algorithm", staticmethod(construct))

  result = VisualPPO.construct_algorithm(observations, env, config, "cpu")

  assert result is marker
  assert len(calls) == 1
  assert calls[0][0] is observations
  assert calls[0][1] is env
  assert calls[0][2] is config
  assert calls[0][3] == "cpu"


def test_neutral_update_is_exactly_native_ppo() -> None:
  native, visual = _matching_native_and_visual_ppo(accumulation_steps=1)

  torch.manual_seed(2025)
  native_losses = native.update()
  torch.manual_seed(2025)
  visual_losses = visual.update()

  assert visual_losses == native_losses
  _assert_ppo_parameters_close(native, visual)


def test_accumulated_update_matches_frozen_pre_refactor_behavior() -> None:
  _, accumulated = _matching_native_and_visual_ppo(
    accumulation_steps=2
  )

  torch.manual_seed(2025)
  accumulated_losses = accumulated.update()

  # The tolerances cover only the justified float32 reduction variability from
  # splitting each logical batch into two microbatches.
  for name, expected in _FROZEN["losses"].items():
    assert accumulated_losses[name] == pytest.approx(
      expected, rel=2e-6, abs=2e-7
    )
  actual_parameters = torch.cat(
    [
      parameter.detach().reshape(-1)
      for model in (accumulated.actor, accumulated.critic)
      for parameter in model.parameters()
    ]
  )
  torch.testing.assert_close(
    actual_parameters,
    torch.tensor(_FROZEN["parameters"]),
    rtol=2e-6,
    atol=2e-7,
  )


def test_cached_construction_reuses_native_models_without_native_rollout(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  observations = TensorDict({"actor": torch.zeros(2, 3)}, batch_size=[2])
  env = SimpleNamespace(num_envs=2, num_actions=1)
  config = {
    "algorithm": {"cache_frozen_features": True},
    "num_steps_per_env": 4,
    "torch_compile_mode": "default",
  }
  algorithm = SimpleNamespace(
    _with_cached_features=lambda obs: obs,
    storage=None,
  )

  def construct(obs, native_env, cfg, device):
    assert obs is observations
    assert native_env is env
    assert device == "cpu"
    assert cfg["num_steps_per_env"] == 0
    assert cfg["torch_compile_mode"] == "default"
    return algorithm

  monkeypatch.setattr(PPO, "construct_algorithm", staticmethod(construct))

  result = VisualPPO.construct_algorithm(observations, env, config, "cpu")

  assert result is algorithm
  assert config["num_steps_per_env"] == 4
  assert algorithm.storage.num_transitions_per_env == 4


class _FrozenFeatureEncoder(nn.Module):
  freeze_backbone = True
  output_dim = 3
  output_channels = None

  def __init__(self) -> None:
    super().__init__()
    self.adapter = nn.Linear(3, 3, bias=False)
    self.encode_calls = 0

  def encode_features(self, images: torch.Tensor) -> torch.Tensor:
    self.encode_calls += 1
    return images.mean(dim=(-2, -1))

  def project_features(self, features: torch.Tensor) -> torch.Tensor:
    return self.adapter(features.float())


def test_frozen_features_replace_images_in_rollout_storage() -> None:
  observations = TensorDict(
    {
      "actor": torch.zeros(2, 5),
      "camera": torch.rand(2, 3, 16, 16),
    },
    batch_size=[2],
  )
  encoder = _FrozenFeatureEncoder()
  model = VisionModel(
    observations,
    {"actor": ["actor", "camera"]},
    "actor",
    1,
    hidden_dims=(4,),
    cnns={"camera": encoder},
  )

  cached = model.add_cached_features(
    observations,
    drop_raw_images=True,
    feature_cache_dtype=torch.bfloat16,
  )
  storage = RolloutStorage("rl", 2, 2, cached, [1], "cpu")

  assert "camera" not in cached
  assert cached["camera_features"].shape == (2, 3)
  assert cached["camera_features"].dtype == torch.bfloat16
  assert encoder.encode_calls == 1
  assert "camera" not in storage.observations
  assert storage.observations["camera_features"].dtype == torch.bfloat16


def test_trainable_storage_recomputes_visual_features_with_gradients() -> None:
  config = VisionConfig(
    encoder="nature_cnn",
    weights="scratch",
    train_encoder=True,
    adapter="spatial_softmax",
  )
  observations = TensorDict(
    {
      "actor": torch.zeros(2, 4),
      "camera": torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8),
    },
    batch_size=[2],
  )
  model = VisionModel(
    observations,
    {"critic": ["actor", "camera"]},
    "critic",
    1,
    hidden_dims=(16,),
    cnn_cfg={"vision": config.asdict()},
  )
  storage = RolloutStorage("rl", 2, 2, observations, [1], "cpu")
  storage.observations["camera"].random_(0, 256)
  storage.distribution_params = (torch.zeros(2, 2, 1),)

  batch = next(storage.mini_batch_generator(num_mini_batches=1, num_epochs=1))
  model(batch.observations).square().mean().backward()

  encoder = model.cnns["camera"]
  assert batch.observations["camera"].dtype == torch.uint8
  assert any(parameter.grad is not None for parameter in encoder.backbone.parameters())
  assert any(parameter.grad is not None for parameter in encoder.adapter.parameters())
