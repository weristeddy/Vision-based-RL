from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.storage import RolloutStorage

from mjlab.rl import RslRlPpoAlgorithmCfg


@dataclass
class VisualPpoCfg(RslRlPpoAlgorithmCfg):
  """Native PPO plus visual caching and accumulated updates."""

  cache_frozen_features: bool = True
  feature_cache_dtype: str = "bfloat16"
  gradient_accumulation_steps: int = 1
  early_stop_kl: bool = False
  class_name: str = "vbrl.training.ppo:VisualPPO"


NATIVE_ADAPTIVE_MIN_LEARNING_RATE = 1.0e-5
NATIVE_ADAPTIVE_MAX_LEARNING_RATE = 1.0e-2


_CACHE_DTYPES = {
  "bfloat16": torch.bfloat16,
  "float16": torch.float16,
  "float32": torch.float32,
}


def resolve_cache_dtype(dtype: str) -> torch.dtype:
  try:
    return _CACHE_DTYPES[str(dtype).lower()]
  except KeyError as exc:
    raise ValueError(
      "feature_cache_dtype must be bfloat16, float16, or float32."
    ) from exc


class VisualPPO(PPO):
  """PPO with cached frozen features and accumulated updates."""

  learning_rate: float

  def __init__(
    self,
    *args,
    cache_frozen_features: bool = True,
    feature_cache_dtype: str = "bfloat16",
    gradient_accumulation_steps: int = 1,
    early_stop_kl: bool = False,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self._cached_feature_models = [
      model
      for model in (self._raw_actor, self._raw_critic)
      if bool(getattr(model, "supports_cached_features", False))
    ]
    self.cache_frozen_features = bool(
      cache_frozen_features and self._cached_feature_models
    )
    self.feature_cache_dtype = resolve_cache_dtype(feature_cache_dtype)
    self.gradient_accumulation_steps = int(gradient_accumulation_steps)
    self.early_stop_kl = bool(early_stop_kl)
    if self.gradient_accumulation_steps < 1:
      raise ValueError("gradient_accumulation_steps must be at least one.")
    if self.early_stop_kl and self.desired_kl is None:
      raise ValueError("early_stop_kl requires desired_kl.")
    if cache_frozen_features and not self.cache_frozen_features:
      print("[INFO] Feature caching requested, but no frozen visual encoder is present.")
    elif self.cache_frozen_features:
      print(f"[INFO] Caching frozen rollout features as {self.feature_cache_dtype}.")

  def act(self, obs: TensorDict) -> torch.Tensor:
    return super().act(self._with_cached_features(obs))

  def compute_returns(self, obs: TensorDict) -> None:
    super().compute_returns(self._with_cached_features(obs))

  def update(self) -> dict[str, float]:
    if self.gradient_accumulation_steps == 1 and not self.early_stop_kl:
      return super().update()
    return self._update_with_gradient_accumulation()

  def _with_cached_features(self, obs: TensorDict) -> TensorDict:
    if not self.cache_frozen_features:
      return obs
    cached = obs
    for model in self._feature_models():
      cached = model.add_cached_features(
        cached,
        drop_raw_images=True,
        feature_cache_dtype=self.feature_cache_dtype,
      )
    return cached

  def _feature_models(self) -> list:
    return self._cached_feature_models

  def _update_with_gradient_accumulation(self) -> dict[str, float]:
    """Accumulate micro-batch gradients, with optional KL early stop.

    A transliteration of native PPO's update loop with the logical minibatch
    split into ``gradient_accumulation_steps`` micro-batches, so a frozen
    visual encoder's activations fit in memory. Feed-forward only, which every
    registered task is -- the guards below protect against a future config
    that is not, since the micro-batch loop slices no hidden state.
    """
    if self.actor.is_recurrent or self.critic.is_recurrent:
      raise NotImplementedError("Gradient accumulation supports feed-forward PPO only.")
    if self.rnd or self.symmetry:
      raise NotImplementedError(
        "Gradient accumulation does not support RND or symmetry."
      )

    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_approx_kl = 0.0
    mean_clip_fraction = 0.0
    performed_updates = 0
    diagnostic_batches = 0
    stopped_early = False
    batches = self.storage.mini_batch_generator(
      self.num_mini_batches, self.num_learning_epochs
    )
    for batch in batches:
      logical_size = int(batch.observations.batch_size[0])
      if logical_size % self.gradient_accumulation_steps:
        raise ValueError(
          f"Logical minibatch {logical_size} is not divisible by "
          f"gradient_accumulation_steps={self.gradient_accumulation_steps}."
        )
      micro_size = logical_size // self.gradient_accumulation_steps
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (
            batch.advantages - batch.advantages.mean()
          ) / (batch.advantages.std() + 1e-8)

      self.optimizer.zero_grad()
      logical_value = 0.0
      logical_surrogate = 0.0
      logical_entropy = 0.0
      logical_kl = torch.zeros((), device=self.device)
      logical_approx_kl = torch.zeros((), device=self.device)
      logical_clip_fraction = torch.zeros((), device=self.device)
      for micro_index in range(self.gradient_accumulation_steps):
        start = micro_index * micro_size
        stop = start + micro_size
        observations = batch.observations[start:stop]
        actions = batch.actions[start:stop]
        values_target = batch.values[start:stop]
        advantages = batch.advantages[start:stop]
        returns = batch.returns[start:stop]
        old_log_probability = batch.old_actions_log_prob[start:stop]
        old_distribution = tuple(
          parameter[start:stop] for parameter in batch.old_distribution_params
        )

        self.actor(observations, stochastic_output=True)
        log_probability = self.actor.get_output_log_prob(actions)
        values = self.critic(observations)
        distribution = self.actor.output_distribution_params
        entropy = self.actor.output_entropy

        if self.desired_kl is not None and self.schedule == "adaptive":
          with torch.inference_mode():
            kl = self.actor.get_kl_divergence(old_distribution, distribution)
            logical_kl += kl.mean() / self.gradient_accumulation_steps

        log_ratio = log_probability - torch.squeeze(old_log_probability)
        ratio = torch.exp(log_ratio)
        with torch.inference_mode():
          logical_approx_kl += (
            (ratio - 1.0 - log_ratio).mean()
            / self.gradient_accumulation_steps
          )
          logical_clip_fraction += (
            (torch.abs(ratio - 1.0) > self.clip_param).float().mean()
            / self.gradient_accumulation_steps
          )
        surrogate = -torch.squeeze(advantages) * ratio
        clipped = -torch.squeeze(advantages) * torch.clamp(
          ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, clipped).mean()
        if self.use_clipped_value_loss:
          value_clipped = values_target + (values - values_target).clamp(
            -self.clip_param, self.clip_param
          )
          value_loss = torch.max(
            (values - returns).pow(2), (value_clipped - returns).pow(2)
          ).mean()
        else:
          value_loss = (returns - values).pow(2).mean()
        entropy_mean = entropy.mean()
        loss = (
          surrogate_loss
          + self.value_loss_coef * value_loss
          - self.entropy_coef * entropy_mean
        )
        (loss / self.gradient_accumulation_steps).backward()
        logical_value += value_loss.item() / self.gradient_accumulation_steps
        logical_surrogate += surrogate_loss.item() / self.gradient_accumulation_steps
        logical_entropy += entropy_mean.item() / self.gradient_accumulation_steps

      if self.is_multi_gpu:
        torch.distributed.all_reduce(logical_approx_kl)
        torch.distributed.all_reduce(logical_clip_fraction)
        logical_approx_kl /= self.gpu_world_size
        logical_clip_fraction /= self.gpu_world_size
      mean_approx_kl += logical_approx_kl.item()
      mean_clip_fraction += logical_clip_fraction.item()
      diagnostic_batches += 1
      # Never abort before the first optimizer step of an iteration. If the very
      # first minibatch could stop the update, a distribution shift freezes the
      # policy for good: no step is taken, so the next iteration's rollout is
      # just as off-policy, so its first minibatch aborts too. A 6,000-iteration
      # run spent its last 2,950 iterations performing zero updates for exactly
      # that reason, after a curriculum rung widened the goal distribution --
      # `Loss/performed_updates` sat at 0 and `Loss/entropy` at exactly 0.
      if (
        self.early_stop_kl
        and performed_updates
        and logical_approx_kl > self.desired_kl
      ):
        self.optimizer.zero_grad()
        stopped_early = True
        break

      if self.desired_kl is not None and self.schedule == "adaptive":
        if self.is_multi_gpu:
          torch.distributed.all_reduce(logical_kl)
          logical_kl /= self.gpu_world_size
        if self.gpu_global_rank == 0:
          if logical_kl > self.desired_kl * 2:
            self.learning_rate = max(
              NATIVE_ADAPTIVE_MIN_LEARNING_RATE,
              self.learning_rate / 1.5,
            )
          elif 0 < logical_kl < self.desired_kl / 2:
            self.learning_rate = min(
              NATIVE_ADAPTIVE_MAX_LEARNING_RATE,
              self.learning_rate * 1.5,
            )
        if self.is_multi_gpu:
          learning_rate = torch.tensor(self.learning_rate, device=self.device)
          torch.distributed.broadcast(learning_rate, src=0)
          self.learning_rate = learning_rate.item()
        for parameter_group in self.optimizer.param_groups:
          parameter_group["lr"] = self.learning_rate

      if self.is_multi_gpu:
        self.reduce_parameters()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()
      mean_value_loss += logical_value
      mean_surrogate_loss += logical_surrogate
      mean_entropy += logical_entropy
      performed_updates += 1

    self.storage.clear()
    updates = max(performed_updates, 1)
    diagnostics = max(diagnostic_batches, 1)
    return {
      "value": mean_value_loss / updates,
      "surrogate": mean_surrogate_loss / updates,
      "entropy": mean_entropy / updates,
      "approx_kl": mean_approx_kl / diagnostics,
      "clip_fraction": mean_clip_fraction / diagnostics,
      "performed_updates": float(performed_updates),
      "kl_stopped_early": float(stopped_early),
    }

  @staticmethod
  def construct_algorithm(
    obs: TensorDict,
    env: VecEnv,
    cfg: dict,
    device: str,
  ) -> "VisualPPO":
    """Reuse native construction, replacing only cache-enabled storage."""
    if not cfg["algorithm"].get("cache_frozen_features", False):
      return PPO.construct_algorithm(obs, env, cfg, device)

    num_steps = cfg["num_steps_per_env"]
    cfg["num_steps_per_env"] = 0
    try:
      algorithm = PPO.construct_algorithm(obs, env, cfg, device)
    finally:
      cfg["num_steps_per_env"] = num_steps
    with torch.inference_mode():
      storage_obs = algorithm._with_cached_features(obs.to(device))
    algorithm.storage = RolloutStorage(
      "rl",
      env.num_envs,
      num_steps,
      storage_obs,
      [env.num_actions],
      device,
    )
    return algorithm
