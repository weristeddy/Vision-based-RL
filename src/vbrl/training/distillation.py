from __future__ import annotations

from dataclasses import dataclass, field

import torch
from mjlab.rl import RslRlModelCfg
from mjlab.rl.config import RslRlBaseRunnerCfg
from rsl_rl.algorithms import Distillation
from rsl_rl.env import VecEnv
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from vbrl.training.ppo import resolve_cache_dtype


@dataclass
class VisualDistillationCfg:
  num_learning_epochs: int = 2
  gradient_length: int = 4
  learning_rate: float = 1.0e-3
  max_grad_norm: float | None = 1.0
  loss_type: str = "mse"
  optimizer: str = "adam"
  feature_cache_dtype: str = "bfloat16"
  class_name: str = "vbrl.training.distillation:VisualDistillation"


@dataclass
class DistillationRunnerCfg(RslRlBaseRunnerCfg):
  student: RslRlModelCfg = field(default_factory=RslRlModelCfg)
  teacher: RslRlModelCfg = field(default_factory=RslRlModelCfg)
  algorithm: VisualDistillationCfg = field(default_factory=VisualDistillationCfg)
  class_name: str = "DistillationRunner"


class VisualDistillation(Distillation):
  def __init__(self, *args, feature_cache_dtype: str = "bfloat16", **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.feature_cache_dtype = resolve_cache_dtype(feature_cache_dtype)

  def act(self, obs: TensorDict) -> torch.Tensor:
    return super().act(self._with_cached_features(obs))

  def _with_cached_features(self, obs: TensorDict) -> TensorDict:
    return self._raw_student.add_cached_features(
      obs, drop_raw_images=True, feature_cache_dtype=self.feature_cache_dtype
    )

  @staticmethod
  def construct_algorithm(
    obs: TensorDict, env: VecEnv, cfg: dict, device: str
  ) -> VisualDistillation:
    num_steps = cfg["num_steps_per_env"]
    cfg["num_steps_per_env"] = 0
    algorithm = Distillation.construct_algorithm(obs, env, cfg, device)
    cfg["num_steps_per_env"] = num_steps
    with torch.inference_mode():
      storage_obs = algorithm._with_cached_features(obs.to(device))
    algorithm.storage = RolloutStorage(
      "distillation", env.num_envs, num_steps, storage_obs, [env.num_actions], device
    )
    return algorithm


__all__ = ["DistillationRunnerCfg", "VisualDistillation", "VisualDistillationCfg"]
