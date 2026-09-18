from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
from rsl_rl.models.cnn_model import CNNModel
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState
from tensordict import TensorDict
from torch.profiler import record_function

from .config import VisionConfig
from .encoder import VisualEncoder
from .registry import build_encoder


class VisionModel(CNNModel):
  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    cnn_cfg: dict[str, Any] | None = None,
    cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
  ) -> None:
    self._get_obs_dim(obs, obs_groups, obs_set)

    if cnns is not None:
      if set(cnns) != set(self.obs_groups_2d):
        raise ValueError("Shared encoders must cover the same image observation groups.")
      resolved_cnns = cnns
    else:
      if cnn_cfg is None:
        raise ValueError("VisionModel requires cnn_cfg when encoders are not shared.")
      built: dict[str, nn.Module] = {}
      for index, observation_group in enumerate(self.obs_groups_2d):
        vision_cfg = VisionConfig.from_mapping(cnn_cfg.get("vision", cnn_cfg))
        visual_encoder = build_encoder(
          vision_cfg,
          input_dim=self.obs_dims_2d[index],
          input_channels=self.obs_channels_2d[index],
        )
        built[observation_group] = visual_encoder
      resolved_cnns = built

    self.cnn_latent_dim = sum(
      cast(VisualEncoder, module).output_dim for module in resolved_cnns.values()
    )

    MLPModel.__init__(
      self,
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )
    self.cnns = resolved_cnns if isinstance(resolved_cnns, nn.ModuleDict) else nn.ModuleDict(resolved_cnns)

  @staticmethod
  def feature_key(observation_group: str) -> str:
    return f"{observation_group}_features"

  @property
  def supports_cached_features(self) -> bool:
    return any(self._can_cache(encoder) for encoder in self.cnns.values())

  def add_cached_features(
    self,
    obs: TensorDict,
    *,
    drop_raw_images: bool = False,
    feature_cache_dtype: torch.dtype | None = None,
  ) -> TensorDict:
    if not self.supports_cached_features:
      return obs
    cached = obs.clone(False)
    for observation_group in self.obs_groups_2d:
      key = self.feature_key(observation_group)
      encoder = cast(VisualEncoder, self.cnns[observation_group])
      if key not in cached.keys() and self._can_cache(encoder):
        with record_function("frozen_backbone"):
          features = encoder.encode_features(obs[observation_group])
        if feature_cache_dtype is not None:
          features = features.to(dtype=feature_cache_dtype)
        cached[key] = features.detach()
        if drop_raw_images and observation_group in cached.keys():
          del cached[observation_group]
    return cached

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    del masks, hidden_state
    latent_1d = MLPModel.get_latent(self, obs)
    visual = []
    for observation_group in self.obs_groups_2d:
      encoder = cast(VisualEncoder, self.cnns[observation_group])
      feature_key = self.feature_key(observation_group)
      if feature_key in obs.keys():
        with record_function("visual_adapter"):
          visual.append(encoder.project_features(obs[feature_key]))
      else:
        with record_function("visual_encoder"):
          visual.append(encoder(obs[observation_group]))
    return torch.cat([latent_1d, *visual], dim=-1)

  @staticmethod
  def _can_cache(encoder: nn.Module) -> bool:
    return bool(
      getattr(encoder, "freeze_backbone", False)
      and hasattr(encoder, "encode_features")
      and hasattr(encoder, "project_features")
    )


__all__ = ["VisionModel"]
