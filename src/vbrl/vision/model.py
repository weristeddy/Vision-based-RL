from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
from torch.profiler import record_function
from rsl_rl.models.cnn_model import CNNModel
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState
from tensordict import TensorDict

from .config import VisionConfig
from .encoder import VisualEncoder
from .registry import build_encoder


class VisionModel(CNNModel):
  """RSL-RL model that joins proprioception with reusable visual features."""

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


class BalancedVisionModel(VisionModel):
  """VisionModel that gives proprioception its own projection before the concat.

  MJLab's model concatenates the normalised 1D observations at their native
  width, so Push-T's actor sees 27 proprioceptive dimensions beside 256 visual
  ones -- 9.5% of the MLP's input, of which the goal pose is five numbers.
  ManiSkill3's ``ppo_rgb.py`` instead projects the state through its own
  ``Linear(state_dim, 256)`` and concatenates 256 with 256, so the two streams
  reach the policy at equal width.

  This subclass is that change and nothing else: same encoders, same adapters,
  same MLP widths. It exists as a separate class because
  ``"vbrl.vision.model:VisionModel"`` is a string inside every registered
  ``rl_cfg`` and inside saved checkpoints, so the behaviour cannot be switched
  on in place without changing what those checkpoints mean.
  """

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
    state_latent_dim: int = 256,
  ) -> None:
    if state_latent_dim <= 0:
      raise ValueError(f"state_latent_dim must be positive, got {state_latent_dim}.")
    # Read by `_get_latent_dim`, which the MLP head calls during construction.
    self._state_latent_dim = state_latent_dim
    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
      cnn_cfg=cnn_cfg,
      cnns=cnns,
    )
    # No activation, matching ManiSkill3: the ReLU that follows belongs to the
    # policy MLP's first layer, which both streams share.
    self.state_proj = nn.Linear(self.obs_dim, self._state_latent_dim)

  def _get_latent_dim(self) -> int:
    return self._state_latent_dim + self.cnn_latent_dim

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    del masks, hidden_state
    latent_1d = self.state_proj(MLPModel.get_latent(self, obs))
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


__all__ = ["VisionModel", "BalancedVisionModel"]
