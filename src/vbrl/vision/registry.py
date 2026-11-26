from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch.nn as nn

from .adapters import AFAAdapter, FlattenReluAdapter, L2Normalize
from .backbones import dinov2, dinov3, r3m, scratch
from .config import (
  ADAPTER_NAMES,
  ENCODER_NAMES,
  AdapterName,
  EncoderName,
  FeatureRequest,
  VisionConfig,
  Weights,
)
from .encoder import Extract, Preprocess, VisualEncoder
from .preprocessing import (
  preprocess_dinov2,
  preprocess_dinov3,
  preprocess_r3m,
  to_unit_interval,
)

BackboneBuilder = Callable[[tuple[int, int]], nn.Module]
ExtractorFactory = Callable[[FeatureRequest, int], Extract]
AdapterBuilder = Callable[
  [VisionConfig, "AdapterSpec", int, bool],
  tuple[nn.Module, int],
]


@dataclass(frozen=True)
class EncoderSpec:
  name: EncoderName
  channels: int
  weights: Weights
  trainable: bool
  spatial_requests: frozenset[FeatureRequest]
  default_encode_batch_size: int | None
  build: BackboneBuilder
  preprocess: Preprocess
  make_extractor: ExtractorFactory
  # Registers a second name for the backbone's ResNet view; retained R3M
  # spatial checkpoints stored both, pointing at the same parameters.
  spatial_alias: Callable[[nn.Module], None] | None = None


@dataclass(frozen=True)
class AdapterSpec:
  name: AdapterName
  feature_request: FeatureRequest
  build: AdapterBuilder


_ALL_FEATURES: frozenset[FeatureRequest] = frozenset({"global", "spatial"})
_SPATIAL_FEATURES: frozenset[FeatureRequest] = frozenset({"spatial"})

def _global_normalized_adapter(
  config: VisionConfig,
  _: AdapterSpec,
  channels: int,
  spatial: bool,
) -> tuple[nn.Module, int]:
  pool = (nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim=1)) if spatial else ()
  return nn.Sequential(*pool, L2Normalize(), nn.LayerNorm(channels)), channels


def _svea_adapter(
  config: VisionConfig,
  _: AdapterSpec,
  channels: int,
  spatial: bool,
) -> tuple[nn.Module, int]:
  pool = (nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim=1)) if spatial else ()
  return nn.Sequential(
    *pool,
    nn.Linear(channels, config.output_dim),
    nn.LayerNorm(config.output_dim),
    nn.Tanh(),
  ), config.output_dim


def _flatten_relu_adapter(
  config: VisionConfig,
  _: AdapterSpec,
  channels: int,
  spatial: bool,
) -> tuple[nn.Module, int]:
  if not spatial:  # pragma: no cover - guarded by capability declarations.
    raise ValueError("FlattenRelu requires spatial features.")
  adapter = FlattenReluAdapter(
    channels,
    grid_size=config.target_grid_size,
    output_dim=config.output_dim,
  )
  return adapter, adapter.output_dim


def _afa_full_adapter(
  config: VisionConfig,
  _: AdapterSpec,
  channels: int,
  spatial: bool,
) -> tuple[nn.Module, int]:
  if not spatial:  # pragma: no cover - guarded by capability declarations.
    raise ValueError("AFA requires spatial features.")
  pool = AFAAdapter(channels, num_heads=config.afa_num_heads)
  return nn.Sequential(pool, nn.LayerNorm(channels)), channels


ENCODERS: dict[EncoderName, EncoderSpec] = {
  "nature_cnn": EncoderSpec(
    "nature_cnn", 64, "scratch", True, _ALL_FEATURES, None,
    scratch.build_nature_cnn, to_unit_interval, scratch.nature_extractor,
  ),
  "compact_vit": EncoderSpec(
    "compact_vit", 128, "scratch", True, _SPATIAL_FEATURES, None,
    scratch.build_compact_vit, to_unit_interval, scratch.compact_vit_extractor,
  ),
  "dinov2_vits14": EncoderSpec(
    "dinov2_vits14", 384, "pretrained", False, _SPATIAL_FEATURES, 128,
    dinov2.build, preprocess_dinov2, dinov2.make_extractor,
  ),
  "dinov3_vits16": EncoderSpec(
    "dinov3_vits16", 384, "pretrained", False, _SPATIAL_FEATURES, 128,
    dinov3.build, preprocess_dinov3, dinov3.make_extractor,
  ),
  "r3m_resnet50": EncoderSpec(
    "r3m_resnet50", 2048, "pretrained", False, _SPATIAL_FEATURES, 256,
    r3m.build, preprocess_r3m, r3m.make_extractor, r3m.install_spatial_alias,
  ),
  # Same frozen network, tapped one stage earlier: stride 16 instead of 32, so the
  # Push-T object spans 3.1 feature cells instead of 1.5.
  "r3m_resnet50_layer3": EncoderSpec(
    "r3m_resnet50_layer3", 1024, "pretrained", False, _SPATIAL_FEATURES, 128,
    r3m.build, preprocess_r3m, r3m.make_layer3_extractor,
  ),
}

ADAPTERS: dict[AdapterName, AdapterSpec] = {
  "global_normalized": AdapterSpec(
    "global_normalized", "global", _global_normalized_adapter
  ),
  # ManiSkill3's head on the native map.
  "flatten_relu": AdapterSpec("flatten_relu", "spatial", _flatten_relu_adapter),
  # SVEA's RLProjection on the global feature; for CompactViT, its CLS token.
  "svea": AdapterSpec("svea", "global", _svea_adapter),
  "afa_full": AdapterSpec("afa_full", "spatial", _afa_full_adapter),
}


def list_encoders() -> tuple[str, ...]:
  return ENCODER_NAMES


def list_adapters() -> tuple[str, ...]:
  return ADAPTER_NAMES


def encoder_spec(name: str) -> EncoderSpec:
  try:
    return ENCODERS[name]  # type: ignore[index]
  except KeyError as exc:
    raise ValueError(
      f"Unknown encoder {name!r}; choose from {ENCODER_NAMES}."
    ) from exc


def adapter_spec(name: str) -> AdapterSpec:
  try:
    return ADAPTERS[name]  # type: ignore[index]
  except KeyError as exc:
    raise ValueError(
      f"Unknown adapter {name!r}; choose from {ADAPTER_NAMES}."
    ) from exc


def check_composition(config: VisionConfig) -> None:
  spec = encoder_spec(config.encoder)
  adapter_spec(config.adapter)
  if config.weights != spec.weights:
    raise ValueError(
      f"{config.encoder!r} currently supports weights={spec.weights!r}, "
      f"not {config.weights!r}."
    )
  if config.train_encoder != spec.trainable:
    mode = "trainable" if spec.trainable else "frozen"
    raise ValueError(f"{config.encoder!r} is supported as a {mode} encoder.")
  if config.output_dim <= 0:
    raise ValueError(f"output_dim must be positive, got {config.output_dim}.")
  if config.adapter == "afa_full":
    if (
      isinstance(config.afa_num_heads, bool)
      or not isinstance(config.afa_num_heads, int)
      or config.afa_num_heads <= 0
    ):
      raise ValueError(
        f"afa_num_heads must be positive, got {config.afa_num_heads}."
      )
    if spec.channels % config.afa_num_heads:
      raise ValueError(
        f"AFA heads ({config.afa_num_heads}) must divide encoder channels "
        f"({spec.channels})."
      )


def build_encoder(
  config: VisionConfig,
  *,
  input_dim: tuple[int, int],
  input_channels: int = 3,
) -> VisualEncoder:
  config.validate()
  if config.encoder == "none":
    raise ValueError("encoder='none' does not create a visual encoder.")
  if input_channels != 3:
    raise ValueError(
      f"Visual encoders require three RGB channels, got {input_channels}."
    )

  spec, adapter_cfg = ENCODERS[config.encoder], ADAPTERS[config.adapter]
  request = adapter_cfg.feature_request
  spatial = request in spec.spatial_requests
  backbone = spec.build(input_dim)
  adapter, output_dim = adapter_cfg.build(config, adapter_cfg, spec.channels, spatial)
  batch_size = config.encode_batch_size
  if batch_size is None and config.frozen:
    batch_size = spec.default_encode_batch_size

  encoder = VisualEncoder(
    backbone=backbone,
    adapter=adapter,
    preprocess=spec.preprocess,
    extract=spec.make_extractor(request, config.target_grid_size),
    output_dim=output_dim,
    frozen=config.frozen,
    encode_batch_size=batch_size,
    autocast=config.encoder_autocast,
    autocast_dtype=config.encoder_autocast_dtype,
  )
  if spec.spatial_alias is not None and spatial:
    spec.spatial_alias(encoder)
  return encoder


__all__ = [
  "ADAPTERS",
  "ENCODERS",
  "AdapterSpec",
  "EncoderSpec",
  "VisualEncoder",
  "adapter_spec",
  "build_encoder",
  "check_composition",
  "encoder_spec",
  "list_adapters",
  "list_encoders",
]
