"""The single registry of visual backbones, adapters, and their composition.

**The module tree built here is checkpoint format.** ``nn.Sequential`` index
positions, attribute names, and construction order are all weight keys for the
26 retained checkpoints; ``tests/test_vision_checkpoint_layout.py`` pins them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch.nn as nn

from .adapters import (
  AFAAdapter,
  FlattenAdapter,
  FlattenReluAdapter,
  LocalGridAdapter,
  SpatialSoftmaxAdapter,
)
from .backbones import dinov2, r3m, scratch
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
from .preprocessing import preprocess_dinov2, preprocess_r3m, to_unit_interval


BackboneBuilder = Callable[[tuple[int, int]], nn.Module]
ExtractorFactory = Callable[[FeatureRequest, int], Extract]
AdapterBuilder = Callable[
  [VisionConfig, "AdapterSpec", int, bool],
  tuple[nn.Module, int],
]


@dataclass(frozen=True)
class EncoderSpec:
  """Everything known about one registered visual backbone."""

  name: EncoderName
  channels: int
  weights: Weights
  trainable: bool
  # Requests for which this backbone's extractor returns [B, C, H, W].
  #
  # This is NOT a capability declaration. It decides whether the global and
  # linear adapters prepend AdaptiveAvgPool2d + Flatten, which shifts every
  # Sequential index -- and it gates the R3M `resnet` alias. Widening it would
  # silently break the eight registered *-Linear-* checkpoints.
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
  """Everything known about one registered adapter head."""

  name: AdapterName
  feature_request: FeatureRequest
  build: AdapterBuilder
  fixed_output_dim: int | None = None


_ALL_FEATURES: frozenset[FeatureRequest] = frozenset(
  {"global", "spatial", "local_grid"}
)
_SPATIAL_FEATURES: frozenset[FeatureRequest] = frozenset({"spatial", "local_grid"})


# --- adapter builders -------------------------------------------------------
#
# Sequential member order and count are checkpoint keys. The AdaptiveAvgPool2d
# prefix is added only for backbones whose extractor returns a spatial map, so
# `global`/`linear` on DINOv2 or R3M start at index 0.


def _global_adapter(
  config: VisionConfig,
  _: AdapterSpec,
  channels: int,
  spatial: bool,
) -> tuple[nn.Module, int]:
  layers: list[nn.Module] = []
  if spatial:
    layers.extend((nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim=1)))
  layers.append(nn.LayerNorm(channels))
  return nn.Sequential(*layers), channels


def _linear_adapter(
  config: VisionConfig,
  _: AdapterSpec,
  channels: int,
  spatial: bool,
) -> tuple[nn.Module, int]:
  # Preserve Sequential indices and therefore existing checkpoint keys.
  layers: list[nn.Module] = []
  if spatial:
    layers.extend((nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim=1)))
  layers.extend(
    (
      nn.LayerNorm(channels),
      nn.Linear(channels, config.output_dim),
      nn.LayerNorm(config.output_dim),
    )
  )
  return nn.Sequential(*layers), config.output_dim


def _flatten_adapter(
  config: VisionConfig,
  _: AdapterSpec,
  channels: int,
  spatial: bool,
) -> tuple[nn.Module, int]:
  if not spatial:  # pragma: no cover - guarded by capability declarations.
    raise ValueError("Flatten requires spatial features.")
  adapter = FlattenAdapter(
    channels,
    grid_size=config.target_grid_size,
    output_dim=config.output_dim,
  )
  return adapter, adapter.output_dim


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


def _local_grid_adapter(
  config: VisionConfig,
  _: AdapterSpec,
  channels: int,
  spatial: bool,
) -> tuple[nn.Module, int]:
  if not spatial:  # pragma: no cover - guarded by capability declarations.
    raise ValueError("LocalGrid requires spatial features.")
  adapter = LocalGridAdapter(
    channels,
    output_dim=config.output_dim,
    projected_channels=config.projected_channels,
    target_grid_size=config.target_grid_size,
  )
  return adapter, adapter.output_dim


def _spatial_softmax_adapter(
  _: VisionConfig,
  spec: AdapterSpec,
  channels: int,
  spatial: bool,
) -> tuple[nn.Module, int]:
  if not spatial:  # pragma: no cover - guarded by capability declarations.
    raise ValueError("SpatialSoftmax requires spatial features.")
  output_dim = spec.fixed_output_dim
  if output_dim is None or output_dim % 2:  # pragma: no cover - invalid declaration.
    raise ValueError("SpatialSoftmax requires an even fixed output dimension.")
  adapter = SpatialSoftmaxAdapter(channels, output_channels=output_dim // 2)
  return adapter, adapter.output_dim


def _afa_adapter(
  config: VisionConfig,
  _: AdapterSpec,
  channels: int,
  spatial: bool,
) -> tuple[nn.Module, int]:
  if not spatial:  # pragma: no cover - guarded by capability declarations.
    raise ValueError("AFA requires spatial features.")
  adapter = AFAAdapter(
    channels,
    output_dim=config.output_dim,
    num_heads=config.afa_num_heads,
  )
  return adapter, adapter.output_dim


# --- the registry -----------------------------------------------------------

ENCODERS: dict[EncoderName, EncoderSpec] = {
  "nature_cnn": EncoderSpec(
    "nature_cnn", 64, "scratch", True, _ALL_FEATURES, None,
    scratch.build_nature_cnn, to_unit_interval, scratch.nature_extractor,
  ),
  "compact_vit": EncoderSpec(
    "compact_vit", 128, "scratch", True, _ALL_FEATURES, None,
    scratch.build_compact_vit, to_unit_interval, scratch.compact_vit_extractor,
  ),
  "dinov2_vits14": EncoderSpec(
    "dinov2_vits14", 384, "pretrained", False, _SPATIAL_FEATURES, 128,
    dinov2.build, preprocess_dinov2, dinov2.make_extractor,
  ),
  "r3m_resnet50": EncoderSpec(
    "r3m_resnet50", 2048, "pretrained", False, _SPATIAL_FEATURES, 256,
    r3m.build, preprocess_r3m, r3m.make_extractor, r3m.install_spatial_alias,
  ),
  # Same frozen network, tapped one stage earlier: stride 16 instead of 32, so
  # the Push-T object spans 3.1 feature cells instead of 1.5. No `resnet` alias
  # -- that exists only to load retained layer4 checkpoints. Halved encode batch
  # because a layer3 map is twice the size of a layer4 one.
  "r3m_resnet50_layer3": EncoderSpec(
    "r3m_resnet50_layer3", 1024, "pretrained", False, _SPATIAL_FEATURES, 128,
    r3m.build, preprocess_r3m, r3m.make_layer3_extractor,
  ),
}

ADAPTERS: dict[AdapterName, AdapterSpec] = {
  "global": AdapterSpec("global", "global", _global_adapter),
  "linear": AdapterSpec("linear", "global", _linear_adapter),
  # Asks for the native map, not `local_grid`, precisely because it must not be
  # resampled: "no adapter" means the encoder's own output reaches the policy.
  "flatten": AdapterSpec("flatten", "spatial", _flatten_adapter),
  # ManiSkill3's head: the same projection as `flatten`, rectified instead of
  # normalised. A separate row because the module tree is checkpoint format.
  "flatten_relu": AdapterSpec("flatten_relu", "spatial", _flatten_relu_adapter),
  "local_grid": AdapterSpec("local_grid", "local_grid", _local_grid_adapter),
  "spatial_softmax": AdapterSpec(
    "spatial_softmax", "spatial", _spatial_softmax_adapter, 256
  ),
  "afa": AdapterSpec("afa", "spatial", _afa_adapter),
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
  """Fail before loading weights when a composition is not registered."""
  spec = encoder_spec(config.encoder)
  adapter = adapter_spec(config.adapter)
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
  if (
    adapter.fixed_output_dim is not None
    and config.output_dim != adapter.fixed_output_dim
  ):
    raise ValueError(
      f"SpatialSoftmax uses a fixed {adapter.fixed_output_dim}-dimensional output."
    )
  if config.adapter == "afa":
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
  """Compose one registered backbone and adapter into a visual encoder."""
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
  # Preserve seeded initialization order: backbone before adapter.
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
