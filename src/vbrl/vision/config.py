"""The serializable visual-policy configuration.

Free of torch and MJLab imports because task ``rl_cfg`` modules import it at
module scope. Field names are wire format inside ``cnn_cfg["vision"]`` and in
W&B run configs: renaming one breaks reading historical runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Literal, Mapping, TypeAlias, cast, get_args


EncoderName: TypeAlias = Literal[
  "none",
  "nature_cnn",
  "compact_vit",
  "dinov2_vits14",
  "r3m_resnet50",
  "r3m_resnet50_layer3",
]
AdapterName: TypeAlias = Literal[
  "none",
  "global",
  "linear",
  "flatten",
  "flatten_relu",
  "local_grid",
  "spatial_softmax",
  "afa",
]
Weights: TypeAlias = Literal["scratch", "pretrained"]
FloatDType: TypeAlias = Literal["bfloat16", "float16", "float32"]


FeatureRequest: TypeAlias = Literal["global", "spatial", "local_grid"]

ENCODER_NAMES: tuple[str, ...] = get_args(EncoderName)
ADAPTER_NAMES: tuple[str, ...] = get_args(AdapterName)
FLOAT_DTYPES: frozenset[str] = frozenset(get_args(FloatDType))

# Fields that older W&B run configs and cnn_cfg blobs may still carry. Every
# one of them only ever held its default, so dropping them cannot change a
# reconstructed policy -- but the keys must still parse.
RETIRED_FIELDS: frozenset[str] = frozenset(
  {
    "image_width",
    "image_height",
    "adapter_activation",
    "normalize_features",
    "afa_positional_encoding",
    "memory_format",
  }
)


@dataclass
class VisionConfig:
  """Complete, serializable visual-policy configuration.

  The four supported encoders deliberately describe the combinations that are
  trained and checkpointed in this repository. New backbones can be registered
  without adding robot- or task-specific branches.
  """

  encoder: EncoderName = "none"
  weights: Weights = "scratch"
  train_encoder: bool = False
  adapter: AdapterName = "none"
  output_dim: int = 256
  projected_channels: int = 64
  target_grid_size: int = 7
  afa_num_heads: int = 8
  encoder_autocast: bool = True
  encoder_autocast_dtype: FloatDType = "bfloat16"
  encode_batch_size: int | None = None

  @classmethod
  def from_mapping(cls, value: Mapping[str, Any] | None) -> "VisionConfig":
    if value is None:
      return cls()
    valid = {field.name for field in fields(cls)}
    unknown = sorted(set(value) - valid - RETIRED_FIELDS)
    if unknown:
      raise ValueError(f"Unknown vision configuration fields: {unknown}.")
    # Historical W&B run configs still carry the retired fields; drop them so
    # an old run stays readable instead of raising.
    normalized = {key: item for key, item in value.items() if key in valid}
    if "encoder" in normalized:
      normalized["encoder"] = normalize_encoder_name(str(normalized["encoder"]))
    if "adapter" in normalized:
      normalized["adapter"] = normalize_adapter_name(str(normalized["adapter"]))
    encoder = cast(EncoderName, normalized.get("encoder", "none"))
    if "weights" not in normalized:
      from .registry import ENCODERS

      normalized["weights"] = (
        "scratch" if encoder == "none" else ENCODERS[encoder].weights
      )
    return cls(**normalized)

  def asdict(self) -> dict[str, Any]:
    return asdict(self)

  @property
  def frozen(self) -> bool:
    return self.encoder != "none" and not self.train_encoder

  def validate(self) -> None:
    if self.encode_batch_size is not None and (
      isinstance(self.encode_batch_size, bool)
      or not isinstance(self.encode_batch_size, int)
      or self.encode_batch_size <= 0
    ):
      raise ValueError(
        "encode_batch_size must be positive or None, "
        f"got {self.encode_batch_size}."
      )
    if self.encoder_autocast_dtype not in FLOAT_DTYPES:
      raise ValueError(
        f"Unsupported autocast dtype {self.encoder_autocast_dtype!r}."
      )

    if self.encoder == "none":
      if self.adapter != "none":
        raise ValueError("encoder='none' requires adapter='none'.")
      return
    if self.adapter == "none":
      raise ValueError("An RGB encoder requires an adapter.")

    # Deferred so this module stays importable without torch: the registry is
    # authoritative for which backbone/adapter combinations exist.
    from .registry import check_composition

    check_composition(self)


def normalize_encoder_name(name: str) -> EncoderName:
  normalized = name.lower()
  if normalized not in ENCODER_NAMES:
    raise ValueError(f"Unknown encoder {name!r}; choose from {ENCODER_NAMES}.")
  return cast(EncoderName, normalized)


def normalize_adapter_name(name: str) -> AdapterName:
  normalized = name.lower()
  if normalized not in ADAPTER_NAMES:
    raise ValueError(f"Unknown adapter {name!r}; choose from {ADAPTER_NAMES}.")
  return cast(AdapterName, normalized)


__all__ = [
  "ADAPTER_NAMES",
  "ENCODER_NAMES",
  "FLOAT_DTYPES",
  "RETIRED_FIELDS",
  "AdapterName",
  "EncoderName",
  "FeatureRequest",
  "FloatDType",
  "VisionConfig",
  "Weights",
  "normalize_adapter_name",
  "normalize_encoder_name",
]
