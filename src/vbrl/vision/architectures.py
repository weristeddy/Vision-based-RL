from __future__ import annotations

from vbrl.vision.config import AdapterName, EncoderName, VisionConfig

PRETRAINED_ENCODERS = frozenset(
  {"dinov2_vits14", "r3m_resnet50", "r3m_resnet50_layer3"}
)

# AFA head count is not a free parameter.
AFA_HEAD_DIM = 64


def vision_cfg(
  encoder: EncoderName,
  adapter: AdapterName,
  *,
  projected_channels: int = 64,
  target_grid_size: int = 7,
  afa_num_heads: int = 8,
) -> VisionConfig:
  frozen = encoder in PRETRAINED_ENCODERS
  cfg = VisionConfig(
    encoder=encoder,
    weights="pretrained" if frozen else "scratch",
    train_encoder=not frozen,
    adapter=adapter,
    output_dim=256,
    projected_channels=projected_channels,
    target_grid_size=target_grid_size,
    afa_num_heads=afa_num_heads,
  )
  cfg.validate()
  return cfg


# The key is the exact token used in a task ID. A row here is a menu entry, not a
# registration: an ID exists only where a `register_mjlab_task` line creates it, so rows.
ARCHITECTURES: dict[str, VisionConfig] = {
  "NatureCnn-Flatten": vision_cfg("nature_cnn", "flatten", target_grid_size=24),
  "NatureCnn-SpatialSoftmax": vision_cfg("nature_cnn", "spatial_softmax"),
  "CompactVit-Flatten": vision_cfg("compact_vit", "flatten", target_grid_size=14),
  "CompactVit-SpatialSoftmax": vision_cfg("compact_vit", "spatial_softmax"),
  # ManiSkill3's exact head for the two trainable trunks: the same projection as
  # `-Flatten`, rectified rather than layer-normed.
  "NatureCnn-FlattenRelu": vision_cfg(
    "nature_cnn", "flatten_relu", target_grid_size=24
  ),
  "CompactVit-FlattenRelu": vision_cfg(
    "compact_vit", "flatten_relu", target_grid_size=14
  ),
  "DinoV2ViTS14-Linear": vision_cfg("dinov2_vits14", "linear"),
  "DinoV2ViTS14-LocalGrid16": vision_cfg(
    "dinov2_vits14", "local_grid", target_grid_size=16
  ),
  "DinoV2ViTS14-SpatialSoftmax": vision_cfg("dinov2_vits14", "spatial_softmax"),
  "DinoV2ViTS14-Afa6": vision_cfg(
    "dinov2_vits14", "afa", afa_num_heads=384 // AFA_HEAD_DIM
  ),
  "R3MResNet50-Linear": vision_cfg("r3m_resnet50", "linear"),
  "R3MResNet50-LocalGrid7": vision_cfg(
    "r3m_resnet50", "local_grid", target_grid_size=7
  ),
  "R3MResNet50-SpatialSoftmax": vision_cfg("r3m_resnet50", "spatial_softmax"),
  "R3MResNet50-Afa32": vision_cfg(
    "r3m_resnet50", "afa", afa_num_heads=2048 // AFA_HEAD_DIM
  ),
  "R3MResNet50L3-LocalGrid14": vision_cfg(
    "r3m_resnet50_layer3", "local_grid", target_grid_size=14
  ),
  "R3MResNet50L3-SpatialSoftmax": vision_cfg(
    "r3m_resnet50_layer3", "spatial_softmax"
  ),
  "R3MResNet50L3-Afa16": vision_cfg(
    "r3m_resnet50_layer3", "afa", afa_num_heads=1024 // AFA_HEAD_DIM
  ),
  # Each row below pools its encoder's feature map down to a grid the encoder does not
  # produce.
  "NatureCnn-LocalGrid7": vision_cfg("nature_cnn", "local_grid", target_grid_size=7),
  "NatureCnn-LocalGrid16": vision_cfg("nature_cnn", "local_grid", target_grid_size=16),
  "CompactVit-LocalGrid8": vision_cfg(
    "compact_vit", "local_grid", projected_channels=32, target_grid_size=8
  ),
  "DinoV2ViTS14-LocalGrid7": vision_cfg(
    "dinov2_vits14", "local_grid", target_grid_size=7
  ),
}


# The 17 rows above the superseded block: what a new registration crosses.
CURRENT_ARCHITECTURES: tuple[str, ...] = tuple(ARCHITECTURES)[:17]


__all__ = [
  "ARCHITECTURES",
  "CURRENT_ARCHITECTURES",
  "PRETRAINED_ENCODERS",
  "vision_cfg",
]
