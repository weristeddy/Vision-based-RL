from __future__ import annotations

from vbrl.vision.config import AdapterName, EncoderName, VisionConfig

PRETRAINED_ENCODERS = frozenset(
  {"dinov2_vits14", "dinov3_vits16", "r3m_resnet50", "r3m_resnet50_layer3"}
)

# AFA head count is not a free parameter.
AFA_HEAD_DIM = 64


def vision_cfg(
  encoder: EncoderName,
  adapter: AdapterName,
  *,
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
    target_grid_size=target_grid_size,
    afa_num_heads=afa_num_heads,
  )
  cfg.validate()
  return cfg


# The key is the exact token used in a task ID. A row here is a menu entry, not a
# registration: an ID exists only where a `register_mjlab_task` line creates it.
ARCHITECTURES: dict[str, VisionConfig] = {
  # ManiSkill3's exact head for the trainable CNN.
  "NatureCnn-FlattenRelu": vision_cfg(
    "nature_cnn", "flatten_relu", target_grid_size=24
  ),
  # SVEA's ViT head: the CLS token through Linear, LayerNorm and Tanh.
  "CompactVit-Svea": vision_cfg("compact_vit", "svea"),
  "DinoV2ViTS14-Global": vision_cfg("dinov2_vits14", "global_normalized"),
  "DinoV2ViTS14-AfaFull6": vision_cfg(
    "dinov2_vits14", "afa_full", afa_num_heads=384 // AFA_HEAD_DIM
  ),
  "R3MResNet50-Global": vision_cfg("r3m_resnet50", "global_normalized"),
  "R3MResNet50-AfaFull32": vision_cfg(
    "r3m_resnet50", "afa_full", afa_num_heads=2048 // AFA_HEAD_DIM
  ),
  "DinoV3ViTS16-Global": vision_cfg("dinov3_vits16", "global_normalized"),
  "DinoV3ViTS16-AfaFull6": vision_cfg(
    "dinov3_vits16", "afa_full", afa_num_heads=384 // AFA_HEAD_DIM
  ),
  "R3MResNet50L3-AfaFull16": vision_cfg(
    "r3m_resnet50_layer3", "afa_full", afa_num_heads=1024 // AFA_HEAD_DIM
  ),
}


__all__ = [
  "ARCHITECTURES",
  "PRETRAINED_ENCODERS",
  "vision_cfg",
]
