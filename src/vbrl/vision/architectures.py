"""Named visual architectures, keyed by the token that appears in a task ID.

One row here is one encoder + adapter pairing. Crossing this table with a task
is what produces registered IDs like
``Mjlab-PushT-RealTexture-DinoV2ViTS14-LocalGrid7-TrossenRealistic``, so adding
a row makes that architecture available to every task that crosses it.

Free of torch and MJLab imports: task ``rl_cfg`` modules import this at module
scope.
"""

from __future__ import annotations

from vbrl.vision.config import AdapterName, EncoderName, VisionConfig


PRETRAINED_ENCODERS = frozenset(
  {"dinov2_vits14", "r3m_resnet50", "r3m_resnet50_layer3"}
)

# AFA head count is not a free parameter. `AttentionPoolLatent` splits the
# encoder's channels into 64-wide heads -- the width DINOv2 itself uses (384/6)
# and the timm default the published layout was taken from (`num_heads =
# dim // 64`). Every `Afa<N>` row below is therefore `channels // 64`, and
# `test_vision.py` pins it so a new encoder cannot pick N by hand.
AFA_HEAD_DIM = 64


def vision_cfg(
  encoder: EncoderName,
  adapter: AdapterName,
  *,
  projected_channels: int = 64,
  target_grid_size: int = 7,
  afa_num_heads: int = 8,
) -> VisionConfig:
  """Build one validated architecture; pretrained backbones stay frozen."""
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


# Adding a row here makes the architecture available to every task that crosses
# this table. The key is the exact token used in the task ID.
ARCHITECTURES: dict[str, VisionConfig] = {
  # Scratch encoders: the two pooling heads, plus the bare flatten that is what
  # "no adapter" means for a trainable trunk. Grid sizes are each encoder's own.
  "NatureCnn-Flatten": vision_cfg("nature_cnn", "flatten", target_grid_size=24),
  "NatureCnn-SpatialSoftmax": vision_cfg("nature_cnn", "spatial_softmax"),
  "NatureCnn-Afa1": vision_cfg("nature_cnn", "afa", afa_num_heads=64 // AFA_HEAD_DIM),
  "CompactVit-Flatten": vision_cfg("compact_vit", "flatten", target_grid_size=14),
  "CompactVit-SpatialSoftmax": vision_cfg("compact_vit", "spatial_softmax"),
  "CompactVit-Afa2": vision_cfg(
    "compact_vit", "afa", afa_num_heads=128 // AFA_HEAD_DIM
  ),
  # ManiSkill3's exact head for the two trainable trunks: the same projection
  # as `-Flatten`, rectified rather than layer-normed. Only the scratch encoders
  # get it -- a ReLU on a coordinate or attention readout would clip half its
  # range, and ManiSkill applies it to a flattened conv map.
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
  # R3M one stage earlier. At its published layer4 tap the object spans 1.5
  # feature cells and no adapter recovers yaw; layer3 gives it 3.1, the same as
  # CompactVit, which does. The layer4 rows above stay as the published baseline.
  "R3MResNet50L3-LocalGrid14": vision_cfg(
    "r3m_resnet50_layer3", "local_grid", target_grid_size=14
  ),
  "R3MResNet50L3-SpatialSoftmax": vision_cfg(
    "r3m_resnet50_layer3", "spatial_softmax"
  ),
  "R3MResNet50L3-Afa16": vision_cfg(
    "r3m_resnet50_layer3", "afa", afa_num_heads=1024 // AFA_HEAD_DIM
  ),
  # --- superseded, kept because trained weights load against them ------------
  #
  # Each row below pools its encoder's feature map down to a grid the encoder
  # does not produce. That is the loss the rows above exist to remove.
  "NatureCnn-LocalGrid7": vision_cfg("nature_cnn", "local_grid", target_grid_size=7),
  "NatureCnn-LocalGrid16": vision_cfg("nature_cnn", "local_grid", target_grid_size=16),
  "CompactVit-LocalGrid8": vision_cfg(
    "compact_vit", "local_grid", projected_channels=32, target_grid_size=8
  ),
  "DinoV2ViTS14-LocalGrid7": vision_cfg(
    "dinov2_vits14", "local_grid", target_grid_size=7
  ),
}


# What a new registration crosses -- every row above the superseded block.
#
# *Native resolution.* Each adapter reads its encoder's own feature grid --
# NatureCnn 24x24, CompactVit 14x14, DINOv2 16x16, R3M 7x7 -- instead of
# average-pooling to a shared one. Pooling to 7 was meant to make encoders
# comparable, but it only ever discarded, and it discarded unequally: R3M was
# left untouched while NatureCnn lost twelve cells out of every thirteen. The
# Push-T target spans about two cells after pooling, which is what a yaw
# estimate then has to survive.
#
# *Head size.* A local grid's final dense layer costs
# grid^2 x projected_channels x output_dim, which at native resolution is 9.4M
# parameters against NatureCnn's 76k trunk. It is kept only for the frozen
# backbones, where 4.2M against DINOv2's 21M is proportionate. The scratch
# encoders instead get SpatialSoftmax (8.8k / 17k) and AFA (30k / 83k), which
# read the whole grid through pooling that has no dense flatten.
#
# Flatten is the deliberate exception: 9.4M / 6.4M, the same cost, because it is
# the plain Nature-CNN head and the point of the row is to have no adapter at
# all. Dropping it would not remove that matrix, only move it into the policy
# MLP's first layer -- so it is a baseline for the pooling heads, not a
# violation of the rule above.
CURRENT_ARCHITECTURES: tuple[str, ...] = tuple(ARCHITECTURES)[:19]


def list_architectures() -> tuple[str, ...]:
  """Return every architecture token, in table order."""
  return tuple(ARCHITECTURES)


def get_architecture(token: str) -> VisionConfig:
  """Resolve one architecture token, listing the choices when it is unknown."""
  try:
    return ARCHITECTURES[token]
  except KeyError as exc:
    choices = ", ".join(ARCHITECTURES)
    raise ValueError(
      f"Unknown architecture {token!r}. Choose one of: {choices}."
    ) from exc


__all__ = [
  "ARCHITECTURES",
  "CURRENT_ARCHITECTURES",
  "PRETRAINED_ENCODERS",
  "get_architecture",
  "list_architectures",
  "vision_cfg",
]
