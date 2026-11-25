from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from vbrl.paths import model_root

from ..config import FeatureRequest
from .weights import (
  DINOV3_FILE,
  DINOV3_MODEL,
  DINOV3_REPO,
  DINOV3_REVISION,
  huggingface_cache,
)

PATCH_SIZE = 16


def load(*, allow_download: bool = False) -> nn.Module:
  import timm
  from huggingface_hub import hf_hub_download
  from safetensors.torch import load_file

  path = hf_hub_download(
    DINOV3_REPO,
    DINOV3_FILE,
    revision=DINOV3_REVISION,
    cache_dir=huggingface_cache(model_root()),
    local_files_only=not allow_download,
  )
  model = timm.create_model(DINOV3_MODEL, pretrained=False, img_size=256)
  model.load_state_dict(load_file(path), strict=True)
  return model


def spatial_features(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
  tokens = backbone.forward_features(images)
  height, width = images.shape[-2] // PATCH_SIZE, images.shape[-1] // PATCH_SIZE
  patches = tokens[:, backbone.num_prefix_tokens :]
  batch, _, channels = patches.shape
  return patches.transpose(1, 2).reshape(batch, channels, height, width)


def local_grid_features(
  backbone: nn.Module, images: torch.Tensor, *, target_grid_size: int
) -> torch.Tensor:
  return F.adaptive_avg_pool2d(
    spatial_features(backbone, images), (target_grid_size, target_grid_size)
  )


# The normalised CLS token, as Meta's x_norm_clstoken; timm's own pooling averages the
# patches instead.
def global_features(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
  return backbone.forward_features(images)[:, 0]


def build(_input_dim: tuple[int, int]) -> nn.Module:
  return load()


def make_extractor(request: FeatureRequest, target_grid_size: int):
  return {
    "global": global_features,
    "spatial": spatial_features,
    "local_grid": partial(local_grid_features, target_grid_size=target_grid_size),
  }[request]


__all__ = [
  "PATCH_SIZE",
  "build",
  "global_features",
  "load",
  "local_grid_features",
  "make_extractor",
  "spatial_features",
]
