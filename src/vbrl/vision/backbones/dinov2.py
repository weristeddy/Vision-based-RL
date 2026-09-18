from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from vbrl.paths import model_root

from ..config import FeatureRequest
from .weights import (
  DINOV2_REPO,
  DINOV2_REVISION,
  huggingface_cache,
)


def load(*, allow_download: bool = False) -> nn.Module:
  # transformers imports an ABI-incompatible torchaudio; DINOv2 has no audio.
  import transformers.utils as transformers_utils
  from transformers.utils import import_utils

  import_utils.is_torchaudio_available = lambda: False
  transformers_utils.is_torchaudio_available = lambda: False
  from transformers import AutoModel

  return AutoModel.from_pretrained(
    DINOV2_REPO,
    revision=DINOV2_REVISION,
    cache_dir=huggingface_cache(model_root()),
    local_files_only=not allow_download,
  )


def _patch_grid(backbone: nn.Module, images: torch.Tensor) -> tuple[int, int]:
  patch = int(backbone.config.patch_size)
  return images.shape[-2] // patch, images.shape[-1] // patch


def spatial_features(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
  tokens = backbone(images)["last_hidden_state"]
  height, width = _patch_grid(backbone, images)
  patches = tokens[:, int(tokens.shape[1]) - height * width :]
  batch, _, channels = patches.shape
  return patches.transpose(1, 2).reshape(batch, channels, height, width)


def local_grid_features(
  backbone: nn.Module, images: torch.Tensor, *, target_grid_size: int
) -> torch.Tensor:
  return F.adaptive_avg_pool2d(
    spatial_features(backbone, images), (target_grid_size, target_grid_size)
  )


def global_features(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
  return backbone(images)["pooler_output"]


def build(_input_dim: tuple[int, int]) -> nn.Module:
  return load()


def make_extractor(request: FeatureRequest, target_grid_size: int):
  return {
    "global": global_features,
    "spatial": spatial_features,
    "local_grid": partial(local_grid_features, target_grid_size=target_grid_size),
  }[request]


__all__ = [
  "build",
  "global_features",
  "load",
  "local_grid_features",
  "make_extractor",
  "spatial_features",
]
