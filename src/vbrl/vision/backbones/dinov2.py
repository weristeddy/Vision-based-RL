"""Pinned DINOv2 loading and feature extraction."""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vbrl.paths import model_root

from .weights import DINOV2_DIRECTORY

from ..config import FeatureRequest




def load() -> nn.Module:
  # Transformers may import an ABI-incompatible torchaudio build even though
  # DINOv2 never uses audio. Mark it unavailable before resolving AutoModel.
  try:
    import transformers.utils as transformers_utils
    from transformers.utils import import_utils

    import_utils.is_torchaudio_available = lambda: False
    transformers_utils.is_torchaudio_available = lambda: False
    from transformers import AutoModel
  except ImportError as exc:
    raise ImportError("DINOv2 requires the pinned transformers dependency.") from exc
  path = model_root() / DINOV2_DIRECTORY
  if not (path / "config.json").is_file() or not (
    path / "model.safetensors"
  ).is_file():
    raise FileNotFoundError(
      f"Pinned DINOv2 assets are missing from {path}. "
      "Rebuild rl.sif from rl.def."
    )
  return AutoModel.from_pretrained(str(path), local_files_only=True)


def _output_tensor(outputs: Any, *names: str) -> torch.Tensor | None:
  for name in names:
    value = getattr(outputs, name, None)
    if isinstance(value, torch.Tensor):
      return value
    if isinstance(outputs, dict) and isinstance(outputs.get(name), torch.Tensor):
      return outputs[name]
  return None


def spatial_features(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
  outputs = backbone(images)
  patches = _output_tensor(outputs, "x_norm_patchtokens", "patch_tokens")
  if patches is None:
    tokens = _output_tensor(outputs, "last_hidden_state", "tokens")
    if tokens is None or tokens.ndim != 3:
      raise ValueError("DINOv2 did not return patch tokens.")
    patch_size = int(getattr(getattr(backbone, "config", None), "patch_size", 14))
    expected = (images.shape[-2] // patch_size) * (images.shape[-1] // patch_size)
    register_tokens = int(
      getattr(getattr(backbone, "config", None), "num_register_tokens", 0) or 0
    )
    start = tokens.shape[1] - expected
    if start not in {0, 1, 1 + register_tokens}:
      raise ValueError(
        f"DINOv2 returned {tokens.shape[1]} tokens for {expected} image patches."
      )
    patches = tokens[:, start:]
  patch_size = int(getattr(getattr(backbone, "config", None), "patch_size", 14))
  height, width = images.shape[-2] // patch_size, images.shape[-1] // patch_size
  if patches.shape[1] != height * width:
    raise ValueError("DINOv2 patch-token count does not match the input image grid.")
  return patches.transpose(1, 2).reshape(patches.shape[0], patches.shape[-1], height, width)


def local_grid_features(
  backbone: nn.Module,
  images: torch.Tensor,
  *,
  target_grid_size: int,
) -> torch.Tensor:
  """Pool before BF16 rollout caching, matching the retained policies."""
  return F.adaptive_avg_pool2d(
    spatial_features(backbone, images),
    output_size=(target_grid_size, target_grid_size),
  )


def global_features(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
  outputs = backbone(images)
  pooled = _output_tensor(outputs, "pooler_output", "x_norm_clstoken")
  if pooled is not None:
    return pooled
  tokens = _output_tensor(outputs, "last_hidden_state", "tokens")
  if tokens is None or tokens.ndim != 3:
    raise ValueError("DINOv2 did not return a global or CLS feature.")
  return tokens[:, 0]


def build(_input_dim: tuple[int, int]) -> nn.Module:
  """Uniform backbone constructor; the image size is fixed by the encoder."""
  return load()


def make_extractor(request: FeatureRequest, target_grid_size: int):
  return {
    "global": global_features,
    "spatial": spatial_features,
    "local_grid": partial(
      local_grid_features,
      target_grid_size=target_grid_size,
    ),
  }[request]


__all__ = [
  "build",
  "global_features",
  "load",
  "local_grid_features",
  "make_extractor",
  "spatial_features",
]
