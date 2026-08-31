"""Visual encoder execution, freezing, autocast, and chunking."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


Preprocess = Callable[[torch.Tensor], torch.Tensor]
Extract = Callable[[nn.Module, torch.Tensor], torch.Tensor]


def _freeze(module: nn.Module) -> None:
  for parameter in module.parameters():
    parameter.requires_grad_(False)


# VisionConfig.validate has already rejected anything outside this table.
_AUTOCAST_DTYPES = {
  "bfloat16": torch.bfloat16,
  "float16": torch.float16,
  "float32": torch.float32,
}


class VisualEncoder(nn.Module):
  """Backbone/adapter composition with a stable feature-cache interface."""

  def __init__(
    self,
    *,
    backbone: nn.Module,
    adapter: nn.Module,
    preprocess: Preprocess,
    extract: Extract,
    output_dim: int,
    frozen: bool,
    encode_batch_size: int | None,
    autocast: bool,
    autocast_dtype: str,
  ) -> None:
    super().__init__()
    self.backbone = backbone
    self.adapter = adapter
    self.preprocess = preprocess
    self.extract = extract
    self._output_dim = int(output_dim)
    self.freeze_backbone = bool(frozen)
    self.encode_batch_size = encode_batch_size
    self.encoder_autocast = bool(autocast)
    self.encoder_autocast_dtype = autocast_dtype
    if frozen:
      _freeze(self.backbone)
      self.backbone.eval()

  @property
  def output_dim(self) -> int:
    return self._output_dim

  def train(self, mode: bool = True) -> "VisualEncoder":
    super().train(mode)
    if self.freeze_backbone:
      self.backbone.eval()
    return self

  def encode_features(self, images: torch.Tensor) -> torch.Tensor:
    if self.freeze_backbone:
      with torch.inference_mode():
        features = self._encode_in_chunks(images)
      # Inference tensors cannot subsequently be saved for adapter autograd.
      return features.clone()
    return self._encode_in_chunks(images)

  def project_features(self, features: torch.Tensor) -> torch.Tensor:
    parameter = next(self.adapter.parameters(), None)
    if parameter is not None:
      features = features.to(dtype=parameter.dtype)
    return self.adapter(features)

  def forward(self, images: torch.Tensor) -> torch.Tensor:
    return self.project_features(self.encode_features(images))

  def _encode_in_chunks(self, images: torch.Tensor) -> torch.Tensor:
    size = self.encode_batch_size
    if size is None or images.shape[0] <= size:
      return self._encode_chunk(images)
    return torch.cat(
      [
        self._encode_chunk(images[start : start + size])
        for start in range(0, images.shape[0], size)
      ],
      dim=0,
    )

  def _encode_chunk(self, images: torch.Tensor) -> torch.Tensor:
    images = self.preprocess(images)
    dtype = _AUTOCAST_DTYPES[self.encoder_autocast_dtype]
    enabled = (
      self.encoder_autocast
      and images.device.type == "cuda"
      and dtype != torch.float32
    )
    with torch.autocast(
      device_type=images.device.type,
      dtype=dtype,
      enabled=enabled,
    ):
      return self.extract(self.backbone, images)


__all__ = ["VisualEncoder"]
