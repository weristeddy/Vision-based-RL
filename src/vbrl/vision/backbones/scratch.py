"""Trainable scratch visual backbones."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn


class NatureCNNBackbone(nn.Module):
  """The three-layer Nature DQN trunk used by scratch policies."""


  def __init__(self) -> None:
    super().__init__()
    self.trunk = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=8, stride=4),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=4, stride=2),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3),
      nn.ReLU(),
    )

  def forward(self, images: torch.Tensor) -> torch.Tensor:
    return self.trunk(images)


class CompactViTBlock(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.norm1 = nn.LayerNorm(128)
    self.attn = nn.MultiheadAttention(
      embed_dim=128,
      num_heads=8,
      dropout=0.0,
      bias=False,
      batch_first=True,
    )
    self.norm2 = nn.LayerNorm(128)
    self.mlp = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128))

  def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    normalized = self.norm1(tokens)
    tokens = tokens + self.attn(
      normalized, normalized, normalized, need_weights=False
    )[0]
    return tokens + self.mlp(self.norm2(tokens))


class CompactViTBackbone(nn.Module):
  """Four-block, 128-wide ViT with a 16-pixel patch embedding."""


  def __init__(self, image_size: int = 224) -> None:
    super().__init__()
    if image_size <= 0 or image_size % 16:
      raise ValueError(
        f"image_size must be positive and divisible by 16, got {image_size}."
      )
    self.image_size = image_size
    self.patch_size = 16
    side = image_size // self.patch_size
    self.grid_size = (side, side)
    self.num_patches = side**2
    self.patch_embed = nn.Conv2d(3, 128, kernel_size=16, stride=16)
    self.cls_token = nn.Parameter(torch.zeros(1, 1, 128))
    self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, 128))
    self.blocks = nn.ModuleList(CompactViTBlock() for _ in range(4))
    self.norm = nn.LayerNorm(128)
    nn.init.trunc_normal_(self.cls_token, std=0.02)
    nn.init.trunc_normal_(self.pos_embed, std=0.02)

  def forward_features(self, images: torch.Tensor) -> torch.Tensor:
    patches = self.patch_embed(images)
    if tuple(patches.shape[-2:]) != self.grid_size:
      raise ValueError(
        f"CompactViT expects {self.image_size}x{self.image_size} images, "
        f"got {tuple(images.shape[-2:])}."
      )
    tokens = patches.flatten(2).transpose(1, 2)
    cls = self.cls_token.expand(tokens.shape[0], -1, -1)
    tokens = torch.cat((cls, tokens), dim=1) + self.pos_embed
    for block in self.blocks:
      tokens = block(tokens)
    return self.norm(tokens)

  def forward(self, images: torch.Tensor) -> torch.Tensor:
    return self.forward_features(images)


def build_nature_cnn(_: tuple[int, int]) -> nn.Module:
  return NatureCNNBackbone()


def build_compact_vit(input_dim: tuple[int, int]) -> nn.Module:
  height, width = input_dim
  if height != width:
    raise ValueError("CompactViT requires square images.")
  return CompactViTBackbone(width)


def nature_features(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
  return backbone(images)


def compact_vit_features(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
  compact_vit = cast(CompactViTBackbone, backbone)
  tokens = compact_vit.forward_features(images)
  patch_tokens = tokens[:, 1:]
  height, width = compact_vit.grid_size
  return patch_tokens.transpose(1, 2).reshape(
    patch_tokens.shape[0], patch_tokens.shape[-1], height, width
  )


def nature_extractor(_request: str, _grid: int):
  """Nature CNN returns spatial maps for every request."""
  return nature_features


def compact_vit_extractor(_request: str, _grid: int):
  """Compact ViT returns spatial maps for every request."""
  return compact_vit_features


__all__ = [
  "compact_vit_extractor",
  "nature_extractor",
  "CompactViTBackbone",
  "CompactViTBlock",
  "NatureCNNBackbone",
  "build_compact_vit",
  "build_nature_cnn",
  "compact_vit_features",
  "nature_features",
]
