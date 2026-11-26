from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlattenReluAdapter(nn.Module):
  def __init__(
    self,
    input_channels: int,
    grid_size: int,
    output_dim: int = 256,
  ) -> None:
    super().__init__()
    if min(input_channels, grid_size, output_dim) <= 0:
      raise ValueError("All FlattenReluAdapter dimensions must be positive.")
    self.input_channels = input_channels
    self.grid_size = grid_size
    self.output_dim = output_dim
    self.flatten = nn.Flatten(start_dim=1)
    self.proj = nn.Linear(input_channels * grid_size**2, output_dim)
    self.activation = nn.ReLU()

  def forward(self, features: torch.Tensor) -> torch.Tensor:
    expected = (self.input_channels, self.grid_size, self.grid_size)
    if features.ndim != 4 or tuple(features.shape[1:]) != expected:
      raise ValueError(
        f"Expected [B,{expected[0]},{expected[1]},{expected[2]}], "
        f"got {tuple(features.shape)}."
      )
    return self.activation(self.proj(self.flatten(features)))


class AttentionPoolLatent(nn.Module):
  def __init__(self, features: int, num_heads: int) -> None:
    super().__init__()
    if features % num_heads:
      raise ValueError(f"features={features} must be divisible by heads={num_heads}.")
    self.num_heads = num_heads
    self.head_dim = features // num_heads
    self.scale = self.head_dim**-0.5
    self.latent = nn.Parameter(torch.zeros(1, 1, features))
    self.q = nn.Linear(features, features)
    self.kv = nn.Linear(features, features * 2)
    self.q_norm = nn.LayerNorm(self.head_dim)
    self.k_norm = nn.LayerNorm(self.head_dim)
    self.norm = nn.LayerNorm(features)  # Published PV-Robo state-dict layout.
    nn.init.trunc_normal_(self.latent, std=features**-0.5)

  def forward(self, tokens: torch.Tensor) -> torch.Tensor:
    batch, count, channels = tokens.shape
    tokens = tokens / tokens.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    query = self.q(self.latent.expand(batch, -1, -1))
    query = query.reshape(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
    key_value = self.kv(tokens).reshape(
      batch, count, 2, self.num_heads, self.head_dim
    ).permute(2, 0, 3, 1, 4)
    key, value = key_value.unbind(0)
    attention = (
      (self.q_norm(query) * self.scale) @ self.k_norm(key).transpose(-2, -1)
    ).softmax(-1)
    return (attention @ value).transpose(1, 2).reshape(batch, channels)


class L2Normalize(nn.Module):
  def forward(self, features: torch.Tensor) -> torch.Tensor:
    return F.normalize(features, dim=-1)


class AFAAdapter(nn.Module):
  def __init__(self, input_channels: int, num_heads: int = 8) -> None:
    super().__init__()
    self.input_channels = input_channels
    self.output_dim = input_channels
    self.pool = AttentionPoolLatent(input_channels, num_heads)

  def forward(self, features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 4 or features.shape[1] != self.input_channels:
      raise ValueError(
        f"Expected [B,{self.input_channels},H,W], got {tuple(features.shape)}."
      )
    return self.pool(features.flatten(2).transpose(1, 2))
