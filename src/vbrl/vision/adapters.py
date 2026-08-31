from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FlattenAdapter(nn.Module):
  """The plain Nature-CNN head: flatten the whole feature map, project once.

  This is what "no adapter" means for a trainable encoder. Nothing is pooled,
  projected, or attended to first -- and nothing can be, because a policy MLP
  needs a vector. Removing this layer would only move the same
  ``channels x grid^2 -> output_dim`` matrix into the MLP's first layer.

  Unlike :class:`LocalGridAdapter` it never resamples: a map of the wrong size
  is a configuration error, not something to average away.
  """

  def __init__(
    self,
    input_channels: int,
    grid_size: int,
    output_dim: int = 256,
  ) -> None:
    super().__init__()
    if min(input_channels, grid_size, output_dim) <= 0:
      raise ValueError("All FlattenAdapter dimensions must be positive.")
    self.input_channels = input_channels
    self.grid_size = grid_size
    self.output_dim = output_dim
    self.flatten = nn.Flatten(start_dim=1)
    self.proj = nn.Linear(input_channels * grid_size**2, output_dim)
    self.output_norm = nn.LayerNorm(output_dim)

  def forward(self, features: torch.Tensor) -> torch.Tensor:
    expected = (self.input_channels, self.grid_size, self.grid_size)
    if features.ndim != 4 or tuple(features.shape[1:]) != expected:
      raise ValueError(
        f"Expected [B,{','.join(map(str, expected))}], got {tuple(features.shape)}."
      )
    return self.output_norm(self.proj(self.flatten(features)))


class FlattenReluAdapter(nn.Module):
  """ManiSkill3's Nature-CNN head: flatten, project once, rectify.

  Identical to :class:`FlattenAdapter` in shape and parameter count; the only
  difference is that the projection is followed by a ReLU rather than a
  LayerNorm, which is what ``ppo_rgb.py`` does. Kept as a separate class rather
  than a flag because the module tree is checkpoint format -- swapping the
  normalisation inside ``FlattenAdapter`` would invalidate every retained
  ``*-Flatten-*`` checkpoint.

  A ReLU here is safe only because the flattened map is already non-negative
  post-ReLU conv output. The same substitution on a coordinate readout would be
  destructive: spatial softmax emits expected x/y in roughly [-1, 1], and a
  rectifier would collapse half the image plane onto zero.
  """

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


class LocalGridAdapter(nn.Module):
  """Compress an ordered feature grid into one policy vector."""

  def __init__(
    self,
    input_channels: int,
    output_dim: int = 256,
    projected_channels: int = 64,
    target_grid_size: int = 7,
  ) -> None:
    super().__init__()
    if min(input_channels, output_dim, projected_channels, target_grid_size) <= 0:
      raise ValueError("All LocalGridAdapter dimensions must be positive.")
    self.input_channels = input_channels
    self.output_dim = output_dim
    self.target_grid_size = target_grid_size
    self.proj = nn.Conv2d(input_channels, projected_channels, 1)
    self.norm = nn.LayerNorm(projected_channels)
    self.activation = nn.ReLU()
    self.flatten_proj = nn.Linear(projected_channels * target_grid_size**2, output_dim)
    self.output_norm = nn.LayerNorm(output_dim)

  def forward(self, features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 4 or features.shape[1] != self.input_channels:
      raise ValueError(
        f"Expected [B,{self.input_channels},H,W], got {tuple(features.shape)}."
      )
    if features.shape[-2:] != (self.target_grid_size, self.target_grid_size):
      features = F.adaptive_avg_pool2d(features, self.target_grid_size)
    features = self.proj(features).permute(0, 2, 3, 1)
    features = self.activation(self.norm(features)).permute(0, 3, 1, 2)
    return self.output_norm(self.flatten_proj(features.flatten(1)))


class SpatialSoftmaxAdapter(nn.Module):
  """Convert spatial heatmaps to interleaved expected x/y coordinates."""

  def __init__(self, input_channels: int, output_channels: int = 128) -> None:
    super().__init__()
    if input_channels <= 0 or output_channels <= 0:
      raise ValueError("SpatialSoftmaxAdapter channel counts must be positive.")
    self.input_channels = input_channels
    self.output_dim = 2 * output_channels
    self.proj = nn.Conv2d(input_channels, output_channels, 1)
    self.output_norm = nn.LayerNorm(self.output_dim)
    self._pos_x: torch.Tensor
    self._pos_y: torch.Tensor
    self.register_buffer("_pos_x", torch.empty(0), persistent=False)
    self.register_buffer("_pos_y", torch.empty(0), persistent=False)

  def _coordinate_grid(
    self, height: int, width: int, reference: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (1, 1, height * width)
    if (
      tuple(self._pos_x.shape) != shape
      or self._pos_x.device != reference.device
      or self._pos_x.dtype != reference.dtype
    ):
      with torch.inference_mode(False), torch.no_grad():
        y = torch.linspace(-1, 1, height, device=reference.device, dtype=reference.dtype)
        x = torch.linspace(-1, 1, width, device=reference.device, dtype=reference.dtype)
        pos_y, pos_x = torch.meshgrid(y, x, indexing="ij")
        self._pos_x, self._pos_y = pos_x.reshape(shape), pos_y.reshape(shape)
    return self._pos_x, self._pos_y

  def forward(self, features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 4 or features.shape[1] != self.input_channels:
      raise ValueError(
        f"Expected [B,{self.input_channels},H,W], got {tuple(features.shape)}."
      )
    logits = self.proj(features)
    batch, channels, height, width = logits.shape
    probabilities = logits.reshape(batch, channels, -1).softmax(dim=-1)
    pos_x, pos_y = self._coordinate_grid(height, width, logits)
    coordinates = torch.stack(
      ((probabilities * pos_x).sum(-1), (probabilities * pos_y).sum(-1)), dim=-1
    )
    return self.output_norm(coordinates.reshape(batch, self.output_dim))


class AttentionPoolLatent(nn.Module):
  """Attention pooling with the trainable latent-query layout used by PV-Robo."""

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


class AFAAdapter(nn.Module):
  def __init__(
    self,
    input_channels: int,
    output_dim: int = 0,
    num_heads: int = 8,
  ) -> None:
    super().__init__()
    self.input_channels = input_channels
    self.output_dim = output_dim or input_channels
    self.pool = AttentionPoolLatent(input_channels, num_heads)
    self.proj = (
      nn.Linear(input_channels, self.output_dim)
      if self.output_dim != input_channels
      else nn.Identity()
    )

  def forward(self, features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 4 or features.shape[1] != self.input_channels:
      raise ValueError(
        f"Expected [B,{self.input_channels},H,W], got {tuple(features.shape)}."
      )
    return self.proj(self.pool(features.flatten(2).transpose(1, 2)))
