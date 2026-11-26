from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def grid_layer(encoder: Any, config: Any) -> Any:
  from vbrl.vision.backbones.r3m import resnet

  backbone = encoder.backbone
  if config.encoder == "nature_cnn":
    return backbone
  if config.encoder.startswith("r3m"):
    net = resnet(backbone)
    return net.layer3 if config.encoder == "r3m_resnet50_layer3" else net.layer4
  # A CLS readout gathers the patches in every block; only the patch embedding
  # still carries all of it. AFA reads the last layer's patches directly.
  if config.adapter == "afa_full":
    return (
      backbone.encoder.layer[-1]
      if config.encoder == "dinov2_vits14"
      else backbone.blocks[-1]
    )
  return (
    backbone.embeddings if config.encoder == "dinov2_vits14" else backbone.patch_embed
  )


def _as_grid(activation: Any) -> Any:
  if activation.ndim == 4:
    return activation
  side = math.isqrt(activation.shape[1])
  tokens = activation[:, -side * side :]
  return tokens.transpose(1, 2).reshape(len(tokens), -1, side, side)


def gradcam(model: Any, obs: Any, config: Any) -> Any:
  encoder = model.cnns["camera"]
  captured = {}

  def hook(_module: Any, _inputs: Any, output: Any) -> Any:
    first = output[0] if isinstance(output, tuple) else output
    leaf = first.detach().float().requires_grad_(True)
    captured["activation"] = leaf
    return (leaf, *output[1:]) if isinstance(output, tuple) else leaf

  handle = grid_layer(encoder, config).register_forward_hook(hook)
  frozen, encoder.freeze_backbone = encoder.freeze_backbone, False
  try:
    with torch.enable_grad():
      actions = model(obs)
      maps = 0.0
      for dimension in range(actions.shape[1]):
        activation = captured["activation"]
        (gradient,) = torch.autograd.grad(
          actions[:, dimension].sum(), activation, retain_graph=True
        )
        grid, gradient = _as_grid(activation), _as_grid(gradient)
        maps = maps + (gradient * grid).sum(1, keepdim=True).abs()
  finally:
    handle.remove()
    encoder.freeze_backbone = frozen
  maps = F.interpolate(
    maps / actions.shape[1], size=obs["camera"].shape[-2:], mode="bilinear"
  )
  return maps[:, 0].detach()


def jaccard(maps: Any, masks: Any) -> Any:
  maps, masks = maps.flatten(1), masks.flatten(1).bool()
  ranks = maps.argsort(1, descending=True).argsort(1)
  active = ranks < masks.sum(1, keepdim=True)
  return (active & masks).sum(1) / (active | masks).sum(1).clamp(min=1)


__all__ = ["gradcam", "grid_layer", "jaccard"]
