from __future__ import annotations

from typing import Any

import torch

FRACTIONS = torch.linspace(0.0, 1.0, 21)


def deletion_auc(model: Any, obs: Any, maps: Any) -> Any:
  from torchvision.transforms.functional import gaussian_blur

  images = obs["camera"]
  blurred = gaussian_blur(images.float(), kernel_size=31, sigma=10.0).to(images.dtype)
  order = maps.flatten(1).argsort(1, descending=True).argsort(1).to(images.device)
  pixels = order.shape[1]
  with torch.no_grad():
    base = model(obs)
    changes = []
    for fraction in FRACTIONS:
      removed = (order < int(fraction * pixels)).view(-1, 1, *images.shape[-2:])
      edited = obs.clone()
      edited["camera"] = torch.where(removed, blurred, images)
      changes.append((model(edited) - base).norm(dim=1))
  curve = torch.stack(changes, 1)
  return (curve / curve[:, -1:].clamp(min=1e-6)).mean(1)


__all__ = ["FRACTIONS", "deletion_auc"]
