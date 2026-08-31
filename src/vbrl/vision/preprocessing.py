from __future__ import annotations

import torch
import torch.nn.functional as F


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def prepare_images(images: torch.Tensor) -> torch.Tensor:
  """Return canonical BCHW float RGB in ``[0, 1]``.

  Camera captures and analysis artifacts commonly use NHWC uint8, while RSL-RL
  supplies BCHW observations. Both layouts are accepted and ambiguity is
  rejected rather than silently transposing the wrong dimension.
  """
  if images.ndim != 4:
    raise ValueError(f"Expected a four-dimensional image batch, got {tuple(images.shape)}.")
  if images.shape[1] == 3:
    result = images
  elif images.shape[-1] == 3:
    result = images.permute(0, 3, 1, 2)
  else:
    raise ValueError(f"Could not find an RGB channel dimension in {tuple(images.shape)}.")
  if result.dtype == torch.uint8:
    return result.float().div(255.0)
  if not torch.is_floating_point(result):
    raise TypeError(f"RGB images must be floating point or uint8, got {result.dtype}.")
  minimum, maximum = result.detach().amin(), result.detach().amax()
  if minimum < -1e-6 or maximum > 1.0 + 1e-6:
    raise ValueError("Floating-point RGB images must be in [0, 1].")
  return result


def to_unit_interval(images: torch.Tensor) -> torch.Tensor:
  # Hot training path: RSL-RL has already established BCHW and camera terms
  # produce values in [0, 1]. Avoid reduction-based range checks that would
  # synchronize every GPU rollout step.
  if images.ndim != 4 or images.shape[1] != 3:
    raise ValueError(f"Expected BCHW RGB images, got {tuple(images.shape)}.")
  if images.dtype == torch.uint8:
    return images.float().div(255.0)
  if not torch.is_floating_point(images):
    raise TypeError(f"RGB images must be floating point or uint8, got {images.dtype}.")
  return images


def resize_images(images: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
  images = to_unit_interval(images)
  if tuple(images.shape[-2:]) == tuple(size):
    return images
  return F.interpolate(images, size=size, mode="bilinear", align_corners=False)


def imagenet_normalize(images: torch.Tensor) -> torch.Tensor:
  images = to_unit_interval(images)
  mean = _IMAGENET_MEAN.to(device=images.device, dtype=images.dtype)
  std = _IMAGENET_STD.to(device=images.device, dtype=images.dtype)
  return (images - mean) / std


def preprocess_dinov2(images: torch.Tensor) -> torch.Tensor:
  return imagenet_normalize(resize_images(images, (224, 224)))


def preprocess_r3m(images: torch.Tensor) -> torch.Tensor:
  # R3M applies its own /255 and normalization inside the published model.
  return resize_images(images, (224, 224)).mul(255.0)
