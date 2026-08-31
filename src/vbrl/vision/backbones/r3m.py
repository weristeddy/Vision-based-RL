"""Pinned R3M loading and feature extraction."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn as nn

from vbrl.paths import model_root

from .weights import R3M_DIRECTORY

from ..config import FeatureRequest

from ..preprocessing import imagenet_normalize


TensorLayer = Callable[[torch.Tensor], torch.Tensor]


def load() -> nn.Module:
  try:
    import hydra
    from omegaconf import OmegaConf
    from r3m import cleanup_config, remove_language_head
  except ImportError as exc:
    raise ImportError("R3M requires the pinned repository dependency in rl.sif.") from exc
  path = model_root() / R3M_DIRECTORY
  config_path = path / "config.yaml"
  model_path = path / "model.pt"
  if not config_path.is_file() or not model_path.is_file():
    raise FileNotFoundError(
      f"Pinned R3M assets are missing from {path}. "
      "Rebuild rl.sif from rl.def."
    )

  model_config = OmegaConf.load(config_path)
  backbone = hydra.utils.instantiate(cleanup_config(model_config))
  wrapped = nn.DataParallel(backbone)
  payload = torch.load(model_path, map_location="cpu", weights_only=True)
  if not isinstance(payload, dict) or not isinstance(payload.get("r3m"), dict):
    raise ValueError(f"Invalid R3M checkpoint payload in {model_path}.")
  state_dict = remove_language_head(dict(payload["r3m"]))
  wrapped.load_state_dict(state_dict, strict=True)
  return wrapped.module


def resnet(backbone: nn.Module) -> nn.Module:
  model = backbone.module if isinstance(backbone, nn.DataParallel) else backbone
  for attribute in ("convnet", "resnet", "encoder", "trunk"):
    candidate = getattr(model, attribute, None)
    if isinstance(candidate, nn.Module):
      return candidate.module if isinstance(candidate, nn.DataParallel) else candidate
  return model


_REQUIRED_LAYERS = (
  "conv1",
  "bn1",
  "relu",
  "maxpool",
  "layer1",
  "layer2",
  "layer3",
  "layer4",
)


def _resnet_forward(
  backbone: nn.Module,
  images: torch.Tensor,
  *,
  stages: int,
) -> torch.Tensor:
  """Run the ResNet stem plus ``stages`` residual stages, and stop there."""
  # preprocess_r3m deliberately supplies [0, 255], matching R3M's public
  # forward contract. The local ResNet path bypasses that forward, so convert
  # deterministically without a GPU-synchronizing range reduction.
  images = imagenet_normalize(images.div(255.0))
  model = resnet(backbone)
  for attribute in _REQUIRED_LAYERS:
    if not hasattr(model, attribute):
      raise TypeError(f"R3M ResNet50 is missing required layer {attribute!r}.")
  conv1 = cast(TensorLayer, getattr(model, "conv1"))
  bn1 = cast(TensorLayer, getattr(model, "bn1"))
  relu = cast(TensorLayer, getattr(model, "relu"))
  maxpool = cast(TensorLayer, getattr(model, "maxpool"))
  x = maxpool(relu(bn1(conv1(images))))
  for index in range(1, stages + 1):
    x = cast(TensorLayer, getattr(model, f"layer{index}"))(x)
  return x


def spatial_features(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
  """R3M's published tap point: layer4, stride 32, 2048 channels, 7x7 at 224px."""
  return _resnet_forward(backbone, images, stages=4)


def layer3_features(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
  """One stage earlier: stride 16, 1024 channels, 14x14 at 224px.

  At layer4 the Push-T object spans about 1.5 feature cells, which is too coarse
  for its orientation to survive; layer3 gives it 3.1, matching the encoders that
  do decode yaw. This deviates from how R3M is normally consumed, so it is a
  separate registered encoder rather than a change to the published tap point.
  """
  return _resnet_forward(backbone, images, stages=3)


def layer3_global_features(
  backbone: nn.Module, images: torch.Tensor
) -> torch.Tensor:
  return layer3_features(backbone, images).mean(dim=(-2, -1))


def global_features(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
  outputs = backbone(images)
  if isinstance(outputs, (tuple, list)):
    outputs = outputs[0]
  if isinstance(outputs, dict):
    for key in ("embedding", "features", "feat"):
      if isinstance(outputs.get(key), torch.Tensor):
        outputs = outputs[key]
        break
  if not isinstance(outputs, torch.Tensor):
    raise TypeError(f"R3M returned unsupported feature type {type(outputs).__name__}.")
  if outputs.ndim == 4:
    outputs = outputs.mean(dim=(-2, -1))
  return outputs.flatten(start_dim=1)


def build(_input_dim: tuple[int, int]) -> nn.Module:
  """Uniform backbone constructor; the image size is fixed by the encoder."""
  return load()


def make_extractor(request: FeatureRequest, target_grid_size: int):
  del target_grid_size
  return {
    "global": global_features,
    "spatial": spatial_features,
    "local_grid": spatial_features,
  }[request]


def make_layer3_extractor(request: FeatureRequest, target_grid_size: int):
  del target_grid_size
  return {
    "global": layer3_global_features,
    "spatial": layer3_features,
    "local_grid": layer3_features,
  }[request]


def install_spatial_alias(encoder: nn.Module) -> None:
  # Retained R3M spatial checkpoints registered this ResNet view alongside the
  # backbone. Both names deliberately point to the same parameters.
  backbone = cast(nn.Module, getattr(encoder, "backbone"))
  setattr(encoder, "resnet", resnet(backbone))


__all__ = [
  "build",
  "global_features",
  "install_spatial_alias",
  "layer3_features",
  "layer3_global_features",
  "load",
  "make_extractor",
  "make_layer3_extractor",
  "resnet",
  "spatial_features",
]
