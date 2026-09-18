from __future__ import annotations

import os

import torch
import torch.nn as nn

from vbrl.paths import model_root

from ..config import FeatureRequest
from ..preprocessing import imagenet_normalize
from .weights import R3M_MODEL, r3m_files, r3m_home


def load(*, allow_download: bool = False) -> nn.Module:
  from r3m import load_r3m

  root = model_root()
  if not allow_download and not all(f.is_file() for f in r3m_files(root)):
    raise FileNotFoundError(
      f"R3M weights missing from {r3m_home(root)}. Run vbrl-fetch-backbones."
    )
  # load_r3m resolves `~/.r3m`, and returns the DataParallel it loads through.
  previous = os.environ.get("HOME")
  os.environ["HOME"] = str(r3m_home(root))
  try:
    return load_r3m(R3M_MODEL).module
  finally:
    if previous is None:
      os.environ.pop("HOME", None)
    else:
      os.environ["HOME"] = previous


def resnet(backbone: nn.Module) -> nn.Module:
  return backbone.convnet


def _stages(backbone: nn.Module, images: torch.Tensor, count: int) -> torch.Tensor:
  # R3M's public forward takes [0, 255].
  images = imagenet_normalize(images.div(255.0))
  model = resnet(backbone)
  x = model.maxpool(model.relu(model.bn1(model.conv1(images))))
  for index in range(1, count + 1):
    x = getattr(model, f"layer{index}")(x)
  return x


def spatial_features(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
  return _stages(backbone, images, 4)


# At layer4 the Push-T object spans ~1.5 feature cells; layer3 gives it 3.1.
def layer3_features(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
  return _stages(backbone, images, 3)


def layer3_global_features(
  backbone: nn.Module, images: torch.Tensor
) -> torch.Tensor:
  return layer3_features(backbone, images).mean(dim=(-2, -1))


def global_features(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
  return backbone(images)


def build(_input_dim: tuple[int, int]) -> nn.Module:
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
  encoder.resnet = resnet(encoder.backbone)


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
