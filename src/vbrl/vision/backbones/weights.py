from __future__ import annotations

from pathlib import Path

DINOV2_REPO = "facebook/dinov2-small"
DINOV2_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
# timm's copy of Meta's ViT-S/16 (LVD-1689M): the same weights, not gated, so the
# image build needs no Hugging Face login.
DINOV3_REPO = "timm/vit_small_patch16_dinov3.lvd1689m"
DINOV3_REVISION = "3bf4720a82ec2066db88137180ff1f83a675cef0"
DINOV3_MODEL = "vit_small_patch16_dinov3"
DINOV3_FILE = "model.safetensors"
R3M_MODEL = "resnet50"
R3M_FOLDER = "r3m_50"


def huggingface_cache(root: Path) -> Path:
  return root / "huggingface"


# `load_r3m` reads `~/.r3m` and takes no override, so HOME is pointed here.
def r3m_home(root: Path) -> Path:
  return root / "r3m"


def r3m_files(root: Path) -> tuple[Path, Path]:
  folder = r3m_home(root) / ".r3m" / R3M_FOLDER
  return folder / "model.pt", folder / "config.yaml"


__all__ = [
  "DINOV2_REPO",
  "DINOV2_REVISION",
  "DINOV3_FILE",
  "DINOV3_MODEL",
  "DINOV3_REPO",
  "DINOV3_REVISION",
  "R3M_FOLDER",
  "R3M_MODEL",
  "huggingface_cache",
  "r3m_files",
  "r3m_home",
]
