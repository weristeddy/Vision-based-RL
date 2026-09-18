from __future__ import annotations

from pathlib import Path

DINOV2_REPO = "facebook/dinov2-small"
DINOV2_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
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
  "R3M_FOLDER",
  "R3M_MODEL",
  "huggingface_cache",
  "r3m_files",
  "r3m_home",
]
