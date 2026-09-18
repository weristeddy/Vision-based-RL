from __future__ import annotations

from pathlib import Path

import pytest

from vbrl.paths import CHECKOUT_MODEL_DIRECTORY, DEFAULT_MODEL_ROOT, model_root
from vbrl.vision.backbones.weights import (
  DINOV2_REPO,
  DINOV2_REVISION,
  R3M_MODEL,
  huggingface_cache,
  r3m_files,
  r3m_home,
)


def test_model_root_prefers_the_environment_then_the_checkout(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  import vbrl.paths as paths

  monkeypatch.setenv("VBRL_MODEL_ROOT", str(tmp_path))
  assert model_root() == tmp_path.resolve()

  monkeypatch.delenv("VBRL_MODEL_ROOT")
  checkout = tmp_path / "checkout"
  (checkout / CHECKOUT_MODEL_DIRECTORY).mkdir(parents=True)
  monkeypatch.setattr(paths, "_SOURCE_CHECKOUT_ROOT", checkout)
  assert model_root() == (checkout / CHECKOUT_MODEL_DIRECTORY).resolve()

  monkeypatch.setattr(paths, "_SOURCE_CHECKOUT_ROOT", None)
  assert model_root() == DEFAULT_MODEL_ROOT


def test_relative_model_root_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("VBRL_MODEL_ROOT", "relative/models")
  with pytest.raises(ValueError, match="absolute"):
    model_root()


def test_both_backbones_cache_under_one_model_root(tmp_path: Path) -> None:
  assert huggingface_cache(tmp_path).is_relative_to(tmp_path)
  assert r3m_home(tmp_path).is_relative_to(tmp_path)
  for path in r3m_files(tmp_path):
    assert path.is_relative_to(r3m_home(tmp_path))


# The revision is a git commit SHA, which is what pins the DINOv2 weights: the
# hub verifies the download against it, so nothing here re-implements that.
def test_the_dinov2_revision_is_pinned_to_a_commit() -> None:
  assert len(DINOV2_REVISION) == 40
  assert set(DINOV2_REVISION) <= set("0123456789abcdef")
  assert DINOV2_REPO == "facebook/dinov2-small"
  assert R3M_MODEL == "resnet50"


def test_fetching_weights_does_not_require_torch() -> None:
  import subprocess
  import sys

  probe = (
    "import sys, vbrl.vision.backbones.weights as w;"
    "assert w.DINOV2_REPO;"
    "print('torch' in sys.modules)"
  )
  result = subprocess.run(
    [sys.executable, "-c", probe], capture_output=True, text=True, check=True
  )
  assert result.stdout.strip() == "False", "importing the weight manifest pulled torch"
