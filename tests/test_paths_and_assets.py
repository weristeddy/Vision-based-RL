"""Where pretrained backbones are looked for, and where they are written.

Two resolutions have to agree or nothing fails loudly: the directory
``vbrl.paths.model_root`` hands the encoders, and the layout
``vbrl.asset_zoo.backbones`` writes into. A mismatch does not raise here -- it
raises on a GPU node, as a missing-file error at the start of a job.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vbrl.paths import CHECKOUT_MODEL_DIRECTORY, DEFAULT_MODEL_ROOT, model_root
from vbrl.vision.backbones.weights import (
  BACKBONE_ASSETS,
  DINOV2_DIRECTORY,
  R3M_DIRECTORY,
  PinnedAsset,
  verify_backbones,
)


def test_model_root_prefers_the_environment_then_the_checkout(
  monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
  """Precedence is what lets one command line work on both hosts.

  A development host sets nothing and gets the checkout's ``.models``; the image
  has no such directory and falls through to the baked path.
  """
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


def test_every_pinned_asset_is_well_formed() -> None:
  assert BACKBONE_ASSETS, "the backbone manifest is empty"
  for asset in BACKBONE_ASSETS:
    assert not Path(asset.relative_path).is_absolute(), asset.relative_path
    assert len(asset.sha256) == 64, asset.relative_path
    assert set(asset.sha256) <= set("0123456789abcdef"), asset.relative_path

  with pytest.raises(ValueError, match="exactly one"):
    PinnedAsset(relative_path="x", sha256="0" * 64)
  with pytest.raises(ValueError, match="exactly one"):
    PinnedAsset(
      relative_path="x", sha256="0" * 64, drive_id="a", huggingface_repo="b/c"
    )


def test_the_manifest_writes_where_the_loaders_read() -> None:
  """The fetcher and the two loaders resolve one set of directory constants.

  They share the constants rather than agreeing by coincidence, so this pins the
  filenames each loader additionally requires -- ``r3m.load`` wants both a
  checkpoint and a config, ``dinov2.load`` wants a safetensors file and a config.
  """
  written = {asset.relative_path for asset in BACKBONE_ASSETS}

  assert {f"{R3M_DIRECTORY}/model.pt", f"{R3M_DIRECTORY}/config.yaml"} <= written
  assert {
    f"{DINOV2_DIRECTORY}/config.json",
    f"{DINOV2_DIRECTORY}/model.safetensors",
  } <= written


def test_fetching_weights_does_not_require_torch() -> None:
  """``vbrl-fetch-backbones`` runs during the image build, before torch matters."""
  import subprocess
  import sys

  probe = (
    "import sys, vbrl.vision.backbones.weights as w;"
    "assert w.BACKBONE_ASSETS;"
    "print('torch' in sys.modules)"
  )
  result = subprocess.run(
    [sys.executable, "-c", probe], capture_output=True, text=True, check=True
  )
  assert result.stdout.strip() == "False", "importing the weight manifest pulled torch"


def test_verify_reports_an_empty_model_root_as_entirely_missing(
  tmp_path: Path,
) -> None:
  assert verify_backbones(tmp_path) == BACKBONE_ASSETS
