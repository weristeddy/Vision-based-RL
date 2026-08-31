"""Installed-package assets and optional source-checkout locations."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent

DEFAULT_MODEL_ROOT = Path("/opt/vbrl-models")
CHECKOUT_MODEL_DIRECTORY = ".models"


def _environment_path(name: str) -> Path | None:
  value = os.environ.get(name)
  if not value:
    return None
  path = Path(value).expanduser()
  if not path.is_absolute():
    raise ValueError(f"{name} must be an absolute path; got {value!r}.")
  return path.resolve()


def _source_checkout_root() -> Path | None:
  """Return the checkout containing this package when using ``src`` layout."""
  source_root = PACKAGE_ROOT.parent
  candidate = source_root.parent
  if (
    source_root.name == "src"
    and (candidate / "configs").is_dir()
    and (source_root / "vbrl").resolve() == PACKAGE_ROOT
  ):
    return candidate.resolve()
  return None


def checkout_root(*, required: bool = True) -> Path | None:
  """Resolve checkout data without treating installed assets as checkout data."""
  root = _environment_path("VBRL_REPO_ROOT") or _SOURCE_CHECKOUT_ROOT
  if root is None and required:
    raise RuntimeError(
      "A source-checkout path was requested, but no checkout is available. "
      "Pass an absolute path or set VBRL_REPO_ROOT."
    )
  return root


def model_root() -> Path:
  """Resolve the pretrained-backbone root.

  ``VBRL_MODEL_ROOT`` wins, then a checkout-local ``.models`` directory, then the
  image path. The middle case is what lets a development host run every command
  with no environment variable set at all; the cluster image has no such
  directory in its checkout, so it still resolves to ``/opt/vbrl-models``.
  """
  requested = _environment_path("VBRL_MODEL_ROOT")
  if requested is not None:
    return requested
  if _SOURCE_CHECKOUT_ROOT is not None:
    local = _SOURCE_CHECKOUT_ROOT / CHECKOUT_MODEL_DIRECTORY
    if local.is_dir():
      return local.resolve()
  return DEFAULT_MODEL_ROOT


_SOURCE_CHECKOUT_ROOT = _source_checkout_root()


def _under_checkout(
  path: str | Path,
  *,
  required: bool = True,
) -> tuple[Path, Path | None]:
  """Return ``(resolved, checkout_root)``.

  An absolute path never needs a checkout, so it is resolved before one is
  looked up -- installed mode has no checkout and must still accept one.
  """
  candidate = Path(path).expanduser()
  if candidate.is_absolute():
    return candidate.resolve(), checkout_root(required=False)
  root = checkout_root(required=required)
  if root is None:
    raise RuntimeError(
      "Relative paths require a checkout; pass an absolute path or "
      "set VBRL_REPO_ROOT."
    )
  return (root / candidate).resolve(), root


def repository_path(path: str | Path) -> Path:
  """Resolve a checkout-relative path; absolute paths need no checkout."""
  return _under_checkout(path)[0]


def analysis_manifest_path(path: str | Path) -> Path:
  """Resolve an absolute manifest or a checkout analysis selector."""
  candidate = Path(path).expanduser()
  resolved, root = _under_checkout(candidate)
  if candidate.is_absolute():
    return resolved
  assert root is not None
  if resolved.is_file() or candidate.parts[:2] == ("configs", "analysis"):
    return resolved
  return (root / "configs" / "analysis" / candidate).resolve()


def artifact_path(path: str | Path) -> Path:
  """Resolve generated output without making installed mode need a checkout."""
  resolved, root = _under_checkout(path, required=False)
  if root is None:
    return resolved

  artifacts = (root / "artifacts").resolve()
  try:
    resolved.relative_to(artifacts)
  except ValueError as exc:
    raise ValueError(
      f"Generated output must be below {artifacts}; got {resolved}."
    ) from exc
  return resolved
