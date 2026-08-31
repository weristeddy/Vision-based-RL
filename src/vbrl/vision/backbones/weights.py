"""Where the pretrained backbone weights come from, as data.

Model weights are not Python packages, so they cannot be declared in
``pyproject.toml``: R3M's are a Google Drive file and DINOv2's are a HuggingFace
revision. They are pinned here instead -- one source and one SHA-256 per file --
in the same declarative style as [`scenes/presets.py`](../../scenes/presets.py).

This lives beside :mod:`dinov2` and :mod:`r3m` rather than in ``asset_zoo``
because it is not packaged data: ``asset_zoo`` holds robot XML, meshes, and
textures that ship inside the wheel and locate themselves from their own package,
while these 459 MB of weights are fetched into ``model_root()`` at image-build or
setup time. Being here also means the directory constants below are the *same*
ones the two loaders resolve, rather than two string literals that have to agree.

Both hosts read this table. ``rl.def`` bakes the files into the image's
``/opt/vbrl-models`` so a cluster job never touches the network or a user's home
cache, and a development workstation fetches the same bytes into a checkout-local
``.models``. Every file is verified against its digest, and one already present
with the right digest is left alone, so fetching is idempotent.

This module must stay importable without ``torch``: ``vbrl-fetch-backbones`` runs
during the image build, before there is any reason to pay for a deep-learning
import. Its siblings here import torch at module scope; this one may not.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


__all__ = [
  "BACKBONE_ASSETS",
  "DINOV2_DIRECTORY",
  "DINOV2_REVISION",
  "PinnedAsset",
  "R3M_DIRECTORY",
  "fetch_backbones",
  "verify_backbones",
]

# The layout under model_root(). dinov2.load() and r3m.load() resolve these, so
# the fetcher cannot write somewhere the loaders do not look.
DINOV2_DIRECTORY = "dinov2-small"
R3M_DIRECTORY = "r3m/r3m_50"

DINOV2_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
_CHUNK = 1 << 20


@dataclass(frozen=True)
class PinnedAsset:
  """One file, its location relative to the model root, and its digest."""

  relative_path: str
  sha256: str
  drive_id: str | None = None
  huggingface_repo: str | None = None

  def __post_init__(self) -> None:
    if bool(self.drive_id) == bool(self.huggingface_repo):
      raise ValueError(
        f"{self.relative_path} must name exactly one of drive_id or "
        "huggingface_repo."
      )

  @property
  def filename(self) -> str:
    return Path(self.relative_path).name

  def path_under(self, model_root: Path) -> Path:
    return model_root / self.relative_path


BACKBONE_ASSETS: tuple[PinnedAsset, ...] = (
  PinnedAsset(
    relative_path=f"{R3M_DIRECTORY}/model.pt",
    sha256="d6cf6f71632907d12ede6987cdc5212bd385988135ed07f1bd59bf488831378b",
    drive_id="1Xu0ssuG0N1zjZS54wmWzJ7-nb0-7XzbA",
  ),
  PinnedAsset(
    relative_path=f"{R3M_DIRECTORY}/config.yaml",
    sha256="0eb7132c15fdf0b6a16993be786bf04305f187423984883f7cb6b7d5ae4d6d57",
    drive_id="10jY2VxrrhfOdNPmsFdES568hjjIoBJx8",
  ),
  PinnedAsset(
    relative_path=f"{DINOV2_DIRECTORY}/config.json",
    sha256="1809f83e3bdb1609a501a610ad4a742f4fd8ae44d72ca4aa0df52d1f2ac8628d",
    huggingface_repo="facebook/dinov2-small",
  ),
  PinnedAsset(
    relative_path=f"{DINOV2_DIRECTORY}/model.safetensors",
    sha256="ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1",
    huggingface_repo="facebook/dinov2-small",
  ),
)


def _digest(path: Path) -> str:
  running = sha256()
  with path.open("rb") as handle:
    while chunk := handle.read(_CHUNK):
      running.update(chunk)
  return running.hexdigest()


def verify_backbones(model_root: Path) -> tuple[PinnedAsset, ...]:
  """Return the assets that are missing or do not match their digest."""
  return tuple(
    asset
    for asset in BACKBONE_ASSETS
    if not asset.path_under(model_root).is_file()
    or _digest(asset.path_under(model_root)) != asset.sha256
  )


def _download(asset: PinnedAsset, destination: Path) -> None:
  destination.parent.mkdir(parents=True, exist_ok=True)
  if asset.drive_id is not None:
    import gdown

    result = gdown.download(
      f"https://drive.google.com/uc?id={asset.drive_id}",
      str(destination),
      quiet=False,
    )
    if result is None:
      raise RuntimeError(f"Download returned nothing for {asset.relative_path}.")
    return

  import shutil

  from huggingface_hub import hf_hub_download

  if asset.huggingface_repo is None:  # pragma: no cover - __post_init__ prevents this
    raise ValueError(f"{asset.relative_path} names no source.")
  source = hf_hub_download(
    repo_id=asset.huggingface_repo,
    filename=asset.filename,
    revision=DINOV2_REVISION,
  )
  shutil.copyfile(source, destination)


def fetch_backbones(model_root: Path) -> Iterator[tuple[PinnedAsset, str]]:
  """Materialise every pinned asset under ``model_root``, verifying each file.

  Yields ``(asset, action)`` where action is ``"kept"`` or ``"downloaded"``, so a
  caller can report progress. Raises ``RuntimeError`` if a freshly downloaded
  file does not match its pinned digest -- a wrong backbone silently changes
  every result that depends on it.
  """
  if not model_root.is_absolute():
    raise ValueError(f"model_root must be an absolute path; got {model_root}.")
  for asset in BACKBONE_ASSETS:
    destination = asset.path_under(model_root)
    if destination.is_file() and _digest(destination) == asset.sha256:
      yield asset, "kept"
      continue
    _download(asset, destination)
    observed = _digest(destination)
    if observed != asset.sha256:
      raise RuntimeError(
        f"Pinned asset verification failed for {asset.relative_path}: "
        f"expected {asset.sha256}, got {observed}."
      )
    yield asset, "downloaded"
