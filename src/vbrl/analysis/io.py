from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

METADATA_KEY = "metadata_json"


def save_npz(
  path: str | Path,
  arrays: Mapping[str, np.ndarray],
  metadata: Mapping[str, Any] | None = None,
) -> Path:
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  payload = dict(arrays)
  payload[METADATA_KEY] = np.asarray(json.dumps(dict(metadata or {}), sort_keys=True))
  np.savez_compressed(destination, **payload)
  return destination


def load_npz(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
  with np.load(Path(path), allow_pickle=False) as data:
    arrays = {key: data[key].copy() for key in data.files if key != METADATA_KEY}
    metadata = json.loads(str(data[METADATA_KEY])) if METADATA_KEY in data else {}
  return arrays, metadata


def prefixed(arrays: Mapping[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
  return {
    key.removeprefix(prefix): value
    for key, value in arrays.items()
    if key.startswith(prefix)
  }


def save_figure(figure: Any, path: str | Path) -> Path:
  import matplotlib.pyplot as plt

  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(destination, dpi=180)
  plt.close(figure)
  return destination


def string_list(value: Any, field: str) -> tuple[str, ...]:
  if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
    raise ValueError(f"{field} must be a list of strings.")
  if not all(isinstance(item, str) for item in value):
    raise ValueError(f"{field} must be a list of strings.")
  return tuple(value)


def resolve_outputs(
  context: Any,
  templates: Mapping[str, str],
  axes: Iterable[Mapping[str, str]],
  *,
  label: str,
) -> list[tuple[Mapping[str, str], dict[str, Path]]]:
  jobs: list[tuple[Mapping[str, str], dict[str, Path]]] = []
  seen: set[Path] = set()
  for values in axes:
    paths = {
      role: context.output(template.format(**values))
      for role, template in templates.items()
    }
    for path in paths.values():
      if path in seen:
        raise ValueError(f"{label} output templates must produce unique paths.")
      seen.add(path)
    jobs.append((values, paths))
  return jobs


__all__ = [
  "METADATA_KEY",
  "load_npz",
  "prefixed",
  "resolve_outputs",
  "save_figure",
  "save_npz",
  "string_list",
]
