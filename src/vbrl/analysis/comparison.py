"""Paired distances and plots for aligned feature artifacts."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .features import flatten_features, load_features
from .io import resolve_outputs, save_figure, string_list


@dataclass(frozen=True)
class FeatureComparison:
  """Per-sample cosine similarities and Euclidean distances."""

  cosine_similarity: np.ndarray
  euclidean_distance: np.ndarray


def compare_features(
  left: np.ndarray,
  right: np.ndarray,
) -> FeatureComparison:
  """Compare corresponding flattened feature samples."""
  left_flat = flatten_features(np.asarray(left))
  right_flat = flatten_features(np.asarray(right))
  if left_flat.shape != right_flat.shape:
    raise ValueError(
      "Paired feature shapes must match, got "
      f"{left_flat.shape} and {right_flat.shape}."
    )
  denominator = (
    np.linalg.norm(left_flat, axis=1)
    * np.linalg.norm(right_flat, axis=1)
  )
  cosine = np.divide(
    np.sum(left_flat * right_flat, axis=1),
    denominator,
    out=np.zeros(len(left_flat), dtype=np.float32),
    where=denominator > 0,
  )
  distance = np.linalg.norm(left_flat - right_flat, axis=1)
  return FeatureComparison(
    cosine_similarity=cosine.astype(np.float32, copy=False),
    euclidean_distance=distance.astype(np.float32, copy=False),
  )


def save_comparison_csv(
  result: FeatureComparison,
  path: str | Path,
) -> Path:
  """Persist paired comparison metrics as a CSV."""
  destination = Path(path)
  destination.parent.mkdir(parents=True, exist_ok=True)
  with destination.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(("sample", "cosine_similarity", "euclidean_distance"))
    values = zip(
      result.cosine_similarity,
      result.euclidean_distance,
      strict=True,
    )
    for index, (cosine, distance) in enumerate(values):
      writer.writerow((index, float(cosine), float(distance)))
  return destination


def plot_feature_comparison(
  result: FeatureComparison,
  *,
  output: str | Path,
  title: str = "Paired feature comparison",
) -> Path:
  """Plot distributions of the paired feature metrics."""
  import matplotlib.pyplot as plt

  figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
  axes[0].hist(result.cosine_similarity, bins=30)
  axes[0].set(xlabel="Cosine similarity", ylabel="Samples")
  axes[1].hist(result.euclidean_distance, bins=30)
  axes[1].set(xlabel="Euclidean distance", ylabel="Samples")
  figure.suptitle(title)
  return save_figure(figure, output)


def run(
  context: Any,
  *,
  left: str,
  right: str,
  output: str,
  plot: str,
  stages: Sequence[str],
) -> tuple[Path, ...]:
  """Compare feature files using explicit input and output templates."""
  left_source = context.input(left)
  right_source = context.input(right)
  jobs = resolve_outputs(
    context,
    {"output": output, "plot": plot},
    (
      {"stage": stage, "left": left_source.stem, "right": right_source.stem}
      for stage in string_list(stages, "comparison.stages")
    ),
    label="Comparison",
  )

  left_batch = load_features(left_source)
  right_batch = load_features(right_source)
  generated: list[Path] = []
  for values, paths in jobs:
    stage = values["stage"]
    if stage not in left_batch.features or stage not in right_batch.features:
      raise KeyError(f"Feature stage {stage!r} must exist in both artifacts.")
    result = compare_features(
      left_batch.features[stage], right_batch.features[stage]
    )
    generated.append(save_comparison_csv(result, paths["output"]))
    generated.append(
      plot_feature_comparison(
        result,
        output=paths["plot"],
        title=f"{context.task_id} / {stage} feature comparison",
      )
    )
  return tuple(generated)
