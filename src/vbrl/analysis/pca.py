"""Plot deterministic two-dimensional feature PCA projections."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .capture import load_capture
from .features import flatten_features, load_features
from .io import resolve_outputs, save_figure, string_list


def plot_feature_pca(
  features: np.ndarray,
  *,
  color: np.ndarray | None = None,
  title: str = "Visual feature PCA",
  output: str | Path,
) -> Path:
  """Plot the leading two singular-vector coordinates of flattened features."""
  import matplotlib.pyplot as plt

  flattened = flatten_features(features)
  if len(flattened) < 2:
    raise ValueError("PCA plotting requires at least two feature samples.")
  centered = flattened - flattened.mean(axis=0, keepdims=True)
  left, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
  points = left[:, :2] * singular_values[:2]

  figure, axis = plt.subplots(figsize=(6.4, 5.0), constrained_layout=True)
  scatter = axis.scatter(points[:, 0], points[:, 1], c=color, s=18, alpha=0.8)
  if color is not None:
    figure.colorbar(scatter, ax=axis)
  axis.set(title=title, xlabel="PC 1", ylabel="PC 2")
  return save_figure(figure, output)


def run(
  context: Any,
  *,
  features: str,
  output: str,
  stages: Sequence[str],
  capture: str | None = None,
  color_target: str | None = None,
) -> tuple[Path, ...]:
  """Plot a PCA projection for every requested feature stage."""
  jobs = resolve_outputs(
    context,
    {"output": output},
    ({"stage": stage} for stage in string_list(stages, "pca.stages")),
    label="PCA",
  )
  feature_batch = load_features(context.input(features))

  color: np.ndarray | None = None
  if color_target is not None:
    if capture is None:
      raise ValueError("capture is required when color_target is configured.")
    capture_batch = load_capture(context.input(capture))
    target = capture_batch.targets[color_target]
    color = target if target.ndim == 1 else target[:, 0]

  return tuple(
    plot_feature_pca(
      feature_batch.features[values["stage"]],
      color=color,
      title=f"{context.task_id} / {values['stage']}",
      output=paths["output"],
    )
    for values, paths in jobs
  )
