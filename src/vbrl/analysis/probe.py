"""Fit deterministic ridge probes and plot held-out predictions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .capture import load_capture
from .features import flatten_features, load_features
from .io import load_npz, resolve_outputs, save_figure, save_npz, string_list


@dataclass(frozen=True)
class ProbeResult:
  """Metrics and held-out predictions from one deterministic ridge probe."""

  train_size: int
  test_size: int
  r2: float
  mean_absolute_error: float
  predictions: np.ndarray
  targets: np.ndarray
  metadata: Mapping[str, Any] = field(default_factory=dict)


def ridge_probe(
  features: np.ndarray,
  targets: np.ndarray,
  *,
  train_fraction: float = 0.7,
  alpha: float = 1.0,
  seed: int = 0,
) -> ProbeResult:
  """Fit a standardized ridge model and evaluate one deterministic split."""
  from sklearn.linear_model import Ridge
  from sklearn.metrics import mean_absolute_error, r2_score
  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import StandardScaler

  sample_count = len(features)
  if sample_count != len(targets):
    raise ValueError("Features and targets must contain the same number of samples.")
  if sample_count < 3:
    raise ValueError("At least three samples are required for a probe.")
  if not 0.0 < train_fraction < 1.0:
    raise ValueError("train_fraction must lie strictly between zero and one.")

  rng = np.random.default_rng(seed)
  order = rng.permutation(sample_count)
  split = min(max(1, round(sample_count * train_fraction)), sample_count - 1)
  train_ids, test_ids = order[:split], order[split:]

  features = flatten_features(features)
  targets = np.asarray(targets, dtype=np.float32)
  model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
  model.fit(features[train_ids], targets[train_ids])
  predictions = np.asarray(model.predict(features[test_ids]))
  expected = targets[test_ids]

  return ProbeResult(
    train_size=len(train_ids),
    test_size=len(test_ids),
    r2=float(r2_score(expected, predictions, multioutput="variance_weighted")),
    mean_absolute_error=float(mean_absolute_error(expected, predictions)),
    predictions=predictions,
    targets=expected,
  )


def save_probe(
  result: ProbeResult,
  path: str | Path,
  *,
  metadata: Mapping[str, Any] | None = None,
) -> Path:
  """Persist one probe result without pickle-backed arrays."""
  return save_npz(
    path,
    {
      "predictions": result.predictions,
      "targets": result.targets,
      "train_size": np.asarray(result.train_size, dtype=np.int64),
      "test_size": np.asarray(result.test_size, dtype=np.int64),
      "r2": np.asarray(result.r2, dtype=np.float64),
      "mean_absolute_error": np.asarray(
        result.mean_absolute_error, dtype=np.float64
      ),
    },
    {**result.metadata, **(metadata or {})},
  )


def load_probe(path: str | Path) -> ProbeResult:
  """Load one probe result from its compressed NumPy artifact."""
  arrays, metadata = load_npz(path)
  return ProbeResult(
    train_size=int(arrays["train_size"]),
    test_size=int(arrays["test_size"]),
    r2=float(arrays["r2"]),
    mean_absolute_error=float(arrays["mean_absolute_error"]),
    predictions=arrays["predictions"],
    targets=arrays["targets"],
    metadata=metadata,
  )


def plot_probe_predictions(
  targets: np.ndarray,
  predictions: np.ndarray,
  *,
  output: str | Path,
) -> Path:
  """Plot predicted values against held-out targets for every target dimension."""
  import matplotlib.pyplot as plt

  targets = np.asarray(targets).reshape(len(targets), -1)
  predictions = np.asarray(predictions).reshape(len(predictions), -1)
  if targets.shape != predictions.shape:
    raise ValueError(
      f"Target/prediction shapes differ: {targets.shape} vs {predictions.shape}."
    )

  figure, axes = plt.subplots(
    1, targets.shape[1], squeeze=False, figsize=(5 * targets.shape[1], 4.5)
  )
  for index, axis in enumerate(axes[0]):
    expected = targets[:, index]
    actual = predictions[:, index]
    axis.scatter(expected, actual, s=16, alpha=0.8)
    lower = float(min(expected.min(), actual.min()))
    upper = float(max(expected.max(), actual.max()))
    axis.plot((lower, upper), (lower, upper), "k--", linewidth=1)
    axis.set(xlabel="Target", ylabel="Prediction", title=f"Dimension {index}")

  figure.tight_layout()
  return save_figure(figure, output)


def run(
  context: Any,
  *,
  capture: str,
  features: str,
  output: str,
  plot: str,
  stages: Sequence[str],
  targets: Sequence[str],
  train_fraction: float = 0.7,
  alpha: float = 1.0,
  seed: int = 0,
) -> tuple[Path, ...]:
  """Fit and plot every requested feature-stage/target probe pair."""
  jobs = resolve_outputs(
    context,
    {"output": output, "plot": plot},
    (
      {"stage": stage, "target": target}
      for stage in string_list(stages, "probe.stages")
      for target in string_list(targets, "probe.targets")
    ),
    label="Probe",
  )

  capture_batch = load_capture(context.input(capture))
  feature_batch = load_features(context.input(features))
  generated: list[Path] = []
  for values, paths in jobs:
    result = ridge_probe(
      feature_batch.features[values["stage"]],
      capture_batch.targets[values["target"]],
      train_fraction=train_fraction,
      alpha=alpha,
      seed=seed,
    )
    generated.append(
      save_probe(result, paths["output"], metadata={**context.provenance(), **values})
    )
    generated.append(
      plot_probe_predictions(result.targets, result.predictions, output=paths["plot"])
    )
  return tuple(generated)
