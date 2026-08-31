"""Turn probe artifacts into figures a reader can interpret.

:mod:`vbrl.analysis.probe` plots predictions against targets in whatever units it
was fitted on. For yaw that is a ``(sin, cos)`` pair, so its axes read
"Dimension 0/1" and a good fit says nothing about how many degrees the estimate
is off by. This converts back: angles in degrees, positions in millimetres, with
the held-out error distribution beside the fit so a tight scatter cannot hide a
tail.

The units are recovered from the target's name, which is the same name
``capture`` recorded it under -- anything ending in ``yaw`` is a ``(sin, cos)``
pair, anything else is treated as a vector and scored by its norm.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["angle_error_degrees", "run", "summarize_probe"]


def angle_error_degrees(targets: Any, predictions: Any) -> Any:
  """Wrapped angular error, in degrees, from ``(sin, cos)`` pairs."""
  import numpy as np

  actual = np.arctan2(targets[:, 0], targets[:, 1])
  estimate = np.arctan2(predictions[:, 0], predictions[:, 1])
  wrapped = (actual - estimate + np.pi) % (2.0 * np.pi) - np.pi
  return np.degrees(np.abs(wrapped))


def summarize_probe(path: str | Path, target: str) -> dict[str, Any]:
  """Load one probe artifact and score it in the target's natural units."""
  import numpy as np

  from .probe import load_probe

  result = load_probe(path)
  targets = np.asarray(result.targets, dtype=np.float64).reshape(
    len(result.targets), -1
  )
  predictions = np.asarray(result.predictions, dtype=np.float64).reshape(
    targets.shape
  )
  if target.endswith("yaw"):
    errors = angle_error_degrees(targets, predictions)
    unit, actual, estimate = "deg", np.degrees(
      np.arctan2(targets[:, 0], targets[:, 1])
    ), np.degrees(np.arctan2(predictions[:, 0], predictions[:, 1]))
  else:
    errors = np.linalg.norm(targets - predictions, axis=1) * 1000.0
    unit, actual, estimate = "mm", targets[:, 0] * 1000.0, predictions[:, 0] * 1000.0
  residual = ((targets - predictions) ** 2).sum()
  total = ((targets - targets.mean(axis=0)) ** 2).sum()
  return {
    "target": target,
    "unit": unit,
    "mean": float(errors.mean()),
    "median": float(np.median(errors)),
    "p95": float(np.percentile(errors, 95)),
    "r2": float(1.0 - residual / total) if total else float("nan"),
    "n": int(len(errors)),
    "spread": float(actual.max() - actual.min()),
    "errors": errors,
    "actual": actual,
    "estimate": estimate,
  }


def _figure(rows: list[dict[str, Any]], destination: Path) -> Path:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  import numpy as np

  fig, axes = plt.subplots(
    len(rows), 2, figsize=(9.0, 3.1 * len(rows)), squeeze=False
  )
  for index, row in enumerate(rows):
    fit, hist = axes[index]
    lo = min(row["actual"].min(), row["estimate"].min())
    hi = max(row["actual"].max(), row["estimate"].max())
    fit.plot([lo, hi], [lo, hi], ls=(0, (3, 3)), lw=0.9, color="#8b8a84", zorder=1)
    fit.scatter(row["actual"], row["estimate"], s=16, alpha=0.7,
                color="#2a78d6", edgecolors="none", zorder=2)
    fit.set_xlabel(f"true ({row['unit']})")
    fit.set_ylabel(f"decoded ({row['unit']})")
    fit.set_title(
      f"{row['label']}  —  mean |error| {row['mean']:.2f} {row['unit']}",
      loc="left", fontsize=10, fontweight="bold")
    fit.spines[["top", "right"]].set_visible(False)

    hist.hist(row["errors"], bins=24, color="#2a78d6", alpha=0.85)
    for value, colour, name in (
      (row["median"], "#0b0b0b", "median"), (row["p95"], "#eb6834", "p95")
    ):
      hist.axvline(value, color=colour, lw=1.4)
      hist.text(value, hist.get_ylim()[1] * 0.95, f" {name} {value:.2f}",
                fontsize=8, color=colour, va="top")
    hist.set_xlabel(f"held-out |error| ({row['unit']})")
    hist.set_ylabel("frames")
    hist.set_title(
      f"n={row['n']}   true range spans {row['spread']:.0f} {row['unit']}",
      loc="left", fontsize=9, color="#52514e")
    hist.spines[["top", "right"]].set_visible(False)
  fig.tight_layout()
  destination.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(destination, dpi=170, bbox_inches="tight")
  plt.close(fig)
  return destination


def run(
  context: Any,
  *,
  probes: dict[str, str],
  output: str,
  table: str | None = None,
) -> Any:
  """Render probe artifacts in interpretable units.

  ``probes`` maps a label to a probe NPZ written earlier in the manifest, e.g.
  ``{"adapter · object yaw": "probe_adapter_object_yaw.npz"}``. The target's
  units are taken from the label's trailing word, so a label naming ``yaw`` is
  scored in degrees.
  """
  rows = []
  for label, path in probes.items():
    target = "yaw" if "yaw" in label else "position"
    row = summarize_probe(context.input(path), target)
    row["label"] = label
    rows.append(row)

  written = [_figure(rows, context.output(output))]
  if table is not None:
    lines = [f"{'probe':34}{'mean':>9}{'median':>9}{'p95':>9}{'R2':>8}{'n':>7}"]
    for row in rows:
      lines.append(
        f"{row['label']:34}{row['mean']:8.2f}{row['unit']:>1}"
        f"{row['median']:9.2f}{row['p95']:9.2f}{row['r2']:8.3f}{row['n']:7d}"
      )
    destination = context.output(table)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n")
    written.append(destination)
  return written
