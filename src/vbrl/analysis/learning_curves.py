from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

METRICS = {
  "success": ("Success rate (%)", 100.0, (0.0, 100.0)),
  "final_success": ("Final success rate (%)", 100.0, (0.0, 100.0)),
  "overlap": ("Final overlap (%)", 100.0, (0.0, 100.0)),
  "position_error": ("Final position error (mm)", 1000.0, (0.0, None)),
  "yaw_error": ("Final yaw error (°)", 180.0 / np.pi, (0.0, None)),
}
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
REFERENCE_COLOR = "#52514e"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


@dataclass(frozen=True)
class Curve:
  name: str
  runs: tuple[str, ...]


def parse_curve(text: str) -> Curve:
  name, separator, runs = text.partition("=")
  ids = tuple(run.strip() for run in runs.split(",") if run.strip())
  if not separator or not name.strip() or not ids:
    raise ValueError(f"Expected 'Name=run,run,run'; got {text!r}.")
  return Curve(name.strip(), ids)


def fetch(project: str, run: str, metrics: tuple[str, ...]) -> dict[int, dict[str, float]]:
  import wandb

  keys = ["eval/iteration", *(f"eval/{metric}" for metric in metrics)]
  rows = wandb.Api().run(f"{project}/{run}").scan_history(keys=keys)
  history = {
    int(row["eval/iteration"]): {metric: float(row[f"eval/{metric}"]) for metric in metrics}
    for row in rows
  }
  if not history:
    raise ValueError(f"Run {run} has no eval/* history.")
  return history


def aggregate(
  histories: list[dict[int, dict[str, float]]], metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  iterations = sorted(set.intersection(*(set(history) for history in histories)))
  values = np.array([[history[i][metric] for i in iterations] for history in histories])
  return np.array(iterations), values.mean(axis=0), values.min(axis=0), values.max(axis=0)


def _style(axis, label: str, limits) -> None:
  axis.set_ylabel(label, color=INK)
  axis.set_xlabel("Training iteration", color=INK)
  axis.xaxis.set_major_formatter(
    lambda value, _: f"{value / 1000:g}k" if value else "0"
  )
  if limits is not None:
    axis.set_ylim(*limits)
  axis.grid(axis="y", color=GRID, linewidth=0.6)
  axis.set_axisbelow(True)
  for side in ("top", "right"):
    axis.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    axis.spines[side].set_color(MUTED)
    axis.spines[side].set_linewidth(0.6)
  axis.tick_params(colors=MUTED, labelcolor=INK, width=0.6, length=3)
  axis.margins(x=0.01)


def plot(
  curves: list[Curve],
  reference: Curve | None,
  metrics: tuple[str, ...],
  data: dict[str, dict[str, dict[int, dict[str, float]]]],
  output: Path,
  name: str,
  seeds: bool = False,
) -> list[dict[str, object]]:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  plt.rcParams.update(
    {"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6}
  )
  figure, axes = plt.subplots(
    1, len(metrics), figsize=(3.3 * len(metrics), 2.6), squeeze=False
  )
  summary = []
  for axis, metric in zip(axes[0], metrics, strict=True):
    label, scale, limits = METRICS[metric]
    if reference is not None:
      histories = [data[reference.name][run] for run in reference.runs]
      final = np.array([history[max(history)][metric] for history in histories])
      axis.axhline(
        final.mean() * scale,
        color=REFERENCE_COLOR,
        linestyle=(0, (4, 3)),
        linewidth=1.2,
        label=f"{reference.name} (final)",
      )
    for curve, color in zip(curves, COLORS, strict=False):
      histories = [data[curve.name][run] for run in curve.runs]
      iterations, mean, low, high = aggregate(histories, metric)
      if seeds:
        for history in histories:
          axis.plot(
            iterations, [history[i][metric] * scale for i in iterations],
            color=color, alpha=0.35, linewidth=0.8,
          )
      else:
        axis.fill_between(
          iterations, low * scale, high * scale,
          color=color, alpha=0.18, linewidth=0,
        )
      axis.plot(iterations, mean * scale, color=color, linewidth=1.5, label=curve.name)
      final = np.array([history[max(history)][metric] for history in histories])
      summary.append({
        "curve": curve.name,
        "metric": metric,
        "seeds": len(curve.runs),
        "mean": final.mean() * scale,
        "std": final.std(ddof=1) * scale if len(final) > 1 else 0.0,
        "runs": " ".join(curve.runs),
      })
    _style(axis, label, limits)
  if len(curves) > 1 or reference is not None:
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(
      handles, labels, loc="lower center", ncol=min(len(labels), 4),
      frameon=False, bbox_to_anchor=(0.5, 0.0),
    )
    figure.tight_layout(rect=(0, 0.1, 1, 1))
  else:
    figure.tight_layout()
  output.mkdir(parents=True, exist_ok=True)
  for suffix in ("pdf", "png"):
    figure.savefig(output / f"{name}.{suffix}", dpi=300)
  plt.close(figure)
  return summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
  with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(
    description="Eval learning curves from W&B: mean across seeds, min-max shaded."
  )
  parser.add_argument("--project", required=True, help="W&B entity/project")
  parser.add_argument("--curve", action="append", required=True, type=parse_curve)
  parser.add_argument("--reference", type=parse_curve)
  parser.add_argument("--metric", action="append", required=True, choices=METRICS)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--name", default="learning_curves", help="figure file stem")
  parser.add_argument(
    "--seeds", action="store_true", help="thin line per seed instead of the band"
  )
  args = parser.parse_args(argv)

  metrics = tuple(args.metric)
  groups = [*args.curve, *([args.reference] if args.reference else [])]
  data = {
    group.name: {run: fetch(args.project, run, metrics) for run in group.runs}
    for group in groups
  }
  summary = plot(
    args.curve, args.reference, metrics, data, args.output, args.name, args.seeds
  )
  _write_csv(
    args.output / f"{args.name}_curves.csv",
    [
      {"curve": name, "run": run, "iteration": iteration, **values}
      for name, runs in data.items()
      for run, history in runs.items()
      for iteration, values in sorted(history.items())
    ],
  )
  _write_csv(args.output / f"{args.name}_final.csv", summary)
  for row in summary:
    print(
      f"{row['curve']:<28} {row['metric']:<15} "
      f"{row['mean']:.2f} ± {row['std']:.2f} ({row['seeds']} seeds)"
    )


__all__ = ["Curve", "aggregate", "main", "parse_curve", "plot"]


if __name__ == "__main__":
  main()
