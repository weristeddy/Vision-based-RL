from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from vbrl.analysis.push_t_measurement import (
  GOAL_COLOUR,
  INK,
  MUTED,
  REFERENCE_COLOUR,
  SUCCESS_THRESHOLD,
  T_COLOUR,
  _style,
  _subtitle,
)


def measure_run(directory: Path) -> dict[str, Any]:
  import cv2

  from vbrl.deployment.config import load_config
  from vbrl.deployment.push_t.silhouette import (
    measure_t,
    overlap,
    render_footprint,
    render_t,
  )
  from vbrl.deployment.push_t.view import ExternalView

  goal = load_config(directory / "manifest.yaml").goal
  image = cv2.cvtColor(cv2.imread(str(directory / "final.png")), cv2.COLOR_BGR2RGB)
  view = ExternalView.load(image.shape[1], image.shape[0])
  fit = measure_t(view, image)
  x, y, yaw = fit.pose
  coverage = overlap(fit.pose, goal)
  goal_mask, t_mask = render_footprint(view, goal), render_t(view, fit.pose)
  rows, columns = np.nonzero(goal_mask | t_mask)
  middle = ((rows.min() + rows.max()) // 2, (columns.min() + columns.max()) // 2)
  crop = (
    slice(middle[0] - 110, middle[0] + 110), slice(middle[1] - 150, middle[1] + 150)
  )
  return {
    "run": directory.name,
    "dx_mm": (x - goal[0]) * 1000.0,
    "dy_mm": (y - goal[1]) * 1000.0,
    "dyaw_deg": math.degrees(math.remainder(yaw - goal[2], math.tau)),
    "overlap": coverage,
    "final_success": coverage >= SUCCESS_THRESHOLD,
    "fit_iou": fit.iou,
    "start": np.load(directory / "episode.npz")["frames"][0],
    "end": image[crop],
    "goal_mask": goal_mask[crop],
    "t_mask": t_mask[crop],
  }


def plot(runs: list[dict[str, Any]], output: Path, title: str) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  from matplotlib.lines import Line2D
  from matplotlib.patches import Rectangle

  plt.rcParams.update(
    {"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6}
  )
  figure = plt.figure(figsize=(6.3, 6.6), layout="constrained")
  figure.suptitle(title, x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK)
  grid = figure.add_gridspec(4, len(runs), height_ratios=(0.8, 0.75, 0.75, 1.5))
  for index, run in enumerate(runs):
    start = figure.add_subplot(grid[0, index])
    start.imshow(run["start"])
    start.set_title(f"run {run['run']}: start", loc="left", color=INK, fontsize=8)
    start.set_axis_off()
    plain = figure.add_subplot(grid[1, index])
    plain.imshow(run["end"])
    plain.set_title("end", loc="left", color=INK, fontsize=8)
    plain.set_axis_off()
    end = figure.add_subplot(grid[2, index])
    end.imshow(run["end"])
    end.contour(run["goal_mask"], levels=(0.5,), colors=GOAL_COLOUR, linewidths=0.9)
    end.contour(run["t_mask"], levels=(0.5,), colors=T_COLOUR, linewidths=0.9)
    end.set_title(f"end: {run['overlap']:.1%}", loc="left", color=INK, fontsize=8)
    end.set_axis_off()
  figure.legend(
    handles=[
      Line2D([], [], color=GOAL_COLOUR, label="goal footprint"),
      Line2D([], [], color=T_COLOUR, label="T fit"),
    ],
    loc="outside upper right", ncol=2, frameon=False, fontsize=7,
  )

  bottom = grid[3, :].subgridspec(1, 3)
  names = [run["run"] for run in runs]
  overlaps = np.array([run["overlap"] for run in runs]) * 100.0
  coverage = figure.add_subplot(bottom[0])
  coverage.bar(names, overlaps, width=0.6, color=T_COLOUR, alpha=0.8)
  coverage.axhline(SUCCESS_THRESHOLD * 100.0, color=REFERENCE_COLOUR, linewidth=0.8,
                   linestyle=(0, (4, 3)))
  coverage.set_ylim(50, 100)
  coverage.set_ylabel("Final overlap (%)")
  coverage.set_xlabel("Run")
  _subtitle(coverage, "(a) Overlap", f"mean {overlaps.mean():.1f} %, dashed: success")

  dx = np.array([run["dx_mm"] for run in runs])
  dy = np.array([run["dy_mm"] for run in runs])
  offset = figure.add_subplot(bottom[1])
  offset.add_patch(
    Rectangle((-5, -5), 10, 10, color=REFERENCE_COLOUR, alpha=0.1, linewidth=0)
  )
  offset.axhline(0.0, color=REFERENCE_COLOUR, linewidth=0.8)
  offset.axvline(0.0, color=REFERENCE_COLOUR, linewidth=0.8)
  offset.scatter(dx, dy, s=22, color=T_COLOUR, zorder=3)
  for index, (name, x, y) in enumerate(zip(names, dx, dy, strict=True)):
    offset.annotate(
      name, (x, y), xytext=(0, 6 if index % 2 else -10), textcoords="offset points",
      ha="center", fontsize=6.5, color=MUTED,
    )
  limit = 1.2 * max(np.abs(dx).max(), np.abs(dy).max(), 5.0)
  offset.set_xlim(-limit, limit)
  offset.set_ylim(-limit, limit)
  offset.set_aspect("equal")
  offset.set_xlabel("Δx (mm)")
  offset.set_ylabel("Δy (mm)")
  stats = f"mean Δx {dx.mean():+.1f}, Δy {dy.mean():+.1f} mm; shaded ±5 mm"
  _subtitle(offset, "(b) Final offset", stats)

  yaw = np.array([run["dyaw_deg"] for run in runs])
  turn = figure.add_subplot(bottom[2])
  turn.axhline(0.0, color=REFERENCE_COLOUR, linewidth=0.8)
  turn.bar(names, yaw, width=0.6, color=T_COLOUR, alpha=0.8)
  turn.set_ylabel("Δyaw (°)")
  turn.set_xlabel("Run")
  _subtitle(turn, "(c) Yaw", f"mean |Δyaw| {np.abs(yaw).mean():.1f}°")
  for axis in (coverage, offset, turn):
    _style(axis)
  output.mkdir(parents=True, exist_ok=True)
  for suffix in ("pdf", "png"):
    figure.savefig(output / f"rig_runs.{suffix}", dpi=300)
  plt.close(figure)


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(
    description="Re-measure the final frames of real Push-T episodes and plot them."
  )
  parser.add_argument("runs", type=Path, help="a folder of episode folders")
  parser.add_argument("--output", type=Path, default=Path("reports/push_t/rig_runs"))
  parser.add_argument("--title", default="Push-T on the real robot")
  args = parser.parse_args(argv)

  runs = [measure_run(d) for d in sorted(p for p in args.runs.iterdir() if p.is_dir())]
  plot(runs, args.output, args.title)
  keys = ("run", "dx_mm", "dy_mm", "dyaw_deg", "overlap", "final_success", "fit_iou")
  with (args.output / "rig_runs.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=keys)
    writer.writeheader()
    writer.writerows({key: run[key] for key in keys} for run in runs)
  for run in runs:
    print(
      f"run {run['run']}: dx {run['dx_mm']:+.1f} mm, dy {run['dy_mm']:+.1f} mm, "
      f"dyaw {run['dyaw_deg']:+.2f} deg, overlap {run['overlap']:.3f}"
    )


__all__ = ["main", "measure_run", "plot"]


if __name__ == "__main__":
  main()
