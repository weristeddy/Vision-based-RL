from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from vbrl.analysis.final_evaluation import ENCODERS, GRID, INK, MUTED, _setup

PUSH_T_SUCCESS = 0.90


def load(path: Path) -> list[dict]:
  rows = list(csv.DictReader(path.open()))
  encoders = {row["encoder"] for row in rows}
  starts: dict[tuple[str, str], set[str]] = {}
  for row in rows:
    starts.setdefault((row["goal"], row["run"]), set()).add(row["encoder"])
  # Paired comparison: a start counts only once every encoder has run it.
  complete = {start for start, done in starts.items() if done == encoders}
  rows = [row for row in rows if (row["goal"], row["run"]) in complete]
  goals = {
    goal: json.loads((path.parent / f"goal{goal}" / "goal.json").read_text())["goal"]
    for goal in {row["goal"] for row in rows}
  }
  for row in rows:
    x, y, _ = json.loads(row["end_pose"])
    goal_x, goal_y, _ = goals[row["goal"]]
    row["position_error_mm"] = 1000 * math.hypot(x - goal_x, y - goal_y)
  return rows


def _strip(axis, encoders, rows, key, scale) -> None:
  for i, encoder in enumerate(encoders):
    color = ENCODERS[encoder][1]
    values = [scale * float(r[key]) for r in rows if r["encoder"] == encoder]
    # Spread the runs sideways so equal values (often 0) stay countable.
    spread = [
      i + 0.36 * (k / max(len(values) - 1, 1) - 0.5) for k in range(len(values))
    ]
    axis.scatter(spread, values, s=14, color=color, alpha=0.6, lw=0)
    axis.scatter(
      [i],
      [sum(values) / len(values)],
      s=80,
      marker="_",
      color=color,
      lw=2.2,
      label=ENCODERS[encoder][0],
    )
  axis.set_xticks([])
  axis.set_xlim(-0.6, len(encoders) - 0.4)
  axis.grid(axis="y", color=GRID, linewidth=0.6)
  axis.set_axisbelow(True)
  axis.tick_params(colors=MUTED, labelcolor=INK, width=0.6, length=3)
  axis.tick_params(axis="x", length=0)
  for side in ("top", "right", "bottom"):
    axis.spines[side].set_visible(False)
  axis.spines["left"].set_color(MUTED)


def _results(overlap, error, encoders, rows) -> None:
  _strip(overlap, encoders, rows, "end_overlap", 100)
  overlap.axhline(100 * PUSH_T_SUCCESS, color=MUTED, linewidth=0.8, linestyle="--")
  overlap.set_ylim(-3, 110)
  overlap.set_yticks(range(0, 101, 20))
  overlap.spines["left"].set_bounds(0, 100)
  overlap.set_title("Final overlap with the goal (%)", loc="left", color=INK)
  for i, encoder in enumerate(encoders):
    runs = [r for r in rows if r["encoder"] == encoder]
    wins = sum(r["success"] == "True" for r in runs)
    overlap.text(
      i, 105, f"{wins}/{len(runs)}", ha="center", va="center", fontsize=8, color=MUTED
    )
  _strip(error, encoders, rows, "position_error_mm", 1)
  step = 50 if max(r["position_error_mm"] for r in rows) <= 300 else 100
  top = step * math.ceil(max(r["position_error_mm"] for r in rows) / step)
  error.set_ylim(-0.03 * top, 1.0 * top + 0.1 * top)
  error.set_yticks(range(0, top + 1, step))
  error.spines["left"].set_bounds(0, top)
  error.set_title("Final position error (mm)", loc="left", color=INK)


def _composite(root: Path, goal: int) -> tuple:
  import cv2
  import numpy as np

  from vbrl.rig.measure import read, render_goal, render_object
  from vbrl.rig.view import ExternalView

  folder = root / f"goal{goal}"
  image = read(folder / "goal_external.png")
  view = ExternalView.load(image.shape[1], image.shape[0])
  encoder = next(e for e in ENCODERS if (folder / e).is_dir())
  runs = sorted((folder / encoder).glob("run*"), key=lambda p: int(p.name[3:]))
  # Each start's real T, cut from its start still: the arm is at home in every
  # still, so only the T differs from the goal photo.
  starts = []
  for run in runs:
    pose = json.loads((run / "episode.json").read_text())["result"]["start_pose"]
    mask = cv2.dilate(
      render_object(view, "push_t", pose), np.ones((5, 5), np.uint8)
    ).astype(bool)
    image[mask] = read(run / "start_external.png")[mask]
    starts.append((run.name[3:], mask))
  goal_pose = json.loads((folder / "goal.json").read_text())["goal"]
  outline = render_goal(view, "push_t", (*goal_pose, 0.0)).astype(bool)
  rows, cols = np.nonzero(outline | np.any([m for _, m in starts], axis=0))
  pad = 60
  top, bottom = max(rows.min() - pad, 0), min(rows.max() + pad, image.shape[0])
  left, right = max(cols.min() - pad, 0), min(cols.max() + pad, image.shape[1])
  labels = []
  for label, mask, style in [
    *((number, mask, "circle,pad=0.25") for number, mask in starts),
    ("goal", outline, "round,pad=0.3"),
  ]:
    y, x = np.argwhere(mask[top:bottom, left:right]).mean(axis=0)
    labels.append((label, x, y, style))
  return image[top:bottom, left:right], labels


def _starts(axis, composite: tuple) -> None:
  image, labels = composite
  axis.imshow(image)
  for label, x, y, style in labels:
    axis.text(
      x,
      y,
      label,
      ha="center",
      va="center",
      color="white",
      fontsize=9,
      fontweight="bold",
      bbox={"boxstyle": style, "facecolor": INK, "edgecolor": "none"},
    )
  axis.set_axis_off()


def _legend(figure, encoders, loc="lower center") -> None:
  from matplotlib.lines import Line2D

  handles = [
    Line2D(
      [], [], marker="o", markersize=5, lw=0, color=ENCODERS[e][1], label=ENCODERS[e][0]
    )
    for e in encoders
  ]
  figure.legend(
    handles=handles,
    loc=loc,
    ncol=len(encoders) // 2,
    frameon=False,
    handletextpad=0.3,
    columnspacing=1.6,
    fontsize=8,
  )


def _save(figure, path: Path) -> None:
  import matplotlib.pyplot as plt

  for suffix in ("pdf", "png"):
    figure.savefig(path.with_suffix(f".{suffix}"), dpi=300)
  plt.close(figure)


def plot(rows: list[dict], path: Path) -> None:
  plt = _setup()
  encoders = [e for e in ENCODERS if any(r["encoder"] == e for r in rows)]
  figure, (overlap, error) = plt.subplots(
    1, 2, figsize=(6.3, 3.0), layout="constrained"
  )
  _results(overlap, error, encoders, rows)
  _legend(figure, encoders, "outside lower center")
  _save(figure, path)


def plot_starts(root: Path, goal: int, path: Path) -> None:
  plt = _setup()
  composite = _composite(root, goal)
  height, width = composite[0].shape[:2]
  figure, axis = plt.subplots(figsize=(6.3, 6.3 * height / width))
  figure.subplots_adjust(0, 0, 1, 1)
  _starts(axis, composite)
  _save(figure, path)


def plot_goal(rows: list[dict], root: Path, goal: int, path: Path) -> None:
  plt = _setup()
  rows = [r for r in rows if int(r["goal"]) == goal]
  encoders = [e for e in ENCODERS if any(r["encoder"] == e for r in rows)]
  composite = _composite(root, goal)
  height, width = composite[0].shape[:2]
  photo = 6.3 * height / width
  figure = plt.figure(figsize=(6.3, photo + 3.2), layout="constrained")
  figure.get_layout_engine().set(h_pad=0.08, hspace=0.04, wspace=0.06)
  grid = figure.add_gridspec(2, 2, height_ratios=(photo, 2.6))
  axes = [figure.add_subplot(grid[0, :]), figure.add_subplot(grid[1, 0])]
  axes.append(figure.add_subplot(grid[1, 1]))
  _starts(axes[0], composite)
  axes[0].set_title("Start poses and goal", loc="left", color=INK)
  _results(axes[1], axes[2], encoders, rows)
  for axis, letter in zip(axes, "abc", strict=True):
    axis.set_title(f"({letter})  " + axis.get_title(loc="left"), loc="left")
  _legend(figure, encoders, "outside lower center")
  _save(figure, path)


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(
    description="Real-robot Push-T results per encoder from collect's results.csv."
  )
  parser.add_argument(
    "--results", type=Path, default=Path("artifacts/real_eval/push_t/results.csv")
  )
  parser.add_argument("--output", type=Path, default=Path("reports/real"))
  args = parser.parse_args(argv)
  rows = load(args.results)
  args.output.mkdir(parents=True, exist_ok=True)
  plot(rows, args.output / "push_t_overlap")
  for goal in sorted({int(r["goal"]) for r in rows}):
    plot_starts(args.results.parent, goal, args.output / f"push_t_goal{goal}_starts")
    plot_goal(rows, args.results.parent, goal, args.output / f"push_t_goal{goal}")
  starts = len({(r["goal"], r["run"]) for r in rows})
  print(f"{len(rows)} episodes, {starts} starts -> {args.output}")


__all__ = ["load", "main", "plot", "plot_goal", "plot_starts"]


if __name__ == "__main__":
  main()
