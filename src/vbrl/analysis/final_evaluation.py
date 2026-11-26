from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from statistics import fmean, stdev

ENTITY_PROJECT = "eduard-nicolae-robot-learning/mjlab"
GROUPS = r"^distill-(u6bm8899|auxcyfuo-seeds)_(?P<task>.+)$"
# Fixed order and colour per encoder, the same in every figure (reference palette
# slots 1-6, validated for adjacent pairs).
ENCODERS = {
  "DinoV2ViTS14-AfaFull6": ("DINOv2 + AFA", "#2a78d6"),
  "DinoV2ViTS14-Global": ("DINOv2 global", "#eb6834"),
  "R3MResNet50-AfaFull32": ("R3M + AFA", "#1baf7a"),
  "R3MResNet50-Global": ("R3M global", "#eda100"),
  "R3MResNet50L3-AfaFull16": ("R3M layer 3 + AFA", "#8c564b"),
  "NatureCnn-FlattenRelu": ("Nature CNN", "#e87ba4"),
  "CompactVit-Svea": ("CompactViT", "#008300"),
  "DinoV3ViTS16-AfaFull6": ("DINOv3 + AFA", "#6250d6"),
  "DinoV3ViTS16-Global": ("DINOv3 global", "#e34948"),
}
SCENES = {
  "wood049": "In distribution",
  "real_table": "Rig table",
  "marble019": "Marble table",
  "ood_light": "Light color",
  "colour_swap": "Color swap",
  "object_color_sweep": "Object color\nsweep",
  "goal_color_sweep": "Goal color\nsweep",
}
# Lift-Cube's goal is never in view, so its "swap" only turns the cube green.
SCENE_OVERRIDES = {("LiftCube", "colour_swap"): "Object green"}
# Lift-Cube as the real robot scores it: holding the cube within 5 cm of the goal
# at the episode's end, not the cube passing the goal once.
SUCCESS = {"LiftCube": "held_at_goal"}
# The colour sweep charts box the object and goal marker defaults: chart cells as
# (hue column, shade counted from the darkest).
DEFAULT_OBJECT_CELL = (0, 2)
DEFAULT_GOAL_CELL = (5, 1)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


def scene_label(task: str, scene: str) -> str:
  return SCENE_OVERRIDES.get((task, scene), SCENES[scene])


def _students(project: str):
  import wandb

  for run in wandb.Api(timeout=120).runs(project, per_page=500):
    match = re.match(GROUPS, run.group or "")
    if not match or run.state != "finished":
      continue
    task, _, rest = match["task"].partition("-WoodTable-")
    encoder = rest.removesuffix("-Trossen")
    yield run, task, encoder, run.config.get("train_cfg", {}).get("seed")


def fetch_colour_sweeps(project: str = ENTITY_PROJECT) -> list[dict]:
  # The 256 per-colour values live in each run's logged table; W&B keeps lists
  # that long out of the run summary.
  rows = []
  for run, task, encoder, seed in _students(project):
    for artifact in run.logged_artifacts():
      match = re.search(r"final(\w+_color_sweep)per_colour", artifact.name)
      if artifact.type != "run_table" or not match:
        continue
      scene = match[1]
      table = artifact.get(f"final/{scene}/per_colour")
      data = sorted(table.data, key=lambda row: row[0])
      for column, metric in enumerate(table.columns):
        if metric in ("colour", "r", "g", "b"):
          continue
        rows.append(
          {
            "task": task,
            "encoder": encoder,
            "run": run.id,
            "seed": seed,
            "scene": scene,
            "metric": metric,
            "values": [float(row[column]) for row in data],
          }
        )
  return rows


def fetch(project: str = ENTITY_PROJECT) -> list[dict]:
  rows = []
  for run, task, encoder, seed in _students(project):
    for key, value in run.summary.items():
      parts = key.split("/")
      if len(parts) != 3 or parts[0] != "final" or parts[1] not in SCENES:
        continue
      if parts[2] == "iteration":
        continue
      if not isinstance(value, (int, float)):
        continue
      rows.append(
        {
          "task": task,
          "encoder": encoder,
          "run": run.id,
          "seed": seed,
          "scene": parts[1],
          "metric": parts[2],
          "value": float(value),
        }
      )
  return rows


def summarise(rows: list[dict]) -> list[dict]:
  cells: dict[tuple, list[float]] = {}
  for row in rows:
    cells.setdefault(
      (row["task"], row["encoder"], row["scene"], row["metric"]), []
    ).append(row["value"])
  out = []
  for (task, encoder, scene, metric), values in cells.items():
    out.append(
      {
        "task": task,
        "encoder": encoder,
        "scene": scene,
        "metric": metric,
        "seeds": len(values),
        "mean": fmean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
      }
    )
  order = {name: i for i, name in enumerate(ENCODERS)}
  scenes = {name: i for i, name in enumerate(SCENES)}
  return sorted(
    out,
    key=lambda r: (
      r["task"],
      order.get(r["encoder"], 99),
      scenes[r["scene"]],
      r["metric"],
    ),
  )


def _write_csv(path: Path, rows: list[dict]) -> None:
  with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def _cell(summary, task, encoder, scene, metric="success"):
  for row in summary:
    if (row["task"], row["encoder"], row["scene"], row["metric"]) == (
      task,
      encoder,
      scene,
      metric,
    ):
      return row
  return None


def _setup():
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  plt.rcParams.update(
    {"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6}
  )
  return plt


def plot_overview(
  summary: list[dict], task: str, path: Path, metric: str = "success"
) -> None:
  from matplotlib.colors import LinearSegmentedColormap

  plt = _setup()
  encoders = [
    e for e in ENCODERS if any(r["task"] == task and r["encoder"] == e for r in summary)
  ]
  scenes = [
    s for s in SCENES if any(r["task"] == task and r["scene"] == s for r in summary)
  ]
  cmap = LinearSegmentedColormap.from_list("ramp", RAMP)
  figure, axis = plt.subplots(
    figsize=(0.95 * len(scenes) + 1.9, 0.42 * len(encoders) + 1.2)
  )
  values = [
    [
      (_cell(summary, task, e, s, metric) or {}).get("mean", float("nan")) * 100
      for s in scenes
    ]
    for e in encoders
  ]
  axis.imshow(values, cmap=cmap, vmin=0, vmax=100, aspect="auto")
  for i, encoder in enumerate(encoders):
    for j, scene in enumerate(scenes):
      cell = _cell(summary, task, encoder, scene, metric)
      if cell is None:
        continue
      mean, std = cell["mean"] * 100, cell["std"] * 100
      colour = "white" if mean > 55 else INK
      axis.text(
        j, i - 0.12, f"{mean:.0f}", ha="center", va="center", color=colour, fontsize=9
      )
      axis.text(
        j,
        i + 0.22,
        f"±{std:.1f}",
        ha="center",
        va="center",
        color=colour,
        fontsize=6.5,
        alpha=0.85,
      )
  axis.set_xticks(
    range(len(scenes)), [scene_label(task, s) for s in scenes], fontsize=8
  )
  axis.set_yticks(range(len(encoders)), [ENCODERS[e][0] for e in encoders])
  axis.xaxis.tick_top()
  axis.tick_params(length=0, colors=INK)
  for spine in axis.spines.values():
    spine.set_visible(False)
  axis.set_xticks([x - 0.5 for x in range(1, len(scenes))], minor=True)
  axis.set_yticks([y - 0.5 for y in range(1, len(encoders))], minor=True)
  axis.grid(which="minor", color="white", linewidth=2)
  axis.tick_params(which="minor", length=0)
  figure.tight_layout()
  for suffix in ("pdf", "png"):
    figure.savefig(path.with_suffix(f".{suffix}"), dpi=300)
  plt.close(figure)


def plot_scenes(rows: list[dict], summary: list[dict], task: str, path: Path) -> None:
  plt = _setup()
  encoders = [
    e for e in ENCODERS if any(r["task"] == task and r["encoder"] == e for r in summary)
  ]
  scenes = [
    s for s in SCENES if any(r["task"] == task and r["scene"] == s for r in summary)
  ]
  figure, axis = plt.subplots(figsize=(0.9 * len(scenes) + 1.6, 3.0))
  width = 0.8 / len(encoders)
  for k, encoder in enumerate(encoders):
    name, colour = ENCODERS[encoder]
    offset = (k - (len(encoders) - 1) / 2) * width
    for j, scene in enumerate(scenes):
      cell = _cell(summary, task, encoder, scene, SUCCESS.get(task, "success"))
      if cell is None:
        continue
      x = j + offset
      axis.vlines(
        x, cell["min"] * 100, cell["max"] * 100, color=colour, linewidth=1.2, alpha=0.6
      )
      seeds = [
        r["value"] * 100
        for r in rows
        if (r["task"], r["encoder"], r["scene"], r["metric"])
        == (task, encoder, scene, SUCCESS.get(task, "success"))
      ]
      axis.scatter([x] * len(seeds), seeds, s=6, color=colour, alpha=0.45, linewidths=0)
      axis.scatter(
        [x],
        [cell["mean"] * 100],
        s=26,
        color=colour,
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
        label=name if j == 0 else None,
      )
  axis.set_xticks(
    range(len(scenes)), [scene_label(task, s) for s in scenes], fontsize=8
  )
  axis.set_ylabel("Success rate (%)", color=INK)
  axis.set_ylim(-3, 103)
  axis.set_xlim(-0.6, len(scenes) - 0.4)
  for x in range(1, len(scenes)):
    axis.axvline(x - 0.5, color=GRID, linewidth=0.6, zorder=0)
  axis.grid(axis="y", color=GRID, linewidth=0.6)
  axis.set_axisbelow(True)
  for side in ("top", "right"):
    axis.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    axis.spines[side].set_color(MUTED)
  axis.tick_params(colors=MUTED, labelcolor=INK, width=0.6, length=3)
  figure.legend(
    loc="lower center",
    ncol=min(len(encoders), 4),
    frameon=False,
    bbox_to_anchor=(0.5, 0.0),
  )
  figure.tight_layout(rect=(0, 0.12, 1, 1))
  for suffix in ("pdf", "png"):
    figure.savefig(path.with_suffix(f".{suffix}"), dpi=300)
  plt.close(figure)


_METRIC_LABELS = {
  "success": "Success rate (%)",
  "held_at_goal": "Success rate (%)",
  "overlap": "Final overlap (%)",
}


def plot_colour_sweep(
  sweeps: list[dict], task: str, scene: str, metric: str, path: Path
) -> None:
  import numpy as np

  from vbrl.scenes.presets import (
    COLOUR_SWEEP_COLOURS,
    COLOUR_SWEEP_HUES,
    COLOUR_SWEEP_SHADES,
    colour_sweep_rgb,
  )

  plt = _setup()
  rows = [
    r for r in sweeps if (r["task"], r["scene"], r["metric"]) == (task, scene, metric)
  ]
  encoders = [e for e in ENCODERS if any(r["encoder"] == e for r in rows)]

  # Columns: the hues, then the greys; rows: shades, light at the top.
  def as_chart(values):
    values = np.asarray(values)
    grid = values.reshape(COLOUR_SWEEP_HUES + 1, COLOUR_SWEEP_SHADES, *values.shape[1:])
    grid = grid.swapaxes(0, 1)
    return grid[::-1]

  colours = as_chart([colour_sweep_rgb(k) for k in range(COLOUR_SWEEP_COLOURS)])

  # Solid: the object's default colour; dashed: the goal marker's. Chart cells as
  # (hue column, shade from the darkest): dark red #690000 and dark green #004E00.
  boxes = tuple(
    ((column, COLOUR_SWEEP_SHADES - 1 - shade), style)
    for (column, shade), style in (
      (DEFAULT_OBJECT_CELL, "solid"),
      (DEFAULT_GOAL_CELL, (0, (2, 1.5))),
    )
  )
  # Columns sit at hue k / COLOUR_SWEEP_HUES of the circle; label the six primaries.
  ticks = [COLOUR_SWEEP_HUES * f for f in (0, 1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6)]
  ticks.append(COLOUR_SWEEP_HUES)
  names = ["red", "yellow", "green", "cyan", "blue", "magenta", "gray"]

  def frame(axis, title):
    axis.set_xticks(ticks, names, fontsize=7, rotation=45, ha="right")
    axis.set_yticks([0, COLOUR_SWEEP_SHADES - 1], ["light", "dark"], fontsize=7)
    axis.tick_params(length=0)
    axis.set_title(title, fontsize=9)
    for spine in axis.spines.values():
      spine.set_visible(False)
    for (x, y), style in boxes:
      axis.add_patch(
        plt.Rectangle(
          (x - 0.5, y - 0.5),
          1,
          1,
          fill=False,
          edgecolor=INK,
          linewidth=1.4,
          linestyle=style,
        )
      )

  columns = min(len(encoders) + 1, 4)
  lines = -(-(len(encoders) + 1) // columns)
  figure, axes = plt.subplots(
    lines, columns, figsize=(3.0 * columns, 3.2 * lines), squeeze=False
  )
  axes = axes.ravel()
  axes[0].imshow(colours, aspect="equal", interpolation="nearest")
  frame(axes[0], "Test colors (default: object solid, goal dashed)")
  for axis, encoder in zip(axes[1:], encoders, strict=False):
    values = np.mean([r["values"] for r in rows if r["encoder"] == encoder], axis=0)
    image = axis.imshow(
      as_chart(values) * 100,
      cmap="Blues",
      vmin=0,
      vmax=100,
      aspect="equal",
      interpolation="nearest",
    )
    frame(axis, ENCODERS[encoder][0])
  for axis in axes[len(encoders) + 1 :]:
    axis.set_visible(False)
  bar = figure.colorbar(image, ax=axes.tolist(), fraction=0.015, pad=0.01)
  bar.set_label(_METRIC_LABELS.get(metric, metric), fontsize=8)
  bar.ax.tick_params(labelsize=7)
  for suffix in ("pdf", "png"):
    figure.savefig(path.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight")
  plt.close(figure)

  figure, (by_hue, by_shade) = plt.subplots(
    1, 2, figsize=(6.3, 2.8), gridspec_kw={"width_ratios": (2, 1)}
  )
  hue_x = np.arange(COLOUR_SWEEP_HUES + 1)
  shade_x = np.arange(COLOUR_SWEEP_SHADES)
  for encoder in encoders:
    name, color = ENCODERS[encoder]
    grid = (
      np.mean([r["values"] for r in rows if r["encoder"] == encoder], axis=0).reshape(
        COLOUR_SWEEP_HUES + 1, COLOUR_SWEEP_SHADES
      )
      * 100
    )
    by_hue.plot(
      hue_x, grid.mean(1), color=color, linewidth=1.3, marker="o", ms=2.5, label=name
    )
    by_shade.plot(
      shade_x,
      grid[:COLOUR_SWEEP_HUES].mean(0),
      color=color,
      linewidth=1.3,
      marker="o",
      ms=2.5,
    )
  rainbow = [
    colour_sweep_rgb(k * COLOUR_SWEEP_SHADES + COLOUR_SWEEP_SHADES // 2) for k in hue_x
  ]
  ramp = [
    colour_sweep_rgb(COLOUR_SWEEP_HUES * COLOUR_SWEEP_SHADES + k) for k in shade_x
  ]
  for axis, strip, x, labels in (
    (by_hue, rainbow, hue_x, (ticks, names)),
    (by_shade, ramp, shade_x, ([0, COLOUR_SWEEP_SHADES - 1], ["dark", "light"])),
  ):
    axis.set_ylim(-12, 100)
    axis.imshow(
      [strip], extent=(x[0] - 0.5, x[-1] + 0.5, -11, -4), aspect="auto", zorder=0
    )
    axis.set_xlim(x[0] - 0.5, x[-1] + 0.5)
    axis.set_xticks(*labels, fontsize=7)
    axis.set_yticks(range(0, 101, 20))
    axis.grid(axis="y", color=GRID, linewidth=0.6)
    axis.tick_params(colors=MUTED, labelcolor=INK, width=0.6, length=0)
    for side in ("top", "right", "bottom"):
      axis.spines[side].set_visible(False)
    axis.spines["left"].set_color(MUTED)
    axis.spines["left"].set_bounds(0, 100)
  by_hue.set_ylabel(_METRIC_LABELS.get(metric, metric), color=INK)
  by_hue.set_title("Per color (over its shades)", fontsize=8, loc="left")
  by_shade.set_title("Per shade (over the colors)", fontsize=8, loc="left")
  figure.legend(
    loc="outside lower center", ncol=min(len(encoders), 4), frameon=False, fontsize=7
  )
  figure.set_layout_engine("constrained")
  for suffix in ("pdf", "png"):
    figure.savefig(
      path.with_name(path.name + "_lines").with_suffix(f".{suffix}"), dpi=300
    )
  plt.close(figure)


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(
    description="Final evaluation of every student across its scenes."
  )
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--project", default=ENTITY_PROJECT)
  args = parser.parse_args(argv)
  rows = fetch(args.project)
  summary = summarise(rows)
  args.output.mkdir(parents=True, exist_ok=True)
  _write_csv(
    args.output / "final_evaluation_runs.csv",
    sorted(
      rows,
      key=lambda r: (r["task"], r["encoder"], r["seed"] or 0, r["scene"], r["metric"]),
    ),
  )
  _write_csv(args.output / "final_evaluation_summary.csv", summary)
  for task in sorted({r["task"] for r in summary}):
    plot_overview(
      summary,
      task,
      args.output / f"{task.lower()}_overview",
      SUCCESS.get(task, "success"),
    )
    if any(r["task"] == task and r["metric"] == "overlap" for r in summary):
      plot_overview(
        summary, task, args.output / f"{task.lower()}_overview_overlap", "overlap"
      )
    plot_scenes(rows, summary, task, args.output / f"{task.lower()}_scenes")
  sweeps = fetch_colour_sweeps(args.project)
  for task, scene, metric in sorted(
    {(r["task"], r["scene"], r["metric"]) for r in sweeps}
  ):
    if metric in (SUCCESS.get(task, "success"), "overlap"):
      name = f"{task.lower()}_{scene}" + ("" if metric != "overlap" else "_overlap")
      plot_colour_sweep(sweeps, task, scene, metric, args.output / name)
  print(
    f"{len(rows)} values from {len({r['run'] for r in rows})} runs -> {args.output}"
  )


__all__ = [
  "ENCODERS",
  "SCENES",
  "SUCCESS",
  "fetch",
  "fetch_colour_sweeps",
  "main",
  "plot_colour_sweep",
  "plot_overview",
  "plot_scenes",
  "summarise",
]


if __name__ == "__main__":
  main()
