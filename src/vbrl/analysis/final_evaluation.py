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
  "NatureCnn-FlattenRelu": ("Nature CNN", "#e87ba4"),
  "CompactVit-Svea": ("CompactViT", "#008300"),
  "DinoV3ViTS16-AfaFull6": ("DINOv3 + AFA", "#6250d6"),
  "DinoV3ViTS16-Global": ("DINOv3 global", "#e34948"),
}
SCENES = {
  "wood049": "In distribution",
  "real_table": "Rig table",
  "marble019": "Marble table",
  "ood_light": "Light colour",
  "object_blue": "Object blue",
  "goal_blue": "Goal blue",
  "object_blue_goal_purple": "Object blue,\ngoal purple",
  "colour_swap": "Colour swap",
}
# Lift-Cube's goal is never in view, so its "swap" only turns the cube green.
SCENE_OVERRIDES = {("LiftCube", "colour_swap"): "Object green"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


def scene_label(task: str, scene: str) -> str:
  return SCENE_OVERRIDES.get((task, scene), SCENES[scene])


def fetch(project: str = ENTITY_PROJECT) -> list[dict]:
  import wandb

  rows = []
  for run in wandb.Api(timeout=120).runs(project, per_page=500):
    match = re.match(GROUPS, run.group or "")
    if not match or run.state != "finished":
      continue
    task, _, rest = match["task"].partition("-WoodTable-")
    encoder = rest.removesuffix("-Trossen")
    seed = run.config.get("train_cfg", {}).get("seed")
    for key, value in run.summary.items():
      parts = key.split("/")
      if len(parts) != 3 or parts[0] != "final" or parts[1] not in SCENES:
        continue
      if parts[2] == "iteration":
        continue
      if value is None or isinstance(value, dict):
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
    [(_cell(summary, task, e, s, metric) or {}).get("mean", float("nan")) * 100
     for s in scenes]
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
      cell = _cell(summary, task, encoder, scene)
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
        == (task, encoder, scene, "success")
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
    plot_overview(summary, task, args.output / f"{task.lower()}_overview")
    if any(r["task"] == task and r["metric"] == "overlap" for r in summary):
      plot_overview(
        summary, task, args.output / f"{task.lower()}_overview_overlap", "overlap"
      )
    plot_scenes(rows, summary, task, args.output / f"{task.lower()}_scenes")
  print(
    f"{len(rows)} values from {len({r['run'] for r in rows})} runs -> {args.output}"
  )


__all__ = [
  "ENCODERS",
  "SCENES",
  "fetch",
  "main",
  "plot_overview",
  "plot_scenes",
  "summarise",
]


if __name__ == "__main__":
  main()
