"""Aggregate completed evaluation episodes and write presentation outputs."""

from __future__ import annotations

import csv
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _sample_std(values: Sequence[float]) -> float:
  return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
  """Aggregate each named checkpoint and scene from equal-weighted seed means."""

  grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
  for row in rows:
    key = (str(row["name"]), str(row["scene"]))
    grouped.setdefault(key, []).append(row)

  summaries = []
  for group in grouped.values():
    first = group[0]
    by_seed: dict[int, list[Mapping[str, Any]]] = {}
    for row in group:
      by_seed.setdefault(int(row["seed"]), []).append(row)

    def seed_means(key: str) -> list[float]:
      return [
        statistics.fmean(float(row[key]) for row in seed_rows)
        for seed_rows in by_seed.values()
      ]

    reward_means = seed_means("reward")
    success_means = seed_means("success")
    summaries.append(
      {
        "name": first["name"],
        "task_id": first["task_id"],
        "checkpoint_file": first["checkpoint_file"],
        "wandb_run_path": first["wandb_run_path"],
        "wandb_checkpoint_name": first["wandb_checkpoint_name"],
        "checkpoint_path": first["checkpoint_path"],
        "architecture": first["architecture"],
        "scene": first["scene"],
        "seeds": len(by_seed),
        "episodes": len(group),
        "mean_reward": statistics.fmean(reward_means),
        "reward_seed_std": _sample_std(reward_means),
        "success_rate": statistics.fmean(success_means),
        "success_seed_std": _sample_std(success_means),
      }
    )
  return summaries


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
  if not rows:
    raise ValueError(f"No evaluation rows to write to {path}.")
  with path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def _display(value: str) -> str:
  return value.replace("_", " ").replace(" + ", "\n")


def plot(
  summaries: Sequence[Mapping[str, Any]],
  output: Path,
  *,
  title: str,
) -> None:
  """Plot reward and success in model and scene order."""

  import matplotlib.pyplot as plt

  scenes = tuple(dict.fromkeys(str(row["scene"]) for row in summaries))

  def model_key(row: Mapping[str, Any]) -> str:
    return str(row["name"])

  models = tuple(dict.fromkeys(model_key(row) for row in summaries))
  lookup = {
    (str(row["scene"]), model_key(row)): row
    for row in summaries
  }
  model_info = {model_key(row): row for row in summaries}
  labels = [
    f"{_display(str(model_info[model]['architecture']))}\n"
    f"({model_info[model]['name']})"
    for model in models
  ]
  metrics = (
    ("Mean reward", "mean_reward", "reward_seed_std"),
    ("Success rate", "success_rate", "success_seed_std"),
  )
  figure, axes = plt.subplots(
    len(metrics),
    len(scenes),
    figsize=(max(9.0, 4.8 * len(scenes), 0.7 * len(models)), 8.5),
    squeeze=False,
  )
  x = list(range(len(models)))

  for column, scene in enumerate(scenes):
    selected = [lookup[(scene, model)] for model in models]
    for row_index, (ylabel, mean_key, std_key) in enumerate(metrics):
      axis = axes[row_index][column]
      axis.bar(
        x,
        [float(result[mean_key]) for result in selected],
        yerr=[float(result[std_key]) for result in selected],
        color="tab:blue",
        capsize=3,
      )
      axis.set_ylabel(ylabel)
      axis.set_title(_display(scene))
      axis.grid(axis="y", alpha=0.25)
      axis.set_axisbelow(True)
      if mean_key == "success_rate":
        axis.set_ylim(0.0, 1.05)
      if row_index == len(metrics) - 1:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, fontsize=8, rotation=40, ha="right")
      else:
        axis.set_xticks([])

  figure.suptitle(title, y=0.995)
  figure.text(
    0.5,
    0.955,
    "Bars = mean across seeds; error bars = sample SD across seed means",
    ha="center",
    fontsize=9,
  )
  figure.tight_layout(rect=(0.0, 0.025, 1.0, 0.92))
  figure.savefig(output, dpi=180)
  plt.close(figure)


def write_report(
  episodes: Sequence[Mapping[str, Any]],
  *,
  output: Path,
  title: str,
) -> Path:
  """Write raw episodes, aggregate metrics, and the comparison figure."""

  output.mkdir(parents=True, exist_ok=True)
  summaries = summarize(episodes)
  _write_csv(output / "episodes.csv", episodes)
  _write_csv(output / "summary.csv", summaries)
  plot(summaries, output / "evaluation.png", title=title)
  return output
