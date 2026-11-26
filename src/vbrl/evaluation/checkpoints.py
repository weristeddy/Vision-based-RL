from __future__ import annotations

import csv
import re
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean

from vbrl.scenes.presets import COLOUR_SWEEP_COLOURS, COLOUR_SWEEP_SCENES, sweep_colour

EPISODES = 512
SEED = 0
_CHECKPOINT = re.compile(r"model_(\d+)\.pt")
_BOOKKEEPING = {"episode_index", "worker_env_id", "seed"}


def saved_checkpoints(log_dir: Path) -> list[tuple[int, Path]]:
  matches = ((_CHECKPOINT.fullmatch(path.name), path) for path in log_dir.iterdir())
  return sorted((int(match[1]), path) for match, path in matches if match)


def _by_colour(episodes: list[dict], keys: list[str]) -> dict[str, list[float]]:
  groups: list[list[dict]] = [[] for _ in range(COLOUR_SWEEP_COLOURS)]
  for episode in episodes:
    groups[sweep_colour(int(episode["worker_env_id"]), EPISODES)].append(episode)
  return {key: [fmean(float(e[key]) for e in group) for group in groups] for key in keys}


def evaluate_checkpoints(
  task_id: str,
  checkpoints: Sequence[tuple[int, Path]],
  *,
  device: str,
  scene: str | None = None,
) -> list[dict[str, float]]:
  from vbrl.runtime import CheckpointRef, build_env, load_trained_policy

  from .rollout import run_episodes

  env = build_env(
    task_id, device=device, num_envs=EPISODES, seed=SEED, scene=scene, auto_reset=False
  )
  results = []
  try:
    for iteration, path in checkpoints:
      ref = CheckpointRef(checkpoint_file=str(path))
      env, _, policy, _ = load_trained_policy(
        env, task_id=task_id, device=device, ref=ref
      )
      episodes = run_episodes(env, policy, seed=SEED)
      keys = sorted(episodes[0].keys() - _BOOKKEEPING)
      means = {key: fmean(float(e[key]) for e in episodes) for key in keys}
      row = {"iteration": iteration, **means}
      if scene in COLOUR_SWEEP_SCENES:
        row["by_colour"] = _by_colour(episodes, keys)
      results.append(row)
  finally:
    env.close()
  return results


def _write_csv(path: Path, rows: list[dict]) -> None:
  with path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def evaluate_run(
  task_id: str,
  saved: Sequence[tuple[int, Path]],
  scenes: Sequence[str],
  out_dir: Path,
  *,
  device: str,
  curve: bool = True,
) -> tuple[list[dict], list[dict], dict[str, dict]]:
  rows = evaluate_checkpoints(task_id, saved, device=device) if curve else []
  final, sweeps = [], {}
  for scene in scenes:
    for row in evaluate_checkpoints(task_id, saved[-1:], device=device, scene=scene):
      if "by_colour" in row:
        sweeps[scene] = row.pop("by_colour")
      final.append({"scene": scene, **row})
  out_dir.mkdir(parents=True, exist_ok=True)
  if rows:
    _write_csv(out_dir / "evaluation.csv", rows)
  if final:
    _write_csv(out_dir / "final_evaluation.csv", final)
  if sweeps:
    _write_csv(
      out_dir / "final_colour_sweeps.csv",
      [
        {"scene": scene, "colour": k, **{key: values[k] for key, values in by.items()}}
        for scene, by in sweeps.items()
        for k in range(COLOUR_SWEEP_COLOURS)
      ],
    )
  return rows, final, sweeps


def log_evaluation(
  run, curve: list[dict], final: list[dict], sweeps: dict[str, dict]
) -> None:
  import wandb

  from vbrl.scenes.presets import colour_sweep_rgb

  run.define_metric("eval/iteration")
  run.define_metric("eval/*", step_metric="eval/iteration")
  for row in curve:
    run.log({f"eval/{key}": value for key, value in row.items()})
  if final:
    table = wandb.Table(columns=list(final[0]), data=[list(r.values()) for r in final])
    run.log({"final/scenes": table})
    for metric in ("success", "overlap", "yaw_error"):
      run.log({f"final/{metric}": wandb.plot.bar(table, "scene", metric, title=metric)})
  for row in final:
    scene = row["scene"]
    run.summary.update(
      {f"final/{scene}/{key}": value for key, value in row.items() if key != "scene"}
    )
  for scene, by in sweeps.items():
    metrics = list(by)
    table = wandb.Table(columns=["colour", "r", "g", "b", *metrics])
    for k in range(COLOUR_SWEEP_COLOURS):
      table.add_data(k, *colour_sweep_rgb(k), *(by[m][k] for m in metrics))
    run.log({f"final/{scene}/per_colour": table})


__all__ = [
  "EPISODES",
  "SEED",
  "evaluate_checkpoints",
  "evaluate_run",
  "log_evaluation",
  "saved_checkpoints",
]
