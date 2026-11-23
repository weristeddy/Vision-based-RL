from __future__ import annotations

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


__all__ = ["EPISODES", "SEED", "evaluate_checkpoints", "saved_checkpoints"]
