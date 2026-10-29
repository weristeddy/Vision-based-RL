from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean

EPISODES = 512
SEED = 0
_CHECKPOINT = re.compile(r"model_(\d+)\.pt")
_BOOKKEEPING = {"episode_index", "worker_env_id", "seed"}


def saved_checkpoints(log_dir: Path) -> list[tuple[int, Path]]:
  matches = ((_CHECKPOINT.fullmatch(path.name), path) for path in log_dir.iterdir())
  return sorted((int(match[1]), path) for match, path in matches if match)


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
      results.append({"iteration": iteration, **means})
  finally:
    env.close()
  return results


__all__ = ["EPISODES", "SEED", "evaluate_checkpoints", "saved_checkpoints"]
