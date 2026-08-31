"""Run one evaluation episode in every MJLab vector environment."""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict


Episode = dict[str, int | float | bool]


def _success_term(env: Any) -> Any:
  """Find the task command that publishes episode success."""

  manager = env.unwrapped.command_manager
  terms = [manager.get_term(name) for name in manager.active_terms]
  terms = [term for term in terms if "episode_success" in term.metrics]
  if len(terms) != 1:
    raise RuntimeError(
      "Evaluation requires exactly one command with an episode_success metric."
    )
  return terms[0]


def _reset_completed(env: Any, done: torch.Tensor) -> TensorDict:
  """Reset finished MJLab slots and rebuild the RSL-RL observation."""

  env_ids = done.nonzero(as_tuple=False).squeeze(-1)
  observations, _ = env.unwrapped.reset(env_ids=env_ids)
  return TensorDict(observations, batch_size=[env.num_envs])


def run_episodes(env: Any, policy: Any, seed: int) -> list[Episode]:
  """Record the first completed episode from every vector worker."""

  env.seed(seed)
  observations, _ = env.reset()
  success_term = _success_term(env)
  rewards = torch.zeros(env.num_envs, device=env.device)
  lengths = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
  recorded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  rows: list[Episode] = []

  for _ in range(env.max_episode_length):
    with torch.inference_mode():
      actions = policy(observations)
    observations, reward, done, extras = env.step(actions)
    done = done.bool()
    timed_out = extras["time_outs"].bool()
    rewards += reward
    lengths += 1

    first_completions = done & ~recorded
    success = success_term.metrics["episode_success"]
    # Every other per-env metric the command publishes -- yaw error, overlap,
    # position error -- read at the step the episode ends. `success` is latched
    # over the episode; these are terminal, which is what "how well did it end"
    # means and what the training curves cannot tell you.
    terminal = {
      name: value
      for name, value in success_term.metrics.items()
      if name != "episode_success"
      and getattr(value, "ndim", 0) == 1
      and len(value) == env.num_envs
    }
    for worker in first_completions.nonzero(as_tuple=False).flatten().tolist():
      rows.append(
        {
          "episode_index": len(rows),
          "worker_env_id": worker,
          "seed": seed,
          "reward": float(rewards[worker].item()),
          "length": int(lengths[worker].item()),
          "success": float(success[worker].item()),
          "terminated": bool((done[worker] & ~timed_out[worker]).item()),
          "timed_out": bool(timed_out[worker].item()),
          **{name: float(value[worker].item()) for name, value in terminal.items()},
        }
      )
    recorded |= first_completions
    if bool(recorded.all()):
      return rows

    if bool(done.any()):
      observations = _reset_completed(env, done)
      rewards[done] = 0.0
      lengths[done] = 0

  raise RuntimeError(
    f"Only {len(rows)}/{env.num_envs} workers completed an episode within "
    f"{env.max_episode_length} steps."
  )
