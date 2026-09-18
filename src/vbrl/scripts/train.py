from __future__ import annotations

import math
import sys
from dataclasses import dataclass

import tyro
from mjlab.scripts.train import TrainConfig as MjlabTrainConfig
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

WORKER_ENV = (
  "VBRL*",
  "WANDB*",
  "WARP*",
  "SSL*",
  "REQUESTS*",
  "CURL*",
  "HF*",
  "TRANSFORMERS*",
  "MPLBACKEND",
  "PYOPENGL*",
  "SLURM*",
)


@dataclass(frozen=True)
class TrainConfig(MjlabTrainConfig):
  label: str = ""
  """Extra W&B tag and run-name prefix, for telling one sweep arm from another."""
  min_action_std: float | None = None
  """Lower bound of the policy's action-std range."""
  action_path_weight: float | None = None
  """Weight of the L1 commanded-travel penalty."""
  action_rate_weight: float | None = None
  """Weight of MJLab's `action_rate_l2`."""
  object_press_weight: float | None = None
  """Weight of the downward-press penalty."""
  goal_yaw_pin_iterations: int | None = None
  """Iterations the goal yaw stays pinned before the range starts widening."""
  goal_yaw_rungs: int | None = None
  """Rungs the goal-yaw range widens in."""
  goal_yaw_rung_iterations: int | None = None
  """Iterations between rungs."""

  @staticmethod
  def from_task(task_id: str) -> TrainConfig:
    return TrainConfig(env=load_env_cfg(task_id), agent=load_rl_cfg(task_id))


def _retune_penalty_weights(cfg: TrainConfig) -> None:
  overrides = (
    ("action_path_length", "--action-path-weight", cfg.action_path_weight),
    ("action_rate_l2", "--action-rate-weight", cfg.action_rate_weight),
    ("object_table_press", "--object-press-weight", cfg.object_press_weight),
  )
  for name, flag, weight in overrides:
    if weight is None:
      continue
    if weight > 0.0:
      raise ValueError(f"{flag} must be <= 0; got {weight}.")
    term = (cfg.env.rewards or {}).get(name)
    if term is None:
      raise ValueError(f"This task has no `{name}` reward term.")
    term.weight = weight


def _rebuild_goal_yaw_stages(cfg: TrainConfig) -> None:
  requested = (
    cfg.goal_yaw_pin_iterations,
    cfg.goal_yaw_rungs,
    cfg.goal_yaw_rung_iterations,
  )
  if all(value is None for value in requested):
    return
  term = (cfg.env.curriculum or {}).get("goal_yaw_range")
  if term is None:
    raise ValueError(
      "This task has no goal-yaw curriculum, so --goal-yaw-* cannot apply."
    )

  per_iteration = cfg.agent.num_steps_per_env
  registered = list(term.params["stages"])
  pin = cfg.goal_yaw_pin_iterations
  if pin is None:
    pin = registered[1]["step"] // per_iteration if len(registered) > 1 else 0
  rungs = cfg.goal_yaw_rungs or max(1, len(registered) - 1)
  gap = cfg.goal_yaw_rung_iterations
  if gap is None:
    gap = (
      (registered[2]["step"] - registered[1]["step"]) // per_iteration
      if len(registered) > 2
      else 0
    )
  if pin < 0 or rungs < 1 or gap < 0:
    raise ValueError(f"Need pin >= 0, rungs >= 1, gap >= 0; got {pin}, {rungs}, {gap}.")

  term.params["stages"] = [{"step": 0, "half_range": 0.0}] + [
    {
      "step": (pin + (rung - 1) * gap) * per_iteration,
      "half_range": math.pi * rung / rungs,
    }
    for rung in range(1, rungs + 1)
  ]
  full = term.params["stages"][-1]["step"] // per_iteration
  print(
    f"[INFO] Goal-yaw curriculum: pinned for {pin} iterations, then {rungs} "
    f"rung(s) every {gap}; full circle at iteration {full} of "
    f"{cfg.agent.max_iterations}."
  )


def _install_action_std_floor(cfg: TrainConfig) -> None:
  if cfg.min_action_std is None:
    return
  distribution = cfg.agent.actor.distribution_cfg
  low, high = distribution.get("std_range") or (None, 1e6)
  distribution["std_range"] = (cfg.min_action_std, high)
  print(f"[INFO] Action std range {low} -> {cfg.min_action_std} (upper {high}).")


def _apply_label(cfg: TrainConfig) -> None:
  if not cfg.label:
    return
  cfg.agent.wandb_tags = (*cfg.agent.wandb_tags, cfg.label)
  run_name = cfg.agent.run_name
  cfg.agent.run_name = f"{cfg.label}_{run_name}" if run_name else cfg.label


def apply_overrides(cfg: TrainConfig) -> None:
  _retune_penalty_weights(cfg)
  _rebuild_goal_yaw_stages(cfg)
  _install_action_std_floor(cfg)
  _apply_label(cfg)


def _print_task_overview() -> None:
  from vbrl.tasks import vbrl_task_ids

  print("usage: vbrl-train <TASK_ID> [OPTIONS]\n")
  print("The task ID fixes the task, robot, scene, camera, and architecture.")
  print("Everything else -- seed, learning rate, iterations, env count, video")
  print("-- is a flag. Run 'vbrl-train <TASK_ID> --help' to see them all.\n")
  print("Registered VBRL tasks:")
  for task_id in vbrl_task_ids():
    print(f"  {task_id}")
  print("\nMJLab's own tasks are selectable here too; see 'list-envs'.")


def main() -> None:
  import mjlab
  import mjlab.tasks  # noqa: F401
  import torchrunx
  from mjlab.scripts.train import launch_training

  import vbrl.tasks  # noqa: F401

  if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    _print_task_overview()
    return

  torchrunx.DEFAULT_ENV_VARS_FOR_COPY += WORKER_ENV

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=f"{sys.argv[0]} {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  apply_overrides(args)
  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()


__all__ = ["WORKER_ENV", "TrainConfig", "apply_overrides", "main"]
