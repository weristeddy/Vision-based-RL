from __future__ import annotations

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

  @staticmethod
  def from_task(task_id: str) -> TrainConfig:
    return TrainConfig(env=load_env_cfg(task_id), agent=load_rl_cfg(task_id))






def _apply_label(cfg: TrainConfig) -> None:
  if not cfg.label:
    return
  cfg.agent.wandb_tags = (*cfg.agent.wandb_tags, cfg.label)
  run_name = cfg.agent.run_name
  cfg.agent.run_name = f"{cfg.label}_{run_name}" if run_name else cfg.label


def apply_overrides(cfg: TrainConfig) -> None:
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
