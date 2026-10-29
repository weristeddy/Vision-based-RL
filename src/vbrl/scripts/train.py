from __future__ import annotations

import csv
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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
  evaluate: bool = True
  """Evaluate the saved checkpoints after training and keep only the final one."""

  @staticmethod
  def from_task(task_id: str) -> TrainConfig:
    return TrainConfig(env=load_env_cfg(task_id), agent=load_rl_cfg(task_id))


def _launch(task_id: str, args: TrainConfig) -> Path:
  import torchrunx
  from mjlab.scripts.train import run_train
  from mjlab.utils.gpu import select_gpus

  name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    name += f"_{args.agent.run_name}"
  log_dir = (Path(args.log_root) / args.agent.experiment_name).resolve() / name
  gpus, count = select_gpus(args.gpu_ids)
  os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus or []))
  if count <= 1:
    run_train(task_id, args, log_dir)
    return log_dir
  os.environ.setdefault(
    "TORCHRUNX_LOG_DIR", args.torchrunx_log_dir or str(log_dir / "torchrunx")
  )
  logging.basicConfig(level=logging.INFO)
  torchrunx.Launcher(
    hostnames=["localhost"],
    workers_per_host=count,
    backend=None,
    copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
  ).run(run_train, task_id, args, log_dir)
  return log_dir


def _write_csv(path: Path, rows: list[dict]) -> None:
  with path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def _evaluate(
  task_id: str, args: TrainConfig, log_dir: Path
) -> tuple[list[dict], list[dict]]:
  import torch

  from vbrl.evaluation.checkpoints import evaluate_checkpoints, saved_checkpoints

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  saved = saved_checkpoints(log_dir)
  curve = evaluate_checkpoints(task_id, saved, device=device)
  final = [
    {"scene": scene, **row}
    for scene in getattr(args.agent, "evaluation_scenes", ())
    for row in evaluate_checkpoints(task_id, saved[-1:], device=device, scene=scene)
  ]
  _write_csv(log_dir / "evaluation.csv", curve)
  if final:
    _write_csv(log_dir / "final_evaluation.csv", final)
  for _, path in saved[:-1]:
    path.unlink()
  return curve, final


def _finish(task_id: str, args: TrainConfig, log_dir: Path) -> None:
  import wandb

  from vbrl.evaluation.checkpoints import saved_checkpoints

  curve, final = _evaluate(task_id, args, log_dir) if args.evaluate else ([], [])
  if args.agent.logger != "wandb":
    return
  run = wandb.init(
    project=args.agent.wandb_project,
    entity=os.environ.get("WANDB_USERNAME"),
    id=os.environ["WANDB_RUN_ID"],
    resume="must",
  )
  run.define_metric("eval/iteration")
  run.define_metric("eval/*", step_metric="eval/iteration")
  for row in curve:
    run.log({f"eval/{key}": value for key, value in row.items()})
  for row in final:
    scene = row.pop("scene")
    run.summary.update({f"final/{scene}/{key}": value for key, value in row.items()})
  if args.agent.upload_model:
    _, checkpoint = saved_checkpoints(log_dir)[-1]
    for path in (checkpoint, log_dir / f"{log_dir.name}.onnx"):
      if path.is_file():
        run.save(str(path), base_path=str(log_dir))
  run.finish()


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
  import wandb

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
  os.environ.setdefault("WANDB_RUN_ID", wandb.util.generate_id())
  upload = args.agent.upload_model
  args.agent.upload_model = False
  log_dir = _launch(chosen_task, args)
  args.agent.upload_model = upload
  _finish(chosen_task, args, log_dir)


if __name__ == "__main__":
  main()


__all__ = ["WORKER_ENV", "TrainConfig", "apply_overrides", "main"]
