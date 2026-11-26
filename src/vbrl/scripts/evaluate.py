from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="vbrl-evaluate",
    description="Evaluate a run as vbrl-train does after training.",
  )
  parser.add_argument("task")
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument("--run-dir", type=Path, help="every model_N.pt in it")
  source.add_argument("--checkpoint", type=Path)
  source.add_argument("--wandb-run-path", help="entity/project/run_id")
  parser.add_argument("--scenes", nargs="*", help="default: the task's scenes")
  parser.add_argument("--log", action="store_true", help="write into the W&B run")
  parser.add_argument("--device", default="cuda:0")
  return parser


def _checkpoints(task: str, args: argparse.Namespace) -> list[tuple[int, Path]]:
  from mjlab.tasks.registry import load_rl_cfg
  from mjlab.utils.os import get_wandb_checkpoint_path

  from vbrl.evaluation.checkpoints import saved_checkpoints

  if args.run_dir:
    return saved_checkpoints(args.run_dir)
  if args.checkpoint:
    return [(0, args.checkpoint)]
  root = Path("logs/rsl_rl") / load_rl_cfg(task).experiment_name
  path, _ = get_wandb_checkpoint_path(root.resolve(), Path(args.wandb_run_path), None)
  return [(0, Path(path))]


def main(argv: Sequence[str] | None = None) -> None:
  import wandb
  from mjlab.tasks.registry import load_rl_cfg

  import vbrl.tasks  # noqa: F401
  from vbrl.evaluation.checkpoints import evaluate_run, log_evaluation
  from vbrl.paths import artifact_path

  parser = _parser()
  args = parser.parse_args(argv)
  if args.log and not args.wandb_run_path:
    parser.error("--log needs --wandb-run-path")
  saved = _checkpoints(args.task, args)
  if not saved:
    parser.error("no checkpoints found")
  scenes = args.scenes
  if scenes is None:
    scenes = getattr(load_rl_cfg(args.task), "evaluation_scenes", ())
  name = (args.wandb_run_path or str(args.run_dir or args.checkpoint)).rstrip("/")
  out_dir = artifact_path(Path("artifacts/evaluation") / Path(name).name)
  results = evaluate_run(
    args.task, saved, scenes, out_dir, device=args.device, curve=bool(args.run_dir)
  )
  print(f"Evaluation results: {out_dir}")
  if args.log:
    entity, project, run_id = args.wandb_run_path.split("/")
    run = wandb.init(entity=entity, project=project, id=run_id, resume="must")
    log_evaluation(run, *results)
    run.finish()


if __name__ == "__main__":
  main()


__all__ = ["main"]
