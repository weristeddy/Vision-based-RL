from __future__ import annotations

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
  from pathlib import Path

  from mjlab.rl.exporter_utils import attach_metadata_to_onnx
  from mjlab.tasks.registry import load_rl_cfg

  from vbrl.paths import checkpoint_path
  from vbrl.runtime import CheckpointRef, build_env, load_trained_policy
  from vbrl.training.runner import policy_metadata

  parser = argparse.ArgumentParser(
    prog="vbrl-export-onnx",
    description="Re-export a checkpoint that predates automatic export on save.",
  )
  parser.add_argument("task_id")
  parser.add_argument("--checkpoint-file")
  parser.add_argument("--wandb-run-path", help="entity/project/run_id")
  parser.add_argument("--wandb-checkpoint-name", help="model_N.pt")
  parser.add_argument("--output", required=True, help="must be below ckpts/")
  parser.add_argument("--device", default="cuda:0")
  arguments = parser.parse_args(argv)

  ref = CheckpointRef.from_args(arguments)
  ref.validate()
  destination = checkpoint_path(arguments.output)
  destination.parent.mkdir(parents=True, exist_ok=True)

  env = build_env(arguments.task_id, device=arguments.device, num_envs=1, seed=0)
  _, runner, _, path = load_trained_policy(
    env, task_id=arguments.task_id, device=arguments.device, ref=ref
  )
  runner.export_policy_to_onnx(str(destination.parent), destination.name)
  metadata = policy_metadata(
    env.unwrapped,
    f"{arguments.task_id}:{Path(path).name}",
    load_rl_cfg(arguments.task_id).clip_actions,
  )
  attach_metadata_to_onnx(str(destination), metadata)
  print(f"Wrote {destination} ({destination.stat().st_size / 1e6:.0f} MB)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
