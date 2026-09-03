"""``vbrl-export-onnx`` -- turn a trained checkpoint into a deployment artifact.

```bash
vbrl-export-onnx Mjlab-LiftCube-Sim2Real-DinoV2ViTS14-SpatialSoftmax-TrossenRealistic \
  --wandb-run-path eduard-nicolae-robot-learning/mjlab/ntl27zt9 \
  --wandb-checkpoint-name model_5998.pt \
  --output ckpts/lift_cube/dinov2_vits14_spatial_softmax_sim2real_5998.onnx
```

Training already exports one on every save, so this exists for checkpoints that
predate that working, or when you want a named artifact for a specific
iteration. It runs wherever the simulator can be built -- deployment itself
needs neither torch nor mjlab, because the metadata attached here carries the
observation and action contract with the graph.

Weights come from exactly one of ``--checkpoint-file`` or ``--wandb-run-path``,
the same rule every other consumer follows, so an export can name a W&B run
directly instead of needing the ``.pt`` downloaded by hand first.

The output goes under ``ckpts/`` rather than ``artifacts/``: an exported graph
is the deployable form of a checkpoint, so it belongs beside the ``.pt`` it came
from and under the same ``provenance.json``, not among the analysis outputs.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
  from pathlib import Path

  from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata

  from vbrl.paths import checkpoint_path
  from vbrl.runtime import CheckpointRef, build_env, load_trained_policy

  parser = argparse.ArgumentParser(prog="vbrl-export-onnx", description=__doc__)
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
    env,
    task_id=arguments.task_id,
    device=arguments.device,
    ref=ref,
  )
  runner.export_policy_to_onnx(str(destination.parent), destination.name)
  attach_metadata_to_onnx(
    str(destination),
    get_base_metadata(env.unwrapped, f"{arguments.task_id}:{Path(path).name}"),
  )
  print(f"Wrote {destination} ({destination.stat().st_size / 1e6:.0f} MB)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
