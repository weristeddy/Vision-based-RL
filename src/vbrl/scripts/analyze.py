"""Run the scripts listed in an analysis YAML file."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vbrl.analysis import (
  attribution,
  capture,
  comparison,
  features,
  occlusion,
  pca,
  probe,
  report,
)
from vbrl.paths import analysis_manifest_path, artifact_path
from vbrl.runtime import (
  AGENTS,
  CheckpointRef,
  default_device,
  read_manifest,
  required_text,
)


STEPS: dict[str, Callable[..., Any]] = {
  "capture": capture.run,
  "features": features.run,
  "probe": probe.run,
  "pca": pca.run,
  "occlusion": occlusion.run,
  "comparison": comparison.run,
  "attribution": attribution.run,
  "report": report.run,
}

# Steps that read the simulator, so a manifest using none of them never builds
# an environment.
_RUNTIME_STEPS = {"capture", "features", "occlusion", "attribution"}


@dataclass
class Context:
  """Configuration and one shared native runtime for an analysis pipeline."""

  path: Path
  task_id: str
  agent: str
  ref: CheckpointRef
  output_dir: Path
  device: str
  scene: str | None = None
  eval_dr: str = "fixed"
  raw_env: Any | None = field(default=None, repr=False)
  env: Any | None = field(default=None, repr=False)
  runner: Any | None = field(default=None, repr=False)
  policy: Any | None = field(default=None, repr=False)
  checkpoint_path: Path | None = field(default=None, repr=False)

  def input(self, path: str | Path) -> Path:
    path = Path(path).expanduser()
    return path.resolve() if path.is_absolute() else (self.output_dir / path).resolve()

  def output(self, path: str | Path) -> Path:
    path = (self.output_dir / Path(path).expanduser()).resolve()
    if not path.is_relative_to(self.output_dir):
      raise ValueError(f"Output must stay below {self.output_dir}: {path}")
    return path

  def provenance(self) -> dict[str, Any]:
    """Where this artifact came from, recorded into every NPZ we write."""
    return {
      "manifest": str(self.path),
      "task_id": self.task_id,
      "scene": self.scene,
      **self.ref.as_metadata(),
      "checkpoint_path": (
        None if self.checkpoint_path is None else str(self.checkpoint_path)
      ),
    }


def load(path: str | Path, device: str) -> tuple[Context, list[dict[str, Any]]]:
  """Read one task/checkpoint reference and its ordered analysis steps."""
  path = analysis_manifest_path(path)
  config = read_manifest(
    path,
    allowed={
      "version",
      "task_id",
      "agent",
      "checkpoint_file",
      "wandb_run_path",
      "wandb_checkpoint_name",
      "output",
      "scene",
      "eval_dr",
      "steps",
    },
    label="analysis",
  )

  agent = required_text(config.get("agent", "trained"), "agent")
  if agent not in AGENTS:
    raise ValueError("agent must be trained, zero, or random.")
  ref = CheckpointRef.from_mapping(config)
  if agent == "trained":
    ref.validate(prefix="agent: trained ")
  elif not ref.is_empty:
    raise ValueError("Checkpoint fields require agent: trained.")

  scene = config.get("scene")
  eval_dr = config.get("eval_dr", "fixed")
  if scene is not None:
    # Validate here so a typo fails before an environment is built.
    from vbrl.scenes.presets import get_preset

    get_preset(required_text(scene, "scene"), eval_dr=eval_dr, require_ood=True)
  elif "eval_dr" in config:
    raise ValueError("eval_dr requires scene.")

  steps = config.get("steps")
  if not isinstance(steps, list) or not steps:
    raise ValueError("steps must be a non-empty list.")
  if not all(isinstance(step, dict) for step in steps):
    raise ValueError("Every analysis step must be a mapping.")
  return (
    Context(
      path=path,
      task_id=required_text(config.get("task_id"), "task_id"),
      agent=agent,
      ref=ref,
      output_dir=artifact_path(required_text(config.get("output"), "output")),
      device=device,
      scene=scene,
      eval_dr=eval_dr,
    ),
    steps,
  )


def _required_num_envs(steps: list[dict[str, Any]]) -> int:
  required = 1
  for step in steps:
    if step.get("script") != "capture":
      continue
    args = step.get("args", {})
    if not isinstance(args, dict):
      raise ValueError("Analysis step args must be a mapping.")
    env_index = int(args.get("env_index", 0))
    num_envs = int(args.get("num_envs", env_index + 1))
    if env_index < 0 or num_envs <= env_index:
      raise ValueError("capture env_index must be inside num_envs.")
    required = max(required, num_envs)
  return required


def _prepare_runtime(context: Context, steps: list[dict[str, Any]]) -> None:
  from vbrl.runtime import build_env, make_policy

  raw_env = build_env(
    context.task_id,
    device=context.device,
    num_envs=_required_num_envs(steps),
    scene=context.scene,
    eval_dr=context.eval_dr,
  )
  context.raw_env = raw_env
  wrapped, runner, policy, checkpoint = make_policy(
    raw_env,
    task_id=context.task_id,
    agent=context.agent,
    ref=context.ref,
    device=context.device,
  )
  context.env = wrapped
  context.runner = runner
  context.policy = policy
  context.checkpoint_path = checkpoint


def execute(context: Context, steps: list[dict[str, Any]]) -> list[Path]:
  """Build at most one runtime and execute the YAML steps in order."""
  generated: list[Path] = []
  try:
    if {step.get("script") for step in steps} & _RUNTIME_STEPS:
      _prepare_runtime(context, steps)

    for step in steps:
      name = step.get("script")
      args = step.get("args", {})
      if name not in STEPS:
        choices = ", ".join(sorted(STEPS))
        raise ValueError(f"Unknown analysis step {name!r}. Choose one of: {choices}.")
      if not isinstance(args, dict):
        raise ValueError(f"Analysis step {name!r} args must be a mapping.")
      result = STEPS[name](context, **args)
      paths = (result,) if isinstance(result, (str, Path)) else result
      generated.extend(context.output(path) for path in paths)
  finally:
    if context.raw_env is not None:
      context.raw_env.close()
  return generated


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
    prog="vbrl-analyze",
    description="Run the scripts listed in an analysis YAML file.",
  )
  parser.add_argument(
    "yaml",
    type=Path,
    help="Repository path or path relative to configs/analysis.",
  )
  parser.add_argument("--device")
  args = parser.parse_args(argv)

  if args.device is None:
    args.device = default_device()
  try:
    context, steps = load(args.yaml, args.device)
    generated = execute(context, steps)
  except (OSError, ImportError, KeyError, RuntimeError, TypeError, ValueError) as exc:
    parser.error(str(exc))
  for path in generated:
    print(path)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
