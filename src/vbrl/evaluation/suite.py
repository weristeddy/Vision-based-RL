"""Load and execute a YAML-defined model × scene × seed evaluation suite."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any, Literal, cast

from vbrl.paths import artifact_path, repository_path
from vbrl.runtime import CheckpointRef, read_manifest, required_text

from .report import write_report
from .rollout import run_episodes


EvaluationTexture = Literal["peacock", "plaster", "wood"]
EvaluationDr = Literal["fixed", "matched"]


@dataclass(frozen=True)
class Scene:
  """A complete evaluation scene preset, including its DR mode."""

  name: str
  base_scene: EvaluationTexture
  eval_dr: EvaluationDr


@dataclass(frozen=True)
class EvaluationModel:
  """One named checkpoint paired with its registered structural task."""

  name: str
  task_id: str
  ref: CheckpointRef


@dataclass(frozen=True)
class EvaluationConfig:
  name: str
  models: tuple[EvaluationModel, ...]
  scenes: tuple[Scene, ...]
  episodes: int
  seeds: tuple[int, ...]
  output: Path


def _names(value: Any, field: str) -> tuple[str, ...]:
  if not isinstance(value, list) or not value:
    raise ValueError(f"{field} must be a non-empty YAML list.")
  names = tuple(str(item).strip() for item in value)
  if any(not name for name in names) or len(names) != len(set(names)):
    raise ValueError(f"{field} must contain unique, non-empty names.")
  return names


def _seeds(value: Any) -> tuple[int, ...]:
  if not isinstance(value, list) or not value:
    raise ValueError("seeds must be a non-empty YAML list.")
  if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in value):
    raise ValueError("seeds must contain integers.")
  seeds = tuple(value)
  if len(seeds) != len(set(seeds)):
    raise ValueError("seeds must not contain duplicates.")
  return seeds


def _scene(name: str) -> Scene:
  from vbrl.scenes.presets import get_preset

  base, eval_dr = name, "fixed"
  if name.endswith("_matched"):
    base, eval_dr = name.removesuffix("_matched"), "matched"
  elif name.endswith("_fixed"):
    base = name.removesuffix("_fixed")
  try:
    get_preset(base, eval_dr=eval_dr, require_ood=True)
  except ValueError as exc:
    raise ValueError(f"Unknown evaluation scene {name!r}: {exc}") from exc
  return Scene(
    name=name,
    base_scene=cast(EvaluationTexture, base),
    eval_dr=cast(EvaluationDr, eval_dr),
  )


def _model(value: Any, index: int) -> EvaluationModel:
  field = f"models[{index}]"
  if not isinstance(value, Mapping):
    raise ValueError(f"{field} must be a YAML mapping.")
  allowed = {
    "name",
    "task_id",
    "checkpoint_file",
    "wandb_run_path",
    "wandb_checkpoint_name",
  }
  unknown = sorted(set(value) - allowed)
  missing = sorted({"name", "task_id"} - set(value))
  if missing or unknown:
    raise ValueError(f"{field} has missing={missing}, unknown={unknown}.")
  ref = CheckpointRef.from_mapping(value)
  ref.validate(prefix=f"{field} ")
  return EvaluationModel(
    name=required_text(value["name"], f"{field}.name"),
    task_id=required_text(value["task_id"], f"{field}.task_id"),
    ref=ref,
  )


def load_config(path: str | Path) -> EvaluationConfig:
  """Read the complete evaluation definition from one YAML file."""

  raw = read_manifest(
    repository_path(path),
    allowed={"version", "name", "models", "scenes", "episodes", "seeds", "output"},
    label="evaluation",
  )

  name = required_text(raw.get("name"), "name")
  output = required_text(raw.get("output"), "output")
  episodes = raw.get("episodes")
  if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
    raise ValueError("episodes must be a positive integer.")
  values = raw.get("models")
  if not isinstance(values, list) or not values:
    raise ValueError("models must be a non-empty YAML list.")
  models = tuple(_model(value, index) for index, value in enumerate(values))
  names = tuple(model.name for model in models)
  if len(names) != len(set(names)):
    raise ValueError("models must have unique names.")

  return EvaluationConfig(
    name=name,
    models=models,
    scenes=tuple(_scene(name) for name in _names(raw.get("scenes"), "scenes")),
    episodes=episodes,
    seeds=_seeds(raw.get("seeds")),
    output=artifact_path(output),
  )


@lru_cache(maxsize=None)
def _architecture(task_id: str) -> str:
  """Human-readable encoder+adapter label. Cached: it depends only on the ID,
  and the evaluation loop asks once per model x scene x seed."""
  from mjlab.rl import RslRlOnPolicyRunnerCfg
  from mjlab.tasks.registry import load_rl_cfg

  import vbrl.tasks  # noqa: F401

  runner_cfg = cast(RslRlOnPolicyRunnerCfg, load_rl_cfg(task_id))
  actor = runner_cfg.actor
  cnn = actor.cnn_cfg
  if not isinstance(cnn, Mapping):
    return "state"
  vision = cnn.get("vision", cnn)
  if not isinstance(vision, Mapping):
    return "vision"
  encoder = str(vision.get("encoder", "vision"))
  adapter = str(vision.get("adapter", "adapter"))
  return f"{encoder} + {adapter}"


def _run_case(
  config: EvaluationConfig,
  model: EvaluationModel,
  scene: Scene,
  seed: int,
  device: str,
) -> list[dict[str, Any]]:
  from vbrl.runtime import build_env, load_trained_policy

  env = build_env(
    model.task_id,
    device=device,
    num_envs=config.episodes,
    seed=seed,
    scene=scene.base_scene,
    eval_dr=scene.eval_dr,
    auto_reset=False,
  )
  try:
    wrapped, _, policy, checkpoint = load_trained_policy(
      env, task_id=model.task_id, device=device, ref=model.ref
    )
    episodes = run_episodes(wrapped, policy, seed=seed)
  finally:
    env.close()

  metadata = {
    "name": model.name,
    "task_id": model.task_id,
    **model.ref.as_metadata(),
    "checkpoint_path": str(checkpoint),
    "architecture": _architecture(model.task_id),
    "scene": scene.name,
  }
  return [{**metadata, **episode} for episode in episodes]


def run_suite(config: EvaluationConfig, device: str) -> Path:
  """Run every configured model × scene × seed combination."""

  rows = []
  total = len(config.models) * len(config.scenes) * len(config.seeds)
  cases = product(config.models, config.scenes, config.seeds)
  for index, (model, scene, seed) in enumerate(cases, start=1):
    print(
      f"[{index}/{total}] {model.name} | {scene.name} | seed {seed}",
      flush=True,
    )
    rows.extend(_run_case(config, model, scene, seed, device))
  return write_report(rows, output=config.output, title=config.name)
