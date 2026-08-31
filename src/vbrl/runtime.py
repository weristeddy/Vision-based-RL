"""Resolve a checkpoint, build a registered environment, and load an actor.

Shared by evaluation, analysis, and playback so each keeps only its CLI.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CHECKPOINT_NAME = re.compile(r"model_\d+\.pt")
AGENTS = ("trained", "zero", "random")


def default_device() -> str:
  """Prefer the first CUDA device, falling back to CPU."""
  import torch

  return "cuda:0" if torch.cuda.is_available() else "cpu"


def required_text(value: Any, field: str) -> str:
  if not isinstance(value, str) or not value.strip():
    raise ValueError(f"{field} must be a non-empty string.")
  return value.strip()


def read_manifest(
  path: Path,
  *,
  allowed: set[str],
  label: str = "config",
) -> Mapping[str, Any]:
  """Load a versioned YAML manifest, rejecting fields it does not declare."""
  import yaml  # type: ignore[import-untyped]

  if not path.is_file():
    raise FileNotFoundError(path)
  with path.open("r", encoding="utf-8") as stream:
    raw = yaml.safe_load(stream)
  if not isinstance(raw, Mapping):
    raise ValueError(f"{label.capitalize()} config must be a YAML mapping.")
  unknown = sorted(set(raw) - allowed)
  if unknown:
    raise ValueError(f"Unknown {label} fields: {unknown}.")
  if raw.get("version") != 1:
    raise ValueError(f"{label.capitalize()} config version must be 1.")
  return raw


@dataclass(frozen=True)
class CheckpointRef:
  """Where an actor's weights come from: a local file or a W&B run.

  Exactly one source may be given. The task ID -- not this reference -- is
  authoritative for architecture, so nothing here describes the model.
  """

  checkpoint_file: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None

  @classmethod
  def from_mapping(cls, source: Mapping[str, Any]) -> CheckpointRef:
    return cls(
      checkpoint_file=source.get("checkpoint_file"),
      wandb_run_path=source.get("wandb_run_path"),
      wandb_checkpoint_name=source.get("wandb_checkpoint_name"),
    )

  @classmethod
  def from_args(cls, args: Any) -> CheckpointRef:
    checkpoint = getattr(args, "checkpoint_file", None)
    return cls(
      checkpoint_file=None if checkpoint is None else str(checkpoint),
      wandb_run_path=getattr(args, "wandb_run_path", None),
      wandb_checkpoint_name=getattr(args, "wandb_checkpoint_name", None),
    )

  @property
  def is_empty(self) -> bool:
    return not any(asdict(self).values())

  def validate(self, *, prefix: str = "") -> None:
    """Enforce the local-file XOR W&B-run rule and each field's shape."""
    # Order matters: a lone wandb_checkpoint_name is a more specific mistake
    # than "neither source given", and reporting it first is more useful.
    if self.wandb_checkpoint_name is not None and self.wandb_run_path is None:
      raise ValueError(f"{prefix}wandb_checkpoint_name requires wandb_run_path.")
    if (self.checkpoint_file is None) == (self.wandb_run_path is None):
      raise ValueError(
        f"{prefix}must provide exactly one of checkpoint_file or wandb_run_path."
      )
    if self.checkpoint_file is not None:
      required_text(self.checkpoint_file, f"{prefix}checkpoint_file")
      return
    run_path = required_text(self.wandb_run_path, f"{prefix}wandb_run_path")
    parts = run_path.split("/")
    if len(parts) != 3 or any(not part for part in parts):
      raise ValueError(f"{prefix}wandb_run_path must be 'entity/project/run_id'.")
    if self.wandb_checkpoint_name is not None:
      name = required_text(self.wandb_checkpoint_name, f"{prefix}wandb_checkpoint_name")
      if CHECKPOINT_NAME.fullmatch(name) is None:
        raise ValueError(f"{prefix}wandb_checkpoint_name must be model_N.pt.")

  def as_metadata(self) -> dict[str, str | None]:
    return asdict(self)


class ConstantPolicy:
  """A zero or uniform-random actor, for baselines and smoke tests."""

  def __init__(self, env: Any, *, random_actions: bool) -> None:
    import torch

    self._torch = torch
    self._random_actions = random_actions
    self._shape = tuple(env.unwrapped.action_space.shape)
    self._device = env.unwrapped.device

  def __call__(self, observations: Any):
    del observations
    if self._random_actions:
      return 2.0 * self._torch.rand(self._shape, device=self._device) - 1.0
    return self._torch.zeros(self._shape, device=self._device)


def load_trained_policy(
  env: Any,
  *,
  task_id: str,
  device: str,
  ref: CheckpointRef,
  log_root: str | Path = "logs/rsl_rl",
) -> tuple[Any, Any, Any, Path]:
  """Build the registered runner and strict-load its actor checkpoint."""
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

  import vbrl.tasks  # noqa: F401

  ref.validate()
  agent_cfg = load_rl_cfg(task_id)
  if ref.checkpoint_file is not None:
    checkpoint_path = Path(ref.checkpoint_file).expanduser()
    if not checkpoint_path.is_file():
      raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
  else:
    from mjlab.utils.os import get_wandb_checkpoint_path

    assert ref.wandb_run_path is not None
    checkpoint_path, _ = get_wandb_checkpoint_path(
      (Path(log_root).expanduser() / agent_cfg.experiment_name).resolve(),
      Path(ref.wandb_run_path),
      ref.wandb_checkpoint_name,
    )
    checkpoint_path = Path(checkpoint_path)

  wrapped = (
    env
    if isinstance(env, RslRlVecEnvWrapper)
    else RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  )
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), log_dir=None, device=device)
  runner.load(
    str(checkpoint_path),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  return wrapped, runner, runner.get_inference_policy(device=device), checkpoint_path


def build_env(
  task_id: str,
  *,
  device: str,
  num_envs: int,
  seed: int | None = None,
  scene: str | None = None,
  eval_dr: str = "fixed",
  auto_reset: bool | None = None,
  drop_terminations: bool = False,
  fixed_lighting: bool = False,
) -> Any:
  """Construct one registered play environment, optionally re-dressed."""
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  import vbrl.tasks  # noqa: F401

  cfg = load_env_cfg(task_id, play=True)
  if scene is not None:
    from vbrl.scenes.builder import replace_scene

    replace_scene(cfg, scene=scene, eval_dr=eval_dr)
  if fixed_lighting:
    from vbrl.scenes.builder import hold_lighting_colour_fixed

    hold_lighting_colour_fixed(cfg)
  cfg.scene.num_envs = num_envs
  if seed is not None:
    cfg.seed = seed
  if auto_reset is not None:
    cfg.auto_reset = auto_reset
  if drop_terminations:
    cfg.terminations = {}
  return ManagerBasedRlEnv(cfg=cfg, device=device)


def make_policy(
  env: Any,
  *,
  task_id: str,
  agent: str,
  ref: CheckpointRef,
  device: str,
) -> tuple[Any, Any, Any, Path | None]:
  """Return ``(wrapped_env, runner, policy, checkpoint)`` for any agent kind."""
  if agent not in AGENTS:
    raise ValueError(f"agent must be one of {AGENTS}.")
  if agent == "trained":
    return load_trained_policy(env, task_id=task_id, device=device, ref=ref)

  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg

  import vbrl.tasks  # noqa: F401

  if not ref.is_empty:
    raise ValueError("Checkpoint fields require agent: trained.")
  wrapped = RslRlVecEnvWrapper(env, clip_actions=load_rl_cfg(task_id).clip_actions)
  return wrapped, None, ConstantPolicy(wrapped, random_actions=agent == "random"), None


__all__ = [
  "AGENTS",
  "CHECKPOINT_NAME",
  "CheckpointRef",
  "ConstantPolicy",
  "build_env",
  "default_device",
  "load_trained_policy",
  "make_policy",
  "read_manifest",
  "required_text",
]
