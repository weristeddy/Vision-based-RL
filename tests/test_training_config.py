"""Registered agent configs and the native W&B sweeps that drive them."""

from __future__ import annotations

from pathlib import Path

import pytest


yaml = pytest.importorskip("yaml")
pytest.importorskip("mjlab")


SWEEPS = sorted((Path(__file__).resolve().parents[1] / "configs" / "sweeps").glob("*.yaml"))
# Sweeps vary run-specific parameters only. Anything structural -- encoder,
# adapter, camera, robot, scene -- selects a different registered task ID.
NATIVE_SWEEP_KEYS = {
  "agent.seed",
  "agent.max-iterations",
  "agent.save-interval",
  "agent.algorithm.learning-rate",
  "agent.algorithm.schedule",
  "agent.algorithm.num-learning-epochs",
  "agent.algorithm.num-mini-batches",
  "agent.algorithm.entropy-coef",
  "agent.algorithm.desired-kl",
  "agent.algorithm.max-grad-norm",
  "agent.algorithm.gradient-accumulation-steps",
  "env.scene.num-envs",
  "video-interval",
  "video-length",
}


def _sweeps() -> dict[str, dict]:
  return {p.name: yaml.safe_load(p.read_text(encoding="utf-8")) for p in SWEEPS}


def _task_ids() -> set[str]:
  from vbrl.tasks import vbrl_task_ids

  return set(vbrl_task_ids())


def test_every_task_logs_to_wandb_under_its_own_id_tag() -> None:
  """The first tag identifies the task, within W&B's 64-character limit."""
  from mjlab.tasks.registry import load_rl_cfg

  from vbrl.tasks.utils import wandb_task_tag

  for task_id in _task_ids():
    agent = load_rl_cfg(task_id)
    tag = wandb_task_tag(task_id)
    assert agent.wandb_tags[0] == tag
    assert agent.wandb_tags.count(tag) == 1
    assert len(tag) <= 64
    assert agent.logger == "wandb"
    assert agent.wandb_project == "mjlab"
    assert agent.upload_model is True


def test_every_run_is_named_after_its_task() -> None:
  """RSL-RL names the W&B run after the log directory, which ends in run_name.

  Without this the runs show up as bare timestamps, and several architectures
  share one experiment_name, so nothing in the W&B list tells them apart.
  """
  from mjlab.tasks.registry import load_rl_cfg

  from vbrl.tasks.utils import wandb_task_tag

  for task_id in _task_ids():
    assert load_rl_cfg(task_id).run_name == wandb_task_tag(task_id), task_id


def test_wandb_tags_never_exceed_the_limit_wandb_enforces() -> None:
  """W&B rejects a run whose tag is over 64 characters, mid-training."""
  from mjlab.tasks.registry import load_rl_cfg

  from vbrl.tasks.utils.tags import WANDB_TAG_MAX_LENGTH, wandb_task_tag

  for task_id in _task_ids():
    for tag in load_rl_cfg(task_id).wandb_tags:
      assert 1 <= len(tag) <= WANDB_TAG_MAX_LENGTH, (task_id, tag)

  with pytest.raises(ValueError, match="the limit is 64"):
    wandb_task_tag("Mjlab-" + "X" * 65)


def test_push_t_rgb_preserves_the_maniskill_style_training_contract() -> None:
  from mjlab.tasks.registry import load_rl_cfg

  from vbrl.training.ppo import VisualPpoCfg

  agent = load_rl_cfg(
    "Mjlab-PushT-RealTexture-DinoV2ViTS14-LocalGrid7-TrossenRealistic"
  )

  assert agent.actor.hidden_dims == (256, 256, 128)
  assert agent.actor.activation == "relu"
  assert agent.actor.distribution_cfg == {
    "class_name": "GaussianDistribution",
    "init_std": pytest.approx(0.6065306597),
    "std_type": "log",
    "std_range": pytest.approx((0.15, 1.0)),
  }
  assert isinstance(agent.algorithm, VisualPpoCfg)
  assert agent.algorithm.cache_frozen_features is True
  assert agent.algorithm.feature_cache_dtype == "bfloat16"
  assert agent.algorithm.gradient_accumulation_steps == 8
  assert agent.algorithm.early_stop_kl is True
  assert agent.clip_actions == pytest.approx(1.0)


def test_state_tasks_keep_native_ppo() -> None:
  from mjlab.rl import RslRlPpoAlgorithmCfg
  from mjlab.tasks.registry import load_rl_cfg

  for task_id in ("Mjlab-PushCube-State-Trossen", "Mjlab-PushT-State-TrossenRealistic"):
    agent = load_rl_cfg(task_id)
    assert type(agent.algorithm) is RslRlPpoAlgorithmCfg
    assert agent.actor.hidden_dims == (512, 256, 128)
    assert agent.actor.cnn_cfg is None


def test_every_sweep_fixes_one_registered_task_and_varies_native_keys_only() -> None:
  task_ids = _task_ids()

  for name, sweep in _sweeps().items():
    command = sweep["command"]
    assert sweep["program"] == "vbrl.scripts.train"
    assert command[:4] == ["${env}", "python", "-m", "${program}"]
    assert command[-1] == "${args_no_boolean_flags}"
    assert len([t for t in command if str(t) in task_ids]) == 1, name
    assert set(sweep.get("parameters", {})) <= NATIVE_SWEEP_KEYS, name

    # W&B renders booleans unreliably, so they stay literal command entries.
    for index, value in enumerate(command[:-1]):
      if value in {"--video", "--headless"}:
        assert command[index + 1] in {"True", "False"}, name
    for spec in sweep.get("parameters", {}).values():
      assert not isinstance(spec.get("value"), bool)
      assert not any(isinstance(v, bool) for v in spec.get("values", []))


def test_every_sweep_flag_path_parses_against_the_train_cli() -> None:
  """Each sweep must resolve against the CLI it actually launches."""
  import mjlab
  import tyro

  from vbrl.scripts.train import TrainConfig

  task_ids = _task_ids()
  for name, sweep in _sweeps().items():
    command = sweep["command"]
    task_id = next(str(t) for t in command if str(t) in task_ids)
    args = [str(v) for v in command[command.index(task_id) + 1 : -1]]
    for key, spec in sweep.get("parameters", {}).items():
      if "value" in spec:
        value = spec["value"]
      elif "values" in spec:
        value = spec["values"][0]
      else:
        value = spec["min"]
      args.extend((f"--{key}", str(value)))

    parsed = tyro.cli(
      TrainConfig,
      args=args,
      default=TrainConfig.from_task(task_id),
      config=mjlab.TYRO_FLAGS,
    )
    assert parsed.env.scene.num_envs == 1024, name
