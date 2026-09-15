"""Native RSL-RL configurations for realistic Trossen Stack-Cubes."""

from __future__ import annotations

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from vbrl.tasks.utils import wandb_task_tag
from vbrl.training.ppo import VisualPpoCfg
from vbrl.vision.config import VisionConfig


STATE_TASK_ID = "Mjlab-StackCubes-State-TrossenRealistic"

# Lift-Cube's rollout length, which is this repository's for a pick-and-place
# task, and ManiSkill3's own Stack-Cube PPO recipe for the update shape (8
# epochs over 32 minibatches at 4,096 environments). Nothing else moves from
# the existing VBRL defaults.
NUM_STEPS_PER_ENV = 24


def trossen_realistic_stack_cubes_state_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    seed=0,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=6000,
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    save_interval=250,
    experiment_name="stack_cubes_state_trossen_realistic",
    run_name=wandb_task_tag(STATE_TASK_ID),
    logger="wandb",
    wandb_project="mjlab",
    wandb_tags=(wandb_task_tag(STATE_TASK_ID), "stack_cubes", "state"),
    upload_model=True,
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
      class_name="MLPModel",
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      class_name="MLPModel",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      num_learning_epochs=8,
      num_mini_batches=32,
      learning_rate=0.00035,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      entropy_coef=0.005,
      desired_kl=0.01,
      max_grad_norm=1.0,
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      normalize_advantage_per_mini_batch=False,
      optimizer="adam",
      share_cnn_encoders=False,
    ),
  )


def trossen_realistic_stack_cubes_rgb_ppo_runner_cfg(
  task_id: str,
  vision: VisionConfig,
  *,
  camera: str,
) -> RslRlOnPolicyRunnerCfg:
  """One RGB Stack-Cubes policy. The camera is the only thing that varies."""
  return RslRlOnPolicyRunnerCfg(
    seed=0,
    num_steps_per_env=NUM_STEPS_PER_ENV,
    max_iterations=6000,
    obs_groups={"actor": ("actor", "camera"), "critic": ("critic",)},
    save_interval=250,
    experiment_name="stack_cubes_rgb_trossen_realistic",
    run_name=wandb_task_tag(task_id),
    logger="wandb",
    wandb_project="mjlab",
    wandb_tags=(
      wandb_task_tag(task_id),
      "stack_cubes",
      "rgb",
      vision.encoder,
      vision.adapter,
      camera,
      "sim2real_dr",
    ),
    upload_model=True,
    actor=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      cnn_cfg={"vision": vision.asdict(), "latent_batchnorm": False},
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
      class_name="vbrl.vision.model:VisionModel",
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="elu",
      obs_normalization=True,
      class_name="MLPModel",
    ),
    algorithm=VisualPpoCfg(
      num_learning_epochs=8,
      num_mini_batches=32,
      learning_rate=0.00035,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      entropy_coef=0.005,
      desired_kl=0.01,
      max_grad_norm=1.0,
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      normalize_advantage_per_mini_batch=False,
      optimizer="adam",
      share_cnn_encoders=False,
      # NatureCnn trains its own trunk, so there is nothing to cache.
      cache_frozen_features=vision.frozen,
      feature_cache_dtype="bfloat16",
      gradient_accumulation_steps=8,
      early_stop_kl=False,
    ),
  )


__all__ = [
  "NUM_STEPS_PER_ENV",
  "STATE_TASK_ID",
  "trossen_realistic_stack_cubes_rgb_ppo_runner_cfg",
  "trossen_realistic_stack_cubes_state_ppo_runner_cfg",
]
