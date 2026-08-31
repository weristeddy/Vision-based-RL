"""Native RSL-RL configurations for realistic Trossen Push-T."""

from __future__ import annotations

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from vbrl.tasks.utils import wandb_task_tag
from vbrl.training.ppo import VisualPpoCfg
from vbrl.vision.config import VisionConfig


STATE_TASK_ID = "Mjlab-PushT-State-TrossenRealistic"


def trossen_realistic_push_t_state_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    seed=0,
    num_steps_per_env=16,
    max_iterations=500,
    obs_groups={
      "actor": ("actor",),
      "critic": ("critic",),
    },
    save_interval=50,
    experiment_name="push_t_state_trossen_realistic_sim2real_dr",
    run_name=wandb_task_tag(STATE_TASK_ID),
    logger="wandb",
    wandb_project="mjlab",
    wandb_tags=(
      wandb_task_tag(STATE_TASK_ID),
      "push_t",
      "state",
      "success_98",
      "sim2real_dr",
    ),
    clip_actions=1.0,
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
      learning_rate=0.0003,
      schedule="fixed",
      gamma=0.99,
      lam=0.9,
      entropy_coef=0.0,
      desired_kl=0.01,
      max_grad_norm=0.5,
      value_loss_coef=0.5,
      use_clipped_value_loss=False,
      clip_param=0.2,
      normalize_advantage_per_mini_batch=False,
      optimizer="adam",
      share_cnn_encoders=False,
    ),
  )


def trossen_realistic_push_t_rgb_ppo_runner_cfg(
  task_id: str,
  vision: VisionConfig,
  *,
  scene: str = "real_texture",
  camera: str = "external_cam",
  success_tag: str = "success_98",
  extra_tags: tuple[str, ...] = (),
  actor_class: str = "vbrl.vision.model:VisionModel",
) -> RslRlOnPolicyRunnerCfg:
  """One RGB Push-T policy. Hyperparameters are shared by every architecture."""
  vision_data = vision.asdict()
  return RslRlOnPolicyRunnerCfg(
    seed=0,
    num_steps_per_env=16,
    max_iterations=3000,
    obs_groups={
      "actor": ("actor", "camera"),
      "critic": ("critic",),
    },
    save_interval=3001,
    experiment_name="push_t_rgb_trossen_realistic_d435",
    run_name=wandb_task_tag(task_id),
    logger="wandb",
    wandb_project="mjlab",
    wandb_tags=(
      wandb_task_tag(task_id),
      "push_t",
      "rgb",
      vision.encoder,
      vision.adapter,
      success_tag,
      "sim2real_dr",
      scene,
      camera,
      *extra_tags,
    ),
    clip_actions=1.0,
    upload_model=True,
    actor=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="relu",
      obs_normalization=True,
      cnn_cfg={
        "vision": vision_data,
        "latent_batchnorm": False,
      },
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.6065306597,
        "std_type": "log",
        "std_range": (0.15, 1.0),
      },
      class_name=actor_class,
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 256, 128),
      activation="relu",
      obs_normalization=True,
      class_name="MLPModel",
    ),
    algorithm=VisualPpoCfg(
      num_learning_epochs=8,
      num_mini_batches=16,
      learning_rate=0.0002,
      schedule="fixed",
      gamma=0.99,
      lam=0.9,
      entropy_coef=0.001,
      desired_kl=0.05,
      max_grad_norm=0.5,
      value_loss_coef=0.5,
      use_clipped_value_loss=False,
      clip_param=0.2,
      normalize_advantage_per_mini_batch=True,
      optimizer="adam",
      share_cnn_encoders=False,
      # Only a frozen backbone may cache its features; the scratch encoders
      # train theirs and must recompute with gradients every update.
      cache_frozen_features=vision.frozen,
      feature_cache_dtype="bfloat16",
      gradient_accumulation_steps=8,
      early_stop_kl=True,
    ),
  )


__all__ = [
  "STATE_TASK_ID",
  "trossen_realistic_push_t_rgb_ppo_runner_cfg",
  "trossen_realistic_push_t_state_ppo_runner_cfg",
]
