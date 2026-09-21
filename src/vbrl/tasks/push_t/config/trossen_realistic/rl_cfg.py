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
_RGB_MAX_ITERATIONS = 6000


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
      "success_90",
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
      gamma=0.9975,
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
) -> RslRlOnPolicyRunnerCfg:
  vision_data = vision.asdict()
  return RslRlOnPolicyRunnerCfg(
    seed=0,
    num_steps_per_env=16,
    max_iterations=_RGB_MAX_ITERATIONS,
    obs_groups={
      "actor": ("actor", "camera"),
      "critic": ("critic",),
    },
    # Above max_iterations, so the only checkpoint is the unconditional final
    # save RSL-RL does after the loop. These runs upload to W&B, where the run
    # files are what fills the 200 GB quota.
    save_interval=_RGB_MAX_ITERATIONS + 1,
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
      "success_90",
      "sim2real_dr",
      "real_texture",
      "external_cam",
      "goal_yaw_curriculum",
      "visual_goal",
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
        "std_range": (0.05, 1.0),
      },
      class_name="vbrl.vision.model:VisionModel",
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
      # Horizon 1/(1-gamma) = 400 steps, the 8 s episode. lam stays at 0.9: at
      # 0.95 the GAE window (1/(1-gamma*lam)) runs past num_steps_per_env, so
      # every advantage leans on the value bootstrap instead of observed reward.
      gamma=0.9975,
      lam=0.9,
      entropy_coef=0.001,
      # ManiSkill3's value. At 0.05 the early stop fired on 97.9% of iterations
      # and threw away 71% of the update budget (36.8 of 128 performed).
      desired_kl=0.1,
      max_grad_norm=0.5,
      value_loss_coef=0.5,
      use_clipped_value_loss=False,
      clip_param=0.2,
      normalize_advantage_per_mini_batch=True,
      optimizer="adam",
      share_cnn_encoders=False,
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
