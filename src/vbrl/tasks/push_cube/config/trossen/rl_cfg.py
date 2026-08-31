"""Native RSL-RL configuration for state Push-Cube."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from vbrl.tasks.utils import wandb_task_tag


TASK_ID = "Mjlab-PushCube-State-Trossen"


def push_cube_rl_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    seed=0,
    num_steps_per_env=24,
    max_iterations=3000,
    obs_groups={
      "actor": ("actor",),
      "critic": ("critic",),
    },
    save_interval=50,
    experiment_name="trossen_push_cube_state_default",
    run_name=wandb_task_tag(TASK_ID),
    logger="wandb",
    wandb_project="mjlab",
    wandb_tags=(wandb_task_tag(TASK_ID), "push_cube", "state"),
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
      num_learning_epochs=5,
      num_mini_batches=8,
      learning_rate=0.0008938003178818718,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      entropy_coef=0.01,
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


__all__ = ["TASK_ID", "push_cube_rl_cfg"]
