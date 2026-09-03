"""Native RSL-RL configuration for Trossen Lift-Cube policies."""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from vbrl.tasks.utils import wandb_task_tag
from vbrl.training.ppo import VisualPpoCfg
from vbrl.vision.config import VisionConfig


def trossen_lift_cube_ppo_runner_cfg(
  task_id: str,
  vision: VisionConfig,
) -> RslRlOnPolicyRunnerCfg:
  vision_data = vision.asdict()
  actor = RslRlModelCfg(
    hidden_dims=(256, 256, 128),
    activation="elu",
    obs_normalization=True,
    cnn_cfg={
      "vision": vision_data,
      "latent_batchnorm": False,
    },
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
    class_name="vbrl.vision.model:VisionModel",
  )
  critic = RslRlModelCfg(
    hidden_dims=(256, 256, 128),
    activation="elu",
    obs_normalization=True,
    class_name="MLPModel",
  )
  algorithm = VisualPpoCfg(
    num_learning_epochs=4,
    num_mini_batches=8,
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
    cache_frozen_features=vision.frozen,
    feature_cache_dtype="bfloat16",
    gradient_accumulation_steps=8,
    early_stop_kl=False,
  )
  return RslRlOnPolicyRunnerCfg(
    seed=0,
    num_steps_per_env=24,
    # 6000, not 3000. Measured on the Sim2Real contract, whose corrected D405
    # optics project the cube 1.72x larger and therefore make the task a
    # different one: 3000 iterations reached 0.361 at_goal with 29% of episodes
    # ending in a ground collision, and 6000 reached 0.800 with none. It is a
    # run-specific flag the task ID deliberately does not encode, so this moves
    # the default without changing any contract -- all twelve retained
    # real-texture Lift-Cube runs were trained at 3000 and stay loadable.
    #
    # RSL-RL counts this from wherever it starts (`total_it = start_it +
    # num_learning_iterations`), so it is iterations *this job* runs, not a
    # total: the 6000 measured above was two 3000-iteration jobs, the second
    # resuming the first, and a resume under this default adds another 6000.
    max_iterations=6000,
    obs_groups={
      "actor": ("actor", "camera"),
      "critic": ("critic",),
    },
    save_interval=50,
    experiment_name="trossen_lift_cube_rgb",
    run_name=wandb_task_tag(task_id),
    logger="wandb",
    wandb_project="mjlab",
    wandb_tags=(
      wandb_task_tag(task_id),
      "lift_cube",
      vision.encoder,
      vision.adapter,
    ),
    upload_model=True,
    actor=actor,
    critic=critic,
    algorithm=algorithm,
  )


__all__ = ["trossen_lift_cube_ppo_runner_cfg"]
