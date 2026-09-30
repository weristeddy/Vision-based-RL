from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg

from vbrl.tasks.utils import wandb_task_tag
from vbrl.training.ppo import VisualPpoCfg
from vbrl.vision.config import VisionConfig

STATE_TASK_ID = "Mjlab-PushT-State-TrossenIdentified"
BETA_ENTROPY_COEF = 0.02
_RGB_MAX_ITERATIONS = 6000
_NETWORK = {
  "hidden_dims": (256, 256, 128),
  "activation": "relu",
  "obs_normalization": True,
}
_BETA = {"class_name": "BetaDistribution", "action_range": (-1.0, 1.0)}


def _algorithm(**vision_batching) -> VisualPpoCfg:
  return VisualPpoCfg(
    num_learning_epochs=8,
    num_mini_batches=16,
    learning_rate=0.0002,
    schedule="fixed",
    # Horizon 1/(1-gamma) = 667 steps against the 800-step episode. lam stays
    # at 0.9: at 0.95 the GAE window (1/(1-gamma*lam)) runs past
    # num_steps_per_env, so every advantage leans on the value bootstrap
    # instead of observed reward.
    gamma=0.9985,
    lam=0.9,
    entropy_coef=BETA_ENTROPY_COEF,
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
    early_stop_kl=True,
    **vision_batching,
  )


def trossen_realistic_push_t_state_ppo_runner_cfg(
  task_id: str = STATE_TASK_ID,
) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    seed=0,
    num_steps_per_env=16,
    max_iterations=1500,
    obs_groups={
      "actor": ("actor",),
      "critic": ("critic",),
    },
    save_interval=50,
    experiment_name="push_t_state_trossen_realistic_sim2real_dr",
    run_name=wandb_task_tag(task_id),
    logger="wandb",
    wandb_project="mjlab",
    wandb_tags=(
      wandb_task_tag(task_id),
      "push_t",
      "state",
      "success_90",
      "sim2real_dr",
    ),
    clip_actions=None,
    upload_model=True,
    actor=RslRlModelCfg(**_NETWORK, distribution_cfg=_BETA, class_name="MLPModel"),
    critic=RslRlModelCfg(**_NETWORK, class_name="MLPModel"),
    algorithm=_algorithm(),
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
    clip_actions=None,
    upload_model=True,
    actor=RslRlModelCfg(
      **_NETWORK,
      cnn_cfg={
        "vision": vision_data,
        "latent_batchnorm": False,
      },
      distribution_cfg=_BETA,
      class_name="vbrl.vision.model:VisionModel",
    ),
    critic=RslRlModelCfg(**_NETWORK, class_name="MLPModel"),
    algorithm=_algorithm(
      cache_frozen_features=vision.frozen,
      feature_cache_dtype="bfloat16",
      gradient_accumulation_steps=8,
    ),
  )


__all__ = [
  "STATE_TASK_ID",
  "trossen_realistic_push_t_rgb_ppo_runner_cfg",
  "trossen_realistic_push_t_state_ppo_runner_cfg",
]
