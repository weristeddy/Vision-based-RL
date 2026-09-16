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
# epochs over 32 minibatches at 4,096 environments).
NUM_STEPS_PER_ENV = 24

# --- provenance -------------------------------------------------------------
#
# ManiSkill3's own Stack-Cube-v1 baseline is the reference, because it is the
# only published recipe for this exact task. Its command is
#
#   ppo_fast.py --env_id=StackCube-v1 --num_envs=4096 --num-steps=16
#               --update_epochs=8 --num_minibatches=32 --total_timesteps=50M
#
# with everything else at ppo_fast.py's defaults: lr 3e-4, no KL adaptation,
# gamma 0.8, gae_lambda 0.9, ent_coef 0.0, vf_coef 0.5, clip 0.2,
# max_grad_norm 0.5, target_kl 0.1, norm_adv True, clip_vloss False.
#
# Adopted verbatim: learning rate, fixed schedule, 8 epochs, 32 minibatches,
# 4,096 environments, vf_coef, max_grad_norm, clip range, unclipped value loss,
# per-minibatch advantage normalization, and the 0.1 KL early-stop threshold.
#
# Four deliberate departures, each because their episode is 50 steps and this
# one is 2,000:
#
#   gamma 0.99, not 0.8. At 0.8 the horizon is five steps, which is sensible
#     for a 50-step episode and meaningless here -- a single pick-and-place
#     runs 150-300 steps. 0.99 gives 100 steps (2 s), and the dense staged
#     reward carries the rest, since there is no gap for gamma to bridge.
#   gae_lambda 0.95, not 0.9. The repository's value; the difference is small
#     and 0.95 is the lower-bias choice over a long episode.
#   ent_coef 0.005, not 0.0. Four cubes need more exploration than one, and
#     this keeps the scalar policy std from collapsing early.
#   num_steps 24, not 16. Longer rollouts estimate a longer-horizon advantage
#     better, and 24 is what the rest of this repository uses for pick-and-
#     place. It makes the batch 98k rather than 65k.
#
# Total interaction is 6,000 x 4,096 x 24 = 590M transitions against their 50M,
# which is the same order per sub-goal once the four placements and the 40x
# longer episode are accounted for.

# ManiSkill3's Stack-Cube learning rate, held fixed, and the schedule is the
# point. The first three runs used RSL-RL's KL-adaptive schedule at
# desired_kl=0.01 and it collapsed: 8 epochs over 32 minibatches is 256
# gradient steps per iteration, which puts approx_kl above 0.01 on 86-92% of
# iterations, so the controller divides the rate every time and never gets it
# back. Measured on those runs, the two visual arms sat pinned at RSL-RL's
# 1e-5 floor -- 30x below nominal -- for 42% and 52% of their iterations, and
# the state arm thrashed between 4.6e-5 and 3.5e-4 every single iteration.
# ManiSkill does not adapt the rate on KL for this task and neither does the
# rest of this repository's recent Lift-Cube work.
LEARNING_RATE = 0.0003


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
      learning_rate=LEARNING_RATE,
      schedule="fixed",
      gamma=0.99,
      lam=0.95,
      entropy_coef=0.005,
      # ManiSkill's target_kl: an early-stop threshold, never an lr controller.
      desired_kl=0.1,
      max_grad_norm=0.5,
      value_loss_coef=0.5,
      use_clipped_value_loss=False,
      clip_param=0.2,
      normalize_advantage_per_mini_batch=True,
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
      learning_rate=LEARNING_RATE,
      schedule="fixed",
      gamma=0.99,
      lam=0.95,
      entropy_coef=0.005,
      # ManiSkill's target_kl: an early-stop threshold, never an lr controller.
      desired_kl=0.1,
      max_grad_norm=0.5,
      value_loss_coef=0.5,
      use_clipped_value_loss=False,
      clip_param=0.2,
      normalize_advantage_per_mini_batch=True,
      optimizer="adam",
      share_cnn_encoders=False,
      # NatureCnn trains its own trunk, so there is nothing to cache.
      cache_frozen_features=vision.frozen,
      feature_cache_dtype="bfloat16",
      gradient_accumulation_steps=8,
      early_stop_kl=True,
    ),
  )


__all__ = [
  "LEARNING_RATE",
  "NUM_STEPS_PER_ENV",
  "STATE_TASK_ID",
  "trossen_realistic_stack_cubes_rgb_ppo_runner_cfg",
  "trossen_realistic_stack_cubes_state_ppo_runner_cfg",
]
