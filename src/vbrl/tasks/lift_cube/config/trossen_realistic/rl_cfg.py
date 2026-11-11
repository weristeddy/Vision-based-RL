from __future__ import annotations

from mjlab.rl import RslRlModelCfg

from vbrl.tasks.utils import wandb_task_tag
from vbrl.training.distillation import DistillationRunnerCfg, VisualDistillationCfg
from vbrl.training.ppo import VisualPpoCfg
from vbrl.training.runner import VbrlOnPolicyRunnerCfg
from vbrl.vision.config import VisionConfig

STATE_TASK_ID = "Mjlab-LiftCube-State-Trossen"
_EVALUATION_SCENES = ("real_texture_wood", "real_texture")
_NETWORK = {
  "hidden_dims": (256, 256, 128),
  "activation": "elu",
  "obs_normalization": True,
}
# Push-T's action bounds. Without them gbm0jo0h ended with a std of 460 and mean
# actions of |a| ~1,400: every joint stepped at the +/-1 clip, bang-bang.
_GAUSSIAN = {
  "class_name": "GaussianDistribution",
  "init_std": 1.0,
  "std_type": "log",
  "std_range": (0.0, 1.5),
}


def trossen_realistic_lift_cube_state_ppo_runner_cfg() -> VbrlOnPolicyRunnerCfg:
  return VbrlOnPolicyRunnerCfg(
    seed=0,
    num_steps_per_env=24,
    max_iterations=2000,
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    save_interval=100,
    experiment_name="lift_cube_state",
    run_name=wandb_task_tag(STATE_TASK_ID),
    logger="wandb",
    wandb_project="mjlab",
    wandb_tags=(wandb_task_tag(STATE_TASK_ID), "lift_cube", "state"),
    clip_actions=1.0,
    upload_model=True,
    actor=RslRlModelCfg(**_NETWORK, distribution_cfg=_GAUSSIAN, class_name="MLPModel"),
    critic=RslRlModelCfg(**_NETWORK, class_name="MLPModel"),
    algorithm=VisualPpoCfg(
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
      early_stop_kl=False,
      bound_loss_coef=1.0,
      action_head_init_scale=0.01,
    ),
  )


def trossen_realistic_lift_cube_rgb_distillation_runner_cfg(
  task_id: str, vision: VisionConfig
) -> DistillationRunnerCfg:
  return DistillationRunnerCfg(
    seed=0,
    num_steps_per_env=16,
    max_iterations=8000,
    obs_groups={"student": ("actor", "camera"), "teacher": ("teacher",)},
    save_interval=100,
    evaluation_scenes=_EVALUATION_SCENES,
    experiment_name="lift_cube_rgb_distillation",
    run_name=wandb_task_tag(task_id),
    logger="wandb",
    wandb_project="mjlab",
    wandb_tags=(
      wandb_task_tag(task_id),
      "lift_cube",
      "rgb",
      "distillation",
      vision.encoder,
      vision.adapter,
    ),
    clip_actions=1.0,
    upload_model=True,
    student=RslRlModelCfg(
      **_NETWORK,
      cnn_cfg={"vision": vision.asdict(), "latent_batchnorm": False},
      distribution_cfg={**_GAUSSIAN, "init_std": 0.1},
      class_name="vbrl.vision.model:VisionModel",
    ),
    teacher=RslRlModelCfg(
      **_NETWORK, distribution_cfg=_GAUSSIAN, class_name="MLPModel"
    ),
    algorithm=VisualDistillationCfg(),
  )


__all__ = [
  "STATE_TASK_ID",
  "trossen_realistic_lift_cube_rgb_distillation_runner_cfg",
  "trossen_realistic_lift_cube_state_ppo_runner_cfg",
]
