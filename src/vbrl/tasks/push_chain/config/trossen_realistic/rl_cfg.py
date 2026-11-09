from __future__ import annotations

from mjlab.rl import RslRlModelCfg

from vbrl.tasks.push_t.config.trossen_realistic.rl_cfg import (
  _DISTILLATION_MAX_ITERATIONS,
  _GAUSSIAN,
  _NETWORK,
  _algorithm,
)
from vbrl.tasks.utils import wandb_task_tag
from vbrl.training.distillation import DistillationRunnerCfg, VisualDistillationCfg
from vbrl.training.runner import VbrlOnPolicyRunnerCfg
from vbrl.vision.config import VisionConfig

STATE_TASK_ID = "Mjlab-PushChain-State-Trossen"
_EVALUATION_SCENES = ("real_table", "real_texture_wood")


def trossen_realistic_push_chain_state_ppo_runner_cfg() -> VbrlOnPolicyRunnerCfg:
  return VbrlOnPolicyRunnerCfg(
    seed=0,
    num_steps_per_env=16,
    max_iterations=8000,
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    save_interval=100,
    experiment_name="push_chain_state_trossen_realistic_sim2real_dr",
    run_name=wandb_task_tag(STATE_TASK_ID),
    logger="wandb",
    wandb_project="mjlab",
    wandb_tags=(
      wandb_task_tag(STATE_TASK_ID),
      "push_chain",
      "state",
      "success_90",
      "sim2real_dr",
    ),
    clip_actions=1.0,
    upload_model=True,
    actor=RslRlModelCfg(**_NETWORK, distribution_cfg=_GAUSSIAN, class_name="MLPModel"),
    critic=RslRlModelCfg(**_NETWORK, class_name="MLPModel"),
    algorithm=_algorithm(),
  )


def trossen_realistic_push_chain_rgb_distillation_runner_cfg(
  task_id: str, vision: VisionConfig
) -> DistillationRunnerCfg:
  return DistillationRunnerCfg(
    seed=0,
    num_steps_per_env=16,
    max_iterations=_DISTILLATION_MAX_ITERATIONS,
    obs_groups={"student": ("actor", "camera"), "teacher": ("teacher",)},
    save_interval=100,
    evaluation_scenes=_EVALUATION_SCENES,
    experiment_name="push_chain_rgb_distillation",
    run_name=wandb_task_tag(task_id),
    logger="wandb",
    wandb_project="mjlab",
    wandb_tags=(
      wandb_task_tag(task_id),
      "push_chain",
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
  "trossen_realistic_push_chain_rgb_distillation_runner_cfg",
  "trossen_realistic_push_chain_state_ppo_runner_cfg",
]
