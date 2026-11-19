from mjlab.tasks.registry import register_mjlab_task

from vbrl.training.runner import VbrlDistillationRunner, VbrlOnPolicyRunner
from vbrl.vision.architectures import ARCHITECTURES

from .env_cfgs import (
  trossen_realistic_push_chain_rgb_env_cfg,
  trossen_realistic_push_chain_state_env_cfg,
)
from .rl_cfg import (
  trossen_realistic_push_chain_rgb_distillation_runner_cfg,
  trossen_realistic_push_chain_state_ppo_runner_cfg,
)


def _distill(task_id: str, architecture: str, scene: str) -> None:
  register_mjlab_task(
    task_id,
    trossen_realistic_push_chain_rgb_env_cfg(scene=scene),
    trossen_realistic_push_chain_rgb_env_cfg(scene=scene, play=True),
    trossen_realistic_push_chain_rgb_distillation_runner_cfg(
      task_id, ARCHITECTURES[architecture]
    ),
    VbrlDistillationRunner,
  )


def _state(task_id: str, num_links: int) -> None:
  register_mjlab_task(
    task_id,
    trossen_realistic_push_chain_state_env_cfg(num_links=num_links),
    trossen_realistic_push_chain_state_env_cfg(num_links=num_links, play=True),
    trossen_realistic_push_chain_state_ppo_runner_cfg(task_id),
    VbrlOnPolicyRunner,
  )


_distill("Mjlab-PushChain3-WoodTable-DinoV2ViTS14-AfaFull6-Trossen", "DinoV2ViTS14-AfaFull6", "real_texture_wood")
_distill("Mjlab-PushChain3-WoodTable-DinoV2ViTS14-Global-Trossen", "DinoV2ViTS14-Global", "real_texture_wood")
_distill("Mjlab-PushChain3-WoodTable-R3MResNet50-AfaFull32-Trossen", "R3MResNet50-AfaFull32", "real_texture_wood")
_distill("Mjlab-PushChain3-WoodTable-R3MResNet50-Global-Trossen", "R3MResNet50-Global", "real_texture_wood")
_distill("Mjlab-PushChain3-WoodTable-NatureCnn-FlattenRelu-Trossen", "NatureCnn-FlattenRelu", "real_texture_wood")
_distill("Mjlab-PushChain3-WoodTable-CompactVit-Svea-Trossen", "CompactVit-Svea", "real_texture_wood")
_state("Mjlab-PushChain3-State-Trossen", 3)
_state("Mjlab-PushChain4-State-Trossen", 4)
