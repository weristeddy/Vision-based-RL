from mjlab.tasks.registry import register_mjlab_task

from vbrl.training.runner import VbrlDistillationRunner, VbrlOnPolicyRunner
from vbrl.vision.architectures import ARCHITECTURES

from .env_cfgs import (
  trossen_realistic_push_t_rgb_env_cfg,
  trossen_realistic_push_t_state_env_cfg,
)
from .rl_cfg import (
  STATE_TASK_ID,
  trossen_realistic_push_t_rgb_distillation_runner_cfg,
  trossen_realistic_push_t_rgb_ppo_runner_cfg,
  trossen_realistic_push_t_state_ppo_runner_cfg,
)


def _distill(task_id: str, architecture: str, scene: str) -> None:
  register_mjlab_task(
    task_id,
    trossen_realistic_push_t_rgb_env_cfg(scene=scene),
    trossen_realistic_push_t_rgb_env_cfg(scene=scene, play=True),
    trossen_realistic_push_t_rgb_distillation_runner_cfg(
      task_id, ARCHITECTURES[architecture]
    ),
    VbrlDistillationRunner,
  )


def _ppo(task_id: str, architecture: str, scene: str) -> None:
  register_mjlab_task(
    task_id,
    trossen_realistic_push_t_rgb_env_cfg(scene=scene, distill=False),
    trossen_realistic_push_t_rgb_env_cfg(scene=scene, distill=False, play=True),
    trossen_realistic_push_t_rgb_ppo_runner_cfg(task_id, ARCHITECTURES[architecture]),
    VbrlOnPolicyRunner,
  )


register_mjlab_task(
  STATE_TASK_ID,
  trossen_realistic_push_t_state_env_cfg(),
  trossen_realistic_push_t_state_env_cfg(play=True),
  trossen_realistic_push_t_state_ppo_runner_cfg(),
  VbrlOnPolicyRunner,
)
_distill("Mjlab-PushT-RealTable-DinoV2ViTS14-AfaFull6-Trossen", "DinoV2ViTS14-AfaFull6", "real_table")
_distill("Mjlab-PushT-RealTable-DinoV2ViTS14-Global-Trossen", "DinoV2ViTS14-Global", "real_table")
_distill("Mjlab-PushT-RealTable-R3MResNet50-AfaFull32-Trossen", "R3MResNet50-AfaFull32", "real_table")
_distill("Mjlab-PushT-RealTable-R3MResNet50-Global-Trossen", "R3MResNet50-Global", "real_table")
_distill("Mjlab-PushT-WoodTable-DinoV2ViTS14-AfaFull6-Trossen", "DinoV2ViTS14-AfaFull6", "real_texture_wood")
_ppo("Mjlab-PushT-RealTablePPO-DinoV2ViTS14-AfaFull6-Trossen", "DinoV2ViTS14-AfaFull6", "real_table")
