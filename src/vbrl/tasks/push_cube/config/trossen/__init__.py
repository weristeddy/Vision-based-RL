"""Register the Trossen state Push-Cube task."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import trossen_push_cube_env_cfg
from .rl_cfg import TASK_ID, push_cube_rl_cfg


register_mjlab_task(
  TASK_ID,
  trossen_push_cube_env_cfg(),
  trossen_push_cube_env_cfg(play=True),
  push_cube_rl_cfg(),
)
