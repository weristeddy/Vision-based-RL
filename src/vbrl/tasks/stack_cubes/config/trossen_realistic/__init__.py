"""Register the realistic Trossen Stack-Cubes contracts, one line per task ID.

Three, and deliberately only three. ``State`` is the task-validation baseline;
``Ext`` and ``Wrist`` are the same environment seen through one camera each,
never both, so the two optics are comparable to each other and to the state
run. One architecture -- ``NatureCnn-SpatialSoftmax``, straight out of the
global registry -- until the task is shown to learn; the encoder sweep comes
after that and crosses the same table every other visual generation does.
"""

from mjlab.tasks.registry import register_mjlab_task

from vbrl.vision.architectures import ARCHITECTURES

from .env_cfgs import (
  trossen_realistic_stack_cubes_rgb_env_cfg,
  trossen_realistic_stack_cubes_state_env_cfg,
)
from .rl_cfg import (
  STATE_TASK_ID,
  trossen_realistic_stack_cubes_rgb_ppo_runner_cfg,
  trossen_realistic_stack_cubes_state_ppo_runner_cfg,
)


_ARCHITECTURE = "NatureCnn-SpatialSoftmax"


def _rgb(task_id: str, camera: str) -> None:
  """Register one single-camera Stack-Cubes policy."""
  register_mjlab_task(
    task_id,
    trossen_realistic_stack_cubes_rgb_env_cfg(camera=camera),
    trossen_realistic_stack_cubes_rgb_env_cfg(camera=camera, play=True),
    trossen_realistic_stack_cubes_rgb_ppo_runner_cfg(
      task_id,
      ARCHITECTURES[_ARCHITECTURE],
      camera=f"{camera}_cam" if camera == "external" else "cam",
    ),
  )


# --- State: no camera, no visual randomization ------------------------------

register_mjlab_task(
  STATE_TASK_ID,
  trossen_realistic_stack_cubes_state_env_cfg(),
  trossen_realistic_stack_cubes_state_env_cfg(play=True),
  trossen_realistic_stack_cubes_state_ppo_runner_cfg(),
)

# --- RGB: one camera each, everything else identical ------------------------

_rgb(
  "Mjlab-StackCubes-Ext-NatureCnn-SpatialSoftmax-TrossenRealistic",
  "external",
)
_rgb(
  "Mjlab-StackCubes-Wrist-NatureCnn-SpatialSoftmax-TrossenRealistic",
  "wrist",
)
