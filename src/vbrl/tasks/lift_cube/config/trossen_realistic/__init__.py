"""Register the realistic-material Trossen Lift-Cube contract, one line per ID.

One architecture only: DinoV2ViTS14-SpatialSoftmax, the best of the 24 retained
Lift-Cube runs (0.754 at_goal), registered for the first sim2real attempt.

``Sim2Real`` rather than ``RealTexture`` as the variant, for two reasons. The
scene is indeed ``real_texture``, but this environment is not the one the
retained ``RealTexture`` checkpoints trained in: every view now renders the
D405's measured 54.49-degree optics instead of the 87 degrees that were its
*horizontal* FOV, so objects project 1.72x larger than they did. And the ID
scheme has a hard ceiling -- W&B rejects tags over 64 characters, and
``LiftCube-RealTexture-DinoV2ViTS14-SpatialSoftmax-TrossenRealistic`` is 65 --
so one token had to shorten. Naming the generation beats abbreviating
``RealTexture`` or ``TrossenRealistic``, either of which would give one scene or
one robot two names and break the grep from any ID to its registration.

The runner config is the sibling package's, imported rather than restated: the
hyperparameters belong to the task, not to the robot asset, and a second copy
would be free to drift from the runs already trained against it.
"""

from mjlab.tasks.registry import register_mjlab_task

from vbrl.vision.architectures import ARCHITECTURES

from ..trossen.rl_cfg import trossen_lift_cube_ppo_runner_cfg
from .env_cfgs import trossen_realistic_lift_cube_env_cfg


_REAL_TEXTURE_ENV = trossen_realistic_lift_cube_env_cfg(scene="real_texture")
_REAL_TEXTURE_PLAY_ENV = trossen_realistic_lift_cube_env_cfg(
  scene="real_texture", play=True
)


def _sim2real(task_id: str, architecture: str) -> None:
  """Register one visual-camera policy. ``architecture`` keys ARCHITECTURES."""
  register_mjlab_task(
    task_id,
    _REAL_TEXTURE_ENV,
    _REAL_TEXTURE_PLAY_ENV,
    trossen_lift_cube_ppo_runner_cfg(task_id, ARCHITECTURES[architecture]),
  )


# --- Sim2Real: visual camera, D405 optics, realistic robot materials ---------

_sim2real(
  "Mjlab-LiftCube-Sim2Real-DinoV2ViTS14-SpatialSoftmax-TrossenRealistic",
  "DinoV2ViTS14-SpatialSoftmax",
)
