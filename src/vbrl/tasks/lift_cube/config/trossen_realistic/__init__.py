"""Register the realistic-material Trossen Lift-Cube contract, one line per ID.

Twelve architectures: **exactly** the grid the 12 retained real-texture
Lift-Cube runs were trained on, three superseded rows included. This is the one
place where ``CURRENT_ARCHITECTURES`` is deliberately not what a new
registration crosses. The point of this generation is to change the camera and
nothing else, so ``NatureCnn-LocalGrid16``, ``CompactVit-LocalGrid8`` and
``DinoV2ViTS14-LocalGrid7`` stay as they are: substituting the native-resolution
rows that replaced them would move the adapter and the optics at once and leave
neither attributable. Each of the twelve was tuned separately too -- its own
Bayesian sweep under ``configs/sweeps/`` -- so a rerun passes that
architecture's own learning rate, KL target and epoch count on the command
line, exactly as the first sim2real run did. None of that is registered here:
the task ID deliberately encodes no hyperparameter.

The three ``R3MResNet50L3`` rows are absent for the same reason: they postdate
the retained grid, so they have no run here to be compared against.

``Sim2Real`` rather than ``RealTexture`` as the variant, for two reasons. The
scene is indeed ``real_texture``, but this environment is not the one the
retained ``RealTexture`` checkpoints trained in: every view now renders the
D405's measured 54.49-degree optics instead of the 87 degrees that were its
*horizontal* FOV, so objects project 1.72x larger than they did, and the camera
housing that used to black out the wrist view has moved to the geometry group no
camera renders. And the ID scheme has a hard ceiling -- W&B rejects tags over 64
characters, and
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
  "Mjlab-LiftCube-Sim2Real-NatureCnn-LocalGrid16-TrossenRealistic",
  "NatureCnn-LocalGrid16",
)
_sim2real(
  "Mjlab-LiftCube-Sim2Real-NatureCnn-SpatialSoftmax-TrossenRealistic",
  "NatureCnn-SpatialSoftmax",
)
_sim2real(
  "Mjlab-LiftCube-Sim2Real-CompactVit-LocalGrid8-TrossenRealistic",
  "CompactVit-LocalGrid8",
)
_sim2real(
  "Mjlab-LiftCube-Sim2Real-CompactVit-SpatialSoftmax-TrossenRealistic",
  "CompactVit-SpatialSoftmax",
)
_sim2real(
  "Mjlab-LiftCube-Sim2Real-DinoV2ViTS14-Linear-TrossenRealistic",
  "DinoV2ViTS14-Linear",
)
_sim2real(
  "Mjlab-LiftCube-Sim2Real-DinoV2ViTS14-LocalGrid7-TrossenRealistic",
  "DinoV2ViTS14-LocalGrid7",
)
_sim2real(
  "Mjlab-LiftCube-Sim2Real-DinoV2ViTS14-SpatialSoftmax-TrossenRealistic",
  "DinoV2ViTS14-SpatialSoftmax",
)
_sim2real(
  "Mjlab-LiftCube-Sim2Real-DinoV2ViTS14-Afa6-TrossenRealistic",
  "DinoV2ViTS14-Afa6",
)
_sim2real(
  "Mjlab-LiftCube-Sim2Real-R3MResNet50-Linear-TrossenRealistic",
  "R3MResNet50-Linear",
)
_sim2real(
  "Mjlab-LiftCube-Sim2Real-R3MResNet50-LocalGrid7-TrossenRealistic",
  "R3MResNet50-LocalGrid7",
)
_sim2real(
  "Mjlab-LiftCube-Sim2Real-R3MResNet50-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50-SpatialSoftmax",
)
_sim2real(
  "Mjlab-LiftCube-Sim2Real-R3MResNet50-Afa32-TrossenRealistic",
  "R3MResNet50-Afa32",
)
