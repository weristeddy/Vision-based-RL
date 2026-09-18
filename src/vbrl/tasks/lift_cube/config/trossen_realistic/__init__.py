from mjlab.tasks.registry import register_mjlab_task

from vbrl.training.runner import VbrlOnPolicyRunner
from vbrl.vision.architectures import ARCHITECTURES

from ..trossen.rl_cfg import trossen_lift_cube_ppo_runner_cfg
from .env_cfgs import trossen_realistic_lift_cube_env_cfg

_REAL_TEXTURE_ENV = trossen_realistic_lift_cube_env_cfg(scene="real_texture")
_REAL_TEXTURE_PLAY_ENV = trossen_realistic_lift_cube_env_cfg(
  scene="real_texture", play=True
)


def _sim2real(task_id: str, architecture: str) -> None:
  register_mjlab_task(
    task_id,
    _REAL_TEXTURE_ENV,
    _REAL_TEXTURE_PLAY_ENV,
    trossen_lift_cube_ppo_runner_cfg(task_id, ARCHITECTURES[architecture]),
    VbrlOnPolicyRunner,
  )


# Exactly the grid the 12 retained real-texture runs used, superseded rows included:
# this generation changes the camera and nothing else.

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
