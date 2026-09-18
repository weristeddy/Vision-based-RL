from mjlab.tasks.registry import register_mjlab_task

from vbrl.training.runner import VbrlOnPolicyRunner
from vbrl.vision.architectures import ARCHITECTURES

from .env_cfgs import trossen_lift_cube_env_cfg
from .rl_cfg import trossen_lift_cube_ppo_runner_cfg

_REAL_TEXTURE_ENV = trossen_lift_cube_env_cfg(scene="real_texture")
_REAL_TEXTURE_PLAY_ENV = trossen_lift_cube_env_cfg(scene="real_texture", play=True)
_COLLISION_CAM_ENV = trossen_lift_cube_env_cfg(
  scene="real_texture", camera_geometry="collision"
)
_COLLISION_CAM_PLAY_ENV = trossen_lift_cube_env_cfg(
  scene="real_texture", camera_geometry="collision", play=True
)


def _real_texture(task_id: str, architecture: str) -> None:
  register_mjlab_task(
    task_id,
    _REAL_TEXTURE_ENV,
    _REAL_TEXTURE_PLAY_ENV,
    trossen_lift_cube_ppo_runner_cfg(task_id, ARCHITECTURES[architecture]),
    VbrlOnPolicyRunner,
  )


def _collision_cam(task_id: str, architecture: str) -> None:
  register_mjlab_task(
    task_id,
    _COLLISION_CAM_ENV,
    _COLLISION_CAM_PLAY_ENV,
    trossen_lift_cube_ppo_runner_cfg(task_id, ARCHITECTURES[architecture]),
    VbrlOnPolicyRunner,
  )


_real_texture(
  "Mjlab-LiftCube-RealTexture-NatureCnn-LocalGrid7-Trossen",
  "NatureCnn-LocalGrid7",
)
_real_texture(
  "Mjlab-LiftCube-RealTexture-NatureCnn-SpatialSoftmax-Trossen",
  "NatureCnn-SpatialSoftmax",
)
_real_texture(
  "Mjlab-LiftCube-RealTexture-CompactVit-LocalGrid8-Trossen",
  "CompactVit-LocalGrid8",
)
_real_texture(
  "Mjlab-LiftCube-RealTexture-CompactVit-SpatialSoftmax-Trossen",
  "CompactVit-SpatialSoftmax",
)
_real_texture(
  "Mjlab-LiftCube-RealTexture-DinoV2ViTS14-Linear-Trossen",
  "DinoV2ViTS14-Linear",
)
_real_texture(
  "Mjlab-LiftCube-RealTexture-DinoV2ViTS14-LocalGrid7-Trossen",
  "DinoV2ViTS14-LocalGrid7",
)
_real_texture(
  "Mjlab-LiftCube-RealTexture-DinoV2ViTS14-SpatialSoftmax-Trossen",
  "DinoV2ViTS14-SpatialSoftmax",
)
_real_texture(
  "Mjlab-LiftCube-RealTexture-DinoV2ViTS14-Afa6-Trossen",
  "DinoV2ViTS14-Afa6",
)
_real_texture(
  "Mjlab-LiftCube-RealTexture-R3MResNet50-Linear-Trossen",
  "R3MResNet50-Linear",
)
_real_texture(
  "Mjlab-LiftCube-RealTexture-R3MResNet50-LocalGrid7-Trossen",
  "R3MResNet50-LocalGrid7",
)
_real_texture(
  "Mjlab-LiftCube-RealTexture-R3MResNet50-SpatialSoftmax-Trossen",
  "R3MResNet50-SpatialSoftmax",
)
_real_texture(
  "Mjlab-LiftCube-RealTexture-R3MResNet50-Afa32-Trossen",
  "R3MResNet50-Afa32",
)


_collision_cam(
  "Mjlab-LiftCube-CollisionCam-NatureCnn-LocalGrid16-Trossen",
  "NatureCnn-LocalGrid16",
)
_collision_cam(
  "Mjlab-LiftCube-CollisionCam-NatureCnn-SpatialSoftmax-Trossen",
  "NatureCnn-SpatialSoftmax",
)
_collision_cam(
  "Mjlab-LiftCube-CollisionCam-CompactVit-LocalGrid8-Trossen",
  "CompactVit-LocalGrid8",
)
_collision_cam(
  "Mjlab-LiftCube-CollisionCam-CompactVit-SpatialSoftmax-Trossen",
  "CompactVit-SpatialSoftmax",
)
_collision_cam(
  "Mjlab-LiftCube-CollisionCam-DinoV2ViTS14-Linear-Trossen",
  "DinoV2ViTS14-Linear",
)
_collision_cam(
  "Mjlab-LiftCube-CollisionCam-DinoV2ViTS14-LocalGrid7-Trossen",
  "DinoV2ViTS14-LocalGrid7",
)
_collision_cam(
  "Mjlab-LiftCube-CollisionCam-DinoV2ViTS14-SpatialSoftmax-Trossen",
  "DinoV2ViTS14-SpatialSoftmax",
)
_collision_cam(
  "Mjlab-LiftCube-CollisionCam-DinoV2ViTS14-Afa6-Trossen",
  "DinoV2ViTS14-Afa6",
)
_collision_cam(
  "Mjlab-LiftCube-CollisionCam-R3MResNet50-Linear-Trossen",
  "R3MResNet50-Linear",
)
_collision_cam(
  "Mjlab-LiftCube-CollisionCam-R3MResNet50-LocalGrid7-Trossen",
  "R3MResNet50-LocalGrid7",
)
_collision_cam(
  "Mjlab-LiftCube-CollisionCam-R3MResNet50-SpatialSoftmax-Trossen",
  "R3MResNet50-SpatialSoftmax",
)
_collision_cam(
  "Mjlab-LiftCube-CollisionCam-R3MResNet50-Afa32-Trossen",
  "R3MResNet50-Afa32",
)
