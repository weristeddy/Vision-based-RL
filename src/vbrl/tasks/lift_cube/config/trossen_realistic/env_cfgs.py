"""Realistic-material Trossen Lift-Cube environments.

Identical to the standard Trossen build except for the robot asset:
``wxai_realistic.xml`` splits five meshes so each part carries its own material.
The control constants are shared, so this changes what the camera sees and
nothing about the dynamics -- which is the point, for a sim2real attempt where
the robot's own appearance is a large part of every wrist-camera frame.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from vbrl.asset_zoo.objects import CUBE_XML
from vbrl.asset_zoo.robots.definition import CameraGeometry
from vbrl.asset_zoo.robots.trossen_wxai import make_wxai_realistic
from vbrl.scenes.builder import apply_scene
from vbrl.tasks.lift_cube.lift_cube_env_cfg import build_env_cfg
from vbrl.tasks.utils import add_rgb_camera


_OBJECT_NAME = "cube"
_OBJECT_XML = CUBE_XML


def trossen_realistic_lift_cube_env_cfg(
  *,
  scene: str = "real_texture",
  camera_geometry: CameraGeometry = "visual",
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build the realistic-material Trossen Lift-Cube environment."""
  robot = make_wxai_realistic()
  cfg = build_env_cfg(
    robot=robot,
    object_name=_OBJECT_NAME,
    rgb=True,
    play=play,
  )
  add_rgb_camera(
    cfg,
    robot=robot,
    camera_view="wrist",
    camera_geometry=camera_geometry,
    width=224,
    height=224,
  )
  apply_scene(
    cfg,
    scene=scene,
    robot=robot,
    camera_view="wrist",
    object_xml=_OBJECT_XML,
    object_name=_OBJECT_NAME,
  )
  cfg.scene.num_envs = 1 if play else 1024
  cfg.seed = 0
  return cfg


__all__ = ["trossen_realistic_lift_cube_env_cfg"]
