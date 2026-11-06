from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from vbrl.asset_zoo.objects import CUBE_XML
from vbrl.asset_zoo.robots.definition import CameraView
from vbrl.asset_zoo.robots.trossen_wxai import make_wxai_measured
from vbrl.scenes.builder import apply_scene
from vbrl.tasks.lift_cube.lift_cube_env_cfg import build_env_cfg
from vbrl.tasks.utils import add_rgb_camera

_OBJECT_NAME = "cube"
_CAMERA_VIEW: CameraView = "wrist"


def trossen_realistic_lift_cube_state_env_cfg(
  *, play: bool = False
) -> ManagerBasedRlEnvCfg:
  robot = make_wxai_measured()
  cfg = build_env_cfg(robot=robot, object_name=_OBJECT_NAME, play=play)
  apply_scene(
    cfg,
    scene="default",
    robot=robot,
    camera_view=None,
    object_xml=CUBE_XML,
    object_name=_OBJECT_NAME,
  )
  cfg.scene.num_envs = 1 if play else 1024
  cfg.seed = 0
  return cfg


def trossen_realistic_lift_cube_rgb_env_cfg(
  *, scene: str, play: bool = False
) -> ManagerBasedRlEnvCfg:
  robot = make_wxai_measured()
  cfg = build_env_cfg(
    robot=robot, object_name=_OBJECT_NAME, rgb=True, distill=True, play=play
  )
  add_rgb_camera(
    cfg,
    robot=robot,
    camera_view=_CAMERA_VIEW,
    camera_geometry="visual",
    width=224,
    height=224,
  )
  apply_scene(
    cfg,
    scene=scene,
    robot=robot,
    camera_view=_CAMERA_VIEW,
    object_xml=CUBE_XML,
    object_name=_OBJECT_NAME,
  )
  cfg.scene.num_envs = 1 if play else 1024
  cfg.seed = 0
  return cfg


__all__ = [
  "trossen_realistic_lift_cube_rgb_env_cfg",
  "trossen_realistic_lift_cube_state_env_cfg",
]
