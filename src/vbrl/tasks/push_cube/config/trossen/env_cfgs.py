"""Trossen state Push-Cube environment."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from vbrl.asset_zoo.objects import CUBE_XML
from vbrl.asset_zoo.robots.trossen_wxai import make_wxai
from vbrl.scenes.builder import apply_scene
from vbrl.tasks.push_cube.push_cube_env_cfg import build_env_cfg


_OBJECT_NAME = "cube"
_OBJECT_XML = CUBE_XML


def trossen_push_cube_env_cfg(
  *,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  robot = make_wxai()
  cfg = build_env_cfg(
    robot=robot,
    object_name=_OBJECT_NAME,
    play=play,
  )
  apply_scene(
    cfg,
    scene="default",
    robot=robot,
    object_xml=_OBJECT_XML,
    object_name=_OBJECT_NAME,
  )
  cfg.scene.num_envs = 1 if play else 1024
  cfg.seed = 0
  return cfg


__all__ = ["trossen_push_cube_env_cfg"]
