from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg

from vbrl.asset_zoo.objects import PUSH_CHAIN_XML
from vbrl.asset_zoo.robots.trossen_wxai import make_wxai_measured
from vbrl.scenes.builder import apply_scene
from vbrl.tasks.push_chain.goal_marker import add_goal_marker
from vbrl.tasks.push_chain.push_chain_env_cfg import build_env_cfg
from vbrl.tasks.push_t.config.trossen_realistic.env_cfgs import (
  _CALIBRATION_NOISE,
  _CAMERA_VIEW,
  _OBJECT_NAME,
)
from vbrl.tasks.utils import add_rgb_camera


def trossen_realistic_push_chain_state_env_cfg(
  *, play: bool = False
) -> ManagerBasedRlEnvCfg:
  robot = make_wxai_measured()
  cfg = build_env_cfg(robot=robot, object_name=_OBJECT_NAME, play=play)
  apply_scene(
    cfg,
    scene="default",
    robot=robot,
    camera_view=None,
    object_xml=PUSH_CHAIN_XML,
    object_name=_OBJECT_NAME,
  )
  cfg.scene.num_envs = 1 if play else 1024
  cfg.seed = 0
  return cfg


def trossen_realistic_push_chain_rgb_env_cfg(
  *, scene: str, distill: bool = True, play: bool = False
) -> ManagerBasedRlEnvCfg:
  robot = make_wxai_measured()
  cfg = build_env_cfg(
    robot=robot,
    object_name=_OBJECT_NAME,
    rgb=True,
    distill=distill,
    play=play,
    goal_observation_noise=_CALIBRATION_NOISE,
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
    object_xml=PUSH_CHAIN_XML,
    object_name=_OBJECT_NAME,
  )
  add_goal_marker(cfg)
  if not play:
    cfg.events["camera_fovy"] = EventTermCfg(
      func=dr.cam_fovy,
      mode="startup",
      params={
        "asset_cfg": cfg.events["camera_position"].params["asset_cfg"],
        "operation": "add",
        "ranges": (-2.0, 2.0),
      },
    )
  cfg.scene.num_envs = 1 if play else 1024
  cfg.seed = 0
  return cfg


__all__ = [
  "trossen_realistic_push_chain_rgb_env_cfg",
  "trossen_realistic_push_chain_state_env_cfg",
]
