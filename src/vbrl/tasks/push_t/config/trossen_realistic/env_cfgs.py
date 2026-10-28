from __future__ import annotations

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers import EventTermCfg

from vbrl.asset_zoo.objects import PUSH_T_XML
from vbrl.asset_zoo.robots.definition import CameraView
from vbrl.asset_zoo.robots.trossen_wxai import make_wxai_measured
from vbrl.scenes.builder import apply_scene
from vbrl.tasks.push_t.goal_marker import (
  GOAL_COLOUR_EVENT,
  GOAL_ENTITY_NAME,
  GOAL_REAL_RGBA_RANGE,
  goal_colour_event,
  goal_marker_spec,
)
from vbrl.tasks.push_t.push_t_env_cfg import build_env_cfg
from vbrl.tasks.utils import add_rgb_camera

_OBJECT_NAME = "object"
# The 45-degree tilted pose, measured against the ChArUco board: it keeps 79% of the
# object's silhouette visible while the gripper is on it, against 37% for the near-.
_CAMERA_VIEW: CameraView = "external"
_CALIBRATION_NOISE = (0.003, 0.026)


def trossen_realistic_push_t_state_env_cfg(
  *, play: bool = False
) -> ManagerBasedRlEnvCfg:
  robot = make_wxai_measured()
  cfg = build_env_cfg(
    robot=robot,
    object_name=_OBJECT_NAME,
    play=play,
  )
  apply_scene(
    cfg,
    scene="default",
    robot=robot,
    camera_view=None,
    object_xml=PUSH_T_XML,
    object_name=_OBJECT_NAME,
  )
  cfg.scene.num_envs = 1 if play else 1024
  cfg.seed = 0
  return cfg


def trossen_realistic_push_t_rgb_env_cfg(
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
    object_xml=PUSH_T_XML,
    object_name=_OBJECT_NAME,
  )
  cfg.scene.entities[GOAL_ENTITY_NAME] = EntityCfg(spec_fn=goal_marker_spec)
  cfg.events[GOAL_COLOUR_EVENT] = goal_colour_event(GOAL_REAL_RGBA_RANGE)
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
  "trossen_realistic_push_t_rgb_env_cfg",
  "trossen_realistic_push_t_state_env_cfg",
]
