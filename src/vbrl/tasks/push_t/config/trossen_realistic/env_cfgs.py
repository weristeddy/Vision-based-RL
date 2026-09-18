from __future__ import annotations

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg

from vbrl.asset_zoo.objects import PUSH_T_XML
from vbrl.asset_zoo.robots.definition import CameraView
from vbrl.asset_zoo.robots.trossen_wxai import make_wxai_realistic
from vbrl.scenes.builder import apply_scene
from vbrl.tasks.push_t.goal_marker import (
  GOAL_COLOUR_EVENT,
  GOAL_ENTITY_NAME,
  goal_colour_event,
  goal_marker_spec,
)
from vbrl.tasks.push_t.push_t_env_cfg import GOAL_YAW_STAGES, build_env_cfg
from vbrl.tasks.utils import add_rgb_camera

_OBJECT_NAME = "object"
# The 45-degree tilted pose, measured against the ChArUco board: it keeps 79% of the
# object's silhouette visible while the gripper is on it, against 37% for the near-.
_CAMERA: CameraView = "external"


def trossen_realistic_push_t_state_env_cfg(
  *, play: bool = False
) -> ManagerBasedRlEnvCfg:
  robot = make_wxai_realistic()
  cfg = build_env_cfg(robot=robot, object_name=_OBJECT_NAME, play=play)
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
  *,
  action_delta: float,
  play: bool = False,
  goal_in_observation: bool = True,
  fixed_target: tuple[float, float, float] | None = None,
) -> ManagerBasedRlEnvCfg:
  robot = make_wxai_realistic()
  cfg = build_env_cfg(
    robot=robot,
    object_name=_OBJECT_NAME,
    rgb=True,
    play=play,
    goal_yaw_stages=GOAL_YAW_STAGES,
    visual_goal=True,
    goal_in_observation=goal_in_observation,
    fixed_target=fixed_target,
    action_delta=action_delta,
  )
  add_rgb_camera(
    cfg,
    robot=robot,
    camera_view=_CAMERA,
    camera_geometry="visual",
    width=224,
    height=224,
  )
  apply_scene(
    cfg,
    scene="real_texture",
    robot=robot,
    camera_view=_CAMERA,
    object_xml=PUSH_T_XML,
    object_name=_OBJECT_NAME,
  )
  cfg.scene.entities[GOAL_ENTITY_NAME] = EntityCfg(spec_fn=goal_marker_spec)
  cfg.events[GOAL_COLOUR_EVENT] = goal_colour_event()
  cfg.scene.num_envs = 1 if play else 1024
  cfg.seed = 0
  return cfg


__all__ = [
  "trossen_realistic_push_t_rgb_env_cfg",
  "trossen_realistic_push_t_state_env_cfg",
]
