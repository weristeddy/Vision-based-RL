"""Realistic Trossen Push-T environments."""

from __future__ import annotations

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg

from vbrl.asset_zoo.objects import PUSH_T_XML
from vbrl.asset_zoo.robots.definition import CameraView
from vbrl.asset_zoo.robots.trossen_wxai import make_wxai_realistic
from vbrl.scenes.builder import apply_scene
from vbrl.tasks.utils import add_rgb_camera
from vbrl.tasks.push_t.goal_marker import GOAL_ENTITY_NAME, goal_marker_spec
from vbrl.tasks.push_t.push_t_env_cfg import build_env_cfg


_OBJECT_NAME = "object"
_OBJECT_XML = PUSH_T_XML


def _env_cfg(
  *,
  rgb: bool,
  scene: str,
  play: bool,
  camera: CameraView = "external",
  success_threshold: float = 0.98,
  goal_yaw_stages=None,
  quadratic_orientation: bool = False,
  visual_goal: bool = False,
  free_start: bool = False,
  near_goal_probability: float = 0.0,
  separation_curriculum: bool = False,
) -> ManagerBasedRlEnvCfg:
  robot = make_wxai_realistic()
  cfg = build_env_cfg(
    robot=robot,
    object_name=_OBJECT_NAME,
    rgb=rgb,
    play=play,
    success_threshold=success_threshold,
    goal_yaw_stages=goal_yaw_stages,
    quadratic_orientation=quadratic_orientation,
    visual_goal=visual_goal,
    free_start=free_start,
    near_goal_probability=near_goal_probability,
    separation_curriculum=separation_curriculum,
  )
  camera_view: CameraView | None = camera if rgb else None
  if rgb:
    add_rgb_camera(
      cfg,
      robot=robot,
      camera_view=camera,
      camera_geometry="visual",
      width=224,
      height=224,
    )
  apply_scene(
    cfg,
    scene=scene,
    robot=robot,
    camera_view=camera_view,
    object_xml=_OBJECT_XML,
    object_name=_OBJECT_NAME,
  )
  if visual_goal:
    # After `apply_scene`, which owns table and object; the marker is neither and
    # takes no part in appearance randomization.
    cfg.scene.entities[GOAL_ENTITY_NAME] = EntityCfg(spec_fn=goal_marker_spec)
  cfg.scene.num_envs = 1 if play else 1024
  cfg.seed = 0
  return cfg


def trossen_realistic_push_t_state_env_cfg(*, play: bool = False) -> ManagerBasedRlEnvCfg:
  return _env_cfg(rgb=False, scene="default", play=play)


def trossen_realistic_push_t_rgb_env_cfg(
  *,
  scene: str = "real_texture",
  play: bool = False,
  camera: CameraView = "external",
  success_threshold: float = 0.98,
  goal_yaw_stages=None,
  quadratic_orientation: bool = False,
  visual_goal: bool = False,
  free_start: bool = False,
  near_goal_probability: float = 0.0,
  separation_curriculum: bool = False,
) -> ManagerBasedRlEnvCfg:
  """One RGB Push-T environment.

  ``camera`` names a camera the robot declares. ``external`` is the calibrated
  D435 pose and the default. ``external_front`` is the near-overhead pose the
  earlier generations used; ``external_tilted`` is that pose tilted back to 45
  degrees, which keeps 79% of the object's silhouette visible while the gripper
  is on it instead of 37%, and is what the current generation uses.

  ``success_threshold`` is 0.98 for the retained generations. The curriculum
  generation uses ManiSkill3's 0.90, which also restores the sparse at-goal
  reward bonus: at 0.98 that bonus needs 2 mm and 2.5 degrees and so effectively
  never fires.
  """
  return _env_cfg(
    rgb=True,
    scene=scene,
    play=play,
    camera=camera,
    success_threshold=success_threshold,
    goal_yaw_stages=goal_yaw_stages,
    quadratic_orientation=quadratic_orientation,
    visual_goal=visual_goal,
    free_start=free_start,
    near_goal_probability=near_goal_probability,
    separation_curriculum=separation_curriculum,
  )


__all__ = [
  "trossen_realistic_push_t_rgb_env_cfg",
  "trossen_realistic_push_t_state_env_cfg",
]
