"""Realistic Trossen Push-T environments."""

from __future__ import annotations

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg

from vbrl.asset_zoo.objects import PUSH_T_XML
from vbrl.asset_zoo.robots.definition import CameraView
from vbrl.asset_zoo.robots.trossen_wxai import make_wxai_realistic
from vbrl.scenes.builder import apply_scene
from vbrl.tasks.utils import add_rgb_camera
from vbrl.tasks.push_t.goal_marker import (
  GOAL_COLOUR_EVENT,
  GOAL_ENTITY_NAME,
  goal_colour_event,
  goal_marker_spec,
)
from vbrl.tasks.push_t.push_t_env_cfg import build_env_cfg


_OBJECT_NAME = "object"
_OBJECT_XML = PUSH_T_XML


def _env_cfg(
  *,
  rgb: bool,
  scene: str,
  play: bool,
  camera: CameraView = "external",
  success_threshold: float = 0.90,
  goal_yaw_stages=None,
  quadratic_orientation: bool = False,
  visual_goal: bool = False,
  free_start: bool = False,
  near_goal_probability: float = 0.0,
  separation_curriculum: bool = False,
  goal_in_observation: bool = True,
  fixed_target: tuple[float, float, float] | None = None,
  action_delta: float | None = None,
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
    goal_in_observation=goal_in_observation,
    fixed_target=fixed_target,
    **({} if action_delta is None else {"action_delta": action_delta}),
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
    # After `apply_scene`, which owns the table and the object: the marker is
    # neither, so its own colour event is added here rather than in the preset.
    cfg.scene.entities[GOAL_ENTITY_NAME] = EntityCfg(spec_fn=goal_marker_spec)
    cfg.events[GOAL_COLOUR_EVENT] = goal_colour_event()
  cfg.scene.num_envs = 1 if play else 1024
  cfg.seed = 0
  return cfg


def trossen_realistic_push_t_state_env_cfg(
  *, play: bool = False
) -> ManagerBasedRlEnvCfg:
  return _env_cfg(rgb=False, scene="default", play=play)


def trossen_realistic_push_t_rgb_env_cfg(
  *,
  scene: str = "real_texture",
  play: bool = False,
  camera: CameraView = "external",
  success_threshold: float = 0.90,
  goal_yaw_stages=None,
  quadratic_orientation: bool = False,
  visual_goal: bool = False,
  free_start: bool = False,
  near_goal_probability: float = 0.0,
  separation_curriculum: bool = False,
  goal_in_observation: bool = True,
  fixed_target: tuple[float, float, float] | None = None,
  action_delta: float | None = None,
) -> ManagerBasedRlEnvCfg:
  """One RGB Push-T environment.

  ``camera`` names a camera the robot declares, and ``external`` is the only
  external one left. It is the 45-degree tilted pose, measured against the
  ChArUco board on 2026-09-09: it keeps 79% of the object's silhouette visible
  while the gripper is on it, against 37% for the near-overhead pose the earlier
  generations used. Those two retired poses (``external_front``, and the old
  near-overhead ``external``) were deleted along with the task IDs that named
  them, because a pose that is no longer in the MJCF cannot be replayed.

  ``success_threshold`` is ManiSkill3's 0.90 everywhere. It used to be 0.98 for
  every generation except the curriculum one, and that was measuring the
  threshold rather than the policy: 0.98 demands 2 mm *and* 2.5 degrees, which
  is below what a 0.1 rad joint increment can resolve, so the sparse at-goal
  bonus effectively never fired and success read near zero for policies that
  were placing the T. Rolled out on one trained checkpoint, the same episodes
  score 0.004 at 0.98 and 0.250 at 0.90. Retained results measured at 0.98 are
  not comparable to anything logged after this change.
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
    goal_in_observation=goal_in_observation,
    fixed_target=fixed_target,
    **({} if action_delta is None else {"action_delta": action_delta}),
  )


__all__ = [
  "trossen_realistic_push_t_rgb_env_cfg",
  "trossen_realistic_push_t_state_env_cfg",
]
