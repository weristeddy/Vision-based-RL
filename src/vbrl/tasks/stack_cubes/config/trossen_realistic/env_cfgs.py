"""Realistic Trossen Stack-Cubes environments."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from vbrl.asset_zoo.objects import CUBE_XML
from vbrl.asset_zoo.robots.definition import CameraView
from vbrl.asset_zoo.robots.trossen_wxai import make_wxai_realistic
from vbrl.scenes.builder import apply_scene
from mjlab.managers import EventTermCfg

from vbrl.tasks.stack_cubes.mdp.events import randomize_cube_colour
from vbrl.tasks.stack_cubes.mdp.tower import CUBE_NAMES
from vbrl.tasks.stack_cubes.stack_cubes_env_cfg import build_env_cfg
from vbrl.tasks.utils import add_rgb_camera


def _env_cfg(
  *,
  rgb: bool,
  scene: str,
  play: bool,
  camera: CameraView | None = None,
) -> ManagerBasedRlEnvCfg:
  robot = make_wxai_realistic()
  cfg = build_env_cfg(robot=robot, rgb=rgb, play=play)
  if rgb:
    assert camera is not None
    add_rgb_camera(
      cfg,
      robot=robot,
      camera_view=camera,
      camera_geometry="visual",
      width=224,
      height=224,
    )
  # Four entities from one MJCF. The scene installs them and dresses the table
  # and the lighting as usual, but the cubes' own colour is handled here: the
  # preset's per-object draw would give each cube a different colour, and cube
  # identity means nothing in this task, so one shared saturated colour is both
  # more honest and easier to see. `real_texture_red` is the hook -- it is
  # `real_texture` with the object colour event turned off and the whole
  # photographic table bank kept.
  apply_scene(
    cfg,
    scene=scene,
    robot=robot,
    camera_view=camera if rgb else None,
    object_xml=CUBE_XML,
    object_name=CUBE_NAMES[0],
    extra_object_names=CUBE_NAMES[1:],
  )
  if rgb:
    # `object_color` is the scene builder's own name for this slot, and using it
    # is deliberate: `real_texture_red` leaves the slot empty so this fills it,
    # and `replace_scene` clears it by name, so an OOD evaluation gets fixed
    # cube colours exactly as it gets a fixed tabletop.
    cfg.events["object_color"] = EventTermCfg(
      func=randomize_cube_colour, mode="reset"
    )
  cfg.scene.num_envs = 1 if play else 1024
  cfg.seed = 0
  return cfg


def trossen_realistic_stack_cubes_state_env_cfg(
  *, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """The state contract: no camera, so no lighting or material randomization.

  ``default`` rather than ``real_texture`` for the same reason Push-T's state
  task uses it -- with ``camera_view=None`` the scene builder derives no visual
  events at all, so a state run pays for none of the appearance DR it could
  never see. The physical randomization (fingertip friction) is unaffected.
  """
  return _env_cfg(rgb=False, scene="default", play=play)


def trossen_realistic_stack_cubes_rgb_env_cfg(
  *,
  camera: CameraView,
  scene: str = "real_texture_red",
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """One RGB Stack-Cubes environment, through exactly one camera.

  ``camera`` selects the wrist D405 or the external one; the two variants are
  identical in everything else, which is what makes them comparable. Neither
  policy ever sees both images.
  """
  return _env_cfg(rgb=True, scene=scene, play=play, camera=camera)


__all__ = [
  "trossen_realistic_stack_cubes_rgb_env_cfg",
  "trossen_realistic_stack_cubes_state_env_cfg",
]
