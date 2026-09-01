"""Camera observation composition shared by RGB task modalities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vbrl.asset_zoo.robots.definition import (
  CameraGeometry,
  CameraView,
)

if TYPE_CHECKING:
  from vbrl.asset_zoo.robots.definition import RobotDefinition


def camera_rgb_uint8(env, sensor_name: str):
  """Return native camera bytes in the BCHW layout expected by policies."""
  rgb = env.scene[sensor_name].data.rgb
  assert rgb is not None, f"Camera {sensor_name!r} has no RGB data."
  return rgb.permute(0, 3, 1, 2)


def add_rgb_camera(
  cfg,
  *,
  robot: RobotDefinition,
  camera_view: CameraView,
  camera_geometry: CameraGeometry = "visual",
  width: int | None = None,
  height: int | None = None,
) -> None:
  """Attach the selected robot camera as a policy observation.

  ``visual`` renders the real meshes and is the default. ``collision`` renders
  the collision proxies instead, and exists only to reproduce the retained
  Lift-Cube checkpoints that were trained against them.
  """
  import mujoco
  from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
  from mjlab.sensor import CameraSensorCfg

  from vbrl.tasks.utils.tabletop_env_cfg import ORIGIN_PLANE_GROUP

  camera = robot.resolve_camera(camera_view)
  geom_groups = robot.resolve_camera_geom_groups(camera_geometry)
  sensor = CameraSensorCfg(
    name=camera.sensor_name,
    camera_name=camera.camera_name,
    height=224 if height is None else height,
    width=224 if width is None else width,
    data_types=("rgb",),
    fovy=camera.fovy,
    enabled_geom_groups=geom_groups,
    use_shadows=camera.use_shadows,
    use_textures=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (sensor,)
  # Draw recorded videos with the geometry the policy is actually fed. Without
  # this the offscreen renderer keeps MuJoCo's default groups 0-2, so a
  # collision-geometry task records footage of the visual meshes instead.
  # The floor is the one addition: it is scenery for whoever is watching, which
  # is why it sits in a group of its own and never in the camera's. The D405 body
  # is in that group too, for the same reason and because the wrist camera is
  # mounted inside it.
  drawn = set(geom_groups) | {ORIGIN_PLANE_GROUP}
  cfg.viewer.geom_group = tuple(
    int(group in drawn) for group in range(mujoco.mjNGROUP)
  )
  cfg.observations["camera"] = ObservationGroupCfg(
    terms={
      f"{camera.sensor_name}_rgb": ObservationTermCfg(
        func=camera_rgb_uint8,
        params={"sensor_name": camera.sensor_name},
      )
    },
    enable_corruption=False,
  )


__all__ = [
  "add_rgb_camera",
  "camera_rgb_uint8",
]
