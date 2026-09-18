from __future__ import annotations

from typing import TYPE_CHECKING

from vbrl.asset_zoo.robots.definition import (
  CameraGeometry,
  CameraView,
)

if TYPE_CHECKING:
  from vbrl.asset_zoo.robots.definition import RobotDefinition


def camera_rgb_uint8(env, sensor_name: str):
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
  # Record the geometry the policy is fed: without this the offscreen renderer keeps
  # MuJoCo's default groups 0-2, so a collision-geometry task films the visual meshes.
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
