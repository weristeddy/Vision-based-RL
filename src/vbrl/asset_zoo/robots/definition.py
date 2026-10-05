from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

ArticulationFactory = Callable[[bool], Any]
CollisionFactory = Callable[[], tuple[Any, ...]]
CameraGeometry = Literal["collision", "visual"]
CameraView = Literal["wrist", "external"]

DEFAULT_CAMERA_VIEW: CameraView = "wrist"

# MuJoCo geom-group conventions shared by every robot in this repository: group 2 holds
# visual meshes, group 3 holds collision proxies, group 0 holds scene geometry.
CAMERA_GEOM_GROUPS: Mapping[CameraGeometry, tuple[int, ...]] = {
  "collision": (0, 3),
  "visual": (0, 2),
}


def _load_mjcf(path: str) -> Any:
  import mujoco

  return mujoco.MjSpec.from_file(path)


@dataclass(frozen=True)
class RobotCameraDefinition:
  sensor_name: str
  camera_name: str
  model_name: str
  fovy: float | None = None
  use_shadows: bool = False


@dataclass(frozen=True)
class RobotDefinition:
  name: str
  xml_path: Path
  articulation_factory: ArticulationFactory
  collision_factory: CollisionFactory
  home_joint_pos: Mapping[str, float]
  action_scale: Mapping[str, float]
  arm_action_scale: Mapping[str, float]
  closed_gripper_joint_pos: Mapping[str, float]
  ee_site: str
  fingertip_geom_pattern: str
  collision_body_pattern: str
  viewer_body: str
  cameras: Mapping[CameraView, RobotCameraDefinition]
  home_position: tuple[float, float, float] = (0.0, 0.0, 0.0)

  def make_entity_cfg(
    self,
    *,
    action_delay: bool = False,
    fixed_closed_gripper: bool = False,
  ) -> Any:
    if not self.xml_path.is_file():
      raise FileNotFoundError(f"Missing robot MJCF: {self.xml_path}")

    from mjlab.entity import EntityCfg

    joint_pos = dict(self.home_joint_pos)
    if fixed_closed_gripper:
      joint_pos.update(self.closed_gripper_joint_pos)
    return EntityCfg(
      init_state=EntityCfg.InitialStateCfg(
        pos=self.home_position,
        joint_pos=joint_pos,
        joint_vel={".*": 0.0},
      ),
      spec_fn=partial(_load_mjcf, str(self.xml_path)),
      articulation=self.articulation_factory(action_delay),
      collisions=self.collision_factory(),
    )

  @property
  def arm_actuator_names(self) -> tuple[str, ...]:
    return tuple(self.arm_action_scale)

  def resolve_camera(
    self,
    camera_view: CameraView | None = None,
  ) -> RobotCameraDefinition:
    selected = DEFAULT_CAMERA_VIEW if camera_view is None else camera_view
    try:
      return self.cameras[selected]
    except KeyError as exc:
      choices = ", ".join(sorted(self.cameras))
      raise ValueError(
        f"Robot {self.name!r} has no {selected!r} camera; choose from: {choices}."
      ) from exc

  def resolve_camera_geom_groups(
    self,
    camera_geometry: CameraGeometry,
  ) -> tuple[int, ...]:
    try:
      return CAMERA_GEOM_GROUPS[camera_geometry]
    except KeyError as exc:
      raise ValueError(
        f"Unknown camera geometry {camera_geometry!r}; choose 'collision' or 'visual'."
      ) from exc


__all__ = [
  "CAMERA_GEOM_GROUPS",
  "CameraGeometry",
  "CameraView",
  "DEFAULT_CAMERA_VIEW",
  "RobotCameraDefinition",
  "RobotDefinition",
]
