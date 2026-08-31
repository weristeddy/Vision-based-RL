"""MJLab control metadata for the Trossen WidowX AI asset.

``wxai.xml`` and ``wxai_realistic.xml`` differ only in appearance -- the
realistic one splits five meshes for per-part materials -- so both share these
control constants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vbrl.asset_zoo.robots.definition import (
  RobotCameraDefinition,
  RobotDefinition,
)


XMLS_DIR = Path(__file__).resolve().parent / "xmls"
WXAI_XML = XMLS_DIR / "wxai.xml"
WXAI_REALISTIC_XML = XMLS_DIR / "wxai_realistic.xml"

_D435_RGB_VERTICAL_FOV_DEG = 42.5

_ACTION_SCALE = {
  "joint_0": 0.25,
  "joint_1": 0.25,
  "joint_2": 0.25,
  "joint_3": 0.25,
  "joint_4": 0.25,
  "joint_5": 0.25,
  "left_carriage_joint": 0.01,
}
_HOME_JOINT_POS = {
  "joint_0": 0.0,
  "joint_1": 1.33,
  "joint_2": 1.42,
  "joint_3": -1.30,
  "joint_4": 0.0,
  "joint_5": 0.0,
  "right_carriage_joint": 0.022,
  "left_carriage_joint": 0.022,
}
_CAMERAS = {
  "wrist": RobotCameraDefinition(
    sensor_name="cam",
    camera_name="robot/cam",
    model_name="cam",
    fovy=87.0,
    use_shadows=True,
  ),
  "external": RobotCameraDefinition(
    sensor_name="external_cam",
    camera_name="robot/external_cam",
    model_name="external_cam",
    fovy=_D435_RGB_VERTICAL_FOV_DEG,
    use_shadows=True,
  ),
  # Same D435 optics from a near-overhead pose in front of the robot. Under
  # evaluation against `external`, which is still the default for every task.
  "external_front": RobotCameraDefinition(
    sensor_name="external_front_cam",
    camera_name="robot/external_front_cam",
    model_name="external_front_cam",
    fovy=_D435_RGB_VERTICAL_FOV_DEG,
    use_shadows=True,
  ),
  # The front pose tilted back to 45 degrees and brought in to 0.70 m. Measured
  # against `external_front`, this keeps 79% of the object's silhouette while the
  # gripper is on it instead of 37%, and projects the largest object of any pose
  # tried -- a near-overhead camera is occluded by anything above the object.
  "external_tilted": RobotCameraDefinition(
    sensor_name="external_tilted_cam",
    camera_name="robot/external_tilted_cam",
    model_name="external_tilted_cam",
    fovy=_D435_RGB_VERTICAL_FOV_DEG,
    use_shadows=True,
  ),
}


def _no_collisions() -> tuple[Any, ...]:
  return ()


def _articulation(enable_delay: bool):
  from mjlab.actuator import XmlActuatorCfg
  from mjlab.entity import EntityArticulationInfoCfg

  delay_max_lag = 1 if enable_delay else 0
  return EntityArticulationInfoCfg(
    actuators=(
      XmlActuatorCfg(
        target_names_expr=(r"joint_[0-5]",),
        delay_max_lag=delay_max_lag,
      ),
      XmlActuatorCfg(
        target_names_expr=("left_carriage_joint",),
        delay_max_lag=delay_max_lag,
      ),
    ),
    soft_joint_pos_limit_factor=0.9,
  )


def _definition(name: str, xml_path: Path) -> RobotDefinition:
  return RobotDefinition(
    name=name,
    xml_path=xml_path,
    articulation_factory=_articulation,
    collision_factory=_no_collisions,
    home_joint_pos=dict(_HOME_JOINT_POS),
    action_scale=dict(_ACTION_SCALE),
    arm_action_scale={
      joint: scale
      for joint, scale in _ACTION_SCALE.items()
      if joint.startswith("joint_")
    },
    closed_gripper_joint_pos={
      "right_carriage_joint": 0.0,
      "left_carriage_joint": 0.0,
    },
    ee_site="ee_site",
    fingertip_geom_pattern=r"(left|right)_finger_pad_[0-2]_collision",
    collision_body_pattern="link_6",
    viewer_body="link_6",
    cameras=dict(_CAMERAS),
  )


def make_wxai() -> RobotDefinition:
  """Return a fresh standard Trossen definition."""
  return _definition("trossen", WXAI_XML)


def make_wxai_realistic() -> RobotDefinition:
  """Return a fresh realistic-material Trossen definition."""
  return _definition("trossen_realistic", WXAI_REALISTIC_XML)


__all__ = [
  "WXAI_REALISTIC_XML",
  "WXAI_XML",
  "make_wxai",
  "make_wxai_realistic",
]
