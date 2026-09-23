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
WXAI_IDENTIFIED_XML = XMLS_DIR / "wxai_identified.xml"

# The arm bolts to a 190 x 80 x 5 mm plate on the tabletop, measured off the real rig
# (nothing about it is published).
MOUNT_PLATE_THICKNESS_M = 0.005
FINGER_PAD_PATTERN = r"(left|right)_finger_pad_[0-2]_collision"
IDENTIFIED_PAD_PATTERN = r"gripper_(left|right)_pad"


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
    use_shadows=True,
  ),
  "external": RobotCameraDefinition(
    sensor_name="external_cam",
    camera_name="robot/external_cam",
    model_name="external_cam",
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


def _definition(
  name: str, xml_path: Path, fingertip_geom_pattern: str = FINGER_PAD_PATTERN
) -> RobotDefinition:
  return RobotDefinition(
    name=name,
    xml_path=xml_path,
    home_position=(0.0, 0.0, MOUNT_PLATE_THICKNESS_M),
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
    fingertip_geom_pattern=fingertip_geom_pattern,
    collision_body_pattern="link_6",
    viewer_body="link_6",
    cameras=dict(_CAMERAS),
  )


def make_wxai() -> RobotDefinition:
  return _definition("trossen", WXAI_XML)


def make_wxai_realistic() -> RobotDefinition:
  return _definition("trossen_realistic", WXAI_REALISTIC_XML)


def make_wxai_identified() -> RobotDefinition:
  return _definition(
    "trossen_identified", WXAI_IDENTIFIED_XML, IDENTIFIED_PAD_PATTERN
  )


__all__ = [
  "WXAI_IDENTIFIED_XML",
  "WXAI_REALISTIC_XML",
  "WXAI_XML",
  "make_wxai",
  "make_wxai_identified",
  "make_wxai_realistic",
]
