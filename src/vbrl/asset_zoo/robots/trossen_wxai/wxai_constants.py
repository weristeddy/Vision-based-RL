"""MJLab control metadata for the Trossen WidowX AI asset.

``wxai.xml`` and ``wxai_realistic.xml`` differ only in appearance -- the
realistic one splits five meshes for per-part materials -- so both share these
control constants.
"""

from __future__ import annotations

import math
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

# The deployment camera is a D405 (see camera_mount_d405.stl in both XMLs), and
# every rendered view is matched to it rather than to a datasheet number.
#
# Measured on the hardware, colour stream at 424x240 -- the smallest rgb8 mode
# the D405 offers, and the only one whose 224x224 centre crop needs no resampling
# at all: fx=217.7, fy=217.5, full frame 88.5 x 57.8 degrees. A centred crop of
# side s therefore spans 2*atan(s / 2f), so the 224x224 crop a policy actually
# sees spans 54.5 degrees, and that is what the simulator must render.
#
# This replaces `fovy=87.0`, which was the D405's *horizontal* FOV written into
# the vertical field. That made the rendered view span 1.72x more world than the
# camera does (tan 43.5 / tan 27.2), so every object projected 1.72x too small.
# The 24 retained lift-cube checkpoints and the Push-T generation were trained
# against the old values below; their input distribution does not match this one,
# so their numbers are not comparable across this change.
_D405_COLOUR_FY_PIXELS = 217.5
_POLICY_IMAGE_PIXELS = 224
_D405_CROP_VERTICAL_FOV_DEG = round(
  2.0 * math.degrees(math.atan(_POLICY_IMAGE_PIXELS / (2.0 * _D405_COLOUR_FY_PIXELS))),
  2,
)
_RETIRED_WRIST_FOV_DEG = 87.0  # horizontal FOV, mistakenly used as fovy
_RETIRED_EXTERNAL_FOV_DEG = _D435_RGB_VERTICAL_FOV_DEG  # a D435, not the D405 in use

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
    fovy=_D405_CROP_VERTICAL_FOV_DEG,
    use_shadows=True,
  ),
  "external": RobotCameraDefinition(
    sensor_name="external_cam",
    camera_name="robot/external_cam",
    model_name="external_cam",
    fovy=_D405_CROP_VERTICAL_FOV_DEG,
    use_shadows=True,
  ),
  # Same D405 optics from a near-overhead pose in front of the robot. Under
  # evaluation against `external`, which is still the default for every task.
  "external_front": RobotCameraDefinition(
    sensor_name="external_front_cam",
    camera_name="robot/external_front_cam",
    model_name="external_front_cam",
    fovy=_D405_CROP_VERTICAL_FOV_DEG,
    use_shadows=True,
  ),
  # The front pose tilted back to 45 degrees and brought in to 0.70 m. Measured
  # against `external_front`, this keeps 79% of the object's silhouette while the
  # gripper is on it instead of 37%, and projects the largest object of any pose
  # tried -- a near-overhead camera is occluded by anything above the object.
  # Those percentages were measured at the retired 42.5-degree FOV; the D405 crop
  # is wider, so the framing this pose gives is not the framing they describe.
  "external_tilted": RobotCameraDefinition(
    sensor_name="external_tilted_cam",
    camera_name="robot/external_tilted_cam",
    model_name="external_tilted_cam",
    fovy=_D405_CROP_VERTICAL_FOV_DEG,
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
