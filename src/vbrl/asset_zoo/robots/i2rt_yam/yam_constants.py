"""MJLab control and collision metadata for the I2RT YAM asset."""

from __future__ import annotations

from pathlib import Path

from vbrl.asset_zoo.robots.definition import (
  RobotCameraDefinition,
  RobotDefinition,
)


XMLS_DIR = Path(__file__).resolve().parent / "xmls"
YAM_XML = XMLS_DIR / "yam.xml"

_ACTION_SCALE = {
  "joint1": 0.3599426554453139,
  "joint2": 0.1597918534090456,
  "joint3": 0.1904427157497401,
  "joint4": 0.5250193985879242,
  "joint5": 1.7347616639294616,
  "joint6": 5.520026131457555,
  "left_finger": 0.08657395590844981,
}
_HOME_JOINT_POS = {
  "joint2": 1.047,
  "joint3": 1.05,
  "joint4": -0.9,
  "left_finger": 0.01875,
  "right_finger": -0.01875,
}
_ACTUATORS = (
  ("joint1", 0.032, 19.447542251806023, 6.190344960903768, 28.0),
  ("joint2", 0.032, 43.80698922166542, 13.944197753601417, 28.0),
  ("joint3", 0.032, 36.756459665271045, 11.699944492905928, 28.0),
  ("joint4", 0.0018, 4.761728817494976, 1.515705357978224, 10.0),
  ("joint5", 0.0018, 1.4411201561470834, 0.458722792893456, 10.0),
  ("joint6", 0.0018, 0.45289640673129905, 0.144161403683808, 10.0),
  (
    "left_finger",
    2.603054949414799,
    109.81411388402572,
    69.90983618559427,
    38.02816901408451,
  ),
)
_FINGERTIP_PATTERN = r"[lr]f_down(6|7|8|9|10|11)_collision"


def _articulation(_enable_delay: bool):
  from mjlab.actuator import BuiltinPositionActuatorCfg
  from mjlab.entity import EntityArticulationInfoCfg

  return EntityArticulationInfoCfg(
    actuators=tuple(
      BuiltinPositionActuatorCfg(
        target_names_expr=(name,),
        armature=armature,
        stiffness=stiffness,
        damping=damping,
        effort_limit=effort,
      )
      for name, armature, stiffness, damping, effort in _ACTUATORS
    ),
    soft_joint_pos_limit_factor=0.9,
  )


def _collisions():
  from mjlab.utils.spec_config import CollisionCfg

  return (
    CollisionCfg(
      geom_names_expr=(".*_collision",),
      contype={
        "(link6|[lr]f)_.*_collision": 1,
        ".*_collision": 0,
      },
      conaffinity={
        "(link6|[lr]f)_.*_collision": 1,
        ".*_collision": 0,
      },
      condim={_FINGERTIP_PATTERN: 6, ".*_collision": 3},
      friction={_FINGERTIP_PATTERN: (1, 5e-3, 5e-4), ".*_collision": (0.6,)},
      solref={_FINGERTIP_PATTERN: (0.01, 1)},
      # MJLab 1.6 treats contype/conaffinity/condim/priority as the collision
      # structure this config fully owns, so every dict must cover every
      # matched geom. Only the fingertips are prioritized; the rest keep 0.
      priority={_FINGERTIP_PATTERN: 1, ".*_collision": 0},
    ),
  )


def make_yam() -> RobotDefinition:
  """Return a fresh YAM definition."""
  return RobotDefinition(
    name="yam",
    xml_path=YAM_XML,
    articulation_factory=_articulation,
    collision_factory=_collisions,
    home_joint_pos=dict(_HOME_JOINT_POS),
    home_position=(0.0, 0.0, 0.01),
    action_scale=dict(_ACTION_SCALE),
    arm_action_scale={
      joint: scale
      for joint, scale in _ACTION_SCALE.items()
      if joint.startswith("joint")
    },
    closed_gripper_joint_pos={
      "left_finger": 0.0,
      "right_finger": 0.0,
    },
    ee_site="grasp_site",
    fingertip_geom_pattern=_FINGERTIP_PATTERN,
    collision_body_pattern="link_6",
    viewer_body="arm",
    cameras={
      "wrist": RobotCameraDefinition(
        sensor_name="camera_d405",
        camera_name="robot/camera_d405",
        model_name="camera_d405",
      )
    },
  )


__all__ = ["YAM_XML", "make_yam"]
