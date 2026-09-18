from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

GOAL_RANGE = {"x": (0.3, 0.5), "y": (-0.2, 0.2), "z": (0.2, 0.4)}
# Push-T's goal is a pose on the table, not a point above it: the third number is a
# heading in radians rather than a height.
PUSH_T_GOAL_RANGE = {"x": (0.3, 0.5), "y": (-0.2, 0.2), "yaw": (-math.pi, math.pi)}
GOAL_SPACES = {"lift_xyz": GOAL_RANGE, "push_t_xy_yaw": PUSH_T_GOAL_RANGE}

# base_link sits on a 5 mm mount plate, so the table is 5 mm below it, and the
# command's target sits at the T's mid-height 12 mm above the table.
TABLE_Z_BASE_M = -0.005
TARGET_Z_BASE_M = 0.0070


@dataclass(frozen=True)
class Motion:
  action_smoothing: float = 0.25
  max_joint_step: float | None = 0.035
  max_gripper_step: float = 0.005
  max_arm_action: float = 6.0
  home_seconds: float = 3.0

  def validate(self) -> None:
    if not 0.0 < self.action_smoothing <= 1.0:
      raise ValueError(
        f"action_smoothing must be in (0, 1]; got {self.action_smoothing}."
      )
    if self.max_joint_step is not None and self.max_joint_step <= 0.0:
      raise ValueError(
        f"max_joint_step must be positive or null; got {self.max_joint_step}."
      )
    for name in (
      "max_gripper_step",
      "max_arm_action",
      "home_seconds",
    ):
      if getattr(self, name) <= 0.0:
        raise ValueError(f"{name} must be positive; got {getattr(self, name)}.")


@dataclass(frozen=True)
class DeploymentConfig:
  onnx_file: str
  arm_ip: str
  goal: tuple[float, float, float] = (0.35, 0.0, 0.35)
  goal_space: str = "lift_xyz"
  control_hz: float = 50.0
  device: str = "cuda:0"
  camera_fps: int = 60
  camera_width: int = 424
  camera_height: int = 240
  camera_exposure_us: float | None = None
  arm_model: str = "wxai_v0"
  motor_parameters: str = "wxai_v0_20260317"
  motion: Motion = field(default_factory=Motion)

  def validate(self) -> None:
    if not Path(self.onnx_file).expanduser().is_file():
      raise FileNotFoundError(
        f"onnx_file does not exist: {self.onnx_file}. Export one with "
        "vbrl-export-onnx."
      )
    ranges = GOAL_SPACES.get(self.goal_space)
    if ranges is None:
      raise ValueError(
        f"goal_space must be one of {sorted(GOAL_SPACES)}; got "
        f"{self.goal_space!r}."
      )
    for axis, value in zip(ranges, self.goal, strict=True):
      low, high = ranges[axis]
      if not low <= value <= high:
        raise ValueError(
          f"goal {axis}={value} is outside the range the policy trained on, "
          f"[{low}, {high}]. Valid for {self.goal_space}: "
          + ", ".join(f"{name}{bounds}" for name, bounds in ranges.items())
          + "."
        )
    if self.camera_exposure_us is not None and self.camera_exposure_us <= 0.0:
      raise ValueError(
        f"camera_exposure_us must be positive or null; got "
        f"{self.camera_exposure_us}."
      )
    if self.control_hz <= 0.0:
      raise ValueError(f"control_hz must be positive; got {self.control_hz}.")
    self.motion.validate()


def load_config(path: str | Path) -> DeploymentConfig:
  import yaml

  source = Path(path).expanduser()
  document = yaml.safe_load(source.read_text())
  if document.pop("version", None) != 1:
    raise ValueError(f"{source}: expected 'version: 1'.")

  motion_fields = document.pop("motion", None) or {}
  for name, fields, known in (
    ("motion", motion_fields, set(Motion.__dataclass_fields__)),
    ("", document, set(DeploymentConfig.__dataclass_fields__) - {"motion"}),
  ):
    unknown = set(fields) - known
    if unknown:
      where = f"{name} " if name else ""
      raise ValueError(
        f"{source}: unknown {where}fields {sorted(unknown)}; "
        f"valid are {sorted(known)}."
      )

  if "goal" in document:
    document["goal"] = tuple(float(value) for value in document["goal"])
  config = DeploymentConfig(**document, motion=Motion(**motion_fields))
  config.validate()
  return config


__all__ = [
  "GOAL_RANGE",
  "GOAL_SPACES",
  "PUSH_T_GOAL_RANGE",
  "DeploymentConfig",
  "Motion",
  "load_config",
]
