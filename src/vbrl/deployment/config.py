from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# The box the lift command sampled its target from, in the robot's base frame.
GOAL_RANGE = {"x": (0.3, 0.5), "y": (-0.2, 0.2), "z": (0.2, 0.4)}


@dataclass(frozen=True)
class Motion:
  """How fast the policy's output may reach the joints."""

  action_smoothing: float = 0.25
  """``action = k * policy_action + (1 - k) * action``; 1.0 passes it straight through."""
  max_joint_step: float = 0.035
  """Largest change in a joint target per step (rad). Times ``control_hz``, the
  arm's speed limit."""
  max_gripper_step: float = 0.005
  """The same, for the gripper carriage (m)."""
  max_arm_action: float = 6.0
  """Abort above this on any of the 6 arm channels. The gripper is excluded: it
  is scaled 0.01 m per unit against a 0.04 m mechanism, so policies drive it
  well past the stop and simulation clamps it harmlessly."""
  home_seconds: float = 3.0
  """Duration of the move to home, and of parking afterwards."""

  def validate(self) -> None:
    if not 0.0 < self.action_smoothing <= 1.0:
      raise ValueError(
        f"action_smoothing must be in (0, 1]; got {self.action_smoothing}."
      )
    for name in (
      "max_joint_step",
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
  control_hz: float = 50.0
  """Match the training decimation: 0.005 s physics x 4."""
  device: str = "cuda:0"
  camera_fps: int = 60
  camera_width: int = 424
  camera_height: int = 240
  """424x240 because its 224x224 centre crop needs no resampling."""
  arm_model: str = "wxai_v0"
  motor_parameters: str = "wxai_v0_20260317"
  """Pinned: the driver's default moved between 1.9.3 and 1.10.0."""
  motion: Motion = field(default_factory=Motion)

  def validate(self) -> None:
    if not Path(self.onnx_file).expanduser().is_file():
      raise FileNotFoundError(
        f"onnx_file does not exist: {self.onnx_file}. Export one with "
        "vbrl-export-onnx."
      )
    for axis, value in zip("xyz", self.goal, strict=True):
      low, high = GOAL_RANGE[axis]
      if not low <= value <= high:
        raise ValueError(
          f"goal {axis}={value} is outside the range the policy trained on, "
          f"[{low}, {high}]. Valid: x{GOAL_RANGE['x']}, y{GOAL_RANGE['y']}, "
          f"z{GOAL_RANGE['z']}."
        )
    if self.control_hz <= 0.0:
      raise ValueError(f"control_hz must be positive; got {self.control_hz}.")
    self.motion.validate()


def load_config(path: str | Path) -> DeploymentConfig:
  """Read a manifest, refusing unknown fields."""
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


__all__ = ["GOAL_RANGE", "DeploymentConfig", "Motion", "load_config"]
