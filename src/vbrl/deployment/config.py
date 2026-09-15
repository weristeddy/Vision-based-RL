from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

# The box the lift command sampled its target from, in the robot's base frame.
GOAL_RANGE = {"x": (0.3, 0.5), "y": (-0.2, 0.2), "z": (0.2, 0.4)}
# Push-T's goal is a pose on the table, not a point above it: the third number
# is a heading in radians rather than a height, and z never varies -- the target
# sits at the T's mid-height, 7 mm in the base frame. The x and y windows are
# the command's own `target_position_range`.
PUSH_T_GOAL_RANGE = {"x": (0.3, 0.5), "y": (-0.2, 0.2), "yaw": (-math.pi, math.pi)}
# Which of the two a manifest's `goal` is written in. Declared rather than
# guessed: both are three floats, and reading (x, y, yaw) as (x, y, z) passes a
# height check while aiming the policy at nothing in particular.
GOAL_SPACES = {"lift_xyz": GOAL_RANGE, "push_t_xy_yaw": PUSH_T_GOAL_RANGE}


@dataclass(frozen=True)
class Motion:
  """How fast the policy's output may reach the joints."""

  action_smoothing: float = 0.25
  """``action = k * policy_action + (1 - k) * action``; 1.0 passes it straight through."""
  max_joint_step: float | None = 0.035
  """Largest change in a joint target per step (rad). Times ``control_hz``, the
  arm's speed limit. ``null`` removes the rate clamp entirely, leaving only the
  joint limits -- the policy then runs at the speed it was trained at."""
  max_gripper_step: float = 0.005
  """The same, for the gripper carriage (m)."""
  response_gain: float = 1.0
  """Scale on the commanded joint delta, for relative action terms.

  Sim realizes only 27% of the commanded delta per step -- its position actuator
  lags -- while a real servo reaches the setpoint. A relative action term sets a
  target offset, never a speed, so the same policy output moves the real arm
  several times further. A rate clamp only truncates the large deltas and leaves
  small ones 3.7x too responsive, which transports the object and then overshoots
  every fine correction. This scales both alike."""
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
    if self.max_joint_step is not None and self.max_joint_step <= 0.0:
      raise ValueError(
        f"max_joint_step must be positive or null; got {self.max_joint_step}."
      )
    if not 0.0 < self.response_gain <= 1.0:
      raise ValueError(
        f"response_gain must be in (0, 1]; got {self.response_gain}."
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
  """How to read `goal`: `lift_xyz` is (x, y, z) in metres, `push_t_xy_yaw` is
  (x, y, yaw) with the heading in radians. See GOAL_SPACES."""
  control_hz: float = 50.0
  """Match the training decimation: 0.005 s physics x 4."""
  device: str = "cuda:0"
  camera_fps: int = 60
  camera_width: int = 424
  camera_height: int = 240
  """424x240 because its 224x224 centre crop needs no resampling."""
  camera_exposure_us: float | None = None
  """Fixed colour exposure in microseconds, or ``null`` for auto.

  Set this when the scene clips. Measured on run1.npz under the SDK default,
  42% of the pixels the policy saw were pinned at 255 and the median was 253 --
  the tabletop carried no texture at all, against 1203 photographed tables in
  training. Auto-exposure meters the whole frame, and a pale tabletop filling
  it drives the sensor to saturation."""
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


__all__ = [
  "GOAL_RANGE",
  "GOAL_SPACES",
  "PUSH_T_GOAL_RANGE",
  "DeploymentConfig",
  "Motion",
  "load_config",
]
