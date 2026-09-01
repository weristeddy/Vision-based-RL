"""The deployment manifest: one YAML per robot setup.

Runs are YAML in this repo, and a deployment is a run. The task ID still fixes
the architecture, observations, camera, and action scaling -- exactly as it does
for training, evaluation, and analysis -- so this file adds only what a real
robot needs and the simulator supplied for free: the arm's address, the camera
serial, the lift target, and the safety envelope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# The training command sampled the lift target uniformly from this box, in the
# robot's base frame (metres). A target outside it is out of distribution, which
# is a silent behavioural change rather than an error, so it is refused.
GOAL_RANGE = {
  "x": (0.3, 0.5),
  "y": (-0.2, 0.2),
  "z": (0.2, 0.4),
}
# Centre of that box, and mjlab's own ``difficulty="fixed"`` value.
DEFAULT_GOAL = (0.4, 0.0, 0.3)

# Training stepped the policy every ``decimation`` physics steps of 0.005 s.
DEFAULT_CONTROL_HZ = 50.0


@dataclass(frozen=True)
class SafetyLimits:
  """Bounds applied to every command before it reaches the arm.

  The task's action term declares ``clip: None``, because a simulated arm may
  be commanded anywhere and simply fails to reach it. On hardware an
  out-of-range target is a collision, so deployment adds the clamps training
  never needed.
  """

  max_joint_step: float = 0.05
  """Largest change in a joint target between consecutive control steps (rad)."""
  max_gripper_step: float = 0.005
  """Same, for the gripper carriage (m)."""
  startup_seconds: float = 2.0
  """Time spent easing from the arm's current pose to the policy's first target."""

  def validate(self) -> None:
    for name in ("max_joint_step", "max_gripper_step", "startup_seconds"):
      value = getattr(self, name)
      if not value > 0.0:
        raise ValueError(f"safety.{name} must be positive; got {value}.")


@dataclass(frozen=True)
class DeploymentConfig:
  task_id: str
  checkpoint_file: str
  arm_ip: str
  arm_model: str = "wxai_v0"
  goal: tuple[float, float, float] = DEFAULT_GOAL
  control_hz: float = DEFAULT_CONTROL_HZ
  device: str = "cuda:0"
  camera_serial: str | None = None
  """RealSense serial; ``None`` takes the first device found."""
  camera_fps: int = 60
  """Sensor frame rate. 60 keeps a fresh frame for every 50 Hz control cycle; a
  D405 sustains it at 424x240. Below ``control_hz`` the newest frame is reused
  for consecutive steps rather than slowing the loop -- see the note in
  loop.run."""
  max_steps: int | None = None
  """Stop after this many control steps; ``None`` runs until interrupted."""
  dry_run: bool = False
  """Read sensors and evaluate the policy, but send no command to the arm."""
  safety: SafetyLimits = field(default_factory=SafetyLimits)

  def validate(self) -> None:
    if not Path(self.checkpoint_file).expanduser().is_file():
      raise FileNotFoundError(f"checkpoint_file does not exist: {self.checkpoint_file}")
    if len(self.goal) != 3:
      raise ValueError(f"goal must be three numbers (x, y, z); got {self.goal!r}.")
    for axis, value in zip("xyz", self.goal, strict=True):
      low, high = GOAL_RANGE[axis]
      if not low <= value <= high:
        raise ValueError(
          f"goal {axis}={value} is outside the range the policy trained on, "
          f"[{low}, {high}]. The lift command sampled uniformly from that box, "
          f"so a target beyond it is out of distribution. Valid ranges: "
          f"x{GOAL_RANGE['x']}, y{GOAL_RANGE['y']}, z{GOAL_RANGE['z']}."
        )
    if not self.control_hz > 0.0:
      raise ValueError(f"control_hz must be positive; got {self.control_hz}.")
    if self.max_steps is not None and self.max_steps <= 0:
      raise ValueError(f"max_steps must be positive when set; got {self.max_steps}.")
    self.safety.validate()


def load_config(path: str | Path) -> DeploymentConfig:
  """Read a deployment manifest, refusing unknown fields."""
  import yaml

  source = Path(path).expanduser()
  document = yaml.safe_load(source.read_text())
  if not isinstance(document, dict):
    raise ValueError(f"{source}: expected a mapping at the top level.")

  version = document.pop("version", None)
  if version != 1:
    raise ValueError(f"{source}: unsupported version {version!r}; expected 1.")

  safety_fields = document.pop("safety", None) or {}
  known_safety = {f for f in SafetyLimits.__dataclass_fields__}
  unknown = set(safety_fields) - known_safety
  if unknown:
    raise ValueError(
      f"{source}: unknown safety fields {sorted(unknown)}; "
      f"valid fields are {sorted(known_safety)}."
    )

  known = {f for f in DeploymentConfig.__dataclass_fields__} - {"safety"}
  unknown = set(document) - known
  if unknown:
    raise ValueError(
      f"{source}: unknown fields {sorted(unknown)}; valid fields are {sorted(known)}."
    )
  if "goal" in document:
    document["goal"] = tuple(float(v) for v in document["goal"])

  config = DeploymentConfig(**document, safety=SafetyLimits(**safety_fields))
  config.validate()
  return config


__all__ = [
  "DEFAULT_CONTROL_HZ",
  "DEFAULT_GOAL",
  "GOAL_RANGE",
  "DeploymentConfig",
  "SafetyLimits",
  "load_config",
]
