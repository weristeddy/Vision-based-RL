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

  max_joint_step: float = 0.015
  """Largest change in a joint target between consecutive control steps (rad)."""
  max_gripper_step: float = 0.005
  """Same, for the gripper carriage (m)."""
  command_goal_time: float = 0.0
  """Seconds the controller is given to reach each commanded pose.

  Zero, meaning "this is the setpoint, track it", is what streaming position
  control wants: the controller's servo loop runs far faster than 50 Hz and
  interpolates for free. Setting it to one control period was measured to make
  the arm jitter badly -- each command restarts a trajectory the previous one
  had not finished, so the interpolator resets every cycle and the velocity
  jumps. Raise it only for a deliberately sluggish response, never for
  smoothness.
  """
  action_smoothing: float = 1.0
  """Exponential smoothing on the policy output: ``a = k*new + (1-k)*previous``.

  1.0 passes the action through unchanged. This is the conventional sim2real
  jitter fix -- it acts on the policy's *intent* rather than on the arm's
  motion, so unlike a rate clamp it does not leave a standing gap between the
  requested and achieved pose. Lower values also slow the arm, at the cost of
  lagging behind what the policy asked for.
  """
  max_action_magnitude: float = 8.0
  """Abort if the policy emits an action larger than this.

  In simulation this policy peaked at |a| = 3.5. On real sensors it reached
  18.95, which asks for 4.7 rad of joint travel and 0.21 m from a 0.04 m
  gripper -- the mechanism cannot do it, and the controller faults trying. The
  absolute clamp in TrossenArm.command keeps that from reaching the motors, but
  an action this far outside the training range means the observation is out of
  distribution, and continuing just grinds the arm against its limits. Stopping
  is the useful response.
  """
  startup_seconds: float = 3.0
  """Seconds the homing move takes, from wherever the arm is to the sim's default
  pose. Joints 1 and 2 travel ~1.3-1.4 rad from the powered-on zero pose, so this
  is also how fast that first motion is."""

  def validate(self) -> None:
    for name in (
      "max_joint_step",
      "max_gripper_step",
      "startup_seconds",
      "max_action_magnitude",
    ):
      value = getattr(self, name)
      if not value > 0.0:
        raise ValueError(f"safety.{name} must be positive; got {value}.")
    if self.command_goal_time < 0.0:
      raise ValueError(
        f"safety.command_goal_time must not be negative; got "
        f"{self.command_goal_time}."
      )
    if not 0.0 < self.action_smoothing <= 1.0:
      raise ValueError(
        f"safety.action_smoothing must be in (0, 1]; got {self.action_smoothing}. "
        "1.0 disables smoothing."
      )


@dataclass(frozen=True)
class DeploymentConfig:
  task_id: str
  checkpoint_file: str
  arm_ip: str
  arm_model: str = "wxai_v0"
  motor_parameters: str = "wxai_v0_20260317"
  """Which ``trossen_arm.StandardMotorParameters`` set to push at connect.

  Named rather than left to the driver's default because that default moves: the
  1.9.3 -> 1.10.0 bump changed it from ``wxai_v0_20250509`` to this one, 17
  fields apart, without anything in this repo changing. Pinning it means a
  future driver bump cannot silently re-tune the control loop under a recorded
  result. This is the value 1.10.0 already applies, so it changes nothing today.
  """
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
  home_first: bool = False
  """Move to the simulator's default pose before the first step.

  Needed whenever the arm is not already there: ``joint_pos`` is measured
  relative to that pose, so starting from the powered-on zero pose feeds the
  policy proprioception ~1.4 rad from anything it trained on. This is motion,
  so it stays opt-in even under ``dry_run``."""
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
    import trossen_arm  # noqa: PLC0415 - hardware-only import, see hardware.py

    available = [
      name for name in dir(trossen_arm.StandardMotorParameters)
      if not name.startswith("_")
    ]
    if self.motor_parameters not in available:
      raise ValueError(
        f"Unknown motor_parameters {self.motor_parameters!r}; "
        f"valid choices are {sorted(available)}."
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
