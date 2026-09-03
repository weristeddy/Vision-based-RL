from __future__ import annotations

import time
from typing import Any

import numpy as np

# The pose the arm rests at unpowered, so the only pose from which releasing
# torque is safe: idle is not gravity-compensated.
REST_POSE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
# Commanding a joint to its exact limit leaves nothing for tracking error.
LIMIT_MARGIN = 0.02
GRIPPER_MARGIN = 0.002


class TrossenArm:
  """Streamed joint-position control, rate and limit clamped."""

  def __init__(self, config: Any) -> None:
    import trossen_arm

    self._api = trossen_arm
    self._driver = trossen_arm.TrossenArmDriver()
    self._driver.configure(
      getattr(trossen_arm.Model, config.arm_model),
      trossen_arm.StandardEndEffector.wxai_v0_base,
      config.arm_ip,
      True,  # clear a stale fault so a crashed run can reconnect
    )
    self._driver.set_motor_parameters(
      getattr(trossen_arm.StandardMotorParameters, config.motor_parameters)
    )
    self._motion = config.motion

    limits = self._driver.get_joint_limits()
    self._low = np.array([limit.position_min for limit in limits])
    self._high = np.array([limit.position_max for limit in limits])
    self._low[:-1] += LIMIT_MARGIN
    self._high[:-1] -= LIMIT_MARGIN
    self._low[-1] += GRIPPER_MARGIN
    self._high[-1] -= GRIPPER_MARGIN
    self._last_sent: Any = None

  def read(self) -> tuple[Any, Any]:
    """Measured joint positions and velocities, gripper last."""
    return (
      np.asarray(self._driver.get_all_positions(), dtype=np.float64),
      np.asarray(self._driver.get_all_velocities(), dtype=np.float64),
    )

  def move_to(self, pose: Any, *, seconds: float) -> None:
    """Interpolate to an absolute pose and hold it. Blocks until arrived."""
    pose = np.clip(np.asarray(pose, dtype=np.float64), self._low, self._high)
    self._driver.set_all_modes(self._api.Mode.position)
    self._driver.set_all_positions(pose.tolist(), seconds, True)
    self._last_sent = pose

  def command(self, target: Any) -> Any:
    """Step the setpoint towards ``target``, clamped by rate then joint limit.

    The step is measured from the last value sent, not from where the arm
    actually is, so a joint that cannot follow lets the setpoint run ahead of
    it -- bounded only by the joint's own range.
    """
    target = np.asarray(target, dtype=np.float64)
    if self._last_sent is None:
      self._last_sent, _ = self.read()

    max_change = np.full_like(target, self._motion.max_joint_step)
    max_change[-1] = self._motion.max_gripper_step
    sent = np.clip(
      self._last_sent + np.clip(target - self._last_sent, -max_change, max_change),
      self._low,
      self._high,
    )
    # Zero goal time means "this is the setpoint": the servo loop runs far
    # faster than the policy and interpolates for free. A goal time near the
    # control period makes the arm jitter instead, because every command
    # restarts a trajectory the previous one had not finished.
    self._driver.set_all_positions(sent.tolist(), 0.0, False)
    self._last_sent = sent
    return sent

  def park(self, *, seconds: float) -> None:
    """Retrace to the resting pose, then release torque."""
    self.move_to(REST_POSE, seconds=seconds)
    time.sleep(0.3)
    # Mode setting is fire-and-forget, so a lost call leaves joints powered.
    self._driver.set_all_modes(self._api.Mode.idle)
    time.sleep(0.1)
    self._driver.set_all_modes(self._api.Mode.idle)

  def close(self) -> None:
    self._driver.cleanup()


__all__ = ["REST_POSE", "TrossenArm"]
