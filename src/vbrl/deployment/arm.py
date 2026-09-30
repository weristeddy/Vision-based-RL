from __future__ import annotations

import time
from typing import Any

import numpy as np

from vbrl.asset_zoo.robots.trossen_wxai import make_wxai

# The pose the arm rests at unpowered, so the only pose from which releasing
# torque is safe: idle is not gravity-compensated.
REST_POSE = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
UPRIGHT_POSE = tuple(make_wxai().home_joint_pos[f"joint_{i}"] for i in range(6))
LIMIT_MARGIN = 0.02
GRIPPER_MARGIN = 0.002
D405_MASS = 0.11718
D405_COM = np.array([0.052321, 0.008151, 0.052651])
D405_INERTIA = np.array([
  [5.519e-05, -1.27e-06, -7.39e-06],
  [-1.27e-06, 4.153e-05, -3.45e-06],
  [-7.39e-06, -3.45e-06, 4.694e-05],
])


def with_d405(api: Any, standard: Any) -> Any:
  palm_com = np.array(standard.palm.origin_xyz)
  palm_mass = standard.palm.mass
  mass = palm_mass + D405_MASS
  com = (palm_mass * palm_com + D405_MASS * D405_COM) / mass
  inertia = np.array(standard.palm.inertia).reshape(3, 3) + D405_INERTIA
  for part_mass, part_com in ((palm_mass, palm_com), (D405_MASS, D405_COM)):
    r = part_com - com
    inertia += part_mass * (r @ r * np.eye(3) - np.outer(r, r))
  palm = api.Link()
  palm.mass = mass
  palm.origin_xyz = com.tolist()
  palm.origin_rpy = list(standard.palm.origin_rpy)
  palm.inertia = inertia.ravel().tolist()
  end_effector = api.EndEffector()
  for field in ("finger_left", "finger_right", "offset_finger_left",
                "offset_finger_right", "pitch_circle_radius", "t_flange_tool"):
    setattr(end_effector, field, getattr(standard, field))
  end_effector.palm = palm
  return end_effector


def with_position_kp(api: Any, standard: Any, position_kp: Any) -> Any:
  def pid(source: Any, kp: float | None = None) -> Any:
    target = api.PIDParameter()
    target.kp = source.kp if kp is None else kp
    target.ki, target.kd, target.imax = source.ki, source.kd, source.imax
    return target

  parameters = []
  for joint, modes in enumerate(standard):
    joint_modes = {}
    for mode, source in modes.items():
      motor = api.MotorParameter()
      override = mode == api.Mode.position and joint < len(position_kp)
      motor.position = pid(source.position, position_kp[joint] if override else None)
      motor.velocity = pid(source.velocity)
      joint_modes[mode] = motor
    parameters.append(joint_modes)
  return parameters


class TrossenArm:
  def __init__(self, config: Any) -> None:
    import trossen_arm

    self._api = trossen_arm
    self._driver = trossen_arm.TrossenArmDriver()
    self._driver.configure(
      getattr(trossen_arm.Model, config.arm_model),
      with_d405(trossen_arm, trossen_arm.StandardEndEffector.wxai_v0_base),
      config.arm_ip,
      True,  # clear a stale fault so a crashed run can reconnect
    )
    motor_parameters = getattr(
      trossen_arm.StandardMotorParameters, config.motor_parameters
    )
    if config.position_kp is not None:
      motor_parameters = with_position_kp(
        trossen_arm, motor_parameters, config.position_kp
      )
    self._driver.set_motor_parameters(motor_parameters)
    self._motion = config.motion
    self._goal_time = 1.0 / config.control_hz

    limits = self._driver.get_joint_limits()
    self._low = np.array([limit.position_min for limit in limits])
    self._high = np.array([limit.position_max for limit in limits])
    self._low[:-1] += LIMIT_MARGIN
    self._high[:-1] -= LIMIT_MARGIN
    self._low[-1] += GRIPPER_MARGIN
    self._high[-1] -= GRIPPER_MARGIN
    self._last_sent: Any = None

  def read(self) -> tuple[Any, Any]:
    return (
      np.asarray(self._driver.get_all_positions(), dtype=np.float64),
      np.asarray(self._driver.get_all_velocities(), dtype=np.float64),
    )

  def output(self) -> Any:
    return self._driver.get_robot_output()

  def move_to(self, pose: Any, *, seconds: float) -> None:
    pose = np.clip(np.asarray(pose, dtype=np.float64), self._low, self._high)
    self._driver.set_all_modes(self._api.Mode.position)
    self._driver.set_all_positions(pose.tolist(), seconds, True)
    self._last_sent = pose

  # The step is measured from the last value sent, not from where the arm is, so
  # a joint that cannot follow lets the setpoint run ahead of it.
  def command(self, target: Any) -> Any:
    target = np.asarray(target, dtype=np.float64)
    if self._last_sent is None:
      self._last_sent, _ = self.read()

    if self._motion.max_joint_step is None:
      sent = np.clip(target, self._low, self._high)
    else:
      max_change = np.full_like(target, self._motion.max_joint_step)
      max_change[-1] = self._motion.max_gripper_step
      sent = np.clip(
        self._last_sent + np.clip(target - self._last_sent, -max_change, max_change),
        self._low,
        self._high,
      )
    self._driver.set_all_positions(sent.tolist(), self._goal_time, False)
    self._last_sent = sent
    return sent

  def lift(self, *, seconds: float) -> None:
    self.move_to((*UPRIGHT_POSE, self.read()[0][-1]), seconds=seconds)

  def park(self, *, seconds: float) -> None:
    self.lift(seconds=seconds)
    self.move_to(REST_POSE, seconds=seconds)
    time.sleep(0.3)
    self._driver.set_all_modes(self._api.Mode.idle)
    time.sleep(0.1)
    self._driver.set_all_modes(self._api.Mode.idle)

  def close(self) -> None:
    self._driver.cleanup()


__all__ = ["REST_POSE", "TrossenArm", "UPRIGHT_POSE"]
