"""Move the arm to the pose every observation is measured against.

``joint_pos`` reaching the policy is ``joint_pos_rel``: the measured position
minus the *default* pose from the MJCF. Powered on, this arm sits at its own zero
-- all joints near 0.0 -- while the sim's default puts joint 1 at 1.33 rad and
joint 2 at 1.42. Starting a rollout there would feed the policy a proprioception
vector some 1.4 rad from anything it saw in training, on its very first step.

So homing is not a convenience: it is what makes the observation mean what the
policy thinks it means. The pose comes from the robot definition rather than a
literal here, so it cannot drift from what the simulator uses.
"""

from __future__ import annotations

from typing import Any

# Both Trossen definitions share one home pose (`_HOME_JOINT_POS` in
# wxai_constants), so either resolves the same numbers. The gripper's two
# carriage joints mirror one another and the hardware reports one, so the arm's
# seventh value is the left carriage.
_ARM_JOINT_ORDER = (
  "joint_0",
  "joint_1",
  "joint_2",
  "joint_3",
  "joint_4",
  "joint_5",
  "left_carriage_joint",
)


def home_pose() -> Any:
  """The 7 hardware joint targets matching the simulator's default pose."""
  import numpy as np

  from vbrl.asset_zoo.robots.trossen_wxai import make_wxai

  home = make_wxai().home_joint_pos
  missing = [name for name in _ARM_JOINT_ORDER if name not in home]
  if missing:
    raise RuntimeError(
      f"The robot definition has no home position for {missing}; "
      f"it defines {sorted(home)}."
    )
  return np.array([float(home[name]) for name in _ARM_JOINT_ORDER], dtype=np.float64)


def move_to_home(config: Any, *, duration: float | None = None) -> int:
  """Ease the arm to the simulator's default pose and hold it there."""
  import numpy as np

  from vbrl.deployment.hardware import TrossenArm

  target = home_pose()
  seconds = config.safety.startup_seconds if duration is None else duration

  arm = TrossenArm(
    ip=config.arm_ip,
    model=config.arm_model,
    motor_parameters=config.motor_parameters,
    max_joint_step=config.safety.max_joint_step,
    max_gripper_step=config.safety.max_gripper_step,
    command_goal_time=config.safety.command_goal_time,
  )
  try:
    measured, _ = arm.read()
    travel = np.abs(target - measured)
    print(f"Arm        {config.arm_model} at {config.arm_ip}")
    print(f"  measured {np.round(measured, 4).tolist()}")
    print(f"  home     {np.round(target, 4).tolist()}")
    print(f"  travel   {np.round(travel, 4).tolist()}  (max {travel.max():.3f})")
    print(f"  moving over {seconds:.1f} s, then holding position")

    arm.move_to(target, duration=seconds, blocking=True)

    reached, velocity = arm.read()
    error = np.abs(target - reached)
    print(f"  reached  {np.round(reached, 4).tolist()}")
    print(f"  error    {np.round(error, 4).tolist()}  (max {error.max():.4f})")
    print(f"  velocity {np.round(velocity, 4).tolist()}")
    # The controller keeps executing its last position command, so the arm holds
    # here without further input.
    print("  holding position. Ctrl-C to retrace to rest and release.")
    parked = False
    try:
      import time

      while True:
        time.sleep(0.5)
    except KeyboardInterrupt:
      # Releasing torque here would drop the arm from 1.33 rad up: idle is not
      # gravity-compensated. Retrace the way we came instead.
      print(f"\n  retracing to rest over {seconds:.1f} s, then releasing")
      released = arm.park(duration=seconds)
      parked = True
      print(
        "  at rest, torque released" if released
        else "  at rest, but the joints are STILL POWERED (see warning above)"
      )
  finally:
    # Already parked on the Ctrl-C path; on any other exit bring it down too.
    arm.close(park=not parked)
  return 0


def park_arm(config: Any, *, duration: float | None = None) -> int:
  """Bring the arm down to its resting pose and release torque.

  For when something left it holding a raised pose -- a crashed run, or an exit
  path that held position instead of parking. Safe to run from anywhere in the
  workspace: the controller interpolates to rest over ``duration``.
  """
  import numpy as np

  from vbrl.deployment.hardware import TrossenArm

  seconds = config.safety.startup_seconds if duration is None else duration
  arm = TrossenArm(
    ip=config.arm_ip,
    model=config.arm_model,
    motor_parameters=config.motor_parameters,
    max_joint_step=config.safety.max_joint_step,
    max_gripper_step=config.safety.max_gripper_step,
    command_goal_time=config.safety.command_goal_time,
  )
  try:
    measured, _ = arm.read()
    rest = np.asarray(arm.REST_POSE, dtype=np.float64)
    print(f"Arm        {config.arm_model} at {config.arm_ip}")
    print(f"  measured {np.round(measured, 4).tolist()}")
    print(f"  travel   {float(np.abs(rest - measured).max()):.3f} rad to rest")
    print(f"  parking over {seconds:.1f} s, then releasing torque")
    released = arm.park(duration=seconds)
    reached, _ = arm.read()
    print(f"  reached  {np.round(reached, 4).tolist()}")
    print(f"  modes    {arm.modes()}")
    print(
      "  at rest, torque released" if released
      else "  at rest, but the joints are STILL POWERED (see warning above)"
    )
  finally:
    arm.close(park=False)
  return 0


__all__ = ["home_pose", "move_to_home", "park_arm"]
