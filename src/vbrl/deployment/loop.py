from __future__ import annotations

import time
from typing import Any

import numpy as np

from vbrl.deployment.arm import TrossenArm
from vbrl.deployment.camera import RealSenseCamera
from vbrl.deployment.policy import load_policy

# Joints settle within a few mrad, so this catches a wrong starting pose
# without tripping on servo error.
HOME_TOLERANCE = 0.10
# What the task scores: LiftingCommandCfg.success_threshold, on the cube.
SUCCESS_THRESHOLD = 0.05
# Carriage travel with nothing between the fingers; a cube blocks it near 0.019.
GRIPPER_WHEN_EMPTY = 0.004


def park(config: Any) -> int:
  """Bring the arm down to rest and release torque."""
  arm = TrossenArm(config)
  arm.park(seconds=config.motion.home_seconds)
  arm.close()
  print("At rest, torque released.")
  return 0


def run(config: Any, *, dry_run: bool = False, max_steps: int | None = None) -> int:
  """Home the arm, then drive it with the policy until stopped."""
  motion = config.motion
  policy = load_policy(config)
  print(f"Policy    {config.onnx_file} on {policy.provider}")
  print(
    f"          from {policy.metadata.source_run}, "
    f"obs {policy.metadata.observation_terms}"
  )

  arm = TrossenArm(config)
  camera = RealSenseCamera(config) if policy.metadata.needs_camera else None
  home = policy.metadata.home_pose
  print(f"Goal      {tuple(config.goal)} in the base frame")
  print(f"Homing    {motion.home_seconds:.1f} s")
  arm.move_to(home, seconds=motion.home_seconds)

  # Every observation is relative to the home pose, so starting away from it
  # feeds the policy proprioception it never saw.
  home_error = float(np.abs(home - arm.read()[0]).max())
  if home_error > HOME_TOLERANCE:
    raise RuntimeError(f"{home_error:.3f} rad from home, above {HOME_TOLERANCE}.")

  joint_pos, joint_vel = arm.read()
  policy.warm_up(joint_pos=joint_pos, joint_vel=joint_vel, image=_image(camera))

  period = 1.0 / config.control_hz
  closest_error, at_goal = float("inf"), False
  step = 0
  started_at = deadline = time.perf_counter()
  try:
    while max_steps is None or step < max_steps:
      joint_pos, joint_vel = arm.read()
      action = policy.act(
        joint_pos=joint_pos, joint_vel=joint_vel, image=_image(camera)
      )

      largest_arm_action = float(np.abs(action[:-1]).max())
      if largest_arm_action > motion.max_arm_action:
        raise RuntimeError(
          f"Step {step}: arm action {largest_arm_action:.2f} exceeds "
          f"{motion.max_arm_action}. The observation is likely out of "
          "distribution."
        )

      # A held cube sits 5.6 mm from the ee site, well inside the 50 mm
      # threshold, so the end effector's position is the cube's -- but only
      # while the cube is really held, hence the gripper check.
      holding = bool(action[-1] < 0.0 and joint_pos[-1] > GRIPPER_WHEN_EMPTY)
      if holding:
        closest_error = min(closest_error, policy.goal_distance)
        at_goal = at_goal or policy.goal_distance < SUCCESS_THRESHOLD

      if not dry_run:
        arm.command(policy.joint_targets(action))

      step += 1
      deadline += period
      remaining = deadline - time.perf_counter()
      if remaining > 0:
        time.sleep(remaining)
      else:
        deadline = time.perf_counter()  # behind: drop the missed slots

      if step % int(config.control_hz) == 0:
        print(
          f"step {step:5d}  goal_err {policy.goal_distance:.3f}"
          f"  {'holding' if holding else '-------'}"
        )
  except KeyboardInterrupt:
    print("\nInterrupted.")
  finally:
    elapsed = time.perf_counter() - started_at
    print(f"\n{step} steps in {elapsed:.1f} s ({step / max(elapsed, 1e-9):.1f} Hz)")
    if closest_error < float("inf"):
      print(
        f"  closest while holding {closest_error:.3f} m"
        f"  ->  at_goal {'YES' if at_goal else 'no'}"
        f" (threshold {SUCCESS_THRESHOLD} m)"
      )
    else:
      print("  the cube was never held, so no goal error was measured")
    if camera is not None:
      camera.close()
    # Park rather than release: idle is not gravity-compensated, and the policy
    # leaves the arm wherever its last action put it.
    arm.park(seconds=motion.home_seconds)
    arm.close()
  return 0


def _image(camera: Any) -> Any:
  return camera.frame() if camera is not None else None


__all__ = ["park", "run"]
