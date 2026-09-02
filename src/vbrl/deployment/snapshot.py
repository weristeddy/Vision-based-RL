"""Save one real camera frame from the pose the policy actually starts in.

The encoder is the only part of this stack whose input cannot be validated
against the simulator arithmetically: proprioception is checked by
``tests/test_deployment_parity.py``, but whether the pixels resemble what the
network trained on is a question only a person looking at two images can answer.

The pose matters as much as the frame. A wrist camera sees whatever the wrist is
pointing at, and the simulator's reference render is taken at its default joint
pose, so a frame captured anywhere else compares nothing. This refuses to
capture away from that pose for the same reason the control loop refuses to run
there.
"""

from __future__ import annotations

from typing import Any

# The sensor's auto-exposure and white balance need a moment after the stream
# opens; the first frames come back darker and colour-shifted.
_SETTLE_FRAMES = 45


def save_frame(config: Any, *, output: str, home_first: bool = False) -> int:
  """Home if asked, capture one settled frame, save it, then park."""
  import time

  import numpy as np
  from PIL import Image

  from vbrl.deployment.hardware import RealSenseCamera, TrossenArm
  from vbrl.deployment.homing import home_pose
  from vbrl.deployment.loop import HOME_TOLERANCE_RAD
  from vbrl.paths import artifact_path

  destination = artifact_path(output)
  destination.parent.mkdir(parents=True, exist_ok=True)

  home = home_pose()
  arm = TrossenArm(
    ip=config.arm_ip,
    model=config.arm_model,
    motor_parameters=config.motor_parameters,
    max_joint_step=config.safety.max_joint_step,
    max_gripper_step=config.safety.max_gripper_step,
    command_goal_time=config.safety.command_goal_time,
  )
  camera = None
  moved = False
  try:
    if home_first:
      measured, _ = arm.read()
      travel = float(np.abs(home - measured).max())
      print(f"Homing    {config.safety.startup_seconds:.1f} s, max travel {travel:.3f} rad")
      arm.move_to(home, duration=config.safety.startup_seconds, blocking=True)
      moved = True

    measured, _ = arm.read()
    offset = float(np.abs(home - measured).max())
    if offset > HOME_TOLERANCE_RAD:
      raise RuntimeError(
        f"The arm is {offset:.3f} rad from the home pose, above the "
        f"{HOME_TOLERANCE_RAD} rad tolerance. A wrist frame captured here shows "
        "a different scene than the simulator's reference render. Pass "
        "--home-first to move there."
      )
    print(f"At home   within {offset:.4f} rad")

    camera = RealSenseCamera(
      width=224, height=224, fps=config.camera_fps, serial=config.camera_serial
    )
    print(f"Camera    RealSense 224x224 at {camera.fps} fps, settling")
    for _ in range(_SETTLE_FRAMES):
      camera.frame()
      time.sleep(1.0 / max(camera.fps, 1))

    frame, age = camera.frame()
    Image.fromarray(frame).save(destination)
    print(f"Saved     {destination}")
    print(
      f"  {frame.shape[1]}x{frame.shape[0]}  age {age * 1e3:.1f} ms  "
      f"mean px {frame.mean():.1f}  per-channel {np.round(frame.mean(axis=(0, 1)), 1).tolist()}"
    )
    print(
      "  compare against the simulator's wrist render at the same pose; the "
      "encoder saw that distribution, not this one."
    )
  finally:
    if camera is not None:
      camera.close()
    arm.close(park=moved, park_duration=config.safety.startup_seconds)
  return 0


__all__ = ["save_frame"]
