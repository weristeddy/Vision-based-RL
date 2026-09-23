from __future__ import annotations

import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import numpy as np

from vbrl.deployment.arm import TrossenArm
from vbrl.deployment.camera import RealSenseCamera
from vbrl.deployment.keyboard import HELP, ArrowKeys, nudge_goal
from vbrl.deployment.policy import load_policy

HOME_TOLERANCE = 0.10
SUCCESS_THRESHOLD = 0.05
# Carriage travel with nothing between the fingers; a cube blocks it near 0.019.
GRIPPER_WHEN_EMPTY = 0.004


def home(config: Any) -> int:
  policy = load_policy(config)
  arm = TrossenArm(config)
  print(f"Homing    {config.motion.home_seconds:.1f} s")
  arm.move_to(policy.metadata.home_pose, seconds=config.motion.home_seconds)
  print("At home, holding. Ctrl-C to park and release torque.")
  try:
    while True:
      time.sleep(0.1)
  except KeyboardInterrupt:
    print()
  finally:
    arm.park(seconds=config.motion.home_seconds)
    arm.close()
  return 0


def park(config: Any) -> int:
  arm = TrossenArm(config)
  arm.park(seconds=config.motion.home_seconds)
  arm.close()
  print("At rest, torque released.")
  return 0


def run(
  config: Any,
  *,
  dry_run: bool = False,
  max_steps: int | None = None,
  keyboard_goal: bool = True,
  log: Any = None,
) -> int:
  # A Push-T goal's third number is a heading, not a z the keys may clamp.
  keyboard_goal = keyboard_goal and config.goal_space == "lift_xyz"
  motion = config.motion
  policy = load_policy(config)
  print(f"Policy    {config.onnx_file} on {policy.provider}")
  print(
    f"          from {policy.metadata.source_run}, "
    f"obs {policy.metadata.observation_terms}"
  )
  if policy.metadata.clip_actions is None:
    print(
      "          clip_actions absent: nothing bounds the action fed back as "
      "the `actions` observation. Correct for a task trained with "
      "clip_actions=None (Lift-Cube); for Push-T it is 1.0, so re-export."
    )

  arm = TrossenArm(config)
  camera = RealSenseCamera(config) if policy.metadata.needs_camera else None
  if camera is not None:
    saturated = camera.saturated_fraction()
    setting = (
      "auto"
      if config.camera_exposure_us is None
      else f"{config.camera_exposure_us:.0f} us"
    )
    print(f"Camera    exposure {setting}, {saturated * 100:.0f}% of pixels clipped")
    if saturated > 0.10:
      print(
        "          training tables were textured photographs, never a flat "
        "field -- lower camera_exposure_us until this is a few percent"
      )
  home = policy.metadata.home_pose
  reads_goal = bool(
    {"goal_position", "target_pose"} & set(policy.metadata.observation_terms)
  )
  print(
    f"Goal      {tuple(config.goal)} in the base frame"
    + ("" if reads_goal else "  (not an input: this policy sees only the marker)")
  )
  if keyboard_goal:
    print(f"Keys      the arrow keys move the goal\n{HELP}")
  print(f"Homing    {motion.home_seconds:.1f} s")
  arm.move_to(home, seconds=motion.home_seconds)

  home_error = float(np.abs(home - arm.read()[0]).max())
  if home_error > HOME_TOLERANCE:
    raise RuntimeError(f"{home_error:.3f} rad from home, above {HOME_TOLERANCE}.")

  joint_pos, joint_vel = arm.read()
  policy.warm_up(joint_pos=joint_pos, joint_vel=joint_vel, image=_image(camera))

  period = 1.0 / config.control_hz
  closest_error, at_goal = float("inf"), False
  step = 0
  trace: dict[str, list] = {
    k: []
    for k in (
      "action",
      "network",
      "joint_pos",
      "joint_vel",
      "target",
      "sent",
      "time",
    )
  }
  frames: list = []
  networks: list = []
  stack = ExitStack()
  started_at = deadline = time.perf_counter()
  try:
    keys = stack.enter_context(ArrowKeys()) if keyboard_goal else None
    if keys is not None and not keys.enabled:
      keys = None
      print("Keys      off: stdin is not a terminal, so no key can be read")
    while max_steps is None or step < max_steps:
      if keys is not None and (pressed := keys.pressed()):
        policy.goal, refused = nudge_goal(policy.goal, pressed)
        for line in refused:
          print(f"  {line}")
        print(f"  goal {np.round(policy.goal, 3).tolist()}")

      joint_pos, joint_vel = arm.read()
      frame = _image(camera)
      action = policy.act(
        joint_pos=joint_pos, joint_vel=joint_vel, image=frame
      )

      # The network's own output: `action` is already clamped and never trips.
      network = policy.network_action
      networks.append(np.asarray(network, dtype=np.float32))
      arm_channels = network[:-1] if policy.has_gripper else network
      largest_arm_action = float(np.abs(arm_channels).max())
      if largest_arm_action > motion.max_arm_action:
        raise RuntimeError(
          f"Step {step}: arm action {largest_arm_action:.2f} exceeds "
          f"{motion.max_arm_action}. The observation is likely out of "
          "distribution."
        )

      # A held cube sits 5.6 mm from the ee site, hence the gripper check.
      holding = policy.has_gripper and bool(
        action[-1] < 0.0 and joint_pos[-1] > GRIPPER_WHEN_EMPTY
      )
      if holding:
        closest_error = min(closest_error, policy.goal_distance)
        at_goal = at_goal or policy.goal_distance < SUCCESS_THRESHOLD

      target = policy.joint_targets(action)
      sent = arm.command(target) if not dry_run else target

      if log is not None:
        trace["action"].append(np.asarray(action, dtype=np.float32))
        trace["network"].append(np.asarray(network, dtype=np.float32))
        trace["joint_pos"].append(np.asarray(joint_pos, dtype=np.float32))
        trace["joint_vel"].append(np.asarray(joint_vel, dtype=np.float32))
        trace["target"].append(np.asarray(target, dtype=np.float32))
        trace["sent"].append(np.asarray(sent, dtype=np.float32))
        trace["time"].append(time.perf_counter() - started_at)
        if step % 10 == 0:
          frames.append(np.asarray(frame, dtype=np.uint8))

      step += 1
      deadline += period
      remaining = deadline - time.perf_counter()
      if remaining > 0:
        time.sleep(remaining)
      else:
        deadline = time.perf_counter()  # behind: drop the missed slots

      if step % int(config.control_hz) == 0:
        tracked = (
          f"  goal_err {policy.goal_distance:.3f}"
          f"  {'holding' if holding else '-------'}"
          if policy.has_gripper
          else f"  action |max| {largest_arm_action:.2f}"
        )
        print(f"step {step:5d}{tracked}")
  except KeyboardInterrupt:
    print("\nInterrupted.")
  finally:
    stack.close()
    if log is not None and trace["action"]:
      destination = Path(log)
      destination.parent.mkdir(parents=True, exist_ok=True)
      np.savez_compressed(
        destination,
        goal=np.asarray(policy.goal, dtype=np.float32),
        max_joint_step=np.float32(
          np.nan if motion.max_joint_step is None else motion.max_joint_step
        ),
        action_smoothing=np.float32(motion.action_smoothing),
        control_hz=np.float32(config.control_hz),
        frames=np.asarray(frames, dtype=np.uint8),
        **{k: np.asarray(v) for k, v in trace.items()},
      )
      print(f"Wrote {destination} ({len(trace['action'])} steps, {len(frames)} frames)")
    elapsed = time.perf_counter() - started_at
    print(f"\n{step} steps in {elapsed:.1f} s ({step / max(elapsed, 1e-9):.1f} Hz)")
    _report_clipping(networks, policy.metadata.clip_actions, policy.has_gripper)
    if closest_error < float("inf"):
      print(
        f"  closest while holding {closest_error:.3f} m"
        f"  ->  at_goal {'YES' if at_goal else 'no'}"
        f" (threshold {SUCCESS_THRESHOLD} m)"
      )
    else:
      print("  the cube was never held, so no goal error was measured")
    if keys is not None and keys.reclaims:
      print(f"  the terminal was taken back {keys.reclaims} time(s)")
    if camera is not None:
      camera.close()
    # Park rather than release: idle is not gravity-compensated.
    arm.park(seconds=motion.home_seconds)
    arm.close()
  return 0


def _report_clipping(networks: list, bound: float | None, has_gripper: bool) -> None:
  """Say how hard the raw network pushed against `clip_actions`.

  A channel that sits at one end and never the other is a standing bias, not
  noise: the same observation in sim leaves the mean near zero.
  """
  if not networks:
    return
  raw = np.asarray(networks)
  arm = raw[:, :-1] if has_gripper else raw
  if bound is None:
    print(
      f"  raw action |max| {np.abs(arm).max():.2f}; clip_actions absent, so "
      "nothing bounded it"
    )
    return
  high, low = arm >= bound - 1e-6, arm <= -bound + 1e-6
  touched = (high | low).any(axis=1).mean()
  print(f"  clipping at +/-{bound:g}: {touched * 100:.1f}% of steps touched it")
  print("    ch     mean      min      max     %at+     %at-")
  for channel in range(arm.shape[1]):
    column = arm[:, channel]
    print(
      f"    {channel}  {column.mean():+8.3f} {column.min():+8.3f} "
      f"{column.max():+8.3f} {high[:, channel].mean() * 100:7.1f} "
      f"{low[:, channel].mean() * 100:7.1f}"
    )


def _image(camera: Any) -> Any:
  return camera.frame() if camera is not None else None


__all__ = ["home", "park", "run"]
