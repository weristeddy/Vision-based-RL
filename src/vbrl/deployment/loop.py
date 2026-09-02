"""The control loop: camera and encoders in, joint targets out, at 50 Hz.

Structured so the interesting failure is visible. Every step records how long
capture, inference, and the arm write took, because a vision policy that misses
its control period does not fail loudly -- it just acts on stale pixels.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# Joints settle to a few mrad of a position target, so this allows for servo
# error and sensor noise while still catching a wrong starting pose.
HOME_TOLERANCE_RAD = 0.10


@dataclass
class StepTiming:
  capture_ms: list[float] = field(default_factory=list)
  frame_age_ms: list[float] = field(default_factory=list)
  inference_ms: list[float] = field(default_factory=list)
  command_ms: list[float] = field(default_factory=list)
  period_ms: list[float] = field(default_factory=list)
  clamp_residual: list[float] = field(default_factory=list)
  """Per step, how far the clamps held the command back from what the policy
  asked for. Persistently large means the arm never catches its target, so the
  policy is closing its loop on dynamics it never trained against -- a slower
  arm is safer but is not the same task."""

  def summary(self) -> str:
    def stats(name: str, samples: list[float], unit: str = "ms") -> str:
      if not samples:
        return f"{name}: no samples"
      ordered = sorted(samples)
      mean = sum(ordered) / len(ordered)
      p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
      precision = 2 if unit == "ms" else 4
      return (
        f"{name}: mean {mean:.{precision}f} {unit}, "
        f"p95 {p95:.{precision}f} {unit}, max {ordered[-1]:.{precision}f} {unit}"
      )

    return "\n".join(
      (
        stats("  capture  ", self.capture_ms),
        stats("  frame age", self.frame_age_ms),
        stats("  inference", self.inference_ms),
        stats("  command  ", self.command_ms),
        stats("  period   ", self.period_ms),
        stats("  clamp gap", self.clamp_residual, unit="rad"),
      )
    )

  def clamp_summary(self) -> str:
    if not self.clamp_residual:
      return "  clamp: no commands sent"
    held = sum(1 for value in self.clamp_residual if value > 1e-6)
    share = 100.0 * held / len(self.clamp_residual)
    worst = max(self.clamp_residual)
    return (
      f"  clamp: held back {held}/{len(self.clamp_residual)} steps ({share:.0f}%), "
      f"worst {worst:.3f} rad short of the requested target"
    )


def run(config: Any) -> int:
  """Load the policy, open the devices, and drive the arm until stopped."""
  import numpy as np
  import torch

  from vbrl.deployment.observations import ObservationAssembler
  from vbrl.deployment.spec import RobotSpec
  from vbrl.runtime import CheckpointRef, build_env, load_trained_policy

  print(f"Task      {config.task_id}")
  print(f"Checkpoint{'':1} {config.checkpoint_file}")
  print(f"Goal      {tuple(config.goal)} (base frame, metres)")

  # One environment, never stepped -- see vbrl.deployment.spec.
  env = build_env(config.task_id, device=config.device, num_envs=1, seed=0)
  robot_definition_site = "ee_site"
  spec = RobotSpec.from_env(env, ee_site_name=robot_definition_site)
  print(f"Joints    {spec.joint_names}")
  print(f"Actor obs {spec.actor_terms}")

  _, _, policy, checkpoint_path = load_trained_policy(
    env,
    task_id=config.task_id,
    device=config.device,
    ref=CheckpointRef(checkpoint_file=config.checkpoint_file),
  )
  print(f"Loaded    {checkpoint_path.name} (strict)")

  assembler = ObservationAssembler(spec, goal=config.goal, device=config.device)

  camera = None
  arm = None
  timing = StepTiming()
  period = 1.0 / config.control_hz
  try:
    if spec.camera_group is not None:
      from vbrl.deployment.hardware import RealSenseCamera

      camera = RealSenseCamera(
        width=224, height=224, fps=config.camera_fps, serial=config.camera_serial
      )
      print(f"Camera    RealSense 224x224 at {camera.fps} fps")
      if config.camera_fps < config.control_hz:
        # Reusing a frame keeps the control period the policy trained at, and
        # only the pixels go stale. Slowing the loop to the sensor instead would
        # change the closed-loop dynamics of every joint.
        reuse = config.control_hz / config.camera_fps
        print(
          f"          {config.control_hz:.0f} Hz control on a {config.camera_fps} fps "
          f"sensor: each frame is seen ~{reuse:.2f} steps, so pixels are up to "
          f"{1e3 / config.camera_fps:.0f} ms old. Set control_hz to "
          f"{config.camera_fps} to be sensor-paced instead."
        )

    from vbrl.deployment.hardware import TrossenArm

    arm = TrossenArm(
      ip=config.arm_ip,
      model=config.arm_model,
      motor_parameters=config.motor_parameters,
      max_joint_step=config.safety.max_joint_step,
      max_gripper_step=config.safety.max_gripper_step,
      command_goal_time=config.safety.command_goal_time,
    )
    print(f"Arm       {config.arm_model} at {config.arm_ip}, {arm.num_joints} joints")
    print(f"Motors    {arm.motor_parameters}")

    from vbrl.deployment.homing import home_pose

    home = home_pose()
    if config.home_first:
      measured, _ = arm.read()
      travel = float(np.abs(home - measured).max())
      print(f"Homing    {config.safety.startup_seconds:.1f} s, max travel {travel:.3f} rad")
      arm.move_to(home, duration=config.safety.startup_seconds, blocking=True)

    # The policy's proprioception is relative to the home pose, so starting away
    # from it is not a small error -- it is an observation the network never saw.
    # Refuse rather than produce plausible-looking nonsense.
    measured, _ = arm.read()
    offset = float(np.abs(home - measured).max())
    if offset > HOME_TOLERANCE_RAD:
      raise RuntimeError(
        f"The arm is {offset:.3f} rad from the simulator's home pose, above the "
        f"{HOME_TOLERANCE_RAD} rad tolerance. Every observation is measured "
        "relative to that pose, so the policy would see proprioception outside "
        "its training distribution. Pass --home-first to move there, or run "
        "vbrl-deploy --home separately."
      )
    print(f"At home   within {offset:.4f} rad")

    if config.dry_run:
      print("Dry run   sensors and policy only; no action will be sent")
    else:
      arm.hold()

    # The first forward pays kernel autotune and lazy initialisation -- measured
    # at 192 ms against a 20 ms period, so the policy's very first action would
    # arrive ten periods late. Spend that cost here instead, on a real
    # observation so the shapes and dtypes match the loop exactly.
    torch.backends.cudnn.benchmark = True  # the input shape never changes
    warm_pos, warm_vel = arm.read()
    warm_frame = camera.frame()[0] if camera is not None else None
    warm_obs = assembler.build(joint_pos=warm_pos, joint_vel=warm_vel, rgb=warm_frame)
    started = time.perf_counter()
    with torch.no_grad():
      for _ in range(10):
        policy(warm_obs)
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    print(f"Warmup    10 forwards in {(time.perf_counter() - started) * 1e3:.0f} ms")

    previous_action = np.zeros(len(spec.action_scale))
    rate_holds = np.zeros(len(spec.action_scale), dtype=int)
    limit_holds = np.zeros(len(spec.action_scale), dtype=int)

    step = 0
    next_deadline = time.perf_counter()
    while config.max_steps is None or step < config.max_steps:
      loop_started = time.perf_counter()

      started = time.perf_counter()
      joint_pos, joint_vel = arm.read()
      frame = None
      if camera is not None:
        frame, frame_age = camera.frame()
        timing.frame_age_ms.append(frame_age * 1e3)
      timing.capture_ms.append((time.perf_counter() - started) * 1e3)

      started = time.perf_counter()
      observations = assembler.build(
        joint_pos=joint_pos, joint_vel=joint_vel, rgb=frame
      )
      with torch.no_grad():
        action = policy(observations)
      action = action.detach().cpu().numpy().reshape(-1)
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      timing.inference_ms.append((time.perf_counter() - started) * 1e3)

      # Smooth the intent, not the motion: a rate clamp on the arm leaves the
      # policy asking for a pose it never reaches, which is what drove |a| from
      # 3.5 to 19. Filtering here keeps request and response consistent.
      smoothing = config.safety.action_smoothing
      if smoothing < 1.0:
        action = smoothing * action + (1.0 - smoothing) * previous_action
      previous_action = action

      magnitude = float(np.abs(action).max())
      if magnitude > config.safety.max_action_magnitude:
        raise RuntimeError(
          f"Step {step}: the policy emitted |a| = {magnitude:.2f}, above the "
          f"{config.safety.max_action_magnitude} limit. In simulation this "
          "policy peaked near 3.5, so an action this large means the "
          "observation is outside its training distribution -- almost always "
          "the camera, since proprioception is checked by "
          "tests/test_deployment_parity.py. Stopping before the arm is driven "
          "into its limits."
        )

      started = time.perf_counter()
      targets = assembler.joint_targets(action)
      if config.dry_run:
        # Nothing is commanded, so nothing was applied; the raw action is the
        # only honest thing to report, and it is why |a| drifts upward here.
        assembler.record_action(action)
      else:
        sent = arm.command(targets)
        timing.clamp_residual.append(float(np.abs(targets - sent).max()))
        rate_holds += (arm.last_rate_held > 1e-6).astype(int)
        limit_holds += (arm.last_limit_held > 1e-6).astype(int)
        assembler.record_action(assembler.effective_action(sent))
      timing.command_ms.append((time.perf_counter() - started) * 1e3)

      step += 1
      next_deadline += period
      remaining = next_deadline - time.perf_counter()
      if remaining > 0:
        time.sleep(remaining)
      else:
        # Behind schedule: give up the missed slots rather than sprinting to
        # catch up, which would send a burst of commands at once.
        next_deadline = time.perf_counter()
      timing.period_ms.append((time.perf_counter() - loop_started) * 1e3)

      if step % int(config.control_hz) == 0:
        print(
          f"step {step:6d}  goal_err {np.linalg.norm(observations['actor'][0, -3:].cpu().numpy()):.3f} m"
          f"  |a| {np.abs(action).max():.2f}"
        )
  except KeyboardInterrupt:
    print("\nInterrupted.")
  finally:
    if arm is not None:
      # Park whenever anything moved the arm. --dry-run alone commands nothing,
      # but --home-first raises it regardless, and leaving it held at the home
      # pose is not a resting state.
      moved = config.home_first or not config.dry_run
      arm.close(park=moved, park_duration=config.safety.startup_seconds)
    if camera is not None:
      camera.close()
    print(f"\n{len(timing.period_ms)} steps at a {period * 1e3:.1f} ms period")
    print(timing.summary())
    print(timing.clamp_summary())
    if timing.clamp_residual:
      total = len(timing.clamp_residual)
      print("  per joint, share of steps held by each clamp:")
      for index in range(len(rate_holds)):
        label = "gripper" if index == len(rate_holds) - 1 else f"joint_{index}"
        print(
          f"    {label:9} rate {100 * rate_holds[index] / total:5.1f}%"
          f"   at-limit {100 * limit_holds[index] / total:5.1f}%"
        )
      if limit_holds.max() > 0.5 * total:
        # Rate holds are transient; a joint sitting at its limit means the
        # policy wants a pose the arm does not have, and no rate will fix it.
        print(
          "  a joint spent most of the run pinned at its limit: the policy is "
          "asking for poses outside the mechanism, not merely moving too slowly."
        )
    if camera is not None and timing.period_ms:
      print(f"  frames captured: {camera.frames_captured} for {len(timing.period_ms)} steps")
    overruns = sum(1 for p in timing.period_ms if p > period * 1e3 * 1.1)
    if overruns:
      # A missed period means the policy acted later than it was trained to.
      print(f"  WARNING {overruns} of {len(timing.period_ms)} steps overran the period")
  return 0


__all__ = ["StepTiming", "run"]
