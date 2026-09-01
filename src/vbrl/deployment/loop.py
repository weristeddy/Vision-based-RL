"""The control loop: camera and encoders in, joint targets out, at 50 Hz.

Structured so the interesting failure is visible. Every step records how long
capture, inference, and the arm write took, because a vision policy that misses
its control period does not fail loudly -- it just acts on stale pixels.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepTiming:
  capture_ms: list[float] = field(default_factory=list)
  frame_age_ms: list[float] = field(default_factory=list)
  inference_ms: list[float] = field(default_factory=list)
  command_ms: list[float] = field(default_factory=list)
  period_ms: list[float] = field(default_factory=list)

  def summary(self) -> str:
    def stats(name: str, samples: list[float]) -> str:
      if not samples:
        return f"{name}: no samples"
      ordered = sorted(samples)
      mean = sum(ordered) / len(ordered)
      p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
      return f"{name}: mean {mean:.2f} ms, p95 {p95:.2f} ms, max {ordered[-1]:.2f} ms"

    return "\n".join(
      (
        stats("  capture  ", self.capture_ms),
        stats("  frame age", self.frame_age_ms),
        stats("  inference", self.inference_ms),
        stats("  command  ", self.command_ms),
        stats("  period   ", self.period_ms),
      )
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
      max_joint_step=config.safety.max_joint_step,
      max_gripper_step=config.safety.max_gripper_step,
    )
    print(f"Arm       {config.arm_model} at {config.arm_ip}, {arm.num_joints} joints")
    if config.dry_run:
      print("Dry run   sensors and policy only; no command will be sent")
    else:
      arm.hold()

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

      started = time.perf_counter()
      targets = assembler.joint_targets(action)
      if not config.dry_run:
        arm.command(targets)
      assembler.record_action(action)
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
      arm.close()
    if camera is not None:
      camera.close()
    print(f"\n{len(timing.period_ms)} steps at a {period * 1e3:.1f} ms period")
    print(timing.summary())
    if camera is not None and timing.period_ms:
      print(f"  frames captured: {camera.frames_captured} for {len(timing.period_ms)} steps")
    overruns = sum(1 for p in timing.period_ms if p > period * 1e3 * 1.1)
    if overruns:
      # A missed period means the policy acted later than it was trained to.
      print(f"  WARNING {overruns} of {len(timing.period_ms)} steps overran the period")
  return 0


__all__ = ["StepTiming", "run"]
