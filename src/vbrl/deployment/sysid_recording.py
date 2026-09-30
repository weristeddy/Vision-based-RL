from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

POSES = {
  "home": (0.0, 1.33, 1.42, -1.3, 0.0, 0.0),
  "low": (-0.13, 1.32, 1.03, -0.97, 0.19, 0.1),
  "left": (0.5, 1.1, 1.1, -0.8, 0.0, 0.0),
  "right": (-0.5, 1.1, 1.1, -0.8, 0.0, 0.0),
  "high": (0.0, 0.7, 0.9, -0.4, 0.0, 0.0),
}
AMPLITUDE = np.array([0.4, 0.25, 0.35, 0.8, 0.8, 0.8])
LEVELS = (1.0, 0.25)
PERIOD_S = 10.0
PERIODS = 2
HARMONICS = np.arange(1, 25)
FADE_S = 2.0
RANDOM_S = 20.0
PULL = 0.05
MAX_STEP = 0.03
TRANSITION_STEP = 0.01
MIN_HEIGHT_M = 0.04
LIMIT_MARGIN = 0.1
GRIPPER_M = 0.002
SAMPLE_HZ = 5000.0
SAMPLES = ("positions", "velocities", "efforts", "compensation_efforts")


def multisine(rng, control_hz: float):
  t = np.arange(int((PERIODS * PERIOD_S + 2 * FADE_S) * control_hz)) / control_hz
  wave = np.stack([
    (np.sin(2 * np.pi * k / PERIOD_S * t[:, None] + rng.uniform(0, 2 * np.pi, len(k))) / k).sum(1)
    for k in (HARMONICS[joint::6] for joint in range(6))
  ], 1)
  wave *= np.minimum(AMPLITUDE / np.abs(wave).max(0), MAX_STEP / np.abs(np.diff(wave, axis=0)).max(0))
  envelope = np.sin(0.5 * np.pi * np.clip(np.minimum(t, t[-1] - t) / FADE_S, 0, 1)) ** 2
  return wave * envelope[:, None]


def random_walk(rng, control_hz: float):
  offset, walk = np.zeros(6), []
  for step in rng.uniform(-1.0, 1.0, (int(RANDOM_S * control_hz), 6)):
    offset = (1.0 - PULL) * offset + step
    walk.append(offset)
  walk = np.array(walk)
  t = np.arange(len(walk)) / control_hz
  envelope = np.sin(0.5 * np.pi * np.clip(np.minimum(t, t[-1] - t) / FADE_S, 0, 1)) ** 2
  walk *= envelope[:, None]
  return walk * MAX_STEP / np.abs(np.diff(walk, axis=0)).max()


def transition(start, stop):
  n = max(2, int(np.ceil(np.abs(stop - start).max() * np.pi / 2 / TRANSITION_STEP)))
  s = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, n + 1) / n)
  return start + s[:, None] * (stop - start)


def make_check():
  import mujoco

  from vbrl.asset_zoo.robots import get_robot

  model = get_robot("trossen_identified").make_entity_cfg().spec_fn().compile()
  data = mujoco.MjData(model)
  joints = [model.joint(f"joint_{i}").id for i in range(6)]
  low, high = model.jnt_range[joints].T
  site = model.site("ee_site").id
  bodies = [model.body(name).id for name in ("link_4", "link_5", "link_6", "gripper_left", "gripper_right")]

  def safe(targets) -> bool:
    if (targets < low + LIMIT_MARGIN).any() or (targets > high - LIMIT_MARGIN).any():
      return False
    for q in targets:
      data.qpos[:6] = q
      data.qpos[6:8] = GRIPPER_M
      mujoco.mj_forward(model, data)
      if data.ncon or min(data.site_xpos[site][2], data.xpos[bodies][:, 2].min()) < MIN_HEIGHT_M:
        return False
    return True

  return safe


def trajectory(seed: int, control_hz: float, safe, broadband: bool = False):
  rng = np.random.default_rng(seed)
  home = np.array(POSES["home"])
  parts, scales = [home[None]], {}
  for name, pose in POSES.items():
    pose = np.array(pose)
    parts.append(transition(parts[-1][-1], pose))
    if broadband:
      segments = [random_walk(rng, control_hz)]
    else:
      segments = [level * multisine(rng, control_hz) for level in LEVELS]
    for index, wave in enumerate(segments):
      scale = next((s for s in np.linspace(1.0, 0.1, 10) if safe(pose + s * wave)), None)
      if scale is None:
        raise ValueError(f"No safe amplitude around the {name} pose.")
      parts.append(pose + scale * wave)
      scales[name, index] = scale
  parts.append(transition(parts[-1][-1], home))
  targets = np.concatenate(parts)
  if not safe(targets):
    raise ValueError("A transition between poses is unsafe.")
  return targets, scales


def record(config, targets, out: Path) -> None:
  from vbrl.deployment.arm import TrossenArm

  arm = TrossenArm(config)
  period = 1.0 / config.control_hz
  capacity = int(len(targets) * period * SAMPLE_HZ)
  command_time, sent = np.zeros(len(targets)), np.zeros((len(targets), 7))
  sample_time, controller_time = np.zeros(capacity), np.zeros(capacity)
  samples = {name: np.zeros((capacity, 7)) for name in SAMPLES}
  commands = count = 0
  try:
    arm.lift(seconds=config.motion.home_seconds)
    gripper = arm.read()[0][-1]
    arm.move_to(np.r_[targets[0], gripper], seconds=config.motion.home_seconds)
    last_id = None
    started = deadline = time.perf_counter()
    for target in targets:
      command_time[commands] = time.perf_counter() - started
      sent[commands] = arm.command(np.r_[target, gripper])
      commands += 1
      deadline += period
      while time.perf_counter() < deadline and count < capacity:
        output = arm.output()
        if output.header.id != last_id:
          last_id = output.header.id
          sample_time[count] = time.perf_counter() - started
          controller_time[count] = output.header.timestamp
          for name in SAMPLES:
            samples[name][count] = getattr(output.joint.all, name)
          count += 1
        time.sleep(0.0002)
      deadline = max(deadline, time.perf_counter())
  finally:
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
      out,
      control_hz=config.control_hz,
      goal_time=config.command_goal_time_s,
      command_time=command_time[:commands],
      sent=sent[:commands],
      sample_time=sample_time[:count],
      controller_time=controller_time[:count],
      **{name: value[:count] for name, value in samples.items()},
    )
    print(f"Wrote {out}: {commands} commands, {count} samples")
    arm.park(seconds=config.motion.home_seconds)
    arm.close()


def main() -> None:
  from vbrl.deployment.config import load_config

  parser = argparse.ArgumentParser()
  parser.add_argument("manifest")
  parser.add_argument("--out", default="artifacts/deployment/sysid_run0.npz")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--broadband", action="store_true")
  parser.add_argument("--check", action="store_true")
  arguments = parser.parse_args()

  config = load_config(arguments.manifest)
  targets, scales = trajectory(
    arguments.seed, config.control_hz, make_check(), arguments.broadband
  )
  for (name, index), scale in scales.items():
    print(f"{name:<6} segment {index} amplitude x{scale:.1f}")
  step = np.abs(np.diff(targets, axis=0)).max(0)
  print(
    f"{len(targets) / config.control_hz:.0f} s; largest step {np.round(step, 3).tolist()} rad; "
    f"range {np.round(targets.max(0) - targets.min(0), 2).tolist()} rad"
  )
  if not arguments.check:
    record(config, targets, Path(arguments.out))


if __name__ == "__main__":
  main()
