from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Sequence


def _gain(text: str) -> float | tuple[float, ...]:
  values = tuple(float(value) for value in text.split(","))
  return values[0] if len(values) == 1 else values


def main(argv: Sequence[str] | None = None) -> int:
  from vbrl.deployment.loop import home, park, run
  from vbrl.deployment.session import load_session

  parser = argparse.ArgumentParser(
    prog="vbrl-deploy",
    description="Run one trained policy on the real arm.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument("task", help="configs/deployment/<task>.yaml")
  parser.add_argument("--policy", help="one of the task's policies; default the first")
  parser.add_argument(
    "--goal", help="goal number (the task's list or its measured goal.json), a goal.json, or x,y,z|yaw"
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="home, read sensors and evaluate the policy, but command no action",
  )
  parser.add_argument(
    "--max-steps", type=int, default=None, help="stop after this many steps"
  )
  parser.add_argument(
    "--no-keyboard-goal",
    dest="keyboard_goal",
    action="store_false",
    help="do not read the arrow keys. They are read by default: up/down moves "
    "the goal z, left/right moves y, 0.02 m per press, clamped to the training "
    "range. With no terminal on stdin this turns itself off anyway.",
  )
  parser.add_argument(
    "--log",
    help="Write a per-step trace to this .npz: action, joint_pos, joint_vel, "
    "the commanded target, what the rate clamp actually sent, and every "
    "camera frame.",
  )
  parser.add_argument(
    "--record-other",
    action="store_true",
    help="record the rig's other camera alongside into the --log trace",
  )
  parser.add_argument(
    "--home",
    action="store_true",
    help="move to the home pose and hold it, needing no camera; Ctrl-C parks",
  )
  parser.add_argument(
    "--park",
    action="store_true",
    help="bring the arm down to rest and release torque, then exit",
  )
  parser.add_argument(
    "--action-gain",
    type=_gain,
    help="overrides motion.action_gain: one value, or six comma-separated",
  )
  parser.add_argument(
    "--action-smoothing",
    type=float,
    help="overrides motion.action_smoothing from the manifest",
  )
  arguments = parser.parse_args(argv)

  session = load_session(arguments.task)
  moves = not (arguments.home or arguments.park)
  if moves and arguments.goal is None and session.task == "push_t":
    parser.error("Push-T needs --goal: the goal block's number or x,y,yaw")
  goal = None if arguments.goal is None else session.goal(arguments.goal)
  config = session.config(arguments.policy, goal)
  overrides = {
    name: value
    for name, value in (
      ("action_gain", arguments.action_gain),
      ("action_smoothing", arguments.action_smoothing),
    )
    if value is not None
  }
  if overrides:
    config = dataclasses.replace(
      config, motion=dataclasses.replace(config.motion, **overrides)
    )
    config.motion.validate()
  if arguments.home:
    return home(config)
  if arguments.park:
    return park(config)
  return run(
    config,
    dry_run=arguments.dry_run,
    max_steps=arguments.max_steps,
    keyboard_goal=arguments.keyboard_goal,
    log=arguments.log,
    record_camera=session.config(arguments.policy, goal, session.other_camera)
    if arguments.record_other
    else None,
  )


if __name__ == "__main__":
  raise SystemExit(main())
