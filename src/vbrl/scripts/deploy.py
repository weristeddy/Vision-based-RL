"""``vbrl-deploy`` -- run one trained policy on a real robot.

```bash
vbrl-deploy configs/deployment/lift_cube.yaml
vbrl-deploy configs/deployment/lift_cube.yaml --dry-run   # no arm commands
```
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="vbrl-deploy",
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument("manifest", help="deployment YAML (see configs/deployment/)")
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="read sensors and evaluate the policy, but send nothing to the arm",
  )
  parser.add_argument(
    "--home",
    action="store_true",
    help=(
      "move the arm to the simulator's default pose, hold it, and exit on "
      "Ctrl-C; runs no policy. Do this before a rollout: observations are "
      "relative to that pose"
    ),
  )
  parser.add_argument(
    "--home-seconds",
    type=float,
    default=None,
    help="how long the homing move takes (default: safety.startup_seconds)",
  )
  parser.add_argument(
    "--save-frame",
    nargs="?",
    const="artifacts/deployment/wrist_real.png",
    default=None,
    metavar="PATH",
    help=(
      "capture one camera frame at the simulator's home pose and save it, then "
      "exit. Combine with --home-first to move there. Defaults to "
      "artifacts/deployment/wrist_real.png"
    ),
  )
  parser.add_argument(
    "--park",
    action="store_true",
    help=(
      "bring the arm down to its resting pose and release torque, then exit. "
      "Use this if a run left it holding a raised pose"
    ),
  )
  parser.add_argument(
    "--home-first",
    action="store_true",
    help=(
      "move to the simulator's default pose before the first step, then run. "
      "This moves the arm even with --dry-run"
    ),
  )
  parser.add_argument(
    "--max-steps",
    type=int,
    default=None,
    help="stop after this many control steps",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  from dataclasses import replace

  from vbrl.deployment.config import load_config
  from vbrl.deployment.loop import run

  arguments = _parser().parse_args(argv)
  config = load_config(arguments.manifest)
  if arguments.save_frame is not None:
    from vbrl.deployment.snapshot import save_frame

    return save_frame(
      config, output=arguments.save_frame, home_first=arguments.home_first
    )

  if arguments.park:
    from vbrl.deployment.homing import park_arm

    return park_arm(config, duration=arguments.home_seconds)

  if arguments.home:
    from vbrl.deployment.homing import move_to_home

    return move_to_home(config, duration=arguments.home_seconds)
  overrides = {}
  if arguments.dry_run:
    overrides["dry_run"] = True
  if arguments.home_first:
    overrides["home_first"] = True
  if arguments.max_steps is not None:
    overrides["max_steps"] = arguments.max_steps
  if overrides:
    config = replace(config, **overrides)
    config.validate()
  return run(config)


if __name__ == "__main__":
  raise SystemExit(main())
