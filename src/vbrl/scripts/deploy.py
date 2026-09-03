"""``vbrl-deploy`` -- run one trained policy on a real robot.

```bash
vbrl-deploy configs/deployment/lift_cube.yaml
vbrl-deploy configs/deployment/lift_cube.yaml --dry-run --max-steps 250
vbrl-deploy configs/deployment/lift_cube.yaml --park
```

Every run homes the arm first, because observations are measured relative to
that pose, and parks it afterwards.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
  from vbrl.deployment.config import load_config
  from vbrl.deployment.loop import park, run

  parser = argparse.ArgumentParser(
    prog="vbrl-deploy",
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument("manifest", help="deployment YAML (see configs/deployment/)")
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="home, read sensors and evaluate the policy, but command no action",
  )
  parser.add_argument(
    "--max-steps", type=int, default=None, help="stop after this many steps"
  )
  parser.add_argument(
    "--park",
    action="store_true",
    help="bring the arm down to rest and release torque, then exit",
  )
  arguments = parser.parse_args(argv)

  config = load_config(arguments.manifest)
  if arguments.park:
    return park(config)
  return run(config, dry_run=arguments.dry_run, max_steps=arguments.max_steps)


if __name__ == "__main__":
  raise SystemExit(main())
