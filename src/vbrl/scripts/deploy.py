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
  overrides = {}
  if arguments.dry_run:
    overrides["dry_run"] = True
  if arguments.max_steps is not None:
    overrides["max_steps"] = arguments.max_steps
  if overrides:
    config = replace(config, **overrides)
    config.validate()
  return run(config)


if __name__ == "__main__":
  raise SystemExit(main())
