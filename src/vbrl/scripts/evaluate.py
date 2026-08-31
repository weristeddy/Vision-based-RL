"""Run a YAML evaluation suite: ``vbrl-evaluate configs/evaluation/....yaml``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from vbrl.evaluation.suite import load_config, run_suite
from vbrl.runtime import default_device


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="vbrl-evaluate",
    description="Evaluate YAML-defined model and scene lists.",
  )
  parser.add_argument("config", type=Path)
  parser.add_argument("--device")
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  parser = _parser()
  args = parser.parse_args(argv)
  try:
    config = load_config(args.config)
  except (FileNotFoundError, KeyError, ValueError) as exc:
    parser.error(str(exc))

  output = run_suite(config, args.device or default_device())
  print(f"Evaluation results: {output}", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
