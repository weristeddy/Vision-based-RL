from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="vbrl-fetch-backbones",
    description="Download the pretrained vision backbones into the model root.",
  )
  parser.add_argument(
    "model_root",
    nargs="?",
    default=None,
    help=(
      "absolute destination directory; defaults to VBRL_MODEL_ROOT, then the "
      "checkout's .models, then the image path"
    ),
  )
  parser.add_argument(
    "--verify",
    action="store_true",
    help="report whether both backbones load offline, and download nothing",
  )
  return parser


def _destination(argument: str | None) -> Path:
  from vbrl.paths import (
    CHECKOUT_MODEL_DIRECTORY,
    DEFAULT_MODEL_ROOT,
    checkout_root,
    model_root,
  )

  if argument:
    root = Path(argument).expanduser()
  else:
    root = model_root()
    if root == DEFAULT_MODEL_ROOT and not root.is_dir():
      checkout = checkout_root(required=False)
      if checkout is not None:
        root = checkout / CHECKOUT_MODEL_DIRECTORY
  if not root.is_absolute():
    raise ValueError(f"model_root must be an absolute path; got {root}.")
  return root


def main(argv: Sequence[str] | None = None) -> int:
  arguments = _parser().parse_args(argv)
  root = _destination(arguments.model_root)
  root.mkdir(parents=True, exist_ok=True)
  os.environ["VBRL_MODEL_ROOT"] = str(root)

  from vbrl.vision.backbones import dinov2, r3m

  if arguments.verify:
    failures = 0
    for name, load in (("dinov2-small", dinov2.load), ("r3m resnet50", r3m.load)):
      try:
        load()
        print(f"  ok       {name}")
      except Exception as error:
        failures += 1
        print(f"  MISSING  {name}: {error}")
    print(f"Model root: {root}")
    return 1 if failures else 0

  print(f"Fetching backbones into {root}")
  dinov2.load(allow_download=True)
  print("  dinov2-small")
  r3m.load(allow_download=True)
  print("  r3m resnet50")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
