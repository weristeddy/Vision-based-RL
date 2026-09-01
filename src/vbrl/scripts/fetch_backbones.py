"""Materialise the pinned pretrained vision backbones.

The image bakes them into ``/opt/vbrl-models`` during ``apptainer build``; a
development host fetches the same bytes into the checkout's ``.models``, which
:func:`vbrl.paths.model_root` resolves without any environment variable.

```bash
vbrl-fetch-backbones            # into the resolved model root
vbrl-fetch-backbones --verify   # report only, download nothing
```

Re-running is cheap: a file already present with the pinned digest is left alone.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    prog="vbrl-fetch-backbones",
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
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
    help="report which pinned assets are missing or corrupt, and download nothing",
  )
  return parser


def main(argv: Sequence[str] | None = None) -> int:
  from vbrl.vision.backbones.weights import (
    BACKBONE_ASSETS,
    fetch_backbones,
    verify_backbones,
  )
  from vbrl.paths import DEFAULT_MODEL_ROOT, model_root

  from vbrl.paths import CHECKOUT_MODEL_DIRECTORY, checkout_root

  arguments = _parser().parse_args(argv)
  if arguments.model_root:
    root = Path(arguments.model_root).expanduser()
  else:
    root = model_root()
    # ``model_root`` only returns the checkout's ``.models`` once it exists, so on
    # a fresh checkout it resolves to the image path -- which no development host
    # can write to. Fetching is the one command that legitimately creates that
    # directory, so it resolves the checkout itself rather than sending a first
    # run at /opt/vbrl-models. The image is unaffected: rl.def passes its
    # destination explicitly, taking the branch above.
    if root == DEFAULT_MODEL_ROOT and not root.is_dir():
      checkout = checkout_root(required=False)
      if checkout is not None:
        root = checkout / CHECKOUT_MODEL_DIRECTORY
        root.mkdir(parents=True, exist_ok=True)
  if not root.is_absolute():
    raise ValueError(f"model_root must be an absolute path; got {root}.")

  if arguments.verify:
    problems = verify_backbones(root)
    print(f"{len(BACKBONE_ASSETS) - len(problems)}/{len(BACKBONE_ASSETS)} pinned "
          f"assets present and verified under {root}")
    for asset in problems:
      print(f"  MISSING OR CORRUPT  {asset.relative_path}")
    return 1 if problems else 0

  print(f"Fetching {len(BACKBONE_ASSETS)} pinned backbone files into {root}")
  for asset, action in fetch_backbones(root):
    print(f"  {action:<10}  {asset.relative_path}")
  print(f"Verified. Model root: {root}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
