"""Print every VBRL registry, read live from its table.

Each section below reads the same table you would edit to add something, so
this stays correct without anyone maintaining a second list.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence


def _tasks() -> tuple[str, ...]:
  from vbrl.tasks import vbrl_task_ids

  return vbrl_task_ids()


def _architectures() -> tuple[str, ...]:
  from vbrl.vision.architectures import ARCHITECTURES, CURRENT_ARCHITECTURES

  return tuple(
    f"{token:<28} {cfg.encoder} + {cfg.adapter}"
    + ("" if token in CURRENT_ARCHITECTURES else "   [legacy: reproduction only]")
    for token, cfg in ARCHITECTURES.items()
  )


def _encoders() -> tuple[str, ...]:
  from vbrl.vision.registry import ENCODERS

  return tuple(
    f"{name:<28} {spec.channels:>5} channels, {spec.weights}"
    for name, spec in ENCODERS.items()
  )


def _adapters() -> tuple[str, ...]:
  from vbrl.vision.registry import ADAPTERS

  return tuple(
    f"{name:<28} wants {spec.feature_request} features"
    for name, spec in ADAPTERS.items()
  )


def _scenes() -> tuple[str, ...]:
  from vbrl.scenes.presets import get_preset, list_scenes

  return tuple(
    f"{name:<28} {'OOD evaluation' if get_preset(name).ood else 'training'}"
    for name in list_scenes()
  )


def _robots() -> tuple[str, ...]:
  from vbrl.asset_zoo.robots import list_robots

  return list_robots()


def _analysis_steps() -> tuple[str, ...]:
  from vbrl.scripts.analyze import STEPS

  return tuple(sorted(STEPS))


# Section name -> (where you add one, how to read what exists).
SECTIONS: dict[str, tuple[str, Callable[[], Sequence[str]]]] = {
  "tasks": ("tasks/<task>/config/<robot>/__init__.py", _tasks),
  "architectures": ("vision/architectures.py: ARCHITECTURES", _architectures),
  "encoders": ("vision/registry.py: ENCODERS", _encoders),
  "adapters": ("vision/registry.py: ADAPTERS", _adapters),
  "scenes": ("scenes/presets.py: _PRESETS", _scenes),
  "robots": ("asset_zoo/robots/__init__.py: ROBOTS", _robots),
  "analysis": ("scripts/analyze.py: STEPS", _analysis_steps),
}


def _print_section(name: str) -> None:
  source, read = SECTIONS[name]
  rows = read()
  print(f"{name}  ({len(rows)})")
  print(f"  add one in: {source}")
  for row in rows:
    print(f"    {row}")
  print()


def main(argv: Sequence[str] | None = None) -> int:
  parser = argparse.ArgumentParser(
    prog="vbrl-list",
    description="Print every VBRL registry and where to extend it.",
  )
  parser.add_argument(
    "section",
    nargs="?",
    choices=tuple(SECTIONS),
    help="Print one registry instead of all of them.",
  )
  args = parser.parse_args(argv)

  for name in (args.section,) if args.section else SECTIONS:
    _print_section(name)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())


__all__ = ["SECTIONS", "main"]
