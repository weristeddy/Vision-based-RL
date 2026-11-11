from __future__ import annotations

from typing import Any


def main(argv: Any = None) -> int:
  from vbrl.deployment.push_t import frame
  from vbrl.deployment.push_t.silhouette import measure_t, overlap, render_t

  arguments = frame.parser(
    "Fit the T in one external-camera frame, with the arm at home, and print its "
    "overlap with the manifest goal."
  ).parse_args(argv)
  config, view, image = frame.load(arguments)
  fit = measure_t(view, image)
  frame.report(arguments, config, image, "T", fit, render_t(view, fit.pose))
  print(f"overlap       {overlap(fit.pose, config.goal):.3f}")
  return 0


__all__ = ["main"]


if __name__ == "__main__":
  raise SystemExit(main())
