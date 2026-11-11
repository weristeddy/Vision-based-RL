from __future__ import annotations

import math
from typing import Any

import numpy as np


def main(argv: Any = None) -> int:
  from vbrl.deployment.push_t import frame
  from vbrl.deployment.push_t.silhouette import measure_goal, render_footprint

  arguments = frame.parser(
    "Fit the T footprint to the hole of the green goal outline in one "
    "external-camera frame, with the arm at home and the T off the table, and "
    "print the manifest's goal line."
  ).parse_args(argv)
  config, view, image = frame.load(arguments)
  fit = measure_goal(view, image)
  footprint = render_footprint(view, fit.pose)
  frame.report(arguments, config, image, "goal", fit, footprint)
  x, y, yaw = fit.pose
  offset = np.subtract(fit.pose, config.goal)
  print(
    f"  goal - manifest: {np.hypot(*offset[:2]) * 1000:.1f} mm, "
    f"{math.degrees(math.remainder(offset[2], 2 * math.pi)):+.2f} deg"
  )
  print(f"goal: [{x:.5f}, {y:.5f}, {yaw:.4f}]")
  return 0


__all__ = ["main"]


if __name__ == "__main__":
  raise SystemExit(main())
