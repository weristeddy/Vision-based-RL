from __future__ import annotations

from typing import Any


def main(argv: Any = None) -> int:
  from vbrl.deployment.push_chain import frame
  from vbrl.deployment.push_chain.silhouette import measure_goal, render_goal

  arguments = frame.parser(
    "Fit the chain's footprint to the hole of the green goal outline in one "
    "external-camera frame, with the arm at home and the chain off the table, and "
    "save the goal (x, y, yaw, hinge 0, hinge 1) for measure_overlap."
  ).parse_args(argv)
  view, image = frame.load(arguments)
  fit = measure_goal(view, image)
  frame.report(arguments, image, "goal", fit, render_goal(view, fit.pose))
  print(f"tape offset {fit.pose[5] * 1000:+.1f} mm")
  frame.write_goal(arguments.goal_file, fit.pose)
  print(f"saved to {arguments.goal_file}")
  return 0


__all__ = ["main"]


if __name__ == "__main__":
  raise SystemExit(main())
