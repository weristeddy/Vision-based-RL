from __future__ import annotations

from typing import Any


def main(argv: Any = None) -> int:
  from vbrl.deployment.push_chain import frame
  from vbrl.deployment.push_chain.silhouette import measure_goal, render_goal

  parser = frame.parser(
    "Fit the chain's footprint to the hole of the green goal outline in one "
    "external-camera frame, with the arm at home and the chain off the table, and "
    "save the goal (centre link x, y, yaw, then the hinge angles) for "
    "measure_overlap."
  )
  parser.add_argument("--links", type=int, default=3, help="Number of chain links.")
  arguments = parser.parse_args(argv)
  view, image = frame.load(arguments)
  fit = measure_goal(view, image, arguments.links)
  *goal, offset = fit.pose
  frame.report(arguments, image, "goal", goal, fit, render_goal(view, fit.pose))
  print(f"tape offset {offset * 1000:+.1f} mm")
  frame.write_goal(arguments.goal_file, goal)
  print(f"saved to {arguments.goal_file}")
  return 0


__all__ = ["main"]


if __name__ == "__main__":
  raise SystemExit(main())
