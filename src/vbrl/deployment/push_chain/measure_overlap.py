from __future__ import annotations

from typing import Any

SUCCESS_THRESHOLD = 0.9


def main(argv: Any = None) -> int:
  from vbrl.deployment.push_chain import frame
  from vbrl.deployment.push_chain.silhouette import measure_chain, overlap, render_chain

  arguments = frame.parser(
    "Fit the chain in one external-camera frame, with the arm at home, and print its "
    "overlap with the goal saved by measure_goal."
  ).parse_args(argv)
  goal = frame.read_goal(arguments.goal_file)
  view, image = frame.load(arguments)
  fit = measure_chain(view, image)
  frame.report(arguments, image, "chain", fit, render_chain(view, fit.pose))
  print(f"goal   {frame.pose(goal)}")
  coverage, lowest_link = overlap(fit.pose, goal)
  print(f"chain overlap      {coverage:.3f}  success {coverage >= SUCCESS_THRESHOLD}")
  print(f"lowest link overlap {lowest_link:.3f}")
  return 0


__all__ = ["SUCCESS_THRESHOLD", "main"]


if __name__ == "__main__":
  raise SystemExit(main())
