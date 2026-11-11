from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

# The outer *black* square, which is what `tagsize` means -- not the 45 mm sheet.
TAG_BLACK_M = 0.031
DEFAULT_MARGIN_M = 0.007
# The origin is 10.7 mm past this along the stem, not the shape's centre.
CROSSBAR_INNER_EDGE_M = -0.0107
# Two different ids, so a swap cannot silently put the goal yaw 180 deg out.
TAG_ID_LEFT = 0
TAG_ID_RIGHT = 1


def detect_tags(image: Any) -> dict[int, Any]:
  import cv2

  detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9),
    cv2.aruco.DetectorParameters(),
  )
  corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
  if ids is None:
    return {}
  return {int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners, strict=True)}


def solve_goal_pose(
  image: Any, view: Any, margin_m: float = DEFAULT_MARGIN_M
) -> tuple[float, float, float]:
  tags = detect_tags(image)
  missing = {TAG_ID_LEFT, TAG_ID_RIGHT} - set(tags)
  if missing:
    raise ValueError(
      f"Missing tag id(s) {sorted(missing)}; detected {sorted(tags)}. "
      "Both must be visible, and at this range the black square needs roughly "
      "30 px -- capture at the camera's full resolution."
    )
  left, right = (
    view.on_table(tags[tag].mean(axis=0))[:2]
    for tag in (TAG_ID_LEFT, TAG_ID_RIGHT)
  )
  dx, dy = right - left
  yaw = math.atan2(dy, dx)
  offset = -(CROSSBAR_INNER_EDGE_M + margin_m + TAG_BLACK_M / 2.0)
  x, y = (left + right) / 2.0 + offset * np.array([-math.sin(yaw), math.cos(yaw)])
  return float(x), float(y), yaw


def main(argv: Any = None) -> int:
  import cv2

  from vbrl.deployment.push_t.view import ExternalView

  parser = argparse.ArgumentParser(
    description="Solve the Push-T goal pose from the two printed AprilTags."
  )
  parser.add_argument("image", type=Path, help="One frame from the external camera.")
  parser.add_argument(
    "--margin-mm",
    type=float,
    default=DEFAULT_MARGIN_M * 1000.0,
    help="White margin between the printed edge and the black square, on the "
    "side butted against the crossbar. Measure it; only this side matters.",
  )
  arguments = parser.parse_args(argv)

  image = cv2.imread(str(arguments.image))
  if image is None:
    raise SystemExit(f"Could not read {arguments.image}.")
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  view = ExternalView.load(image.shape[1], image.shape[0])
  x, y, yaw = solve_goal_pose(image, view, arguments.margin_mm / 1000.0)
  print(f"goal: [{x:.5f}, {y:.5f}, {yaw:.4f}]  ({math.degrees(yaw):+.2f} deg)")
  return 0


__all__ = [
  "CROSSBAR_INNER_EDGE_M",
  "DEFAULT_MARGIN_M",
  "TAG_BLACK_M",
  "TAG_ID_LEFT",
  "TAG_ID_RIGHT",
  "detect_tags",
  "solve_goal_pose",
]


if __name__ == "__main__":
  raise SystemExit(main())
