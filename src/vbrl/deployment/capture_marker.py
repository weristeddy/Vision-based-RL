from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from vbrl.deployment.capture_board import (
  EXTERNAL_SERIAL,
  HEIGHT,
  MOVE_SECONDS,
  SETTLE_SECONDS,
  WIDTH,
  _settle_and_grab,
  _write,
)

MARKER_ID = 41
MARKER_M = 0.035
DICTIONARY = "DICT_5X5_50"
# Offsets from home in radians, joint_0 to joint_5: the gripper over the push
# workspace (x 0.20-0.50 m, y +-0.20 m), the marker 7-15 cm above the table and
# within 55 deg of facing the external camera, every link 46+ mm above the table at
# each pose and 35+ mm on the joint-space path between neighbours.
DELTAS: tuple[tuple[float, ...], ...] = (
  (-0.07, +0.28, -0.18, -0.10, -0.02, +0.10),
  (-0.60, +0.51, +0.07, +0.18, +0.02, -0.57),
  (-0.34, +0.69, +0.22, +0.78, -0.04, -0.52),
  (-0.13, +0.52, -0.04, +0.96, -0.32, -0.49),
  (+0.24, +0.69, +0.09, +0.98, -0.04, -0.43),
  (+0.28, +0.66, -0.06, +0.77, +0.55, -0.33),
  (-0.13, +0.46, -0.21, +0.69, +0.10, -0.82),
  (-0.44, -0.03, -0.66, +0.59, -0.13, -1.05),
  (-0.11, -0.12, -0.77, +0.49, -0.73, -0.95),
  (-0.14, +0.37, -0.13, +0.29, -0.74, -0.96),
  (+0.22, +0.63, -0.15, +0.89, -0.70, -0.27),
  (+0.48, +0.20, -0.44, +0.65, -0.33, -0.09),
  (-0.05, +0.15, -0.56, +0.85, -0.40, +0.56),
  (-0.38, +0.40, -0.35, +0.93, +0.04, +0.17),
  (-0.07, +0.53, -0.01, +0.81, +0.58, +0.64),
  (+0.09, +0.50, -0.09, +1.00, +0.22, +1.07),
  (+0.35, +0.36, -0.22, +0.75, -0.07, +1.11),
  (+0.56, +0.10, -0.52, +0.45, +0.14, +1.12),
  (+0.15, +0.41, -0.42, +0.92, +0.66, +1.11),
  (-0.07, -0.14, -0.80, +0.74, +0.73, +0.79),
)


def detect(image: Any) -> Any:
  import cv2

  detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICTIONARY)),
    cv2.aruco.DetectorParameters(),
  )
  corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))
  if ids is None or MARKER_ID not in ids.ravel():
    return None
  return corners[list(ids.ravel()).index(MARKER_ID)].reshape(4, 2)


def capture(destination: Path, arm_ip: str) -> int:
  from vbrl.asset_zoo.robots.trossen_wxai import make_wxai_measured
  from vbrl.deployment.arm import TrossenArm
  from vbrl.deployment.config import DeploymentConfig
  from vbrl.deployment.intrinsics import open_stream

  config = DeploymentConfig.__new__(DeploymentConfig)
  object.__setattr__(config, "arm_ip", arm_ip)
  object.__setattr__(config, "arm_model", "wxai_v0")
  object.__setattr__(config, "motor_parameters", "wxai_v0_20260317")
  object.__setattr__(config, "motion", None)
  object.__setattr__(config, "command_goal_time_s", 0.0)

  joints = make_wxai_measured().home_joint_pos
  home = np.array([joints[f"joint_{i}"] for i in range(6)])
  destination.mkdir(parents=True, exist_ok=True)
  arm = TrossenArm(config)
  gripper = float(arm.read()[0][-1])
  views = []
  camera, _ = open_stream(EXTERNAL_SERIAL, WIDTH, HEIGHT)
  try:
    arm.lift(seconds=MOVE_SECONDS)
    for index, delta in enumerate(DELTAS):
      arm.move_to([*(home + delta), gripper], seconds=MOVE_SECONDS)
      time.sleep(0.4)
      image = _settle_and_grab(camera, SETTLE_SECONDS)
      name = f"pose_{index:02d}.png"
      _write(destination / name, image)
      seen = detect(image) is not None
      views.append({"image": name, "joint_pos": arm.read()[0].tolist()})
      print(f"  pose {index:2d}  marker {'seen' if seen else 'MISSING'}")
    arm.move_to([*home, gripper], seconds=MOVE_SECONDS)
  finally:
    camera.stop()
    (destination / "capture.json").write_text(
      json.dumps(
        {
          "external_serial": EXTERNAL_SERIAL,
          "width": WIDTH,
          "height": HEIGHT,
          "marker": {"id": MARKER_ID, "size_m": MARKER_M, "dictionary": DICTIONARY},
          "views": views,
        },
        indent=2,
      )
      + "\n"
    )
  print("At home, holding the gripper closed. Ctrl-C parks the arm.")
  try:
    while True:
      time.sleep(0.1)
  except KeyboardInterrupt:
    arm.park(seconds=MOVE_SECONDS)
    arm.close()
  return 0


def main(argv: Any = None) -> int:
  parser = argparse.ArgumentParser(
    description="Move the arm with the marker on its gripper through the workspace "
    "and photograph it with the external camera at every pose."
  )
  parser.add_argument("--out", type=Path, default=Path("artifacts/deployment/marker"))
  parser.add_argument("--arm-ip", default="192.168.1.2")
  arguments = parser.parse_args(argv)
  return capture(arguments.out, arguments.arm_ip)


__all__ = ["DELTAS", "MARKER_ID", "MARKER_M", "capture", "detect"]


if __name__ == "__main__":
  raise SystemExit(main())
