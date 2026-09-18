from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from vbrl.deployment.capture_board import EXTERNAL_SERIAL
from vbrl.deployment.config import TABLE_Z_BASE_M, TARGET_Z_BASE_M

# The outer *black* square, which is what `tagsize` means -- not the 45 mm sheet.
TAG_BLACK_M = 0.031
DEFAULT_MARGIN_M = 0.007
# The origin is 10.7 mm past this along the stem, not the shape's centre.
CROSSBAR_INNER_EDGE_M = -0.0107
# Two different ids, so a swap cannot silently put the goal yaw 180 deg out.
TAG_ID_LEFT = 0
TAG_ID_RIGHT = 1

_CAMERA_RE = re.compile(
  r'<camera\s+name="external_cam"\s+pos="([^"]+)"\s*\n?\s*quat="([^"]+)"', re.M
)


def external_camera_pose(xml_path: Path) -> Any:
  import mujoco

  from vbrl.deployment.calibration import CV_TO_MUJOCO, transform

  match = _CAMERA_RE.search(xml_path.read_text())
  if match is None:
    raise ValueError(f"No external_cam <camera> with pos and quat in {xml_path}.")
  position = np.array([float(v) for v in match.group(1).split()])
  quaternion = np.array([float(v) for v in match.group(2).split()])
  rotation_mj = np.empty(9)
  mujoco.mju_quat2Mat(rotation_mj, quaternion)
  return transform(rotation_mj.reshape(3, 3) @ CV_TO_MUJOCO, position)


def camera_intrinsics(
  intrinsics_path: Path, serial: str, width: int, height: int
) -> tuple[Any, Any]:
  cameras = json.loads(intrinsics_path.read_text())
  for camera in cameras:
    if camera["serial"] != serial:
      continue
    mode = camera["modes"].get(f"{width}x{height}")
    if mode is None:
      raise ValueError(
        f"Camera {serial} has no {width}x{height} mode; it carries "
        f"{sorted(camera['modes'])}. Capture at one of those."
      )
    native = mode["native"]
    matrix = np.array(
      [
        [native["fx"], 0.0, native["cx"]],
        [0.0, native["fy"], native["cy"]],
        [0.0, 0.0, 1.0],
      ]
    )
    return matrix, np.array(mode["distortion"]["coeffs"], dtype=float)
  raise ValueError(f"No camera with serial {serial} in {intrinsics_path}.")


def detect_tags(image: Any) -> dict[int, Any]:
  import cv2

  detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9),
    cv2.aruco.DetectorParameters(),
  )
  grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  corners, ids, _ = detector.detectMarkers(grey)
  if ids is None:
    return {}
  return {int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners, strict=True)}


def pixel_on_table(
  pixel: Any, pose_cv: Any, matrix: Any, distortion: Any, plane_z: float
) -> Any:
  import cv2

  undistorted = cv2.undistortPoints(
    np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2), matrix, distortion
  ).reshape(2)
  ray = pose_cv[:3, :3] @ np.array([undistorted[0], undistorted[1], 1.0])
  origin = pose_cv[:3, 3]
  if abs(ray[2]) < 1e-9:
    raise ValueError("The camera ray is parallel to the table; check the pose.")
  return origin + ray * ((plane_z - origin[2]) / ray[2])


def solve_goal_pose(
  image: Any,
  pose_cv: Any,
  matrix: Any,
  distortion: Any,
  margin_m: float = DEFAULT_MARGIN_M,
  plane_z: float = TABLE_Z_BASE_M,
) -> dict[str, Any]:
  tags = detect_tags(image)
  missing = {TAG_ID_LEFT, TAG_ID_RIGHT} - set(tags)
  if missing:
    raise ValueError(
      f"Missing tag id(s) {sorted(missing)}; detected {sorted(tags)}. "
      "Both must be visible, and at this range the black square needs roughly "
      "30 px -- capture at the camera's full resolution."
    )
  centres = {
    tag: pixel_on_table(tags[tag].mean(axis=0), pose_cv, matrix, distortion, plane_z)
    for tag in (TAG_ID_LEFT, TAG_ID_RIGHT)
  }
  left, right = centres[TAG_ID_LEFT], centres[TAG_ID_RIGHT]
  baseline = right - left
  yaw = math.atan2(baseline[1], baseline[0])
  midpoint = (left + right) / 2.0
  offset_y = -(CROSSBAR_INNER_EDGE_M + margin_m + TAG_BLACK_M / 2.0)
  origin = midpoint[:2] + np.array(
    [-math.sin(yaw) * offset_y, math.cos(yaw) * offset_y]
  )
  return {
    "tag_left_xy": left[:2].tolist(),
    "tag_right_xy": right[:2].tolist(),
    "baseline_m": float(np.linalg.norm(baseline[:2])),
    "yaw_rad": yaw,
    "yaw_deg": math.degrees(yaw),
    "offset_y_m": offset_y,
    "origin_xy": origin.tolist(),
    "target_pose": [
      float(origin[0]),
      float(origin[1]),
      TARGET_Z_BASE_M,
      math.sin(yaw),
      math.cos(yaw),
    ],
  }


def main(argv: Any = None) -> int:
  import cv2

  from vbrl.paths import checkout_root

  parser = argparse.ArgumentParser(description="Solve the Push-T goal pose from the two printed AprilTags.")
  parser.add_argument("image", type=Path, help="One frame from the external camera.")
  parser.add_argument(
    "--intrinsics",
    type=Path,
    default=Path("artifacts/deployment/intrinsics/intrinsics.json"),
  )
  parser.add_argument("--serial", default=EXTERNAL_SERIAL)
  parser.add_argument(
    "--margin-mm",
    type=float,
    default=DEFAULT_MARGIN_M * 1000.0,
    help="White margin between the printed edge and the black square, on the "
    "side butted against the crossbar. Measure it; only this side matters.",
  )
  parser.add_argument(
    "--robot-xml",
    type=Path,
    default=None,
    help="MJCF carrying the calibrated external_cam pose.",
  )
  parser.add_argument("--annotate", type=Path, default=None)
  arguments = parser.parse_args(argv)

  xml_path = arguments.robot_xml or (
    checkout_root()
    / "src/vbrl/asset_zoo/robots/trossen_wxai/xmls/wxai_realistic.xml"
  )
  image = cv2.imread(str(arguments.image))
  if image is None:
    raise SystemExit(f"Could not read {arguments.image}.")
  height, width = image.shape[:2]
  matrix, distortion = camera_intrinsics(
    arguments.intrinsics, arguments.serial, width, height
  )
  result = solve_goal_pose(
    image,
    external_camera_pose(xml_path),
    matrix,
    distortion,
    margin_m=arguments.margin_mm / 1000.0,
  )

  print(f"image            {arguments.image}  ({width}x{height})")
  print(f"camera           {arguments.serial}")
  print(f"margin           {arguments.margin_mm:.2f} mm")
  print(f"tag {TAG_ID_LEFT} (left)    {result['tag_left_xy'][0]:+.4f}, "
        f"{result['tag_left_xy'][1]:+.4f} m")
  print(f"tag {TAG_ID_RIGHT} (right)   {result['tag_right_xy'][0]:+.4f}, "
        f"{result['tag_right_xy'][1]:+.4f} m")
  print(f"baseline         {result['baseline_m'] * 1000:.1f} mm "
        f"(expect {31.0 + 2 * arguments.margin_mm + 30.0:.1f})")
  print(f"yaw              {result['yaw_deg']:+.2f} deg")
  print(f"origin           {result['origin_xy'][0]:+.4f}, "
        f"{result['origin_xy'][1]:+.4f} m")
  print("\ntarget_pose = [" + ", ".join(f"{v:+.5f}" for v in result["target_pose"]) + "]")

  if arguments.annotate is not None:
    tags = detect_tags(image)
    for tag, corners in tags.items():
      cv2.polylines(image, [corners.astype(int)], True, (0, 220, 0), 2)
      centre = corners.mean(axis=0).astype(int)
      cv2.drawMarker(image, tuple(centre), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
      cv2.putText(image, str(tag), tuple(centre + np.array([8, -8])),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    arguments.annotate.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(arguments.annotate), image)
    print(f"wrote {arguments.annotate}")
  return 0


__all__ = [
  "DEFAULT_MARGIN_M",
  "TAG_BLACK_M",
  "TAG_ID_LEFT",
  "TAG_ID_RIGHT",
  "camera_intrinsics",
  "detect_tags",
  "external_camera_pose",
  "pixel_on_table",
  "solve_goal_pose",
]


if __name__ == "__main__":
  raise SystemExit(main())
