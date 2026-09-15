"""Solve the Push-T goal pose from two AprilTags on the printed goal marker.

The policy is fed ``target_pose`` -- the goal's position and heading in the
robot base frame -- and that is a *choice*, not a measurement: you decide where
the T should end up. What has to be measured is where the printed goal marker
actually landed on the table, because the visual policy also sees that marker
drawn in its camera image, and the numbers and the picture must agree.

Two tag25h9 markers do that. They tuck into the notches either side of the
marker's stem, each butted against the stem's side edge and the crossbar's inner
edge, and they come off again before the policy runs -- so nothing about them
reaches the policy's observation.

Why two rather than one. A single tag's own orientation estimate is the weak
measurement, and its yaw error becomes position error scaled by the lever arm to
the marker's origin. Two tags give a 75 mm baseline instead, and because the
notches are symmetric the baseline runs parallel to the marker's +x axis by
construction: yaw is ``atan2(R - L)`` with no offset to apply, the midpoint sits
on the centreline, and what is left is one scalar along the stem.

That scalar is the only quantity that depends on how the tags were printed::

    offset_y = -(4.8 mm + m)

where ``m`` is the white margin between the printed edge and the black square on
the side butted against the crossbar. It follows from the 31 mm black square --
the only thing the detector measures -- plus the marker's own geometry: the
crossbar's inner edge is at y = -10.7 mm and the origin at y = 0. The total
printed size never enters. A symmetric margin cancels out of the yaw entirely,
and out of the midpoint's x by symmetry.

    python -m vbrl.deployment.goal_pose artifacts/deployment/goal_frame.png
    python -m vbrl.deployment.goal_pose frame.png --margin-mm 7.0 --annotate out.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from vbrl.deployment.capture_board import EXTERNAL_SERIAL

# tag25h9 detection returns the corners of the outer *black* square. This is the
# number a detector's `tagsize` means, and it is not the printed square: the
# supplied artwork is 31 mm black inside a 45 mm sheet.
TAG_BLACK_M = 0.031
# White between the printed edge and the black square, on the side butted
# against the crossbar. Measured on the rig's own prints; override if you
# reprint. The SVG as generated implies 4.43 mm.
DEFAULT_MARGIN_M = 0.007
# The marker's crossbar inner edge, in its own frame. The origin is 10.7 mm past
# it along the stem -- not the shape's centre, which is 19.3 mm further still.
CROSSBAR_INNER_EDGE_M = -0.0107
# Tag ids, left then right in the marker's own frame. Two different ids so a
# swap cannot silently reverse the baseline and put the goal yaw 180 deg out.
TAG_ID_LEFT = 0
TAG_ID_RIGHT = 1
# The tabletop in the base frame. base_link sits on a 5 mm mount plate whose
# bottom is the table's top face, so the table -- and the paper on it -- is 5 mm
# below the frame the camera pose is written in.
TABLE_Z_BASE_M = -0.005
# What `target_pose` carries as its z: the command's target sits at the T's
# mid-height, 12 mm above the table, which is 7 mm in the base frame.
TARGET_Z_BASE_M = 0.0070

_CAMERA_RE = re.compile(
  r'<camera\s+name="external_cam"\s+pos="([^"]+)"\s*\n?\s*quat="([^"]+)"', re.M
)


def external_camera_pose(xml_path: Path) -> Any:
  """The external camera's 4x4 pose in the base frame, OpenCV convention.

  The MJCF is authoritative: `recalibrate` writes its solve straight into the
  `<camera>` element, so reading it back here cannot drift from the calibration
  the policy's own renders were matched against. MuJoCo cameras look down -z
  with +y up and OpenCV down +z with +y down, which is the flip
  `calibration.CV_TO_MUJOCO` carries.
  """
  import mujoco

  from vbrl.deployment.calibration import CV_TO_MUJOCO, transform

  match = _CAMERA_RE.search(xml_path.read_text())
  if match is None:
    raise ValueError(f"No external_cam <camera> with pos and quat in {xml_path}.")
  position = np.array([float(v) for v in match.group(1).split()])
  quaternion = np.array([float(v) for v in match.group(2).split()])
  rotation_mj = np.empty(9)
  mujoco.mju_quat2Mat(rotation_mj, quaternion)
  # mjcf_camera() builds the MuJoCo rotation as R_cv @ CV_TO_MUJOCO, and the
  # flip is its own inverse, so the same product undoes it.
  return transform(rotation_mj.reshape(3, 3) @ CV_TO_MUJOCO, position)


def camera_intrinsics(
  intrinsics_path: Path, serial: str, width: int, height: int
) -> tuple[Any, Any]:
  """(camera_matrix, distortion) for one unit at one capture resolution.

  Per unit and per mode rather than one shared constant: the two D405s on this
  rig differ by 0.6% in focal length and neither has its principal point at the
  image centre.
  """
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
  """Tag id -> 4x2 corner pixels, for every tag25h9 marker in the image."""
  import cv2

  detector = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_25h9),
    cv2.aruco.DetectorParameters(),
  )
  grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  corners, ids, _ = detector.detectMarkers(grey)
  if ids is None:
    return {}
  return {int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners)}


def pixel_on_table(
  pixel: Any, pose_cv: Any, matrix: Any, distortion: Any, plane_z: float
) -> Any:
  """Where a pixel's ray meets a horizontal plane, in the base frame.

  Ray-casting onto the known tabletop rather than trusting the tag's own solved
  depth: the plane is exact and a 31 mm tag's depth estimate at this range is
  not. Everything here lies on the table, so the plane costs nothing and removes
  the noisiest term.
  """
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
  """The marker's origin and yaw in the base frame, plus the policy's vector."""
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
  # One scalar along the marker's own +y, from the black square alone.
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

  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
