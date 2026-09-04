"""Where the external camera is, from one chessboard seen by two cameras.

The wrist camera's pose in the base frame is known exactly: it is fixed in the
MJCF and the arm reports its joints, so forward kinematics places it. Point both
cameras at one board and that known pose transfers:

    T_base_board  = T_base_wrist @ T_wrist_board          (wrist image)
    T_base_extcam = T_base_board @ inv(T_extcam_board)    (external image)

which is the frame an MJCF <camera> inside base_link is written in.

    python -m vbrl.deployment.calibration --probe-intrinsics
    python -m vbrl.deployment.calibration \
      --wrist-image artifacts/deployment/wrist_cam_chess_848x480_Color.png \
      --external-image artifacts/deployment/external_tilted_848x480_Color.png
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

# MuJoCo cameras look down -z with +y up; OpenCV looks down +z with +y down.
# Flipping y and z converts between them, and the flip is its own inverse.
CV_TO_MUJOCO = np.diag([1.0, -1.0, -1.0])

# Hardware-measured pinhole intrinsics of the deployed 424x240 D405 colour
# stream. These are also the values from which the simulator's 224x224 crop FOV
# is derived in wxai_constants.py. The supplied 848x480 images have exactly
# twice the sampling density, so their focal lengths are twice these values.
# A single view of a planar board cannot independently identify focal length
# and camera-to-board distance, so the known camera intrinsics must remain fixed
# during the extrinsic solve.
D405_FX_424 = 217.7
D405_FY_424 = 217.5


def d405_intrinsics(width: int, height: int) -> dict[str, float]:
  scale = width / 424.0
  return {
    "fx": D405_FX_424 * scale,
    "fy": D405_FY_424 * scale,
    "cx": width / 2.0,
    "cy": height / 2.0,
  }


def read_intrinsics(path: str | None, width: int, height: int) -> dict[str, float]:
  if path is None:
    return d405_intrinsics(width, height)
  values = json.loads(open(path).read())
  missing = {"fx", "fy", "cx", "cy"} - set(values)
  if missing:
    raise ValueError(f"{path} is missing {sorted(missing)}.")
  return {key: float(values[key]) for key in ("fx", "fy", "cx", "cy")}


def camera_matrix(intrinsics: dict[str, float]) -> Any:
  return np.array(
    [
      [intrinsics["fx"], 0.0, intrinsics["cx"]],
      [0.0, intrinsics["fy"], intrinsics["cy"]],
      [0.0, 0.0, 1.0],
    ]
  )




def transform(rotation: Any, translation: Any) -> Any:
  pose = np.eye(4)
  pose[:3, :3] = rotation
  pose[:3, 3] = translation
  return pose


def board_points(columns: int, rows: int, square: float) -> Any:
  """Corner positions in the board frame: z=0, x along columns, y along rows."""
  grid = np.zeros((columns * rows, 3))
  grid[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square
  return grid


def find_corners(path: str, columns: int, rows: int) -> tuple[Any, tuple[int, int]]:
  import cv2

  image = cv2.imread(path, cv2.IMREAD_COLOR)
  if image is None:
    raise FileNotFoundError(f"Cannot read {path}.")
  grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  found, corners = cv2.findChessboardCorners(
    grey,
    (columns, rows),
    flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
  )
  if not found:
    raise RuntimeError(
      f"No {columns}x{rows} chessboard in {path}. The pattern counts inner "
      "corners, so a board of 10x7 squares is 9x6 here."
    )
  # The refinement window must be narrower than the gap between corners. A
  # fixed (11, 11) spans 23 px, wider than the 12 px spacing a small board in
  # an 848x480 frame gives, so neighbouring corners land in each other's window
  # and are dragged together -- a grid that still solves, at 4.1 px instead of
  # 0.2 px.
  window = max(2, int(median_spacing(corners, columns, rows) / 3))
  cv2.cornerSubPix(
    grey,
    corners,
    (window, window),
    (-1, -1),
    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
  )
  reject_irregular_grid(corners, columns, rows, path)
  return corners, (image.shape[1], image.shape[0])


def median_spacing(corners: Any, columns: int, rows: int) -> float:
  """Median gap between neighbouring corners, in pixels."""
  grid = corners.reshape(rows, columns, 2)
  return float(
    np.median(
      np.concatenate(
        [
          np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(),
          np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel(),
        ]
      )
    )
  )


def reject_irregular_grid(
  corners: Any, columns: int, rows: int, path: str, tolerance: float = 0.15
) -> None:
  """Refuse a grid too irregular to be a perspective view of a flat board.

  Mis-associated rows still converge in solvePnP and still report a pose, so
  the spacing is checked rather than trusted.
  """
  grid = corners.reshape(rows, columns, 2)
  for axis, label in ((1, "along rows"), (0, "down columns")):
    spacing = np.linalg.norm(np.diff(grid, axis=axis), axis=2)
    spread = float(spacing.std() / spacing.mean())
    if spread > tolerance:
      raise RuntimeError(
        f"{path}: corner spacing {label} is {spacing.mean():.2f} +/- "
        f"{spacing.std():.2f} px, a {spread:.0%} spread against a "
        f"{tolerance:.0%} limit. The detected grid does not match the board; "
        "reshoot with the board larger in frame."
      )


def solve_board_pose(
  corners: Any, columns: int, rows: int, square: float, intrinsics: dict[str, float]
) -> tuple[Any, float]:
  """Board pose in the OpenCV camera frame, and RMS reprojection error in px."""
  import cv2

  objects = board_points(columns, rows, square)
  matrix = camera_matrix(intrinsics)
  ok, rvec, tvec = cv2.solvePnP(objects, corners, matrix, None)
  if not ok:
    raise RuntimeError("solvePnP failed on a detected board.")
  projected, _ = cv2.projectPoints(objects, rvec, tvec, matrix, None)
  residual = projected.reshape(-1, 2) - corners.reshape(-1, 2)
  error = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
  rotation, _ = cv2.Rodrigues(rvec)
  return transform(rotation, tvec.reshape(3)), error


def flip_board_180(pose: Any, columns: int, rows: int, square: float) -> Any:
  """The other corner ordering a detector may return for the same board.

  findChessboardCorners scans from whichever corner comes first in the image,
  so the labelling is unique only up to a half turn about the board centre. Two
  cameras looking from different sides can disagree, and the disagreement would
  place the external camera reflected through the board. Reprojection error
  cannot separate them -- each fits its own image perfectly -- so the caller
  picks by where the camera lands.
  """
  half = np.diag([-1.0, -1.0, 1.0])
  centre = np.array([(columns - 1) * square / 2.0, (rows - 1) * square / 2.0, 0.0])
  return pose @ transform(half, centre - half @ centre)


def mjcf_camera(pose_cv: Any) -> tuple[Any, Any, Any]:
  """(pos, quat, xyaxes) for an MJCF <camera>, from an OpenCV pose."""
  import mujoco

  rotation = (pose_cv @ transform(CV_TO_MUJOCO, np.zeros(3)))[:3, :3]
  quaternion = np.empty(4)
  mujoco.mju_mat2Quat(quaternion, rotation.flatten())
  # xyaxes names the camera's x and y axes in the parent frame; MuJoCo derives
  # the viewing direction from their cross product.
  return pose_cv[:3, 3], quaternion, np.concatenate([rotation[:, 0], rotation[:, 1]])


def look_direction(pose_cv: Any) -> Any:
  """Unit vector the camera looks along, in the parent frame."""
  return pose_cv[:3, :3] @ np.array([0.0, 0.0, 1.0])


def wrist_camera_at_home() -> Any:
  """The wrist camera's pose in the base frame at the home pose, OpenCV axes.

  The home pose comes from the robot definition rather than a hard-coded
  vector, and is applied by joint name: the MJCF's qpos order and the model's
  joint order need not agree, and silently mismatching them would rotate the
  camera by a joint.
  """
  import mujoco

  from vbrl.asset_zoo.robots.trossen_wxai import WXAI_XML, make_wxai

  robot = make_wxai()
  model = mujoco.MjModel.from_xml_path(str(WXAI_XML))
  data = mujoco.MjData(model)
  for joint, angle in robot.home_joint_pos.items():
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    if index < 0:
      raise RuntimeError(f"No joint {joint!r} in {WXAI_XML}.")
    data.qpos[model.jnt_qposadr[index]] = angle
  mujoco.mj_kinematics(model, data)
  mujoco.mj_camlight(model, data)  # mj_kinematics alone does not place cameras

  name = robot.resolve_camera("wrist").model_name
  index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
  if index < 0:
    raise RuntimeError(f"No camera {name!r} in {WXAI_XML}.")
  in_mujoco = transform(data.cam_xmat[index].reshape(3, 3), data.cam_xpos[index])
  return in_mujoco @ transform(CV_TO_MUJOCO, np.zeros(3))


def probe_intrinsics() -> int:
  """Print the colour intrinsics of every connected RealSense, per mode."""
  import pyrealsense2 as rs

  devices = list(rs.context().devices)
  if not devices:
    print("No RealSense connected.")
    return 1
  for device in devices:
    print(
      f"{device.get_info(rs.camera_info.name)}  "
      f"serial {device.get_info(rs.camera_info.serial_number)}"
    )
    for sensor in device.sensors:
      for profile in sensor.get_stream_profiles():
        if profile.stream_type() != rs.stream.color:
          continue
        if not profile.is_video_stream_profile():
          continue
        video = profile.as_video_stream_profile()
        if (video.width(), video.height()) not in ((424, 240), (848, 480)):
          continue
        i = video.get_intrinsics()
        print(
          f"  {video.width()}x{video.height()}  "
          f'{{"fx": {i.fx:.3f}, "fy": {i.fy:.3f}, '
          f'"cx": {i.ppx:.3f}, "cy": {i.ppy:.3f}}}'
        )
  return 0


def main(argv: Any = None) -> int:
  parser = argparse.ArgumentParser(
    prog="python -m vbrl.deployment.calibration",
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument("--wrist-image")
  parser.add_argument("--external-image")
  parser.add_argument("--pattern", default="9x6", help="inner corners, columns x rows")
  parser.add_argument("--square", type=float, default=0.024, help="metres")
  parser.add_argument("--wrist-intrinsics", help="JSON with fx, fy, cx, cy")
  parser.add_argument("--external-intrinsics", help="JSON with fx, fy, cx, cy")
  parser.add_argument(
    "--probe-intrinsics",
    action="store_true",
    help="print connected RealSense colour intrinsics and exit",
  )
  arguments = parser.parse_args(argv)

  if arguments.probe_intrinsics:
    return probe_intrinsics()
  if not (arguments.wrist_image and arguments.external_image):
    parser.error("--wrist-image and --external-image are both required")

  columns, rows = (int(value) for value in arguments.pattern.lower().split("x"))
  square = arguments.square

  wrist_corners, wrist_size = find_corners(arguments.wrist_image, columns, rows)
  ext_corners, ext_size = find_corners(arguments.external_image, columns, rows)
  wrist_k = read_intrinsics(arguments.wrist_intrinsics, *wrist_size)
  ext_k = read_intrinsics(arguments.external_intrinsics, *ext_size)
  board_in_wrist, wrist_error = solve_board_pose(
    wrist_corners, columns, rows, square, wrist_k
  )
  board_in_ext, ext_error = solve_board_pose(ext_corners, columns, rows, square, ext_k)

  print(f"pattern    {columns}x{rows} inner corners, {square * 1000:.1f} mm squares")
  for label, size, given, k, error in (
    ("wrist   ", wrist_size, arguments.wrist_intrinsics, wrist_k, wrist_error),
    ("external", ext_size, arguments.external_intrinsics, ext_k, ext_error),
  ):
    source = "given" if given else "default D405 estimate, scaled"
    print(
      f"{label}   {size[0]}x{size[1]}  fx {k['fx']:.1f} fy {k['fy']:.1f} "
      f"cx {k['cx']:.1f} cy {k['cy']:.1f}  reproj {error:.3f} px  ({source})"
    )

  wrist_in_base = wrist_camera_at_home()
  board_in_base = wrist_in_base @ board_in_wrist
  normal = board_in_base[:3, :3] @ np.array([0.0, 0.0, 1.0])
  print("\nwrist camera at the home pose, in the base frame:")
  print(f"  pos               {np.round(wrist_in_base[:3, 3], 4)}")
  print("\nboard, placed by the wrist camera:")
  print(f"  origin in base    {np.round(board_in_base[:3, 3], 4)}")
  print(f"  normal in base    {np.round(normal, 4)}")
  print(
    f"  tilt off table    {np.degrees(np.arccos(min(1.0, abs(normal[2])))):.2f} deg"
    "  (a board lying flat reads near 0)"
  )

  print("\nexternal camera in the base frame, for both corner orderings:")
  chosen = None
  for label, candidate in (
    ("as detected", board_in_ext),
    ("half-turned", flip_board_180(board_in_ext, columns, rows, square)),
  ):
    pose = board_in_base @ np.linalg.inv(candidate)
    position, forward = pose[:3, 3], look_direction(pose)
    ok = position[2] > 0.05 and position[0] > 0.0 and forward[2] < 0.0
    print(
      f"  {label:12s} pos {np.round(position, 4)}  looking "
      f"{np.round(forward, 3)}   {'PLAUSIBLE' if ok else 'rejected'}"
    )
    if ok and chosen is None:
      chosen = (label, pose)

  if chosen is None:
    print(
      "\nNeither ordering puts the camera above the table in front of the base.\n"
      "The usual cause is the intrinsics: a wrong focal length scales the whole\n"
      "baseline. Run --probe-intrinsics on the cameras and pass the JSON."
    )
    return 1

  label, pose = chosen
  position, quaternion, xyaxes = mjcf_camera(pose)
  forward = look_direction(pose)
  print(f"\nusing the {label} ordering:\n")
  print('    <camera name="external_tilted_cam"')
  print(f'            pos="{position[0]:.6f} {position[1]:.6f} {position[2]:.6f}"')
  print(f'            xyaxes="{" ".join(f"{v:.6f}" for v in xyaxes)}"')
  print('            mode="fixed" fovy="42.5"/>')
  print(f"\n  quat equivalent   {np.round(quaternion, 6)}")
  print(f"  distance to base  {np.linalg.norm(position):.4f} m")
  print(f"  height above base {position[2]:.4f} m")
  tilt = np.degrees(np.arcsin(max(-1.0, min(1.0, -forward[2]))))
  print(f"  down-tilt         {tilt:.2f} deg below horizontal")
  return 0


__all__ = [
  "CV_TO_MUJOCO",
  "d405_intrinsics",
  "find_corners",
  "flip_board_180",
  "main",
  "mjcf_camera",
  "solve_board_pose",
  "wrist_camera_at_home",
]


if __name__ == "__main__":
  raise SystemExit(main())
