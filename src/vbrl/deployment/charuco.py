"""ChArUco board detection, intrinsic calibration and single-view board pose.

The rig's board: 40 mm squares with 30 mm markers from a DICT_4X4 dictionary,
280 x 400 mm, holding 54 interior corners and 35 markers. Given to OpenCV as
7 x 10 squares with the legacy layout -- see SQUARES and LEGACY_PATTERN, both of
which were identified by fitting rather than assumed, because getting either
wrong interpolates zero corners while still decoding every marker.

Two reasons this is a module rather than inline in the calibration script. The
OpenCV 5 ArUco API dropped ``estimatePoseCharucoBoard`` and
``calibrateCameraCharuco``, so both the pose and the intrinsics now go through
``CharucoDetector.detectBoard`` -> ``board.matchImagePoints`` -> ``solvePnP`` /
``calibrateCamera``; that path is worth writing once. And a ChArUco board, unlike
a plain chessboard, identifies every corner by marker ID, which means a partial
view still yields an unambiguous pose -- the property the wrist camera needs,
because at the home pose the gripper occludes part of the board.

Which DICT_4X4 size the board was generated from is not recorded anywhere, so
:func:`detect_dictionary` tries all four and keeps whichever recognises the most
markers rather than making the caller guess.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# (7, 10), not the (10, 7) the board is described by: OpenCV's first dimension
# counts squares across the board's own x axis, and this board's marker layout
# only matches when it is given as 7 wide by 10 tall. Identified by fitting: with
# (10, 7) the markers still decode but no chessboard corner interpolates at all,
# and a pose forced through the marker corners reprojects at 60 px instead of 1.2.
SQUARES = (7, 10)
SQUARE_M = 0.040
MARKER_M = 0.030
# This board predates OpenCV 4.6's change to the ChArUco layout, so the marker
# placement is the legacy one. Not a detail: under the modern layout the markers
# decode identically and NOT ONE chessboard corner is interpolated -- 45 of 54
# with it, 0 without. Silent, and fatal to the solve.
LEGACY_PATTERN = True
# A 10 x 7 board has (10-1) x (7-1) interior corners and (10*7)//2 markers.
CORNERS = (SQUARES[0] - 1) * (SQUARES[1] - 1)
MARKERS = (SQUARES[0] * SQUARES[1]) // 2
DICTIONARIES = ("DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000")
# solvePnP needs 4 points for a solution, but a pose from 4 near-collinear
# corners of a planar target is worthless. Twelve keeps a usable spread and
# still tolerates the gripper covering a third of the board.
MIN_CORNERS = 12


def make_board(dictionary: str = DICTIONARIES[0]) -> Any:
  import cv2

  name = getattr(cv2.aruco, dictionary, None)
  if name is None:
    raise ValueError(f"Unknown ArUco dictionary {dictionary!r}.")
  board = cv2.aruco.CharucoBoard(
    SQUARES, SQUARE_M, MARKER_M, cv2.aruco.getPredefinedDictionary(name)
  )
  board.setLegacyPattern(LEGACY_PATTERN)
  return board


def _grey(image: Any) -> Any:
  import cv2

  array = np.asarray(image)
  if array.ndim == 2:
    return array
  if array.ndim == 3 and array.shape[2] == 3:
    # Detection is on luminance, and the RGB/BGR channel order changes it only
    # through the weights -- immaterial for a black-and-white target.
    return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
  raise ValueError(f"Expected a grey or 3-channel image, got {array.shape}.")


def detect(image: Any, dictionary: str) -> dict[str, Any]:
  """Interior corners and their IDs, plus how many markers were recognised."""
  import cv2

  detector = cv2.aruco.CharucoDetector(make_board(dictionary))
  corners, ids, marker_corners, marker_ids = detector.detectBoard(_grey(image))
  return {
    "dictionary": dictionary,
    # float32, not float64: matchImagePoints asserts CV_32F on the corners.
    "corners": None if corners is None else np.asarray(corners, dtype=np.float32),
    "ids": None if ids is None else np.asarray(ids).reshape(-1),
    "n_corners": 0 if ids is None else int(len(ids)),
    "n_markers": 0 if marker_ids is None else int(len(marker_ids)),
  }


def detect_dictionary(image: Any) -> dict[str, Any]:
  """Best of the four DICT_4X4 sizes, ranked by interpolated corners.

  Corners first, markers only as a tie-break: the corners are what the pose is
  solved from, and a larger dictionary can decode a few spurious markers whose
  IDs fall outside the board and interpolate nothing. Ranking by markers picked
  DICT_4X4_1000 with 4 markers and 0 corners over DICT_4X4_50 on a real frame.
  """
  attempts = [detect(image, name) for name in DICTIONARIES]
  best = max(attempts, key=lambda a: (a["n_corners"], a["n_markers"]))
  if best["n_markers"] == 0:
    raise RuntimeError(
      "No ArUco markers found under any DICT_4X4 size. Check the board is in "
      "frame, lit without glare, and that it really is a 4x4 dictionary."
    )
  return best


def board_pose(
  detection: dict[str, Any], camera_matrix: Any, distortion: Any
) -> dict[str, Any]:
  """``T_camera_board`` in OpenCV axes, with the reprojection error it achieves.

  SQPNP rather than the default iterative solver: it is a global method, so a
  planar target cannot settle into the mirrored local minimum that an iterative
  solve starting from a bad guess can fall into.
  """
  import cv2

  if detection["n_corners"] < MIN_CORNERS:
    raise RuntimeError(
      f"Only {detection['n_corners']} interior corners; need {MIN_CORNERS}."
    )
  board = make_board(detection["dictionary"])
  object_points, image_points = board.matchImagePoints(
    detection["corners"], detection["ids"]
  )
  ok, rotation, translation = cv2.solvePnP(
    object_points,
    image_points,
    camera_matrix,
    distortion,
    flags=cv2.SOLVEPNP_SQPNP,
  )
  if not ok:
    raise RuntimeError("solvePnP failed on the ChArUco corners.")
  rotation, translation = cv2.solvePnPRefineLM(
    object_points, image_points, camera_matrix, distortion, rotation, translation
  )
  projected, _ = cv2.projectPoints(
    object_points, rotation, translation, camera_matrix, distortion
  )
  residuals = np.linalg.norm(
    projected.reshape(-1, 2) - image_points.reshape(-1, 2), axis=1
  )
  matrix = np.eye(4)
  matrix[:3, :3] = cv2.Rodrigues(rotation)[0]
  matrix[:3, 3] = translation.reshape(3)
  return {
    "pose": matrix,
    "rms_px": float(np.sqrt(np.mean(residuals**2))),
    "max_px": float(residuals.max()),
    "n_corners": detection["n_corners"],
    "object_points": object_points,
    "image_points": image_points,
  }


def calibrate_intrinsics(
  detections: list[dict[str, Any]], size: tuple[int, int]
) -> dict[str, Any]:
  """Refine fx, fy, cx, cy and distortion from many views of the board.

  Reported alongside the factory intrinsics rather than instead of them: a
  RealSense ships factory-calibrated, and a self-calibration from a handful of
  views can easily be worse. Agreement between the two is the evidence that
  either is trustworthy; disagreement says the view set is too narrow.
  """
  import cv2

  usable = [d for d in detections if d["n_corners"] >= MIN_CORNERS]
  if len(usable) < 6:
    raise RuntimeError(
      f"Only {len(usable)} views with >= {MIN_CORNERS} corners; "
      "intrinsic calibration needs at least 6 and prefers 15+."
    )
  object_points, image_points = [], []
  for detection in usable:
    board = make_board(detection["dictionary"])
    obj, img = board.matchImagePoints(detection["corners"], detection["ids"])
    object_points.append(obj)
    image_points.append(img)
  rms, matrix, distortion, _, _ = cv2.calibrateCamera(
    object_points, image_points, size, None, None
  )
  return {
    "n_views": len(usable),
    "rms_px": float(rms),
    "fx": float(matrix[0, 0]),
    "fy": float(matrix[1, 1]),
    "cx": float(matrix[0, 2]),
    "cy": float(matrix[1, 2]),
    "distortion": [float(c) for c in np.asarray(distortion).reshape(-1)],
  }


__all__ = [
  "CORNERS",
  "DICTIONARIES",
  "MARKERS",
  "LEGACY_PATTERN",
  "MARKER_M",
  "MIN_CORNERS",
  "SQUARES",
  "SQUARE_M",
  "board_pose",
  "calibrate_intrinsics",
  "detect",
  "detect_dictionary",
  "make_board",
]
