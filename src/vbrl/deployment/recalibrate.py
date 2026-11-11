from __future__ import annotations

import argparse
import functools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from vbrl.deployment import charuco
from vbrl.deployment.calibration import mjcf_camera, transform, wrist_camera_pose
from vbrl.deployment.push_t.view import EXTRINSICS, INTRINSICS

WRIST_PARENT = "link_6"
# The board lies on a plate as thick as the robot's mount plate, so the table is this
# far below the board's printed face.
BOARD_PLATE_M = 0.005


def _load_image(path: Path) -> Any:
  import cv2

  image = cv2.imread(str(path))
  if image is None:
    raise FileNotFoundError(f"Could not read {path}.")
  return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _intrinsics(intrinsics: Path, serial: str, mode: str) -> tuple[Any, Any]:
  camera = next(c for c in json.loads(intrinsics.read_text()) if c["serial"] == serial)
  native = camera["modes"][mode]["native"]
  matrix = np.array(
    [[native["fx"], 0.0, native["cx"]], [0.0, native["fy"], native["cy"]], [0, 0, 1]]
  )
  coefficients = camera["modes"][mode]["distortion"]["coeffs"]
  return matrix, np.asarray(coefficients, dtype=np.float64)


@functools.cache
def _kinematics() -> tuple[Any, Any]:
  import mujoco

  from vbrl.asset_zoo.robots.trossen_wxai import WXAI_MEASURED_XML

  model = mujoco.MjModel.from_xml_path(str(WXAI_MEASURED_XML))
  return model, mujoco.MjData(model)


def parent_pose(joint_pos: Any) -> Any:
  import mujoco

  model, data = _kinematics()
  data.qpos[:6] = joint_pos[:6]
  mujoco.mj_kinematics(model, data)
  body = model.body(WRIST_PARENT)
  return transform(data.xmat[body.id].reshape(3, 3), data.xpos[body.id])


def _matrix(vector: Any) -> Any:
  from scipy.spatial.transform import Rotation

  return transform(Rotation.from_rotvec(vector[:3]).as_matrix(), vector[3:])


def _vector(matrix: Any) -> Any:
  from scipy.spatial.transform import Rotation

  return np.r_[Rotation.from_matrix(matrix[:3, :3]).as_rotvec(), matrix[:3, 3]]


def _project(in_camera: Any, points: Any, matrix: Any, distortion: Any) -> Any:
  import cv2

  rotation = cv2.Rodrigues(in_camera[:3, :3])[0]
  pixels, _ = cv2.projectPoints(points, rotation, in_camera[:3, 3], matrix, distortion)
  return pixels.reshape(-1, 2)


def _rms(errors: Any) -> float:
  return float(np.sqrt(np.mean(np.square(errors))))


def _square(size: float) -> Any:
  half = size / 2.0
  return np.array(
    [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]]
  )


def hand_eye(
  views: list[dict[str, Any]],
  matrix: Any,
  distortion: Any,
  mount_guess: Any,
  board_guesses: list[Any],
) -> tuple[Any, list[Any], Any]:
  from scipy.optimize import least_squares

  def unpack(parameters: Any) -> tuple[Any, list[Any]]:
    boards = parameters[6:].reshape(-1, 6)
    return _matrix(parameters[:6]), [_matrix(board) for board in boards]

  def residuals(parameters: Any) -> Any:
    mount, boards = unpack(parameters)
    errors = []
    for view in views:
      in_camera = np.linalg.inv(view["parent"] @ mount) @ boards[view["board"]]
      projected = _project(in_camera, view["object_points"], matrix, distortion)
      errors.append((projected - view["image_points"]).ravel())
    return np.concatenate(errors)

  start = np.concatenate([_vector(mount_guess), *map(_vector, board_guesses)])
  fit = least_squares(residuals, start)
  mount, boards = unpack(fit.x)
  return mount, boards, fit.fun


def eye_to_hand(
  views: list[dict[str, Any]],
  matrix: Any,
  distortion: Any,
  camera_guess: Any,
  marker_guess: Any,
  size: float,
) -> tuple[Any, Any, Any]:
  from scipy.optimize import least_squares

  def residuals(parameters: Any) -> Any:
    camera, marker = _matrix(parameters[:6]), _matrix(parameters[6:])
    errors = []
    for view in views:
      in_camera = np.linalg.inv(camera) @ view["parent"] @ marker
      projected = _project(in_camera, _square(size), matrix, distortion)
      errors.append((projected - view["corners"]).ravel())
    return np.concatenate(errors)

  fit = least_squares(residuals, np.r_[_vector(camera_guess), _vector(marker_guess)])
  return _matrix(fit.x[:6]), _matrix(fit.x[6:]), fit.fun


def _board(image: Any, matrix: Any, distortion: Any) -> dict[str, Any] | None:
  detection = charuco.detect_dictionary(image)
  if detection["n_corners"] < charuco.MIN_CORNERS:
    return None
  return charuco.board_pose(detection, matrix, distortion)


def wrist_mount(boards: list[Path], intrinsics: Path) -> tuple[Any, float]:
  views, guesses, mount_guess = [], [], None
  for index, capture in enumerate(boards):
    manifest = json.loads((capture / "capture.json").read_text())
    for entry in manifest["views"]:
      if entry["camera"] != "wrist":
        continue
      mode = f"{entry['width']}x{entry['height']}"
      matrix, distortion = _intrinsics(intrinsics, manifest["wrist_serial"], mode)
      board = _board(_load_image(capture / entry["image"]), matrix, distortion)
      if board is None:
        continue
      parent = parent_pose(entry["joint_pos"])
      if mount_guess is None:
        mount_guess = np.linalg.inv(parent) @ wrist_camera_pose(entry["joint_pos"][:6])
      if len(guesses) == index:
        guesses.append(parent @ mount_guess @ board["pose"])
      views.append(
        {
          "board": index,
          "parent": parent,
          "object_points": board["object_points"].reshape(-1, 3),
          "image_points": board["image_points"].reshape(-1, 2),
        }
      )
  mount, _, errors = hand_eye(views, matrix, distortion, mount_guess, guesses)
  return mount, _rms(errors)


def external_camera(marker: Path) -> tuple[Any, float]:
  import cv2

  from vbrl.deployment.capture_marker import MARKER_M, detect
  from vbrl.deployment.push_t.view import ExternalView

  manifest = json.loads((marker / "capture.json").read_text())
  view = ExternalView.load(manifest["width"], manifest["height"])
  views = []
  for entry in manifest["views"]:
    corners = detect(_load_image(marker / entry["image"]))
    if corners is not None:
      views.append({"parent": parent_pose(entry["joint_pos"]), "corners": corners})
  _, rotation, translation = cv2.solvePnP(
    _square(MARKER_M),
    views[0]["corners"],
    view.matrix,
    view.distortion,
    flags=cv2.SOLVEPNP_IPPE_SQUARE,
  )
  in_camera = transform(cv2.Rodrigues(rotation)[0], translation.ravel())
  marker_guess = np.linalg.inv(views[0]["parent"]) @ view.pose @ in_camera
  camera, _, errors = eye_to_hand(
    views, view.matrix, view.distortion, view.pose, marker_guess, MARKER_M
  )
  return camera, _rms(errors)


def table_plane(boards: list[Path], camera: Any) -> tuple[list[float], float]:
  from vbrl.deployment.push_t.view import ExternalView

  points = []
  for capture in boards:
    image = _load_image(capture / "external.png")
    view = ExternalView.load(image.shape[1], image.shape[0])
    board = _board(image, view.matrix, view.distortion)
    in_base = camera @ board["pose"]
    corners = board["object_points"].reshape(-1, 3)
    points.append(corners @ in_base[:3, :3].T + in_base[:3, 3])
  points = np.concatenate(points)
  design = np.c_[np.ones(len(points)), points[:, :2]]
  coefficients = np.linalg.lstsq(design, points[:, 2], rcond=None)[0]
  residual = points[:, 2] - design @ coefficients
  coefficients[0] -= BOARD_PLATE_M
  return [round(float(v), 6) for v in coefficients], _rms(residual) * 1000.0


def _record(name: str, parent: str, new: Any, old: Any, rms: float) -> dict[str, Any]:
  from scipy.spatial.transform import Rotation

  position, quaternion, _ = mjcf_camera(new)
  old_position, old_quaternion, _ = mjcf_camera(old)
  rotation = Rotation.from_matrix(old[:3, :3].T @ new[:3, :3])
  return {
    "name": name,
    "parent": parent,
    "pos": [round(float(v), 6) for v in position],
    "quat": [round(float(v), 6) for v in quaternion],
    "rms_px": round(rms, 3),
    "previous": {
      "pos": [round(float(v), 6) for v in old_position],
      "quat": [round(float(v), 6) for v in old_quaternion],
    },
    "change": {
      "position_mm": [round(float(v) * 1000, 1) for v in position - old_position],
      "distance_mm": round(float(np.linalg.norm(position - old_position)) * 1000, 1),
      "rotation_deg": round(math.degrees(rotation.magnitude()), 2),
    },
  }


def solve(boards: list[Path], marker: Path, intrinsics: Path) -> dict[str, Any]:
  from vbrl.asset_zoo.robots.trossen_wxai import make_wxai_measured
  from vbrl.deployment.push_t.view import ExternalView

  mount, wrist_rms = wrist_mount(boards, intrinsics)
  camera, external_rms = external_camera(marker)
  plane, plane_rms = table_plane(boards, camera)
  joints = make_wxai_measured().home_joint_pos
  home = np.array([joints[f"joint_{i}"] for i in range(6)])
  old_mount = np.linalg.inv(parent_pose(home)) @ wrist_camera_pose(home)
  old_camera = ExternalView.load(1280, 720).pose
  return {
    "cameras": [
      _record("external_cam", "base_link", camera, old_camera, external_rms),
      _record("cam", WRIST_PARENT, mount, old_mount, wrist_rms),
    ],
    "table_plane": {
      "z_base_m": "c + a * x + b * y",
      "c_a_b": plane,
      "rms_mm": round(plane_rms, 3),
    },
    "data": {"boards": [str(b) for b in boards], "marker": str(marker)},
  }


def report(result: dict[str, Any]) -> None:
  for camera in result["cameras"]:
    change = camera["change"]
    print(
      f"{camera['name']} on {camera['parent']}: {camera['rms_px']:.2f} px rms, moved "
      f"{change['position_mm']} mm, {change['rotation_deg']:.2f} deg"
    )
    print(f'  pos="{" ".join(f"{v:.6f}" for v in camera["pos"])}"')
    print(f'  quat="{" ".join(f"{v:.6f}" for v in camera["quat"])}"')
  c, a, b = result["table_plane"]["c_a_b"]
  print(
    f"table: z = {c:+.5f} {a:+.5f} x {b:+.5f} y in the base frame "
    f"({result['table_plane']['rms_mm']:.2f} mm rms)"
  )
  print("Paste pos and quat into wxai_measured.xml; keep every other attribute.")


def main(argv: Any = None) -> int:
  parser = argparse.ArgumentParser(
    description="Solve the wrist camera on link_6 from board captures, the external "
    "camera from a marker capture and the table plane from the boards' external "
    "frames, and write the extrinsics."
  )
  parser.add_argument(
    "--board",
    type=Path,
    action="append",
    required=True,
    help="a capture_board directory; repeat for several",
  )
  parser.add_argument(
    "--marker", type=Path, required=True, help="a capture_marker directory"
  )
  parser.add_argument(
    "--intrinsics",
    type=Path,
    default=Path(INTRINSICS),
  )
  parser.add_argument("--out", type=Path, default=Path(EXTRINSICS))
  arguments = parser.parse_args(argv)

  result = solve(arguments.board, arguments.marker, arguments.intrinsics)
  report(result)
  arguments.out.parent.mkdir(parents=True, exist_ok=True)
  arguments.out.write_text(json.dumps(result, indent=2) + "\n")
  print(f"wrote {arguments.out}")
  return 0


__all__ = [
  "eye_to_hand",
  "external_camera",
  "hand_eye",
  "parent_pose",
  "report",
  "solve",
  "table_plane",
  "wrist_mount",
]


if __name__ == "__main__":
  raise SystemExit(main())
