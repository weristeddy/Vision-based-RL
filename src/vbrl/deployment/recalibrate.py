"""Solve ``external_cam``'s pose from a capture_board run.

The chain, all in OpenCV axes until the last step:

    T_base_board  = T_base_wristcam(joints_i) @ T_wristcam_board(image_i)
    T_base_extcam = T_base_board @ inv(T_extcam_board)

The first line is evaluated once per wrist view. The board does not move, so the
views must agree; their spread is the error bar, and it is reported rather than
hidden. The combined estimate uses the geometric median of the translations and
the chordal mean of the rotations, so one view that lost half its corners cannot
drag the answer.

The result is expressed in the BASE frame, which is exactly the frame an MJCF
``<camera>`` inside ``base_link`` is written in -- every external camera in this
robot is parented there. So it is written to the XML unchanged. In particular the
5 mm mounting plate is NOT added: the plate lifts the arm and its cameras
together in world coordinates, leaving their relative pose untouched, and adding
it here would move the camera 5 mm relative to the arm it is measured against.

    python -m vbrl.deployment.recalibrate --capture artifacts/deployment/charuco
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from vbrl.deployment import charuco
from vbrl.deployment.calibration import mjcf_camera, wrist_camera_pose

# Fallback only. Each view records the mode it was shot in, and the intrinsics
# are looked up per view, so the capture resolution can change without touching
# this file.
DEFAULT_MODE = "1280x720"


def _camera_matrix(values: dict[str, float]) -> Any:
  return np.array(
    [
      [values["fx"], 0.0, values["cx"]],
      [0.0, values["fy"], values["cy"]],
      [0.0, 0.0, 1.0],
    ]
  )


def _chordal_mean_rotation(rotations: list[Any]) -> Any:
  """The rotation minimising summed squared Frobenius distance to the inputs.

  The arithmetic mean of rotation matrices is not a rotation; projecting it back
  onto SO(3) through an SVD is, and it is the closed-form L2 mean. Averaging
  Euler angles or raw quaternion components would both be wrong here -- the
  former is not even well defined near a gimbal, the latter needs sign handling.
  """
  average = np.mean(np.stack(rotations), axis=0)
  u, _, vt = np.linalg.svd(average)
  rotation = u @ vt
  if np.linalg.det(rotation) < 0.0:  # reflection, not a rotation
    u[:, -1] *= -1.0
    rotation = u @ vt
  return rotation


def _geometric_median(points: Any, iterations: int = 128) -> Any:
  """Weiszfeld's algorithm: robust to a view that is simply wrong."""
  points = np.asarray(points, dtype=np.float64)
  estimate = np.median(points, axis=0)
  for _ in range(iterations):
    offsets = points - estimate
    distances = np.linalg.norm(offsets, axis=1)
    if np.any(distances < 1e-12):
      return estimate
    weights = 1.0 / distances
    update = (points * weights[:, None]).sum(axis=0) / weights.sum()
    if np.linalg.norm(update - estimate) < 1e-12:
      return update
    estimate = update
  return estimate


def _load_image(path: Path) -> Any:
  import cv2

  image = cv2.imread(str(path))
  if image is None:
    raise FileNotFoundError(f"Could not read {path}.")
  return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _intrinsics(
  cameras: list[dict[str, Any]], serial: str, mode_name: str = DEFAULT_MODE
) -> dict[str, Any]:
  for entry in cameras:
    if entry["serial"] == serial:
      mode = entry["modes"].get(mode_name)
      if mode is None or "error" in mode:
        raise RuntimeError(f"Serial {serial} has no usable {mode_name} intrinsics.")
      return {
        "native": mode["native"],
        "distortion": np.asarray(mode["distortion"]["coeffs"], dtype=np.float64),
      }
  raise RuntimeError(f"No intrinsics recorded for serial {serial}.")


def solve(capture_dir: Path, intrinsics_path: Path) -> dict[str, Any]:
  manifest = json.loads((capture_dir / "capture.json").read_text())
  cameras = json.loads(intrinsics_path.read_text())

  def mode_of(view: dict[str, Any]) -> str:
    return f"{view.get('width', 0)}x{view.get('height', 0)}"

  wrist_views = [v for v in manifest["views"] if v["camera"] == "wrist"]
  if not wrist_views:
    raise RuntimeError("The capture has no wrist views.")
  wrist = _intrinsics(cameras, manifest["wrist_serial"], mode_of(wrist_views[0]))
  wrist_k = _camera_matrix(wrist["native"])

  board_in_base: list[Any] = []
  per_view: list[dict[str, Any]] = []
  detections: list[dict[str, Any]] = []
  external_view = None
  for view in manifest["views"]:
    if view["camera"] == "external":
      external_view = view
      continue
    detection = charuco.detect_dictionary(_load_image(capture_dir / view["image"]))
    detections.append(detection)
    if detection["n_corners"] < charuco.MIN_CORNERS:
      per_view.append({"image": view["image"], "skipped": detection["n_corners"]})
      continue
    pose = charuco.board_pose(detection, wrist_k, wrist["distortion"])
    base_wristcam = wrist_camera_pose(view["joint_pos"][:6])
    base_board = base_wristcam @ pose["pose"]
    board_in_base.append(base_board)
    per_view.append(
      {
        "image": view["image"],
        "n_corners": detection["n_corners"],
        "rms_px": round(pose["rms_px"], 4),
        "board_xyz": [round(float(v), 5) for v in base_board[:3, 3]],
        "camera_distance_m": round(float(np.linalg.norm(pose["pose"][:3, 3])), 4),
      }
    )

  if len(board_in_base) < 3:
    raise RuntimeError(
      f"Only {len(board_in_base)} usable wrist views; need at least 3 to have "
      "any cross-check at all."
    )
  if external_view is None:
    raise RuntimeError("The capture has no external view.")

  translations = np.stack([m[:3, 3] for m in board_in_base])
  board = np.eye(4)
  board[:3, 3] = _geometric_median(translations)
  board[:3, :3] = _chordal_mean_rotation([m[:3, :3] for m in board_in_base])

  spread = np.linalg.norm(translations - board[:3, 3], axis=1)
  angles = []
  for matrix in board_in_base:
    relative = board[:3, :3].T @ matrix[:3, :3]
    angles.append(
      float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))))
    )

  external = _intrinsics(
    cameras, manifest["external_serial"], mode_of(external_view)
  )
  external_k = _camera_matrix(external["native"])
  external_detection = charuco.detect_dictionary(
    _load_image(capture_dir / external_view["image"])
  )
  external_pose = charuco.board_pose(
    external_detection, external_k, external["distortion"]
  )
  base_extcam = board @ np.linalg.inv(external_pose["pose"])
  position, quaternion, xyaxes = mjcf_camera(base_extcam)

  intrinsic_check: dict[str, Any] | None = None
  try:
    intrinsic_check = charuco.calibrate_intrinsics(
      detections, (int(wrist["native"]["width"]), int(wrist["native"]["height"]))
    )
  except RuntimeError as error:
    intrinsic_check = {"error": str(error)}

  return {
    "per_view": per_view,
    "n_used": len(board_in_base),
    "board_in_base": board.tolist(),
    "board_spread_mm": {
      "median": round(float(np.median(spread)) * 1000, 3),
      "max": round(float(spread.max()) * 1000, 3),
    },
    "board_angle_spread_deg": {
      "median": round(float(np.median(angles)), 4),
      "max": round(float(np.max(angles)), 4),
    },
    "external": {
      "n_corners": external_detection["n_corners"],
      "rms_px": round(external_pose["rms_px"], 4),
      "distance_m": round(float(np.linalg.norm(external_pose["pose"][:3, 3])), 4),
    },
    "external_cam": {
      "pos": [round(float(v), 6) for v in position],
      "quat": [round(float(v), 6) for v in quaternion],
      "xyaxes": [round(float(v), 6) for v in xyaxes],
    },
    "wrist_intrinsic_self_check": intrinsic_check,
  }


def report(result: dict[str, Any]) -> None:
  print(f"{'image':<16}{'corners':>8}{'rms px':>9}   board xyz in base (m)")
  print("-" * 74)
  for view in result["per_view"]:
    if "skipped" in view:
      print(f"{view['image']:<16}{view['skipped']:>8}   skipped (too few corners)")
      continue
    print(
      f"{view['image']:<16}{view['n_corners']:>8}{view['rms_px']:>9.4f}   "
      f"{view['board_xyz']}"
    )
  spread = result["board_spread_mm"]
  angle = result["board_angle_spread_deg"]
  print(
    f"\n{result['n_used']} views agree on the board to a median "
    f"{spread['median']:.2f} mm / {angle['median']:.3f} deg "
    f"(worst {spread['max']:.2f} mm / {angle['max']:.3f} deg)."
  )
  print(
    "That spread is the REPEATABILITY of the answer below, not a residual of the\n"
    "fit -- but it is not the whole error either. It cannot see anything common\n"
    "to every view: the focal lengths, the board's square size, or the wrist\n"
    "camera's own position in the MJCF. Those bias all 15 views alike and stay\n"
    "invisible here. The board landing on z = 0 is the check that catches them."
  )
  board_z = result["board_in_base"][2][3]
  print(
    f"Board came out {board_z * 1000:+.2f} mm off the tabletop plane; nothing in "
    f"the solve\nwas told it is flat on the table, so that is a free end-to-end "
    "check on the\nscale and the forward kinematics together."
  )
  ext = result["external"]
  print(
    f"\nexternal view: {ext['n_corners']} corners, {ext['rms_px']:.4f} px rms, "
    f"board {ext['distance_m']:.3f} m away"
  )
  camera = result["external_cam"]
  print("\nexternal_cam, in the base frame -- paste into both robot MJCFs:")
  print('      <camera name="external_cam"')
  print(f'              pos="{" ".join(f"{v:.6f}" for v in camera["pos"])}"')
  print(f'              quat="{" ".join(f"{v:.6f}" for v in camera["quat"])}"')
  print('              mode="fixed" fovy="54.284"/>')
  print("   Keep the fovy already in the file. It is this unit's own 224-crop")
  print("   vertical field of view from the factory intrinsics, nothing in Python")
  print("   overrides it any more, and this solve does not measure it -- a single")
  print("   view of a planar board cannot separate focal length from distance.")
  check = result["wrist_intrinsic_self_check"]
  if "error" in check:
    print(f"\nwrist intrinsic self-check: {check['error']}")
  else:
    print(
      f"\nwrist intrinsic self-check over {check['n_views']} views: "
      f"fx {check['fx']:.3f} fy {check['fy']:.3f} "
      f"cx {check['cx']:.3f} cy {check['cy']:.3f}  (rms {check['rms_px']:.4f} px)"
    )
    print("   Compare with the factory values; agreement is the evidence that")
    print("   either is trustworthy. The solve above used the factory numbers.")


def main(argv: Any = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--capture", type=Path, default=Path("artifacts/deployment/charuco"))
  parser.add_argument(
    "--intrinsics",
    type=Path,
    default=Path("artifacts/deployment/intrinsics/intrinsics.json"),
  )
  parser.add_argument("--json", type=Path, default=None, help="Write the full result.")
  arguments = parser.parse_args(argv)

  result = solve(arguments.capture, arguments.intrinsics)
  report(result)
  if arguments.json is not None:
    arguments.json.parent.mkdir(parents=True, exist_ok=True)
    arguments.json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {arguments.json}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
