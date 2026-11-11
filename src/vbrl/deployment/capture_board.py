from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from vbrl.deployment.intrinsics import open_stream

WRIST_SERIAL = "412622271761"
EXTERNAL_SERIAL = "260322270027"
# The highest colour mode: the old 7 x 10 board gave 46 of 54 corners, 31 at 848x480.
WIDTH, HEIGHT = 1280, 720
# Wall-clock, not a frame count, so the settle does not depend on the rate.
SETTLE_SECONDS = 3.0
MOVE_SECONDS = 4.0
# Home, then 19 poses chosen for spread in shoulder, elbow and wrist (camera 24-47 cm
# above the base) so the joint zero offsets separate. Each keeps 16+ board corners above
# the gripper fingers, and the tool 12+ cm above the base on every path between poses.
# Offsets from home in radians, joint_0 to joint_5.
DELTAS: tuple[tuple[float, ...], ...] = (
  (+0.00, +0.00, +0.00, +0.00, +0.00, +0.00),
  (-0.23, -0.35, -0.09, -0.20, +0.04, +0.01),
  (-0.13, -0.34, +0.29, -0.22, -0.07, +0.05),
  (+0.18, -0.34, +0.32, -0.19, +0.32, -0.19),
  (+0.13, -0.26, -0.01, -0.07, +0.16, -0.60),
  (-0.09, -0.08, -0.18, +0.26, +0.07, -0.54),
  (+0.23, -0.32, -0.34, +0.35, +0.29, -0.57),
  (+0.02, -0.32, -0.21, -0.08, -0.25, -0.49),
  (-0.24, -0.08, +0.31, -0.09, -0.04, -0.51),
  (-0.11, +0.23, +0.12, -0.16, -0.21, -0.58),
  (-0.08, +0.32, +0.35, -0.07, +0.12, -0.57),
  (-0.25, +0.17, -0.15, +0.22, -0.31, -0.24),
  (+0.14, -0.30, -0.32, +0.29, -0.29, +0.21),
  (+0.21, -0.06, -0.33, +0.23, +0.12, +0.16),
  (+0.02, -0.35, -0.33, +0.04, +0.33, +0.52),
  (+0.25, -0.22, +0.13, -0.03, +0.19, +0.59),
  (+0.21, +0.24, +0.27, -0.18, +0.24, +0.57),
  (-0.16, +0.26, +0.34, -0.10, -0.35, +0.32),
  (-0.12, -0.08, -0.09, +0.15, -0.24, +0.55),
  (-0.07, -0.33, +0.26, -0.07, -0.32, +0.60),
)
ARM_JOINTS = ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5")


def poses(scale: float = 1.0) -> list[dict[str, Any]]:
  from vbrl.asset_zoo.robots.trossen_wxai import make_wxai_measured

  home = make_wxai_measured().home_joint_pos
  gripper = home["right_carriage_joint"]
  out = []
  for index, offsets in enumerate(DELTAS):
    delta = {j: v * scale for j, v in zip(ARM_JOINTS, offsets, strict=True) if v}
    angles = [home[joint] + delta.get(joint, 0.0) for joint in ARM_JOINTS]
    out.append(
      {
        "index": index,
        "delta": dict(delta),
        "command": [*angles, gripper],
      }
    )
  return out


def _settle_and_grab(pipeline: Any, seconds: float = SETTLE_SECONDS) -> Any:
  deadline = time.monotonic() + seconds
  frame = None
  while time.monotonic() < deadline:
    frame = pipeline.wait_for_frames(timeout_ms=10000).get_color_frame()
  if frame is None:
    frame = pipeline.wait_for_frames(timeout_ms=10000).get_color_frame()
  return np.asanyarray(frame.get_data())


def _write(path: Path, image: Any) -> None:
  import cv2

  path.parent.mkdir(parents=True, exist_ok=True)
  cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def _inspect(image: Any) -> dict[str, Any]:
  from vbrl.deployment import charuco

  grey = np.asarray(image).mean(axis=2) if np.asarray(image).ndim == 3 else image
  report: dict[str, Any] = {
    "mean_brightness": round(float(np.mean(grey)), 1),
    "saturated_fraction": round(float(np.mean(grey > 250)), 4),
  }
  try:
    found = charuco.detect_dictionary(image)
    report.update(
      dictionary=found["dictionary"],
      n_markers=found["n_markers"],
      n_corners=found["n_corners"],
    )
  except RuntimeError as error:
    report["detect_error"] = str(error)
  return report


def dry_run(scale: float = 1.0) -> int:
  from vbrl.deployment.calibration import wrist_camera_pose

  print(f"{len(DELTAS)} wrist poses (pose 0 IS the home pose), then one external")
  print("frame with the arm folded to zero so it occludes nothing.\n")
  print(f"{'i':>3}  {'delta':<34}{'camera xyz in base frame (m)':<34}")
  base = None
  for pose in poses(scale):
    matrix = wrist_camera_pose(pose["command"][:6])
    position = matrix[:3, 3]
    if base is None:
      base = position
    print(
      f"{pose['index']:>3}  {str(pose['delta']) or '{} (home)':<34}"
      f"{np.round(position, 4)!s:<34}"
      f"moved {np.linalg.norm(position - base) * 1000:6.1f} mm"
    )
  print("\nNothing was sent to the arm. Re-run with --execute to capture.")
  return 0


def capture(destination: Path, arm_ip: str, settle: float, scale: float) -> int:
  from vbrl.deployment import charuco
  from vbrl.deployment.arm import REST_POSE, TrossenArm
  from vbrl.deployment.calibration import wrist_camera_pose
  from vbrl.deployment.config import DeploymentConfig

  destination.mkdir(parents=True, exist_ok=True)
  config = DeploymentConfig.__new__(DeploymentConfig)
  object.__setattr__(config, "arm_ip", arm_ip)
  object.__setattr__(config, "arm_model", "wxai_v0")
  object.__setattr__(config, "motor_parameters", "wxai_v0_20260317")
  object.__setattr__(config, "motion", None)

  def log(label: str, entry: dict[str, Any]) -> None:
    print(
      f"  {label:<11} markers {entry.get('n_markers', 0):>2}"
      f"  corners {entry.get('n_corners', 0):>2}"
      f"  brightness {entry['mean_brightness']:>5.1f}"
      f"  saturated {entry['saturated_fraction'] * 100:4.1f}%"
      f"  {entry.get('detect_error', '')}"
    )

  views: list[dict[str, Any]] = []
  arm = TrossenArm(config)
  try:
    wrist, wrist_fps = open_stream(WRIST_SERIAL, WIDTH, HEIGHT)
    print(f"wrist camera streaming {WIDTH}x{HEIGHT} at {wrist_fps} fps")
    try:
      for pose in poses(scale):
        arm.move_to(pose["command"], seconds=MOVE_SECONDS)
        time.sleep(0.4)  # let the servo settle before trusting the reading
        measured, _ = arm.read()
        image = _settle_and_grab(wrist, settle)
        path = destination / f"wrist_{pose['index']:02d}.png"
        _write(path, image)
        entry = {
          "image": path.name,
          "camera": "wrist",
          "serial": WRIST_SERIAL,
          "width": WIDTH,
          "height": HEIGHT,
          "fps": wrist_fps,
          # The MEASURED joints, not the commanded ones: forward kinematics has
          # to describe where the arm actually is.
          "joint_pos": [round(float(v), 6) for v in measured],
          "commanded": pose["command"],
          "delta": pose["delta"],
        }
        entry.update(_inspect(image))
        views.append(entry)
        log(f"wrist {pose['index']:>2}", entry)
    finally:
      wrist.stop()

    print("\nFolding the arm to zero so it does not occlude the external view.")
    arm.move_to(REST_POSE, seconds=MOVE_SECONDS)
    time.sleep(1.0)
    external, external_fps = open_stream(EXTERNAL_SERIAL, WIDTH, HEIGHT)
    print(f"external camera streaming {WIDTH}x{HEIGHT} at {external_fps} fps")
    try:
      image = _settle_and_grab(external, max(settle, SETTLE_SECONDS))
    finally:
      external.stop()
    path = destination / "external.png"
    _write(path, image)
    entry = {
      "image": path.name,
      "camera": "external",
      "serial": EXTERNAL_SERIAL,
      "width": WIDTH,
      "height": HEIGHT,
      "fps": external_fps,
    }
    entry.update(_inspect(image))
    views.append(entry)
    log("external", entry)
  finally:
    try:
      arm.park(seconds=MOVE_SECONDS)
    finally:
      arm.close()
    if views:
      manifest = {
        "version": 1,
        "board": {
          "squares": list(charuco.SQUARES),
          "square_m": charuco.SQUARE_M,
          "marker_m": charuco.MARKER_M,
        },
        "wrist_serial": WRIST_SERIAL,
        "external_serial": EXTERNAL_SERIAL,
        "home_camera_in_base": wrist_camera_pose().tolist(),
        "views": views,
      }
      path = destination / "capture.json"
      path.write_text(json.dumps(manifest, indent=2) + "\n")
      print(f"\nwrote {path} with {len(views)} views")

  usable = sum(
    1 for v in views if v["camera"] == "wrist" and v.get("n_corners", 0) >= charuco.MIN_CORNERS
  )
  print(f"{usable} of {len(DELTAS)} wrist views have enough corners to solve.")
  return 0 if usable >= 3 else 1


def main(argv: Any = None) -> int:
  global WIDTH, HEIGHT  # noqa: PLW0603
  parser = argparse.ArgumentParser(description="Photograph the ChArUco board from many wrist poses.")
  parser.add_argument(
    "--out",
    type=Path,
    default=Path("artifacts/deployment/charuco"),
    help="Directory for the images and capture.json.",
  )
  parser.add_argument("--arm-ip", default="192.168.1.2")
  parser.add_argument(
    "--resolution",
    default=f"{WIDTH}x{HEIGHT}",
    help="Colour mode for the board images. Must be one this camera reports "
    "intrinsics for.",
  )
  parser.add_argument(
    "--delta-scale",
    type=float,
    default=1.0,
    help="Multiply every offset from home. >1 widens the baseline, <1 keeps a "
    "board that is dropping out of view.",
  )
  parser.add_argument(
    "--settle",
    type=float,
    default=SETTLE_SECONDS,
    help="Seconds to drain frames before each capture, for auto-exposure.",
  )
  mode = parser.add_mutually_exclusive_group()
  mode.add_argument(
    "--execute",
    action="store_true",
    help="Actually move the arm and capture. Moving the arm is the default OFF.",
  )
  mode.add_argument(
    "--dry-run",
    action="store_true",
    help="Print the poses and their forward kinematics without touching the arm "
    "(this is also what happens with no flag at all).",
  )
  arguments = parser.parse_args(argv)
  WIDTH, HEIGHT = (int(v) for v in arguments.resolution.lower().split("x"))
  if not arguments.execute:
    return dry_run(arguments.delta_scale)
  return capture(
    arguments.out, arguments.arm_ip, arguments.settle, arguments.delta_scale
  )


if __name__ == "__main__":
  raise SystemExit(main())
