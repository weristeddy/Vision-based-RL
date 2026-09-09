"""Photograph the ChArUco board from many wrist poses, plus once from outside.

What the extrinsic solve needs, and why this script exists to produce it:

  * The wrist camera's pose in the base frame is *known* for any joint vector --
    the MJCF fixes the camera on link_6 and the arm reports its joints -- so a
    wrist view of the board places the board in the base frame. Recording the
    joint angles beside each image is therefore not optional; an image without
    them is useless.
  * One wrist view is enough in principle. Many views are what makes the answer
    trustworthy: the board does not move, so every view must agree on
    ``T_base_board``, and the spread across views measures the error directly
    instead of leaving it unknown. A single view offers no such check.
  * The external camera is fixed, so it needs exactly one frame -- taken with the
    arm folded to zero so it occludes nothing.

Images are captured at 848x480. That mode is a provable exact 2x of the deployed
424x240 (fx, fy and the principal point all scale by exactly 2.0000, and the
field of view is identical), so it is the same optics at twice the angular
precision: a 0.2 px corner error costs 0.026 deg instead of 0.053. The 224 crop
the policy sees is the wrong thing to calibrate a physical pose through.

    python -m vbrl.deployment.capture_board --dry-run
    python -m vbrl.deployment.capture_board --execute --arm-ip 192.168.1.2
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

WRIST_SERIAL = "412622271761"
EXTERNAL_SERIAL = "260322270027"
# 1280x720, the highest colour mode. Measured on the real board: 46 of 54
# corners here against 31 at 848x480, and fx 653 px against 435 -- so both more
# corners and finer angular resolution. It is not an exact scaling of the
# deployed 424x240 the way 848x480 is, which does not matter: the camera's
# optical centre is physically fixed, and every mode's own intrinsics are
# measured separately, so a pose solved here is the same physical pose.
WIDTH, HEIGHT = 1280, 720
# Two D405s on this Jetson advertise 30 fps at 848x480 but do not deliver it --
# wait_for_frames times out. Try the slowest rate first: a calibration frame
# needs no throughput, and a mode that delivers beats a fast one that stalls.
FPS_CANDIDATES = (5, 15, 30)
# Auto-exposure needs time, and the external view was washed out at 20 frames.
# Drain for a fixed wall-clock instead of a frame count so the settle does not
# depend on the rate a mode happens to deliver.
SETTLE_SECONDS = 3.0
MOVE_SECONDS = 4.0
# The board sits still, so views must differ by where the camera looks from.
# joint_0 swings the whole arm sideways; joint_3/4/5 re-aim the wrist without
# moving it far, which keeps a board that is visible at home in frame. Kept
# small deliberately: a pose that loses the board yields nothing, and the solve
# gains more from 12 modest views that all see it than from 4 dramatic ones.
DELTAS: tuple[dict[str, float], ...] = (
  {},
  {"joint_0": +0.12},
  {"joint_0": -0.12},
  {"joint_3": +0.15},
  {"joint_3": -0.15},
  {"joint_4": +0.18},
  {"joint_4": -0.18},
  {"joint_5": +0.30},
  {"joint_5": -0.30},
  {"joint_0": +0.10, "joint_4": +0.14},
  {"joint_0": -0.10, "joint_4": -0.14},
  {"joint_1": +0.08, "joint_3": +0.10},
  {"joint_1": -0.08, "joint_3": -0.10},
  {"joint_0": +0.08, "joint_5": +0.22},
  {"joint_0": -0.08, "joint_5": -0.22},
)
ARM_JOINTS = ("joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5")


def poses(scale: float = 1.0) -> list[dict[str, Any]]:
  """The commanded joint vectors, as the arm wants them: 6 joints + gripper.

  Pose 0 is the home pose itself; the rest are small offsets from it, so the
  wrist sees the board from a spread of angles without leaving the region where
  it is visible at all. ``scale`` widens or narrows every offset at once: after
  a run that kept the board in all 15 views, raise it for more baseline; if
  views are dropping out, lower it.
  """
  from vbrl.asset_zoo.robots.trossen_wxai import make_wxai

  home = make_wxai().home_joint_pos
  gripper = home["right_carriage_joint"]
  out = []
  for index, delta in enumerate(DELTAS):
    delta = {joint: value * scale for joint, value in delta.items()}
    angles = [home[joint] + delta.get(joint, 0.0) for joint in ARM_JOINTS]
    out.append(
      {
        "index": index,
        "delta": dict(delta),
        "command": [*angles, gripper],
      }
    )
  return out


def _open(serial: str) -> tuple[Any, int]:
  """A streaming pipeline that has actually delivered a frame, and its rate.

  Starting a pipeline is not evidence it works: ``start`` succeeds at 30 fps on
  a bus that then never produces a frame. So every candidate rate is proven by
  pulling one frame before it is accepted.
  """
  import pyrealsense2 as rs

  errors = []
  for fps in FPS_CANDIDATES:
    pipeline = rs.pipeline()
    stream = rs.config()
    stream.enable_device(serial)
    stream.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.rgb8, fps)
    try:
      pipeline.start(stream)
      pipeline.wait_for_frames(timeout_ms=8000)
      return pipeline, fps
    except Exception as error:  # noqa: BLE001 -- report every failure together
      errors.append(f"{fps} fps: {type(error).__name__}")
      try:
        pipeline.stop()
      except Exception:  # noqa: BLE001 -- already failing; nothing to salvage
        pass
  raise RuntimeError(
    f"Serial {serial} delivered no frame at {WIDTH}x{HEIGHT} "
    f"({'; '.join(errors)}). Is another process holding the camera?"
  )


def _settle_and_grab(pipeline: Any, seconds: float = SETTLE_SECONDS) -> Any:
  """Drain frames for ``seconds`` so auto-exposure converges, then take one."""
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
  """Board detection plus a brightness read, so a washed-out frame is visible."""
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
  """Print the poses, and where each one aims the camera on the tabletop."""
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
  """Drive the arm through the poses, photograph the board, write a manifest.

  The arm is parked in a ``finally`` that wraps *everything*: an earlier version
  parked it only after the external frame, so a camera timeout mid-sweep left
  the arm powered and holding a pose.
  """
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
    wrist, wrist_fps = _open(WRIST_SERIAL)
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
    external, external_fps = _open(EXTERNAL_SERIAL)
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
    # Whatever happened above, hand the arm back safely.
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
  # One CLI knob reaching module state, declared before any use of the names.
  global WIDTH, HEIGHT  # noqa: PLW0603
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
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
