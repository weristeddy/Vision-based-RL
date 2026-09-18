from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

MODES = ((424, 240), (848, 480), (1280, 720))
CROP = 224
# Both cameras advertise 30 fps at every mode, but two D405s on one bus do not always
# deliver the high-resolution ones there.
FPS_CANDIDATES = (5, 15, 30)


def _distortion(intrinsics: Any) -> dict[str, Any]:
  return {
    "model": str(intrinsics.model).rsplit(".", 1)[-1],
    "coeffs": [round(float(c), 8) for c in intrinsics.coeffs],
  }


def crop_intrinsics(values: dict[str, float], size: int = CROP) -> dict[str, float]:
  left = (values["width"] - size) / 2.0
  top = (values["height"] - size) / 2.0
  return {
    "width": float(size),
    "height": float(size),
    "fx": values["fx"],
    "fy": values["fy"],
    "cx": values["cx"] - left,
    "cy": values["cy"] - top,
  }


def field_of_view_deg(values: dict[str, float]) -> tuple[float, float]:
  return (
    2.0 * math.degrees(math.atan(values["width"] / (2.0 * values["fx"]))),
    2.0 * math.degrees(math.atan(values["height"] / (2.0 * values["fy"]))),
  )


def open_stream(serial: str, width: int, height: int) -> tuple[Any, int]:
  import pyrealsense2 as rs

  errors = []
  for fps in FPS_CANDIDATES:
    pipeline = rs.pipeline()
    stream = rs.config()
    stream.enable_device(serial)
    stream.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    try:
      pipeline.start(stream)
      pipeline.wait_for_frames(timeout_ms=8000)
      return pipeline, fps
    except Exception as error:  # noqa: BLE001 -- report every failure together
      errors.append(f"{fps} fps: {type(error).__name__}")
      try:
        pipeline.stop()
      except Exception:  # noqa: BLE001 -- already failing, nothing to salvage
        pass
  raise RuntimeError(
    f"Serial {serial} delivered no frame at {width}x{height} "
    f"({'; '.join(errors)}). Is another process holding the camera?"
  )


def _exposure_sensor(pipeline: Any) -> Any:
  import pyrealsense2 as rs

  for sensor in pipeline.get_active_profile().get_device().query_sensors():
    if sensor.supports(rs.option.exposure):
      return sensor
  return None


def saturated_fraction(image: Any) -> float:
  array = np.asarray(image)
  grey = array.mean(axis=2) if array.ndim == 3 else array
  return float(np.mean(grey > 250))


def grab_settled(
  pipeline: Any, seconds: float = 3.0, max_saturated: float = 0.02
) -> tuple[Any, dict[str, Any]]:
  import pyrealsense2 as rs

  deadline = time.monotonic() + seconds
  frame = None
  while time.monotonic() < deadline:
    frame = pipeline.wait_for_frames(timeout_ms=10000).get_color_frame()
  if frame is None:
    frame = pipeline.wait_for_frames(timeout_ms=10000).get_color_frame()
  image = np.asanyarray(frame.get_data())
  info: dict[str, Any] = {"auto_exposure": True}

  sensor = _exposure_sensor(pipeline)
  if sensor is not None:
    info["exposure_us"] = float(sensor.get_option(rs.option.exposure))
  clipped = saturated_fraction(image)
  info["saturated_fraction"] = round(clipped, 4)
  if sensor is None or clipped <= max_saturated:
    return image, info

  exposure = info.get("exposure_us", 33000.0)
  low = sensor.get_option_range(rs.option.exposure).min
  sensor.set_option(rs.option.enable_auto_exposure, 0)
  info["auto_exposure"] = False
  for _ in range(12):
    exposure = max(low, exposure * 0.55)
    sensor.set_option(rs.option.exposure, exposure)
    for _ in range(6):
      frame = pipeline.wait_for_frames(timeout_ms=10000).get_color_frame()
    image = np.asanyarray(frame.get_data())
    clipped = saturated_fraction(image)
    if clipped <= max_saturated or exposure <= low:
      break
  info["exposure_us"] = float(exposure)
  info["saturated_fraction"] = round(clipped, 4)
  return image, info


def read_cameras(
  save: Path | None = None, settle: float = 3.0
) -> list[dict[str, Any]]:
  import pyrealsense2 as rs

  devices = list(rs.context().devices)
  if not devices:
    raise RuntimeError("No RealSense connected.")

  results = []
  for device in devices:
    serial = device.get_info(rs.camera_info.serial_number)
    entry: dict[str, Any] = {
      "name": device.get_info(rs.camera_info.name),
      "serial": serial,
      "firmware": device.get_info(rs.camera_info.firmware_version),
      "modes": {},
    }
    for width, height in MODES:
      try:
        pipeline, fps = open_stream(serial, width, height)
      except RuntimeError as error:
        entry["modes"][f"{width}x{height}"] = {"error": str(error)}
        continue
      profile = pipeline.get_active_profile()
      try:
        video = profile.get_stream(rs.stream.color).as_video_stream_profile()
        native = video.get_intrinsics()
        values = {
          "width": float(native.width),
          "height": float(native.height),
          "fx": round(float(native.fx), 4),
          "fy": round(float(native.fy), 4),
          "cx": round(float(native.ppx), 4),
          "cy": round(float(native.ppy), 4),
        }
        mode = {
          "native": values,
          "distortion": _distortion(native),
          "fov_deg": [round(v, 3) for v in field_of_view_deg(values)],
          f"crop_{CROP}": crop_intrinsics(values),
        }
        mode[f"crop_{CROP}_fov_deg"] = [
          round(v, 3) for v in field_of_view_deg(mode[f"crop_{CROP}"])
        ]
        entry["modes"][f"{width}x{height}"] = mode

        mode["fps"] = fps
        if save is not None:
          # One frame per camera per mode, exposed so it is actually legible.
          # Never fatal: the intrinsics above are the point of this script.
          try:
            image, exposure = grab_settled(pipeline, settle)
            save.mkdir(parents=True, exist_ok=True)
            path = save / f"{serial}_{width}x{height}.png"
            import cv2

            cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            mode["frame"] = str(path)
            mode["exposure"] = exposure
          except Exception as error:  # noqa: BLE001
            mode["frame_error"] = f"{type(error).__name__}: {error}"
      finally:
        pipeline.stop()
    results.append(entry)
  return results


def report(cameras: list[dict[str, Any]]) -> None:
  for entry in cameras:
    print(f"\n{entry['name']}  serial {entry['serial']}  fw {entry['firmware']}")
    for mode, values in entry["modes"].items():
      if "error" in values:
        print(f"  {mode:<9} {values['error']}")
        continue
      n = values["native"]
      centre_x, centre_y = n["width"] / 2.0, n["height"] / 2.0
      print(
        f"  {mode:<9} fx {n['fx']:8.3f}  fy {n['fy']:8.3f}  "
        f"cx {n['cx']:8.3f}  cy {n['cy']:8.3f}"
      )
      print(
        f"  {'':<9} principal point offset from centre: "
        f"({n['cx'] - centre_x:+.3f}, {n['cy'] - centre_y:+.3f}) px"
      )
      print(
        f"  {'':<9} FOV {values['fov_deg'][0]:.2f} x {values['fov_deg'][1]:.2f} deg"
        f"   distortion {values['distortion']['model']} "
        f"{values['distortion']['coeffs']}"
      )
      crop = values[f"crop_{CROP}"]
      fov = values[f"crop_{CROP}_fov_deg"]
      print(
        f"  {'':<9} {CROP}x{CROP} centre crop: cx {crop['cx']:.3f} cy {crop['cy']:.3f}"
        f"   FOV {fov[0]:.2f} x {fov[1]:.2f} deg"
      )
      if "exposure" in values:
        e = values["exposure"]
        mode_word = "auto" if e["auto_exposure"] else "fixed"
        print(
          f"  {'':<9} frame {values.get('frame', '-')}  at {values.get('fps', '?')} fps"
          f"   exposure {e.get('exposure_us', float('nan')):.0f} us ({mode_word})"
          f"   clipped {e['saturated_fraction'] * 100:.2f}%"
        )
      elif "frame_error" in values:
        print(f"  {'':<9} frame unavailable ({values['frame_error']})")


def main(argv: Any = None) -> int:
  parser = argparse.ArgumentParser(description="Per-unit colour intrinsics of every connected RealSense.")
  parser.add_argument(
    "--settle",
    type=float,
    default=3.0,
    help="Seconds of auto-exposure settling before each saved frame.",
  )
  parser.add_argument(
    "--save",
    type=Path,
    default=None,
    help="Directory for one frame per camera plus intrinsics.json.",
  )
  arguments = parser.parse_args(argv)

  cameras = read_cameras(arguments.save, arguments.settle)
  report(cameras)
  if arguments.save is not None:
    arguments.save.mkdir(parents=True, exist_ok=True)
    destination = arguments.save / "intrinsics.json"
    destination.write_text(json.dumps(cameras, indent=2) + "\n")
    print(f"\nwrote {destination}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
