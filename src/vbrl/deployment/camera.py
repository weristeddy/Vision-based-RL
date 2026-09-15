from __future__ import annotations

import threading
from typing import Any

import numpy as np

SIZE = 224


def _colour_sensor(device: Any, rs: Any) -> Any:
  """The sensor carrying the colour stream.

  Not ``first_color_sensor()``: the D405 has no separate RGB module, its colour
  frames come off the stereo sensor, and that call raises there. Asking which
  sensor advertises a colour profile works on both layouts.
  """
  for sensor in device.query_sensors():
    for profile in sensor.get_stream_profiles():
      if profile.stream_type() == rs.stream.color:
        return sensor
  raise RuntimeError("No sensor on this device advertises a colour stream.")


class RealSenseCamera:
  """The newest 224x224 RGB frame.

  Capture runs on its own thread, so a sensor slower than the control rate
  cannot become the loop's clock. Frames are centre cropped rather than
  resized: on the 424x240 mode that is a 1:1 pixel map, and its 54.49-degree
  span is the ``fovy`` the simulator renders.
  """

  def __init__(self, config: Any) -> None:
    import pyrealsense2 as rs

    self._pipeline = rs.pipeline()
    stream = rs.config()
    stream.enable_stream(
      rs.stream.color,
      config.camera_width,
      config.camera_height,
      rs.format.rgb8,
      config.camera_fps,
    )
    profile = self._pipeline.start(stream)
    sensor = _colour_sensor(profile.get_device(), rs)
    if config.camera_exposure_us is None:
      sensor.set_option(rs.option.enable_auto_exposure, 1)
    else:
      # Manual, because auto meters the whole frame and a pale tabletop filling
      # it drives the sensor to saturation -- 42% of pixels pinned at 255 on the
      # first logged run, which is a uniform white field where training had
      # texture.
      sensor.set_option(rs.option.enable_auto_exposure, 0)
      sensor.set_option(rs.option.exposure, float(config.camera_exposure_us))

    self._lock = threading.Lock()
    self._frame = self._crop(self._await_frame())
    self._stop = threading.Event()
    self._thread = threading.Thread(target=self._capture, daemon=True)
    self._thread.start()

  def frame(self) -> Any:
    """The newest frame. Never blocks."""
    with self._lock:
      return self._frame

  def saturated_fraction(self) -> float:
    """Share of channels pinned at 255 in the newest frame.

    Printed at startup because a clipped frame is invisible from the numbers
    the loop otherwise reports: the policy keeps producing plausible actions
    from an image whose background has been erased.
    """
    with self._lock:
      return float((self._frame >= 255).mean())

  def close(self) -> None:
    self._stop.set()
    self._thread.join(timeout=2.0)
    self._pipeline.stop()

  def _await_frame(self) -> Any:
    return self._pipeline.wait_for_frames(timeout_ms=5000).get_color_frame()

  def _crop(self, colour: Any) -> Any:
    image = np.asanyarray(colour.get_data())
    height, width = image.shape[:2]
    top, left = (height - SIZE) // 2, (width - SIZE) // 2
    return np.ascontiguousarray(image[top : top + SIZE, left : left + SIZE])

  def _capture(self) -> None:
    while not self._stop.is_set():
      colour = self._await_frame()
      if not colour:
        continue
      cropped = self._crop(colour)
      with self._lock:
        self._frame = cropped


__all__ = ["SIZE", "RealSenseCamera"]


def _probe(argv: Any = None) -> int:
  """Sweep colour exposures and report how much of the frame clips.

      python -m vbrl.deployment.camera
      python -m vbrl.deployment.camera --width 424 --height 240

  The number to put in a manifest's ``camera_exposure_us`` is the largest one
  whose clipped share is a few percent: the policy needs the tabletop to carry
  texture, because that is what every one of the 1203 training tables had.
  """
  import argparse

  import pyrealsense2 as rs

  parser = argparse.ArgumentParser(description=_probe.__doc__.splitlines()[0])
  parser.add_argument("--width", type=int, default=424)
  parser.add_argument("--height", type=int, default=240)
  parser.add_argument("--fps", type=int, default=None)
  parser.add_argument(
    "--exposures",
    type=float,
    nargs="+",
    default=(200, 400, 800, 1200, 1600, 2400, 3200, 5490),
  )
  arguments = parser.parse_args(argv)

  # Same reason intrinsics.py tries the slow rates first: two D405s on one bus
  # do not always deliver a mode at its advertised rate, and a probe needs one
  # frame rather than throughput.
  rates = (arguments.fps,) if arguments.fps else (5, 15, 30, 60)
  pipeline = rs.pipeline()
  for rate in rates:
    stream = rs.config()
    stream.enable_stream(
      rs.stream.color, arguments.width, arguments.height, rs.format.rgb8, rate
    )
    try:
      profile = pipeline.start(stream)
      pipeline.wait_for_frames(timeout_ms=4000)
    except RuntimeError:
      try:
        pipeline.stop()
      except RuntimeError:
        pass
      continue
    print(f"{arguments.width}x{arguments.height} at {rate} fps")
    break
  else:
    raise SystemExit(
      f"No frame at {arguments.width}x{arguments.height} at any of {rates}."
    )
  sensor = _colour_sensor(profile.get_device(), rs)
  print(f"{'exposure':>10} {'clipped':>9} {'>=250':>8} {'median':>7} {'mean':>7}")
  try:
    sensor.set_option(rs.option.enable_auto_exposure, 1)
    for _ in range(15):
      pipeline.wait_for_frames(timeout_ms=5000)
    auto = np.asanyarray(
      pipeline.wait_for_frames(timeout_ms=5000).get_color_frame().get_data()
    )
    print(
      f"{'auto':>10} {(auto >= 255).mean() * 100:8.1f}% "
      f"{(auto >= 250).mean() * 100:7.1f}% {np.median(auto):7.0f} {auto.mean():7.1f}"
    )
    sensor.set_option(rs.option.enable_auto_exposure, 0)
    for exposure in arguments.exposures:
      sensor.set_option(rs.option.exposure, float(exposure))
      # The sensor needs a few frames to apply a new exposure.
      for _ in range(8):
        pipeline.wait_for_frames(timeout_ms=5000)
      image = np.asanyarray(
        pipeline.wait_for_frames(timeout_ms=5000).get_color_frame().get_data()
      )
      print(
        f"{exposure:9.0f}u {(image >= 255).mean() * 100:8.1f}% "
        f"{(image >= 250).mean() * 100:7.1f}% {np.median(image):7.0f} "
        f"{image.mean():7.1f}"
      )
  finally:
    pipeline.stop()
  return 0


if __name__ == "__main__":
  raise SystemExit(_probe())
