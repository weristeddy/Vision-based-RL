"""The two devices: the Trossen arm and the RealSense camera.

Both are thin. Everything about *what* to send and *what* the pixels mean lives
in the task ID and the policy; these classes only move bytes, and refuse to
guess when the hardware disagrees with the configuration.
"""

from __future__ import annotations

from typing import Any


class TrossenArm:
  """Joint-position control over Ethernet, with the clamps the sim did not need."""

  def __init__(
    self,
    *,
    ip: str,
    model: str = "wxai_v0",
    max_joint_step: float = 0.05,
    max_gripper_step: float = 0.005,
  ) -> None:
    import trossen_arm

    available = [name for name in dir(trossen_arm.Model) if not name.startswith("_")]
    if model not in available:
      raise ValueError(f"Unknown arm model {model!r}; valid choices are {available}.")

    self._trossen_arm = trossen_arm
    self._driver = trossen_arm.TrossenArmDriver()
    self._driver.configure(
      getattr(trossen_arm.Model, model),
      trossen_arm.StandardEndEffector.wxai_v0_base,
      ip,
      False,
    )
    self._max_joint_step = max_joint_step
    self._max_gripper_step = max_gripper_step
    self._last_target: Any = None

  @property
  def num_joints(self) -> int:
    return int(self._driver.get_num_joints())

  def read(self) -> tuple[Any, Any]:
    """Measured joint positions and velocities, gripper last."""
    import numpy as np

    positions = np.asarray(self._driver.get_all_positions(), dtype=np.float64)
    velocities = np.asarray(self._driver.get_all_velocities(), dtype=np.float64)
    return positions, velocities

  def hold(self) -> None:
    """Put every joint in position mode at wherever it currently is."""
    positions, _ = self.read()
    self._driver.set_all_modes(self._trossen_arm.Mode.position)
    self._driver.set_all_positions(positions.tolist(), 2.0, True)
    self._last_target = positions

  def command(self, targets: Any) -> Any:
    """Clamp a target against the previous one and send it. Returns what was sent."""
    import numpy as np

    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    if self._last_target is None:
      self._last_target = self.read()[0]

    # Per-step clamps, not absolute ones: the policy was trained without action
    # clipping, so the failure mode to guard is a large jump between steps.
    limits = np.full_like(targets, self._max_joint_step)
    limits[-1] = self._max_gripper_step
    delta = np.clip(targets - self._last_target, -limits, limits)
    clamped = self._last_target + delta

    self._driver.set_all_positions(clamped.tolist(), 0.0, False)
    self._last_target = clamped
    return clamped

  def relax(self) -> None:
    self._driver.set_all_modes(self._trossen_arm.Mode.idle)

  def close(self) -> None:
    try:
      self.relax()
    finally:
      self._driver.cleanup()


class RealSenseCamera:
  """RGB frames at the resolution the encoder was trained on."""

  def __init__(
    self,
    *,
    width: int,
    height: int,
    fps: int = 60,
    serial: str | None = None,
    square_crop: bool = True,
  ) -> None:
    import threading

    import pyrealsense2 as rs

    self._rs = rs
    self._width = width
    self._height = height
    self._square_crop = square_crop

    # Ask the device what it actually offers rather than naming a mode: a D435
    # has 640x480, a D405's colour comes off its stereo imagers with its own
    # list, and enable_stream on a mode the device lacks fails outright.
    modes = _colour_modes(rs, serial=serial)
    capture_width, capture_height, rates = _choose_size(
      modes, target_size=max(width, height)
    )
    # A D405 advertises 60 and 90 fps colour modes it does not always deliver:
    # the stream starts and no frame ever arrives. So open the fastest requested
    # rate, demand a frame, and step down -- rather than hanging on a rate the
    # device only claims to support.
    self._pipeline, self._profile, capture_fps, first = _open_colour_stream(
      rs,
      serial=serial,
      width=capture_width,
      height=capture_height,
      rates=[rate for rate in rates if rate <= fps] or rates,
    )
    self._fps = capture_fps
    self._capture_size = (capture_width, capture_height)

    # Capture runs on its own thread and the control loop only ever reads the
    # newest frame. Calling wait_for_frames() inline instead would make the
    # camera the clock: at 30 fps every step would block for up to 33 ms, and a
    # 50 Hz loop would quietly become a 30 Hz one -- changing the control period
    # the policy was trained at, which is a behavioural change rather than a
    # dropped frame.
    self._lock = threading.Lock()
    self._latest: Any = self._prepare(first)
    self._latest_at = _now()
    self._frames_captured = 1
    self._stop = threading.Event()
    self._thread = threading.Thread(target=self._capture, name="realsense", daemon=True)
    self._thread.start()

  def _prepare(self, color: Any) -> Any:
    """Crop (and if unavoidable resize) one colour frame to the policy's input."""
    import numpy as np

    image = np.asanyarray(color.get_data())
    if self._square_crop:
      image = _centre_crop(image, self._height, self._width)
    if image.shape[:2] != (self._height, self._width):
      image = _resize_nearest(image, self._height, self._width)
    return np.ascontiguousarray(image)

  def _capture(self) -> None:
    while not self._stop.is_set():
      try:
        frames = self._pipeline.wait_for_frames(timeout_ms=1000)
      except Exception:  # noqa: BLE001 - a dropped frame must not kill the thread
        continue
      color = frames.get_color_frame()
      if not color:
        continue
      image = self._prepare(color)
      with self._lock:
        self._latest = image
        self._latest_at = _now()
        self._frames_captured += 1

  @property
  def fps(self) -> int:
    return self._fps

  @property
  def capture_size(self) -> tuple[int, int]:
    return self._capture_size

  @property
  def frames_captured(self) -> int:
    return self._frames_captured

  def frame(self) -> Any:
    """The newest RGB frame, and how old it is in seconds.

    Never blocks. A frame is reused when the control loop is faster than the
    sensor, which is the intended trade: stale pixels for an unchanged control
    period.
    """
    import time

    with self._lock:
      if self._latest is None:
        raise RuntimeError("No RealSense frame available.")
      return self._latest, time.perf_counter() - self._latest_at

  def close(self) -> None:
    self._stop.set()
    self._thread.join(timeout=2.0)
    self._pipeline.stop()


def _now() -> float:
  import time

  return time.perf_counter()


def _colour_modes(rs: Any, *, serial: str | None) -> set[tuple[int, int, int]]:
  """Every rgb8 colour mode the connected device advertises."""
  devices = list(rs.context().query_devices())
  if not devices:
    raise RuntimeError("No RealSense device connected.")

  device = devices[0]
  if serial is not None:
    matching = [
      d for d in devices if d.get_info(rs.camera_info.serial_number) == serial
    ]
    if not matching:
      found = [d.get_info(rs.camera_info.serial_number) for d in devices]
      raise ValueError(f"No RealSense with serial {serial!r}; connected: {found}.")
    device = matching[0]

  modes: set[tuple[int, int, int]] = set()
  for sensor in device.query_sensors():
    for profile in sensor.get_stream_profiles():
      if profile.stream_type() != rs.stream.color:
        continue
      if str(profile.format()) != "format.rgb8":
        continue
      video = profile.as_video_stream_profile()
      modes.add((video.width(), video.height(), profile.fps()))
  return modes


def _choose_size(
  modes: set[tuple[int, int, int]], *, target_size: int
) -> tuple[int, int, list[int]]:
  """Smallest frame that still covers the policy's input, and its rates.

  Smallest, because the crop takes the same angular slice at any resolution
  while a larger capture only adds bytes to move and pixels to throw away. On a
  D405 this selects 424x240, whose 224x224 centre crop is a 1:1 pixel map.
  """
  usable = [m for m in modes if min(m[0], m[1]) >= target_size]
  if not usable:
    raise RuntimeError(
      f"No rgb8 colour mode of at least {target_size} px on the short side; "
      f"the device offers {sorted(modes)}."
    )
  width, height = min({(w, h) for w, h, _ in usable}, key=lambda s: s[0] * s[1])
  rates = sorted(
    {f for w, h, f in usable if (w, h) == (width, height)}, reverse=True
  )
  return width, height, rates


def _open_colour_stream(
  rs: Any,
  *,
  serial: str | None,
  width: int,
  height: int,
  rates: list[int],
  allow_reset: bool = True,
) -> tuple[Any, Any, int, Any]:
  """Start the fastest rate that actually delivers a frame.

  Kept to a single ``start`` on the common path on purpose: cycling a D405
  through several start/stop pairs leaves it enumerating but no longer
  streaming, so probing every advertised rate is what *causes* the failure it
  was meant to detect. A wedged device recovers only from a hardware reset, so
  that is the last resort here rather than a step in the search.
  """
  import time

  failures = []
  for rate in rates:
    pipeline = rs.pipeline()
    config = rs.config()
    if serial is not None:
      config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, rate)
    try:
      profile = pipeline.start(config)
    except Exception as error:  # noqa: BLE001 - an unusable mode is a normal answer
      failures.append(f"{rate} fps: start failed ({str(error)[:60]})")
      continue
    try:
      frames = pipeline.wait_for_frames(timeout_ms=3000)
      color = frames.get_color_frame()
      if color:
        return pipeline, profile, rate, color
      failures.append(f"{rate} fps: frames without a colour plane")
    except Exception as error:  # noqa: BLE001 - advertised but not delivered
      failures.append(f"{rate} fps: {str(error)[:60]}")
    try:
      pipeline.stop()
    except Exception:  # noqa: BLE001 - nothing to unwind if the start half-failed
      pass
    # librealsense needs a moment between pipeline cycles on the same device.
    time.sleep(0.4)

  if allow_reset:
    _hardware_reset(rs, serial=serial)
    return _open_colour_stream(
      rs,
      serial=serial,
      width=width,
      height=height,
      rates=rates,
      allow_reset=False,
    )

  raise RuntimeError(
    f"The device advertises {width}x{height} but delivered no colour frame, "
    f"even after a hardware reset:\n  " + "\n  ".join(failures)
  )


def _hardware_reset(rs: Any, *, serial: str | None, timeout_s: float = 20.0) -> None:
  """Reset the camera and wait for it to come back on the bus."""
  import time

  devices = list(rs.context().query_devices())
  if not devices:
    raise RuntimeError("No RealSense device to reset.")
  for device in devices:
    if serial is None or device.get_info(rs.camera_info.serial_number) == serial:
      device.hardware_reset()
      break

  deadline = time.perf_counter() + timeout_s
  time.sleep(4.0)  # it disappears from the bus before it reappears
  while time.perf_counter() < deadline:
    if list(rs.context().query_devices()):
      time.sleep(2.0)  # settle before the first stream
      return
    time.sleep(1.0)
  raise RuntimeError(f"The camera did not re-enumerate within {timeout_s:.0f}s.")


def _centre_crop(image: Any, height: int, width: int) -> Any:
  """Crop the centred window the simulator's field of view was matched to.

  Cropping straight to the policy's own 224x224 rather than to the largest
  square is what makes the pipeline resample-free on a D405: its smallest rgb8
  mode is 424x240, so a 224x224 centre crop is a 1:1 pixel map into the network,
  and its 54.49-degree span is exactly the `fovy` the renderer now uses.
  Squashing the 16:9 frame instead would compress it horizontally by 1.77x, so a
  cube would arrive as an ellipse the encoder never saw.
  """
  frame_height, frame_width = image.shape[:2]
  if frame_height < height or frame_width < width:
    # Too small to crop: fall back to the largest square, and let the caller
    # resize. The field of view then no longer matches the render.
    side = min(frame_height, frame_width)
    height = width = side
  top = (frame_height - height) // 2
  left = (frame_width - width) // 2
  return image[top : top + height, left : left + width]


def _resize_nearest(image: Any, height: int, width: int) -> Any:
  """Nearest-neighbour resize, so no extra dependency is needed for one call."""
  import numpy as np

  rows = (np.arange(height) * (image.shape[0] / height)).astype(np.int64)
  cols = (np.arange(width) * (image.shape[1] / width)).astype(np.int64)
  return image[rows[:, None], cols[None, :]]


__all__ = ["RealSenseCamera", "TrossenArm"]
