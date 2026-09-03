from __future__ import annotations

import threading
from typing import Any

import numpy as np

SIZE = 224


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
    self._pipeline.start(stream)

    self._lock = threading.Lock()
    self._frame = self._crop(self._await_frame())
    self._stop = threading.Event()
    self._thread = threading.Thread(target=self._capture, daemon=True)
    self._thread.start()

  def frame(self) -> Any:
    """The newest frame. Never blocks."""
    with self._lock:
      return self._frame

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
