from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from vbrl.deployment.capture_board import EXTERNAL_SERIAL
from vbrl.deployment.config import TABLE_Z_BASE_M

INTRINSICS = "artifacts/deployment/intrinsics/intrinsics.json"
EXTRINSICS = "artifacts/deployment/extrinsics/extrinsics.json"


@dataclass(frozen=True)
class ExternalView:
  matrix: Any
  distortion: Any
  pose: Any
  width: int
  height: int
  table: tuple[float, float, float] = (TABLE_Z_BASE_M, 0.0, 0.0)

  @classmethod
  def load(cls, width: int, height: int) -> ExternalView:
    import mujoco

    from vbrl.asset_zoo.robots.trossen_wxai import WXAI_MEASURED_XML
    from vbrl.deployment.calibration import CV_TO_MUJOCO, transform
    from vbrl.paths import checkout_root

    camera = mujoco.MjModel.from_xml_path(str(WXAI_MEASURED_XML)).camera("external_cam")
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, camera.quat)
    pose = transform(rotation.reshape(3, 3) @ CV_TO_MUJOCO, camera.pos)

    cameras = {
      c["serial"]: c for c in json.loads((checkout_root() / INTRINSICS).read_text())
    }
    modes = cameras[EXTERNAL_SERIAL]["modes"]
    name = f"{width}x{height}"
    if name not in modes:
      raise ValueError(f"No {name} intrinsics; calibrated are {sorted(modes)}.")
    mode = modes[name]
    native = mode["native"]
    matrix = np.array(
      [[native["fx"], 0.0, native["cx"]], [0.0, native["fy"], native["cy"]], [0, 0, 1]]
    )
    distortion = np.array(mode["distortion"]["coeffs"], dtype=float)
    extrinsics = json.loads((checkout_root() / EXTRINSICS).read_text())
    table = tuple(extrinsics["table_plane"]["c_a_b"])
    return cls(matrix, distortion, pose, width, height, table)

  def cropped(self, size: int) -> ExternalView:
    matrix = self.matrix.copy()
    matrix[0, 2] -= (self.width - size) // 2
    matrix[1, 2] -= (self.height - size) // 2
    return replace(self, matrix=matrix, width=size, height=size)

  def project(self, points: Any) -> Any:
    import cv2

    to_camera = np.linalg.inv(self.pose)
    rotation = cv2.Rodrigues(to_camera[:3, :3])[0]
    pixels, _ = cv2.projectPoints(
      np.asarray(points, dtype=np.float64),
      rotation,
      to_camera[:3, 3],
      self.matrix,
      self.distortion,
    )
    return pixels.reshape(-1, 2)

  def table_z(self, x: Any, y: Any) -> Any:
    c, a, b = self.table
    return c + a * x + b * y

  def on_table(self, pixel: Any, height: float = 0.0) -> Any:
    import cv2

    point = np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2)
    u, v = cv2.undistortPoints(point, self.matrix, self.distortion).reshape(2)
    ray = self.pose[:3, :3] @ np.array([u, v, 1.0])
    origin = self.pose[:3, 3]
    c, a, b = self.table
    level = c + height + a * origin[0] + b * origin[1] - origin[2]
    return origin + ray * level / (ray[2] - a * ray[0] - b * ray[1])


__all__ = ["EXTRINSICS", "INTRINSICS", "ExternalView"]
