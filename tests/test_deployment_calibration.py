"""Frame conventions and pose recovery for the external-camera calibration."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mjlab")
cv2 = pytest.importorskip("cv2")

from vbrl.deployment.calibration import (  # noqa: E402
  CV_TO_MUJOCO,
  board_points,
  camera_matrix,
  d405_intrinsics,
  flip_board_180,
  look_direction,
  mjcf_camera,
  solve_board_pose,
  transform,
  wrist_camera_at_home,
)


def test_the_mujoco_conversion_reproduces_the_xml() -> None:
  """Ground truth: the sim's own external camera, through the same conversion.

  MuJoCo looks down -z with +y up and OpenCV down +z with +y down, so getting
  this backwards is easy and silent -- it would place a calibrated camera
  mirrored. Feeding the conversion a camera whose MJCF numbers are known pins
  it against those numbers.
  """
  import mujoco

  from vbrl.asset_zoo.robots.trossen_wxai import WXAI_REALISTIC_XML

  model = mujoco.MjModel.from_xml_path(str(WXAI_REALISTIC_XML))
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  mujoco.mj_camlight(model, data)
  index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "external_tilted_cam")

  in_mujoco = transform(data.cam_xmat[index].reshape(3, 3), data.cam_xpos[index])
  in_opencv = in_mujoco @ transform(CV_TO_MUJOCO, np.zeros(3))
  position, quaternion, _ = mjcf_camera(in_opencv)

  # Compared against the model, not against literals: this pins the conversion,
  # and the camera's pose is calibration output that has already moved once.
  truth = np.empty(4)
  mujoco.mju_mat2Quat(truth, np.ascontiguousarray(data.cam_xmat[index]).reshape(9))
  np.testing.assert_allclose(position, data.cam_xpos[index], atol=1e-9)
  assert (
    min(np.linalg.norm(quaternion - truth), np.linalg.norm(quaternion + truth)) < 1e-9
  )
  # And it looks down at the table rather than up at the ceiling.
  assert look_direction(in_opencv)[2] < 0.0


def test_the_flip_is_its_own_inverse_and_keeps_the_board_in_place() -> None:
  pose = transform(np.eye(3), np.array([0.1, -0.2, 0.6]))
  once = flip_board_180(pose, 9, 6, 0.024)
  twice = flip_board_180(once, 9, 6, 0.024)
  np.testing.assert_allclose(twice, pose, atol=1e-12)
  # The board's centre is the fixed point of a half turn about its centre.
  centre = np.array([(9 - 1) * 0.024 / 2, (6 - 1) * 0.024 / 2, 0.0, 1.0])
  np.testing.assert_allclose((pose @ centre)[:3], (once @ centre)[:3], atol=1e-12)


def test_the_intrinsics_scale_without_changing_the_field_of_view() -> None:
  small, large = d405_intrinsics(424, 240), d405_intrinsics(848, 480)
  assert large["fx"] == pytest.approx(2 * small["fx"])
  assert (large["cx"], large["cy"]) == (424.0, 240.0)
  fov = lambda k, px: 2 * np.degrees(np.arctan(px / (2 * k)))  # noqa: E731
  assert fov(small["fy"], 240) == pytest.approx(fov(large["fy"], 480), abs=1e-9)
  # The 224 centre crop the policy is fed, which is what the sim renders.
  assert fov(small["fy"], 224) == pytest.approx(54.49, abs=0.01)


def test_the_board_grid_matches_the_printed_square_size() -> None:
  points = board_points(9, 6, 0.024)
  assert points.shape == (54, 3)
  assert np.all(points[:, 2] == 0.0)
  assert np.linalg.norm(points[1] - points[0]) == pytest.approx(0.024)


def test_a_synthetic_board_pose_is_recovered() -> None:
  """Project a board from a known pose, then solve for it and compare."""
  intrinsics = d405_intrinsics(848, 480)
  rotation, _ = cv2.Rodrigues(np.array([0.35, -0.2, 0.1]))
  truth = transform(rotation, np.array([0.03, -0.02, 0.55]))

  objects = board_points(9, 6, 0.024)
  rvec, _ = cv2.Rodrigues(truth[:3, :3])
  projected, _ = cv2.projectPoints(
    objects, rvec, truth[:3, 3], camera_matrix(intrinsics), None
  )
  recovered, error = solve_board_pose(
    projected.astype(np.float32), 9, 6, 0.024, intrinsics
  )
  assert error < 1e-3, "a noiseless projection must reproject exactly"
  np.testing.assert_allclose(recovered, truth, atol=1e-6)


def test_the_wrist_camera_sits_where_the_arm_puts_it() -> None:
  """Home pose is read by joint name, so a qpos-ordering change cannot skew it."""
  pose = wrist_camera_at_home()
  position = pose[:3, 3]
  assert position[0] > 0.0, "the wrist reaches forward of the base at home"
  assert position[2] > 0.1, "and above it"
  assert look_direction(pose)[2] < 0.0, "the wrist camera looks down at the table"
  np.testing.assert_allclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-9)
