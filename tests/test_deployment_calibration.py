"""Frame conventions and pose recovery for the external-camera calibration."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mjlab")
cv2 = pytest.importorskip("cv2")

from vbrl.deployment import charuco  # noqa: E402
from vbrl.deployment.calibration import (  # noqa: E402
  CV_TO_MUJOCO,
  look_direction,
  mjcf_camera,
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
  index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "external_cam")

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


def test_the_robot_xmls_carry_the_measured_factory_optics() -> None:
  """Two cameras per robot, each with its own measured vertical field of view.

  Both units are D405s but they are not interchangeable: the wrist crop spans
  54.489 degrees and the external one 54.284, from each unit's own factory
  intrinsics (`python -m vbrl.deployment.intrinsics`) reduced to the 224x224
  centre crop the policy is fed. Nothing in Python overrides these any more --
  `CameraSensorCfg.fovy` is left None -- so the MJCF is the only place they
  live, and a silent edit here would change every observation the policy sees.
  """
  import mujoco

  from vbrl.asset_zoo.robots.trossen_wxai import WXAI_REALISTIC_XML, WXAI_XML

  expected = {"cam": 54.489, "external_cam": 54.284}
  for path in (WXAI_XML, WXAI_REALISTIC_XML):
    model = mujoco.MjModel.from_xml_path(str(path))
    names = {
      mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i): i
      for i in range(model.ncam)
    }
    assert names.keys() == expected.keys(), f"{path.name} declares {sorted(names)}"
    for name, fovy in expected.items():
      assert model.cam_fovy[names[name]] == pytest.approx(fovy), (path.name, name)


def _render_board(rvec, tvec, camera_matrix, size=(640, 480), px=60):
  """The real board, projected from a known pose onto a synthetic image.

  ``generateImage`` with no margin spans exactly ``SQUARES * SQUARE_M``, so the
  image's four corners map to known board coordinates and the homography that
  takes them to their projections is exact. Guessing a margin's scale instead
  costs about 6 mm of pose error and would be mistaken for detector error.
  """
  board = charuco.make_board()
  columns, rows = charuco.SQUARES
  image = board.generateImage((columns * px, rows * px), marginSize=0)
  width_m, height_m = columns * charuco.SQUARE_M, rows * charuco.SQUARE_M

  object_corners = np.float32(
    [[0, 0, 0], [width_m, 0, 0], [width_m, height_m, 0], [0, height_m, 0]]
  )
  projected, _ = cv2.projectPoints(
    object_corners, rvec, tvec, camera_matrix, np.zeros(5)
  )
  source = np.float32(
    [
      [0, 0],
      [image.shape[1], 0],
      [image.shape[1], image.shape[0]],
      [0, image.shape[0]],
    ]
  )
  homography = cv2.getPerspectiveTransform(
    source, projected.reshape(-1, 2).astype(np.float32)
  )
  return cv2.warpPerspective(
    cv2.cvtColor(image, cv2.COLOR_GRAY2RGB),
    homography,
    size,
    borderValue=(255, 255, 255),
  )


@pytest.mark.parametrize(
  ("rvec", "tvec"),
  [
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.90)),
    ((0.25, -0.18, 0.06), (0.02, -0.01, 0.80)),
    ((-0.40, 0.30, -0.10), (-0.03, 0.02, 0.70)),
  ],
)
def test_a_synthetic_charuco_pose_is_recovered(rvec, tvec) -> None:
  """The board model plus the solve, round-tripped against a known pose.

  This is the end-to-end guard on ``SQUARES``, ``SQUARE_M``, ``MARKER_M`` and
  ``LEGACY_PATTERN``. Every one of them was identified by fitting rather than
  read off the board, and getting the layout wrong is silent: the markers still
  decode, and only the interpolated corner count collapses. Here a wrong model
  cannot interpolate the corners it needs and the pose does not come back.
  """
  rvec = np.array(rvec, dtype=np.float64).reshape(3, 1)
  tvec = np.array(tvec, dtype=np.float64).reshape(3, 1)
  camera_matrix = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])

  detection = charuco.detect_dictionary(_render_board(rvec, tvec, camera_matrix))
  assert detection["n_corners"] >= charuco.MIN_CORNERS

  pose = charuco.board_pose(detection, camera_matrix, np.zeros(5))
  assert pose["rms_px"] < 1.0

  truth, _ = cv2.Rodrigues(rvec)
  assert np.linalg.norm(pose["pose"][:3, 3] - tvec.ravel()) < 0.003
  relative = truth.T @ pose["pose"][:3, :3]
  angle = np.degrees(np.arccos(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)))
  assert angle < 1.0


def test_the_board_model_is_the_one_that_was_fitted() -> None:
  """A modern-layout board of the same size interpolates nothing.

  Recorded because it is the failure that cost the most: with
  ``LEGACY_PATTERN`` off, the markers decode identically and the corner count
  goes to zero, which looks like a bad photograph rather than a wrong model.
  """
  camera_matrix = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
  scene = _render_board(
    np.zeros((3, 1)), np.array([[0.0], [0.0], [0.9]]), camera_matrix
  )

  board = cv2.aruco.CharucoBoard(
    charuco.SQUARES,
    charuco.SQUARE_M,
    charuco.MARKER_M,
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
  )
  board.setLegacyPattern(False)
  corners, ids, _, marker_ids = cv2.aruco.CharucoDetector(board).detectBoard(
    cv2.cvtColor(scene, cv2.COLOR_RGB2GRAY)
  )
  assert marker_ids is not None and len(marker_ids) > 0, "markers still decode"
  assert ids is None or len(ids) < charuco.MIN_CORNERS, (
    "the wrong layout must not interpolate a usable corner set"
  )


def test_the_wrist_camera_sits_where_the_arm_puts_it() -> None:
  """Home pose is read by joint name, so a qpos-ordering change cannot skew it."""
  pose = wrist_camera_at_home()
  position = pose[:3, 3]
  assert position[0] > 0.0, "the wrist reaches forward of the base at home"
  assert position[2] > 0.1, "and above it"
  assert look_direction(pose)[2] < 0.0, "the wrist camera looks down at the table"
  np.testing.assert_allclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-9)
