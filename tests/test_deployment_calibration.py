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
  assert look_direction(in_opencv)[2] < 0.0


def test_the_robot_xmls_carry_the_measured_factory_optics() -> None:
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
  pose = wrist_camera_at_home()
  position = pose[:3, 3]
  assert position[0] > 0.0, "the wrist reaches forward of the base at home"
  assert position[2] > 0.1, "and above it"
  assert look_direction(pose)[2] < 0.0, "the wrist camera looks down at the table"
  np.testing.assert_allclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-9)


def test_goal_pose_offset_places_the_origin_not_the_tag_midpoint() -> None:
  from vbrl.deployment.goal_pose import (
    CROSSBAR_INNER_EDGE_M,
    DEFAULT_MARGIN_M,
    TAG_BLACK_M,
  )

  midpoint_y = CROSSBAR_INNER_EDGE_M + DEFAULT_MARGIN_M + TAG_BLACK_M / 2.0
  assert midpoint_y == pytest.approx(0.0118)

  offset_y = -(CROSSBAR_INNER_EDGE_M + DEFAULT_MARGIN_M + TAG_BLACK_M / 2.0)
  assert offset_y == pytest.approx(-0.0118)
  assert midpoint_y + offset_y == pytest.approx(0.0, abs=1e-12)

  # -(4.8 mm + m): the printed size never enters, only the margin on the side
  # butted against the crossbar, so a reprint changes exactly this one number.
  for margin in (0.00443, 0.005, 0.006, 0.007, 0.008):
    expected = -(0.0048 + margin)
    actual = -(CROSSBAR_INNER_EDGE_M + margin + TAG_BLACK_M / 2.0)
    assert actual == pytest.approx(expected, abs=1e-9)


def test_goal_pose_reads_the_calibrated_external_camera_from_the_mjcf() -> None:
  import numpy as np

  from vbrl.asset_zoo.robots import make_wxai_realistic
  from vbrl.deployment.goal_pose import external_camera_pose

  pose = external_camera_pose(make_wxai_realistic().xml_path)
  assert pose.shape == (4, 4)
  rotation = pose[:3, :3]
  assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
  assert float(np.linalg.det(rotation)) == pytest.approx(1.0, abs=1e-6)
  assert pose[0, 3] > 0.5 and pose[2, 3] > 0.3
  assert float(rotation @ np.array([0.0, 0.0, 1.0]) @ np.array([0, 0, 1])) < -0.3
