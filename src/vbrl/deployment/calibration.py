"""Frame conventions and forward kinematics for camera calibration.

The geometry the calibration pipeline is built on, and nothing else. The wrist
camera's pose in the base frame is known exactly -- it is fixed in the MJCF and
the arm reports its joints, so forward kinematics places it -- which is what
lets a board seen by the wrist camera locate a camera that cannot see the arm:

    T_base_board  = T_base_wristcam @ T_wristcam_board     (wrist images)
    T_base_extcam = T_base_board @ inv(T_extcam_board)     (external image)

That is the frame an MJCF ``<camera>`` inside ``base_link`` is written in, so
the result is pasted into the file unchanged.

This module used to also carry a chessboard detector, a solver CLI, an
intrinsics probe, and a pair of hardcoded ``D405_FX_424``/``D405_FY_424``
constants shared by both cameras. All of it is gone. The two D405s on this rig
have measurably different optics and different principal points -- neither at
the image centre -- so one shared constant stood for two physically different
cameras. Intrinsics now come per unit from :mod:`vbrl.deployment.intrinsics`,
board geometry from :mod:`vbrl.deployment.charuco`, capture from
:mod:`vbrl.deployment.capture_board`, and the solve from
:mod:`vbrl.deployment.recalibrate`. There is one pipeline, not two.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# MuJoCo cameras look down -z with +y up; OpenCV looks down +z with +y down.
# Flipping y and z converts between them, and the flip is its own inverse.
CV_TO_MUJOCO = np.diag([1.0, -1.0, -1.0])


def transform(rotation: Any, translation: Any) -> Any:
  pose = np.eye(4)
  pose[:3, :3] = rotation
  pose[:3, 3] = translation
  return pose


def mjcf_camera(pose_cv: Any) -> tuple[Any, Any, Any]:
  """(pos, quat, xyaxes) for an MJCF <camera>, from an OpenCV pose."""
  import mujoco

  rotation = (pose_cv @ transform(CV_TO_MUJOCO, np.zeros(3)))[:3, :3]
  quaternion = np.empty(4)
  mujoco.mju_mat2Quat(quaternion, rotation.flatten())
  # xyaxes names the camera's x and y axes in the parent frame; MuJoCo derives
  # the viewing direction from their cross product.
  return pose_cv[:3, 3], quaternion, np.concatenate([rotation[:, 0], rotation[:, 1]])


def look_direction(pose_cv: Any) -> Any:
  """Unit vector the camera looks along, in the parent frame."""
  return pose_cv[:3, :3] @ np.array([0.0, 0.0, 1.0])


def wrist_camera_pose(joint_pos: Any | None = None) -> Any:
  """The wrist camera's pose in the BASE frame, OpenCV axes, for any joint pose.

  Base frame, not world: every ``<camera>`` in the MJCF hangs off ``base_link``
  (checked -- both ``cam`` and ``external_cam`` do), so a pose solved in this
  frame is written into the file unchanged. The 5 mm
  mounting plate that lifts the arm off the table is a world-frame offset and
  must NOT be added here; it would move the camera relative to the arm it is
  bolted beside.

  ``mj_camlight`` is required after ``mj_kinematics``: kinematics alone leaves
  ``cam_xpos`` at whatever the last call left there, which silently returns the
  home pose for every joint vector.

  ``joint_pos`` is applied by joint name when None (the definition's home pose)
  and positionally otherwise, matching how the arm reports ``get_all_positions``.
  """
  import mujoco

  from vbrl.asset_zoo.robots.trossen_wxai import WXAI_XML, make_wxai

  robot = make_wxai()
  model = mujoco.MjModel.from_xml_path(str(WXAI_XML))
  data = mujoco.MjData(model)
  if joint_pos is None:
    for joint, angle in robot.home_joint_pos.items():
      index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
      if index < 0:
        raise RuntimeError(f"No joint {joint!r} in {WXAI_XML}.")
      data.qpos[model.jnt_qposadr[index]] = angle
  else:
    angles = np.asarray(joint_pos, dtype=np.float64).reshape(-1)
    if len(angles) > model.nq:
      raise ValueError(f"{len(angles)} joint values for {model.nq} qpos slots.")
    data.qpos[: len(angles)] = angles
  mujoco.mj_kinematics(model, data)
  mujoco.mj_camlight(model, data)

  name = robot.resolve_camera("wrist").model_name
  index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
  if index < 0:
    raise RuntimeError(f"No camera {name!r} in {WXAI_XML}.")
  in_mujoco = transform(data.cam_xmat[index].reshape(3, 3), data.cam_xpos[index])
  return in_mujoco @ transform(CV_TO_MUJOCO, np.zeros(3))


def wrist_camera_at_home() -> Any:
  """The wrist camera's pose in the base frame at the home pose, OpenCV axes.

  The home pose comes from the robot definition rather than a hard-coded
  vector, and is applied by joint name: the MJCF's qpos order and the model's
  joint order need not agree, and silently mismatching them would rotate the
  camera by a joint.
  """
  import mujoco

  from vbrl.asset_zoo.robots.trossen_wxai import WXAI_XML, make_wxai

  robot = make_wxai()
  model = mujoco.MjModel.from_xml_path(str(WXAI_XML))
  data = mujoco.MjData(model)
  for joint, angle in robot.home_joint_pos.items():
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
    if index < 0:
      raise RuntimeError(f"No joint {joint!r} in {WXAI_XML}.")
    data.qpos[model.jnt_qposadr[index]] = angle
  mujoco.mj_kinematics(model, data)
  mujoco.mj_camlight(model, data)  # mj_kinematics alone does not place cameras

  name = robot.resolve_camera("wrist").model_name
  index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
  if index < 0:
    raise RuntimeError(f"No camera {name!r} in {WXAI_XML}.")
  in_mujoco = transform(data.cam_xmat[index].reshape(3, 3), data.cam_xpos[index])
  return in_mujoco @ transform(CV_TO_MUJOCO, np.zeros(3))


__all__ = [
  "CV_TO_MUJOCO",
  "look_direction",
  "mjcf_camera",
  "transform",
  "wrist_camera_at_home",
  "wrist_camera_pose",
]
