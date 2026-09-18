from __future__ import annotations

from typing import Any

import numpy as np

CV_TO_MUJOCO = np.diag([1.0, -1.0, -1.0])


def transform(rotation: Any, translation: Any) -> Any:
  pose = np.eye(4)
  pose[:3, :3] = rotation
  pose[:3, 3] = translation
  return pose


def mjcf_camera(pose_cv: Any) -> tuple[Any, Any, Any]:
  import mujoco

  rotation = (pose_cv @ transform(CV_TO_MUJOCO, np.zeros(3)))[:3, :3]
  quaternion = np.empty(4)
  mujoco.mju_mat2Quat(quaternion, rotation.flatten())
  return pose_cv[:3, 3], quaternion, np.concatenate([rotation[:, 0], rotation[:, 1]])


def look_direction(pose_cv: Any) -> Any:
  return pose_cv[:3, :3] @ np.array([0.0, 0.0, 1.0])


# Base frame, not world: the 5 mm mounting plate is a world-frame offset and must NOT be
# added. mj_camlight after mj_kinematics is required, or every joint vector silently.
def wrist_camera_pose(joint_pos: Any | None = None) -> Any:
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
