from __future__ import annotations

from typing import Any

import numpy as np


class Kinematics:
  """Where the end effector is, given the joint angles.

  Deployment needs this because ``goal_position`` is the target expressed in
  the end effector's frame, and the arm's MJCF is what defines that frame.
  Checked against the arm's own Cartesian reading: the two agree to 3 mm.
  """

  def __init__(self) -> None:
    import mujoco

    from vbrl.asset_zoo.robots.trossen_wxai import WXAI_XML, make_wxai

    # Both Trossen variants share these kinematics and differ only in appearance.
    self._model = mujoco.MjModel.from_xml_path(str(WXAI_XML))
    self._data = mujoco.MjData(self._model)
    self._ee_site = mujoco.mj_name2id(
      self._model, mujoco.mjtObj.mjOBJ_SITE, make_wxai().ee_site
    )

  def ee_pose(self, joint_pos: Any) -> tuple[Any, Any]:
    """End-effector position and orientation quaternion, in the base frame."""
    import mujoco

    self._data.qpos[: len(joint_pos)] = joint_pos
    mujoco.mj_kinematics(self._model, self._data)
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, self._data.site_xmat[self._ee_site])
    return np.array(self._data.site_xpos[self._ee_site]), quaternion


__all__ = ["Kinematics"]
