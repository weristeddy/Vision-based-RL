from __future__ import annotations

from typing import Any

import numpy as np


class Kinematics:
  def __init__(self) -> None:
    import mujoco

    from vbrl.asset_zoo.robots.trossen_wxai import WXAI_XML, make_wxai

    self._model = mujoco.MjModel.from_xml_path(str(WXAI_XML))
    self._data = mujoco.MjData(self._model)
    self._ee_site = mujoco.mj_name2id(
      self._model, mujoco.mjtObj.mjOBJ_SITE, make_wxai().ee_site
    )

  def ee_pose(self, joint_pos: Any) -> tuple[Any, Any]:
    import mujoco

    self._data.qpos[: len(joint_pos)] = joint_pos
    mujoco.mj_kinematics(self._model, self._data)
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, self._data.site_xmat[self._ee_site])
    return np.array(self._data.site_xpos[self._ee_site]), quaternion


__all__ = ["Kinematics"]
