from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


# 1 - cos of the angle between the gripper's pointing axis and straight down; yaw
# about the vertical stays free. Push-T policies held the gripper 25-50 deg off
# vertical while pushing; Lift-Cube carried the cube at a median 55 deg.
def ee_tilt(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  quat = asset.data.site_quat_w[:, asset_cfg.site_ids[0]]
  pointing = torch.tensor(axis, dtype=quat.dtype, device=quat.device).expand(
    quat.shape[0], 3
  )
  return 1.0 + quat_apply(quat, pointing)[:, 2]


__all__ = ["ee_tilt"]
