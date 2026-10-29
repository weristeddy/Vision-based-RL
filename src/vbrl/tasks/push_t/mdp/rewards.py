from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from mjlab.entity import Entity
from mjlab.managers import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, wrap_to_pi

from ..geometry import footprint_signed_distance, yaw_from_quat
from .commands import push_t_command

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_ROBOT = SceneEntityCfg("robot")
_DISTANCE_SCALE = 5.0
_MAX_REWARD = 3.0
_VERTICAL = 0.7
_REACH_WEIGHT = 0.05


def _contact(env: ManagerBasedRlEnv, sensor_name: str, *fields: str) -> Any:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  missing = [f for f in fields if getattr(data, f) is None]
  if missing:
    raise RuntimeError(f"Contact sensor {sensor_name!r} needs {', '.join(missing)}.")
  return data


# The centre of mass lies inside the T, so the closest a closed gripper gets to it
# is on the top face; t2fn9uc9 avoided the T for 2,500 iterations between that
# pull and the top_face_contact termination.
def _outline_distance(
  env: ManagerBasedRlEnv, obj: Entity, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  offset = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  offset = offset - obj.data.root_link_pos_w
  yaw = yaw_from_quat(obj.data.root_link_quat_w)
  cos, sin = torch.cos(yaw), torch.sin(yaw)
  dx, dy = offset[:, 0], offset[:, 1]
  local = torch.stack((cos * dx + sin * dy, cos * dy - sin * dx), dim=-1)
  return torch.hypot(footprint_signed_distance(local).abs(), offset[:, 2])


def maniskill_dense_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  command = push_t_command(env, command_name)
  obj: Entity = env.scene[object_name]
  yaw_error = wrap_to_pi(
    command.target_yaw - yaw_from_quat(obj.data.root_link_quat_w)
  ).abs()
  goal_distance = torch.linalg.vector_norm(
    command.target_pos[:, :2] - obj.data.root_link_pos_w[:, :2], dim=-1
  )
  tcp_distance = _outline_distance(env, obj, asset_cfg)
  weight = float(command.cfg.orientation_weight)
  # Linear, not ManiSkill's ((cos e + 1) / 2)**2: any function of cos e has zero
  # gradient at 0 and at pi -- 0.302 at 45 degrees, 0.000 at 180 -- so a push
  # that reduces yaw error is punished by the competing position term wherever
  # the orientation gradient has decayed. 1 - |e|/pi is 0.159 everywhere.
  reward = (
    weight * (1.0 - yaw_error / torch.pi)
    + (1.0 - weight) * (1.0 - torch.tanh(_DISTANCE_SCALE * goal_distance)).square()
    + _REACH_WEIGHT
    * torch.sqrt((1.0 - torch.tanh(_DISTANCE_SCALE * tcp_distance)).clamp_min(0.0))
  )
  reward = torch.where(command.get_at_goal(), _MAX_REWARD, reward)
  return reward / _MAX_REWARD


# 1 - cos of the angle between the gripper's pointing axis and straight down; yaw
# about the vertical stays free. Policies held the gripper 25-50 deg off vertical
# while pushing, touching the T with pad edges and the hand instead of the pads.
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


def at_goal_static(
  env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  command = push_t_command(env, command_name)
  vel = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
  static = 1.0 - torch.tanh(5.0 * torch.linalg.vector_norm(vel, dim=-1))
  return static * command.get_at_goal().to(static.dtype)


# Reads force, not force_history: the classifying normal is only published at
# policy-step resolution.
def max_contact_force_on_face(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  vertical: bool,
  verticality_threshold: float = _VERTICAL,
) -> torch.Tensor:
  data = _contact(env, sensor_name, "found", "force", "normal")
  is_vertical = data.normal[..., 2].abs() > verticality_threshold
  selected = (data.found > 0) & (is_vertical if vertical else ~is_vertical)
  magnitude = torch.nan_to_num(
    torch.linalg.vector_norm(data.force, dim=-1), nan=0.0
  )
  return torch.where(selected, magnitude, 0.0).amax(dim=-1)


def table_touch(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  return (_contact(env, sensor_name, "found").found > 0).any(dim=-1).float()


__all__ = [
  "at_goal_static",
  "ee_tilt",
  "maniskill_dense_reward",
  "max_contact_force_on_face",
  "table_touch",
]
