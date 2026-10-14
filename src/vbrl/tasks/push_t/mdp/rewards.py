from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from mjlab.entity import Entity
from mjlab.managers import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.utils.lab_api.math import wrap_to_pi

from ..geometry import yaw_from_quat
from .commands import push_t_command

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_ROBOT = SceneEntityCfg("robot")
_DISTANCE_SCALE = 5.0
_MAX_REWARD = 3.0
_VERTICAL = 0.7


def _contact(env: ManagerBasedRlEnv, sensor_name: str, *fields: str) -> Any:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  missing = [f for f in fields if getattr(data, f) is None]
  if missing:
    raise RuntimeError(f"Contact sensor {sensor_name!r} needs {', '.join(missing)}.")
  return data


def _press(data: Any) -> torch.Tensor:
  return torch.nan_to_num(data.force[..., 2], nan=0.0).sum(dim=-1).abs()


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
  tcp_distance = torch.linalg.vector_norm(
    manipulation_mdp.ee_to_object_distance(env, object_name, asset_cfg), dim=-1
  )
  weight = float(command.cfg.orientation_weight)
  # Linear, not ManiSkill's ((cos e + 1) / 2)**2: any function of cos e has zero
  # gradient at 0 and at pi -- 0.302 at 45 degrees, 0.000 at 180 -- so a push
  # that reduces yaw error is punished by the competing position term wherever
  # the orientation gradient has decayed. 1 - |e|/pi is 0.159 everywhere.
  reward = (
    weight * (1.0 - yaw_error / torch.pi)
    + (1.0 - weight) * (1.0 - torch.tanh(_DISTANCE_SCALE * goal_distance)).square()
    + torch.sqrt((1.0 - torch.tanh(_DISTANCE_SCALE * tcp_distance)).clamp_min(0.0))
    / 20.0
  )
  reward = torch.where(command.get_at_goal(), _MAX_REWARD, reward)
  return reward / _MAX_REWARD


def top_contact_share(
  env: ManagerBasedRlEnv, sensor_name: str, verticality_threshold: float = _VERTICAL
) -> torch.Tensor:
  data = _contact(env, sensor_name, "found", "normal")
  vertical = data.normal[..., 2].abs() > verticality_threshold
  return ((data.found > 0) & vertical).any(dim=-1).float()


def fingertip_height_excess(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  ceiling: float,
  sensor_name: str | None = None,
) -> torch.Tensor:
  if ceiling <= 0.0:
    raise ValueError("fingertip_height_excess needs ceiling > 0.")
  asset: Entity = env.scene[asset_cfg.name]
  lowest = asset.data.geom_pos_w[:, asset_cfg.geom_ids, 2].min(dim=-1).values
  excess = ((lowest - ceiling) / ceiling).clamp_min(0.0)
  if sensor_name is None:
    return excess
  found = _contact(env, sensor_name, "found").found
  return excess * (found.amax(dim=-1) <= 0).to(excess.dtype)


def at_goal_static(
  env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  command = push_t_command(env, command_name)
  vel = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
  static = 1.0 - torch.tanh(5.0 * torch.linalg.vector_norm(vel, dim=-1))
  return static * command.get_at_goal().to(static.dtype)


# Gated on position error alone, so a T in place with its yaw off still gets
# slow corrective pushes.
def near_goal_ee_speed(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  asset_cfg: SceneEntityCfg,
  length_scale: float,
) -> torch.Tensor:
  command = push_t_command(env, command_name)
  goal_distance = torch.linalg.vector_norm(
    command.target_pos[:, :2] - env.scene[object_name].data.root_link_pos_w[:, :2],
    dim=-1,
  )
  speed = env.scene[asset_cfg.name].data.site_lin_vel_w[:, asset_cfg.site_ids[0]]
  return torch.exp(-goal_distance / length_scale) * speed.square().sum(dim=-1)


# Zero-set is "do not press down", so pushing stays free at any magnitude --
# unlike every earlier attempt, whose zero-set was "do not touch the object".
def object_table_press(
  env: ManagerBasedRlEnv, sensor_name: str, onset: float, scale: float
) -> torch.Tensor:
  if scale <= 0.0:
    raise ValueError("object_table_press needs scale > 0.")
  data = _contact(env, sensor_name, "force")
  # Squared: a light touch on the top face stays nearly free, a hard press does
  # not. Capped so one jammed contact cannot dominate an update.
  excess = ((_press(data) - onset) / scale).clamp_min(0.0)
  return excess.square().clamp_max(100.0)


def peak_object_press(
  env: ManagerBasedRlEnv, sensor_name: str, weight_n: float
) -> torch.Tensor:
  return (_press(_contact(env, sensor_name, "force")) - weight_n).clamp_min(0.0)


def max_contact_force(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  data = _contact(env, sensor_name)
  force = data.force_history if data.force_history is not None else data.force
  if force is None:
    raise RuntimeError(f"Contact sensor {sensor_name!r} needs force.")
  magnitude = torch.linalg.vector_norm(force, dim=-1)
  return torch.nan_to_num(magnitude, nan=0.0).flatten(1).amax(dim=-1)


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


# `episode_success` latches on the first at-goal step, so it says whether the
# goal was ever reached, not whether the arm held it. This says how much of the
# episode was spent there.
def at_goal_share(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return push_t_command(env, command_name).get_at_goal().float()


__all__ = [
  "at_goal_static",
  "at_goal_share",
  "fingertip_height_excess",
  "maniskill_dense_reward",
  "max_contact_force",
  "max_contact_force_on_face",
  "near_goal_ee_speed",
  "object_table_press",
  "peak_object_press",
  "table_touch",
  "top_contact_share",
]
