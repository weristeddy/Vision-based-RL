"""Push-T rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

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


def maniskill_dense_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Normalized ManiSkill3 Push-T dense reward."""
  command = push_t_command(env, command_name)
  obj: Entity = env.scene[object_name]
  yaw_error = wrap_to_pi(
    command.target_yaw - yaw_from_quat(obj.data.root_link_quat_w)
  )
  goal_distance = torch.linalg.vector_norm(
    command.target_pos[:, :2] - obj.data.root_link_pos_w[:, :2], dim=-1
  )
  tcp_distance = torch.linalg.vector_norm(
    manipulation_mdp.ee_to_object_distance(env, object_name, asset_cfg),
    dim=-1,
  )
  weight = float(command.cfg.orientation_weight)
  reward = (
    weight * ((torch.cos(yaw_error) + 1.0) / 2.0).square()
    + (1.0 - weight)
    * (1.0 - torch.tanh(_DISTANCE_SCALE * goal_distance)).square()
    + torch.sqrt(
      (1.0 - torch.tanh(_DISTANCE_SCALE * tcp_distance)).clamp_min(0.0)
    )
    / 20.0
  )
  reward = torch.where(
    command.get_at_goal(),
    torch.full_like(reward, _MAX_REWARD),
    reward,
  )
  return reward / _MAX_REWARD


def quadratic_orientation_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """ManiSkill's dense reward with the orientation term's dead zone removed.

  Identical to :func:`maniskill_dense_reward` in every term, weight and cap; the
  only change is the shape of the orientation factor. Measured on the term as
  written, ManiSkill's gradient is 0.250 at 90 degrees and **0.000** at 180: any
  function of ``cos e`` is flat there. Under a goal-yaw curriculum that never
  matters, because episodes start at zero error; without one the initial error is
  uniform on [0, pi], so roughly a quarter of episodes begin past 138 degrees
  with almost no orientation signal at all.

  ``1 - (|e| / pi)**2`` inverts the profile: 0.000 at 0 degrees, so a clumsy
  first contact is not punished, rising to **0.318** at 180 where ManiSkill has
  none. Same [0, 0.5] range, so normalisation and the sparse at-goal bonus are
  untouched.
  """
  command = push_t_command(env, command_name)
  obj: Entity = env.scene[object_name]
  yaw_error = wrap_to_pi(
    command.target_yaw - yaw_from_quat(obj.data.root_link_quat_w)
  ).abs()
  goal_distance = torch.linalg.vector_norm(
    command.target_pos[:, :2] - obj.data.root_link_pos_w[:, :2], dim=-1
  )
  tcp_distance = torch.linalg.vector_norm(
    manipulation_mdp.ee_to_object_distance(env, object_name, asset_cfg),
    dim=-1,
  )
  weight = float(command.cfg.orientation_weight)
  reward = (
    weight * (1.0 - (yaw_error / torch.pi).square())
    + (1.0 - weight)
    * (1.0 - torch.tanh(_DISTANCE_SCALE * goal_distance)).square()
    + torch.sqrt(
      (1.0 - torch.tanh(_DISTANCE_SCALE * tcp_distance)).clamp_min(0.0)
    )
    / 20.0
  )
  reward = torch.where(
    command.get_at_goal(),
    torch.full_like(reward, _MAX_REWARD),
    reward,
  )
  return reward / _MAX_REWARD


def vertical_contact_force(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_scale: float = 10.0,
) -> torch.Tensor:
  """Force-weighted vertical contact; clean side pushes score zero."""
  if force_scale <= 0.0:
    raise ValueError("force_scale must be positive.")
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.found is None or data.force is None or data.normal is None:
    raise RuntimeError(
      f"Contact sensor {sensor_name!r} requires found, force, and normal."
    )
  verticality = torch.abs(data.normal[..., 2]).clamp(0.0, 1.0)
  force = torch.linalg.vector_norm(data.force, dim=-1)
  bounded_force = torch.tanh(
    torch.nan_to_num(force, nan=0.0, posinf=force_scale) / force_scale
  )
  return torch.amax(
    torch.where(
      data.found > 0,
      verticality * bounded_force,
      torch.zeros_like(verticality),
    ),
    dim=-1,
  )


def linear_orientation_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """ManiSkill's dense reward with an orientation factor that never goes flat.

  Both existing shapes have a dead zone, at opposite ends. ManiSkill's
  ``((cos e + 1) / 2)**2`` has zero gradient at 0 *and* at pi -- 0.302 at 45
  degrees but 0.004 at 162 and 0.000 at 180 -- and
  :func:`quadratic_orientation_reward` removes the far one only by introducing a
  near one, 0.318 at 180 degrees against 0.000 at 0, which takes away the fine
  alignment the 0.90 overlap threshold is made of. It scored 2 of 15 where the
  matched ManiSkill control scored 4.

  ``1 - |e| / pi`` is flat nowhere: gradient 0.159 at every angle. That matters
  because the two terms compete. Rotating the T toward the goal also shifts it,
  and the position factor's gradient peaks near the goal, so a push that
  correctly reduces yaw error is *punished* under ManiSkill's shape wherever the
  orientation gradient has decayed -- measured at -0.00101 at 15 cm and 162
  degrees, and still negative at 22 cm. Roughly a quarter of episodes with a
  uniform goal start past 135 degrees, inside that region. Under the linear
  shape the same push pays at every distance and every angle.

  Shares the ``orientation_weight`` split with the other two shapes, so the
  weight and the shape are independent knobs, and reduces to the same
  ``[0, 0.5]`` contribution at the registered 0.5 -- normalisation, the cap
  and the sparse at-goal bonus are untouched.
  """
  command = push_t_command(env, command_name)
  obj: Entity = env.scene[object_name]
  yaw_error = wrap_to_pi(
    command.target_yaw - yaw_from_quat(obj.data.root_link_quat_w)
  ).abs()
  goal_distance = torch.linalg.vector_norm(
    command.target_pos[:, :2] - obj.data.root_link_pos_w[:, :2], dim=-1
  )
  tcp_distance = torch.linalg.vector_norm(
    manipulation_mdp.ee_to_object_distance(env, object_name, asset_cfg),
    dim=-1,
  )
  weight = float(command.cfg.orientation_weight)
  reward = (
    weight * (1.0 - yaw_error / torch.pi)
    + (1.0 - weight)
    * (1.0 - torch.tanh(_DISTANCE_SCALE * goal_distance)).square()
    + torch.sqrt(
      (1.0 - torch.tanh(_DISTANCE_SCALE * tcp_distance)).clamp_min(0.0)
    )
    / 20.0
  )
  reward = torch.where(
    command.get_at_goal(), torch.full_like(reward, _MAX_REWARD), reward
  )
  return reward / _MAX_REWARD


__all__ = [
  "linear_orientation_reward",
  "maniskill_dense_reward",
  "quadratic_orientation_reward",
  "vertical_contact_force",
]
