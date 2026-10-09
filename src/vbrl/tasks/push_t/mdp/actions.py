from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class TargetRelativeJointPositionAction(BaseAction):
  def __init__(self, cfg: TargetRelativeJointPositionActionCfg, env) -> None:
    super().__init__(cfg=cfg, env=env)
    if cfg.ramp_s < 0.0:
      raise ValueError(f"ramp_s must be >= 0; got {cfg.ramp_s}.")
    limits = self._entity.data.soft_joint_pos_limits[:, self._target_ids]
    self.low, self.high = limits[..., 0], limits[..., 1]
    self._target = self._entity.data.joint_pos[:, self._target_ids].clone()
    self._setpoint = self._target.clone()
    self._ramp = torch.zeros_like(self._target)
    self._ramp_substeps = max(1, round(cfg.ramp_s / env.physics_dt))
    self._physics_dt = env.physics_dt
    model = env.sim.mj_model
    actuators = [model.actuator(f"{cfg.entity_name}/{n}") for n in self._target_names]
    # A position actuator's kv/kp turns a feedforward velocity into a target lead.
    self._lead_per_velocity = torch.tensor(
      [-a.biasprm[2] / a.gainprm[0] for a in actuators],
      dtype=torch.float,
      device=env.device,
    ) * (cfg.ramp_s > env.physics_dt)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self._target[ids] = self._entity.data.joint_pos[ids][:, self._target_ids]
    self._setpoint[ids] = self._target[ids]
    self._ramp[ids] = 0.0
    # mjlab seeds the delay buffer with the zeroed target otherwise
    self._entity.set_joint_position_target(
      self._target[ids], joint_ids=self._target_ids, env_ids=ids
    )

  @property
  def target(self) -> torch.Tensor:
    return self._target

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    if self.cfg.reference == "measured":
      base = self._entity.data.joint_pos[:, self._target_ids]
    else:
      base = self._target
    self._target = torch.clamp(base + self._processed_actions, self.low, self.high)
    self._ramp = (self._target - self._setpoint) / self._ramp_substeps

  # The Trossen SDK moves its setpoint linearly over goal_time and, unless
  # ramp_feedforward is off, also commands the ramp's speed.
  def apply_actions(self) -> None:
    remaining = self._target - self._setpoint
    step = torch.where(
      self._ramp.abs() < remaining.abs(), self._ramp, remaining
    )
    self._setpoint = self._setpoint + step
    command = self._setpoint
    if self.cfg.ramp_feedforward:
      command = command + step / self._physics_dt * self._lead_per_velocity
    self._entity.set_joint_position_target(command, joint_ids=self._target_ids)


@dataclass(kw_only=True)
class TargetRelativeJointPositionActionCfg(BaseActionCfg):
  reference: Literal["target", "measured"] = "target"
  ramp_s: float = 0.0
  ramp_feedforward: bool = True

  def __post_init__(self) -> None:
    self.transmission_type = TransmissionType.JOINT

  def build(self, env: ManagerBasedRlEnv) -> TargetRelativeJointPositionAction:
    return TargetRelativeJointPositionAction(self, env)


__all__ = [
  "TargetRelativeJointPositionAction",
  "TargetRelativeJointPositionActionCfg",
]
