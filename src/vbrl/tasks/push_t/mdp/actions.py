from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class TargetRelativeJointPositionAction(BaseAction):
  def __init__(self, cfg: TargetRelativeJointPositionActionCfg, env) -> None:
    super().__init__(cfg=cfg, env=env)
    limits = self._entity.data.soft_joint_pos_limits[:, self._target_ids]
    self.low, self.high = limits[..., 0], limits[..., 1]
    self._target = self._entity.data.joint_pos[:, self._target_ids].clone()
    self._previous = self._target.clone()
    self._substeps = env.cfg.decimation
    self._substep = 0

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self._target[ids] = self._entity.data.joint_pos[ids][:, self._target_ids]
    self._previous[ids] = self._target[ids]
    # mjlab seeds the delay buffer with the zeroed target otherwise
    self._entity.set_joint_position_target(
      self._target[ids], joint_ids=self._target_ids, env_ids=ids
    )

  @property
  def target(self) -> torch.Tensor:
    return self._target

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    self._previous = self._target
    self._target = torch.clamp(
      self._target + self._processed_actions, self.low, self.high
    )
    self._substep = 0

  def apply_actions(self) -> None:
    self._substep += 1
    setpoint = torch.lerp(self._previous, self._target, self._substep / self._substeps)
    self._entity.set_joint_position_target(setpoint, joint_ids=self._target_ids)


@dataclass(kw_only=True)
class TargetRelativeJointPositionActionCfg(BaseActionCfg):
  def __post_init__(self) -> None:
    self.transmission_type = TransmissionType.JOINT

  def build(self, env: ManagerBasedRlEnv) -> TargetRelativeJointPositionAction:
    return TargetRelativeJointPositionAction(self, env)


__all__ = [
  "TargetRelativeJointPositionAction",
  "TargetRelativeJointPositionActionCfg",
]
