from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.tasks.manipulation.mdp import LiftingCommand, LiftingCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# The real arm's gripper reads 0.004 closed on nothing; a cube blocks it near 0.019
# (vbrl.deployment.loop.GRIPPER_WHEN_EMPTY).
GRIPPER_WHEN_EMPTY = 0.004


class TcpLiftingCommand(LiftingCommand):
  cfg: TcpLiftingCommandCfg

  def __init__(self, cfg: TcpLiftingCommandCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self._robot = env.scene["robot"]
    self._site = self._robot.find_sites((cfg.ee_site,))[0][0]
    self._gripper = self._robot.find_joints((cfg.gripper_joint,))[0][0]
    for name in ("tcp_goal_error", "holding", "held_at_goal"):
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

  # What the real arm can measure: forward kinematics to the goal, and a closing
  # gripper blocked by something. A held cube sits 5.6 mm from the ee site.
  def _update_metrics(self) -> None:
    super()._update_metrics()
    tcp = self._robot.data.site_pos_w[:, self._site]
    error = torch.linalg.vector_norm(self.target_pos - tcp, dim=-1)
    closing = self._env.action_manager.action[:, -1] < 0.0
    blocked = self._robot.data.joint_pos[:, self._gripper] > GRIPPER_WHEN_EMPTY
    holding = closing & blocked
    self.metrics["tcp_goal_error"] = error
    self.metrics["holding"] = holding.float()
    self.metrics["held_at_goal"] = (
      holding & (error < self.cfg.success_threshold)
    ).float()


@dataclass(kw_only=True)
class TcpLiftingCommandCfg(LiftingCommandCfg):
  ee_site: str
  gripper_joint: str

  def build(self, env: ManagerBasedRlEnv) -> TcpLiftingCommand:
    return TcpLiftingCommand(self, env)


__all__ = ["GRIPPER_WHEN_EMPTY", "TcpLiftingCommand", "TcpLiftingCommandCfg"]
