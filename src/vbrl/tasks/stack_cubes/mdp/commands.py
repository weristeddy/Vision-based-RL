"""The Stack-Cubes command: where the next cube belongs.

Nothing here is sampled, which is the point of the task -- the stack sits at a
fixed spot and the policy is never told to go somewhere else. What makes it a
command rather than a constant is that the target rises one course every time
the tower grows, so "where does the next cube go" is a real per-step question.
Publishing it as a command is also what gives the task the two things every
other task in this repository gets for free: ``episode_success`` for
``vbrl-evaluate``, and a drawn target in the viewer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .tower import MAX_CUBES, stack_point, tower_state

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class StackCommand(CommandTerm):
  """The next course's placement point, and whether the tower was ever built."""

  cfg: StackCommandCfg

  def __init__(self, cfg: StackCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    cfg.asset_cfg.resolve(env.scene)
    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)
    for name in ("tower_height", "at_goal", "episode_success"):
      self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  def _update_metrics(self) -> None:
    state = tower_state(self._env, self.cfg.asset_cfg)
    self.target_pos = (
      stack_point(state.height.clamp(max=MAX_CUBES - 1)) + self._env.scene.env_origins
    )
    at_goal = state.complete.float()
    # Latched, so an episode counts as solved even if its tower is knocked down
    # afterwards -- which this task does to itself once per episode on purpose.
    self.episode_success = torch.maximum(self.episode_success, at_goal)
    self.metrics["tower_height"] = state.height.float()
    self.metrics["at_goal"] = at_goal
    self.metrics["episode_success"] = self.episode_success

  def compute_success(self) -> torch.Tensor:
    return self.metrics["at_goal"] > 0.0

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # There is nothing to draw: the stack is where it always is. A reset only
    # clears the latch.
    self.episode_success[env_ids] = 0.0

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    del env_ids

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    for batch in visualizer.get_env_indices(self.num_envs):
      visualizer.add_sphere(
        center=self.target_pos[batch].cpu().numpy(),
        radius=0.022,
        color=(1.0, 0.5, 0.0, 0.35),
        label=f"stack_target_{batch}",
      )


@dataclass(kw_only=True)
class StackCommandCfg(CommandTermCfg):
  """``asset_cfg`` names the robot, its end effector and its gripper joint."""

  asset_cfg: SceneEntityCfg
  # Never resampled mid-episode: the target follows the tower, not a timer.
  resampling_time_range: tuple[float, float] = field(
    default_factory=lambda: (1.0e9, 1.0e9)
  )

  def build(self, env: ManagerBasedRlEnv) -> StackCommand:
    return StackCommand(self, env)


__all__ = ["StackCommand", "StackCommandCfg"]
