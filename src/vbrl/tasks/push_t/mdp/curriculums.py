from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers import CurriculumTermCfg


class GoalYawStage(TypedDict):
  step: int
  half_range: float


def _validate(stages: list[GoalYawStage]) -> None:
  if not stages:
    raise ValueError("goal_yaw_curriculum requires at least one stage.")
  if stages[0]["step"] != 0:
    raise ValueError("The first goal-yaw stage must start at step 0.")
  steps = [stage["step"] for stage in stages]
  if steps != sorted(steps) or len(set(steps)) != len(steps):
    raise ValueError(f"Goal-yaw stages must have strictly increasing steps: {steps}.")
  for stage in stages:
    half = stage["half_range"]
    if not 0.0 <= half <= torch.pi:
      raise ValueError(
        f"half_range must lie in [0, pi], got {half} at step {stage['step']}."
      )


# Goal-space, not domain randomization: the easy end pins the goal while the
# object's yaw stays uniform, so rotating pays from the first episode.
class goal_yaw_curriculum:
  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
    stages: list[GoalYawStage] = cfg.params["stages"]
    _validate(stages)
    self._stages = stages
    self._command_cfg = env.command_manager.get_term(cfg.params["command_name"]).cfg

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    stages: list[GoalYawStage],
  ) -> dict[str, torch.Tensor]:
    del env_ids, command_name, stages
    step = int(env.common_step_counter)
    half = self._stages[0]["half_range"]
    for stage in self._stages:
      if step >= stage["step"]:
        half = stage["half_range"]
    self._command_cfg.target_yaw_range = (-half, half)
    return {"half_range": torch.tensor(half, device=env.device)}


__all__ = [
  "GoalYawStage",
  "goal_yaw_curriculum",
]
