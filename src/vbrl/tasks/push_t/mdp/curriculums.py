from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers import CurriculumTermCfg


class penalty_weight_ramp:
  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
    start, end = int(cfg.params["start_step"]), int(cfg.params["end_step"])
    if not 0 <= start < end:
      raise ValueError(f"penalty_weight_ramp needs 0 <= start < end; got {start}, {end}.")
    self._start, self._end = start, end
    self._terms = {
      name: env.reward_manager.get_term_cfg(name) for name in cfg.params["reward_names"]
    }
    self._weights = {name: term.weight for name, term in self._terms.items()}

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    reward_names: list[str],
    start_step: int,
    end_step: int,
  ) -> dict[str, torch.Tensor]:
    del env_ids, reward_names, start_step, end_step
    elapsed = int(env.common_step_counter) - self._start
    scale = min(1.0, max(0.0, elapsed / (self._end - self._start)))
    for name, term in self._terms.items():
      term.weight = self._weights[name] * scale
    return {"scale": torch.tensor(scale, device=env.device)}


__all__ = [
  "penalty_weight_ramp",
]
