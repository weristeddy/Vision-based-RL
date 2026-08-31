"""Curriculum terms owned by Push-T."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers import CurriculumTermCfg


class GoalYawStage(TypedDict):
  """One rung: from ``step`` on, sample the goal yaw from ``+/- half_range``."""

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


class goal_yaw_curriculum:
  """Widen the goal-yaw sampling range as training progresses.

  The policy must estimate the T's absolute yaw from pixels and compose it with
  a goal yaw supplied as a bare sin/cos pair. Starting from a fixed goal makes
  that a single canonical orientation to recognise, so the reward gradient points
  the same way at every reset instead of somewhere new each episode; the range
  then widens back to the full circle.

  This is deliberately a *goal-space* curriculum and not a domain-randomization
  one. ManiSkill3's published Push-T keeps its goal pose fixed for the whole of
  training, so a schedule that ends at ``+/- pi`` finishes strictly harder than
  the benchmark it is compared against.

  The command term re-reads ``target_yaw_range`` on every resample, so writing
  to its config takes effect at the next episode reset.
  """

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

class separation_curriculum:
  """Grow the cap on object-goal separation linearly over training.

  A reverse curriculum: episodes begin with the goal within ``start`` metres of
  the object, held there for ``pin_iterations`` before the ramp begins, and the
  bound then widens until it exceeds the largest separation the
  target range allows -- from which point the filter never fires and the
  distribution is exactly the uncapped one. That endpoint is the reason the cap
  is scheduled rather than the sampling ranges: the curriculum provably finishes
  on the task it is compared against.

  Linear rather than staged on purpose. The goal-yaw curriculum widens in rungs
  and a single 180-degree rung destroyed every run that tried it; a continuous
  ramp has no handover for a converged policy to fall off.

  The command term re-reads ``max_xy_separation`` on every resample, so writing
  to its config takes effect at the next episode reset.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
    start = float(cfg.params["start"])
    end = float(cfg.params["end"])
    iterations = int(cfg.params["iterations"])
    pin = int(cfg.params.get("pin_iterations", 0))
    if start <= 0.0 or end < start or iterations <= 0 or pin < 0:
      raise ValueError(
        f"separation_curriculum needs 0 < start <= end, iterations > 0 and "
        f"pin_iterations >= 0; got start={start}, end={end}, "
        f"iterations={iterations}, pin_iterations={pin}."
      )
    self._start, self._end = start, end
    per_iteration = int(cfg.params["steps_per_iteration"])
    self._pin = pin * per_iteration
    self._steps = iterations * per_iteration
    self._command_cfg = env.command_manager.get_term(cfg.params["command_name"]).cfg

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    start: float,
    end: float,
    iterations: int,
    steps_per_iteration: int,
    pin_iterations: int = 0,
  ) -> dict[str, torch.Tensor]:
    del env_ids, command_name, start, end, iterations, steps_per_iteration
    del pin_iterations
    elapsed = int(env.common_step_counter) - self._pin
    fraction = min(1.0, max(0.0, elapsed / self._steps))
    cap = self._start + fraction * (self._end - self._start)
    self._command_cfg.max_xy_separation = cap
    return {"max_separation": torch.tensor(cap, device=env.device)}


class goal_yaw_resolution_curriculum:
  """Coarsen the goal yaw to a few angles, then refine to the full continuum.

  A curriculum on the goal's *cardinality* rather than its range. The goal
  spans the whole circle from the first episode and the object's yaw stays
  uniform, so the starting state is never made easy -- the failure mode every
  distance-based curriculum here ran into, where the object begins near-optimal,
  contact can only lose reward, and the policy converges on not touching it.

  What it does reduce is how many distinct goal orientations the policy must
  hold at once. `goal_yaw_curriculum` reduces that to one by pinning; this
  reduces it to four and then relaxes, which is why running both separates
  "fewer mappings" from "narrower range" as the reason pinning works.

  Levels double at each rung until `None`, which restores the continuum.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
    self._start = int(cfg.params["start_levels"])
    self._steps = int(cfg.params["rung_iterations"]) * int(
      cfg.params["steps_per_iteration"]
    )
    if self._start < 2 or self._steps <= 0:
      raise ValueError(
        f"goal_yaw_resolution_curriculum needs start_levels >= 2 and "
        f"rung_iterations > 0; got {self._start}, {cfg.params['rung_iterations']}."
      )
    self._rungs = int(cfg.params["rungs"])
    self._command_cfg = env.command_manager.get_term(cfg.params["command_name"]).cfg

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    start_levels: int,
    rungs: int,
    rung_iterations: int,
    steps_per_iteration: int,
  ) -> dict[str, torch.Tensor]:
    del env_ids, command_name, start_levels, rungs, rung_iterations
    del steps_per_iteration
    rung = int(env.common_step_counter) // self._steps
    levels = None if rung >= self._rungs else self._start * (2**rung)
    self._command_cfg.target_yaw_levels = levels
    return {
      "goal_yaw_levels": torch.tensor(
        float(levels) if levels else 0.0, device=env.device
      )
    }


class _LinearToRegistered:
  """Ramp one command-config float from a starting value to its registered one."""

  _field: str

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
    self._command_cfg = env.command_manager.get_term(cfg.params["command_name"]).cfg
    self._start = float(cfg.params["start"])
    self._end = float(cfg.params["end"])
    self._steps = int(cfg.params["iterations"]) * int(
      cfg.params["steps_per_iteration"]
    )
    if self._steps <= 0:
      raise ValueError(f"{type(self).__name__} needs iterations > 0.")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    start: float,
    end: float,
    iterations: int,
    steps_per_iteration: int,
  ) -> dict[str, torch.Tensor]:
    del env_ids, command_name, start, end, iterations, steps_per_iteration
    fraction = min(1.0, int(env.common_step_counter) / self._steps)
    value = self._start + fraction * (self._end - self._start)
    setattr(self._command_cfg, self._field, value)
    return {self._field: torch.tensor(value, device=env.device)}


class success_threshold_curriculum(_LinearToRegistered):
  """Loosen the overlap needed for the sparse bonus, then tighten to the real one.

  The bonus replaces the whole reward with 3.0 and is by far the largest signal
  in the task, but at 0.90 it needs roughly 5 mm *and* 5 degrees together, so a
  policy that has not already solved the task never sees it -- FreeStart ends at
  a 0.010 success rate. Starting loose puts it within reach of roughly-aligned
  states, and overlap is dominated by orientation once the object is anywhere
  near the goal, so what it reinforces is rotation.
  """

  _field = "success_threshold"


class orientation_weight_curriculum(_LinearToRegistered):
  """Weight the dense reward toward rotation early, then rebalance to ManiSkill's.

  The object still starts anywhere, so contact still pays and no do-nothing
  attractor appears -- this changes what is worth doing, not where the episode
  begins. The risk it carries is the one the 250-step ManiSkill run showed: the
  two shaped terms trade against each other (correlation -0.755 between yaw and
  position error), so a weight shift may move which one is sacrificed rather
  than buying both.
  """

  _field = "orientation_weight"


__all__ = [
  "GoalYawStage",
  "goal_yaw_curriculum",
  "goal_yaw_resolution_curriculum",
  "orientation_weight_curriculum",
  "separation_curriculum",
  "success_threshold_curriculum",
]
