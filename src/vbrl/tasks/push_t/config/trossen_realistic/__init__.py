import functools

from mjlab.tasks.registry import register_mjlab_task

from vbrl.tasks.push_t.push_t_env_cfg import DEPLOYABLE_ACTION_DELTA
from vbrl.training.runner import VbrlOnPolicyRunner
from vbrl.vision.architectures import ARCHITECTURES

from .env_cfgs import (
  trossen_realistic_push_t_rgb_env_cfg,
  trossen_realistic_push_t_state_env_cfg,
)
from .rl_cfg import (
  STATE_TASK_ID,
  trossen_realistic_push_t_rgb_ppo_runner_cfg,
  trossen_realistic_push_t_state_ppo_runner_cfg,
)


def _register(
  task_id: str,
  architecture: str,
  *,
  action_delta: float,
  goal_in_observation: bool = True,
  fixed_target: tuple[float, float, float] | None = None,
) -> None:
  env = functools.partial(
    trossen_realistic_push_t_rgb_env_cfg,
    action_delta=action_delta,
    goal_in_observation=goal_in_observation,
    fixed_target=fixed_target,
  )
  register_mjlab_task(
    task_id,
    env(),
    env(play=True),
    trossen_realistic_push_t_rgb_ppo_runner_cfg(
      task_id, ARCHITECTURES[architecture]
    ),
    VbrlOnPolicyRunner,
  )


register_mjlab_task(
  STATE_TASK_ID,
  trossen_realistic_push_t_state_env_cfg(),
  trossen_realistic_push_t_state_env_cfg(play=True),
  trossen_realistic_push_t_state_ppo_runner_cfg(),
  VbrlOnPolicyRunner,
)
# `VisualSlowStep` is the current generation: goal drawn on the table, the goal-yaw
# schedule, and the per-step delta cap at 0.03 so the raw policy output is deployable.
_register(
  "Mjlab-PushT-VisualSlowStep-DinoV2ViTS14-Afa6-TrossenRealistic",
  "DinoV2ViTS14-Afa6",
  action_delta=DEPLOYABLE_ACTION_DELTA,
)
_register(
  "Mjlab-PushT-PixelGoalFixed-DinoV2ViTS14-Afa6-TrossenRealistic",
  "DinoV2ViTS14-Afa6",
  action_delta=0.05,
  goal_in_observation=False,
  fixed_target=(0.38442, 0.01567, 0.0124),
)
