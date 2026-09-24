import functools

from mjlab.tasks.registry import register_mjlab_task

from vbrl.tasks.push_t.push_t_env_cfg import ACTION_SCALE
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
  action_scale: float,
  goal_in_observation: bool = True,
  fixed_target: tuple[float, float, float] | None = None,
  scene: str = "real_texture",
  episode_length_s: float = 16.0,
  goal_outline: bool = False,
  real_goal_colour: bool = False,
  goal_observation_noise: tuple[float, float] = (0.0, 0.0),
) -> None:
  env = functools.partial(
    trossen_realistic_push_t_rgb_env_cfg,
    action_scale=action_scale,
    goal_in_observation=goal_in_observation,
    fixed_target=fixed_target,
    scene=scene,
    episode_length_s=episode_length_s,
    goal_outline=goal_outline,
    real_goal_colour=real_goal_colour,
    goal_observation_noise=goal_observation_noise,
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
# The marker is a hollow frame so it cannot read as a second copy of the object,
# which leaves colour randomized over the whole RGB cube for object and goal.
#
# The goal-conditioned variant carries a per-episode bias on the observed goal --
# 3 mm and 1.5 deg -- the rig's measured calibration chain (1.33 mm extrinsic
# repeatability, ~0.4 mm tag detection, +/-1 mm hand-measured margin) with
# headroom. The reward keeps the true goal, so only the actor is misled.
_OUTLINE = {
  "action_scale": ACTION_SCALE,
  "goal_outline": True,
}
_CALIBRATION_NOISE = (0.003, 0.026)

_register(
  "Mjlab-PushT-GoalOutline-DinoV2ViTS14-Afa6-TrossenIdentified",
  "DinoV2ViTS14-Afa6",
  goal_observation_noise=_CALIBRATION_NOISE,
  **_OUTLINE,
)
_register(
  "Mjlab-PushT-GoalOutlinePixel-DinoV2ViTS14-Afa6-TrossenIdentified",
  "DinoV2ViTS14-Afa6",
  goal_in_observation=False,
  **_OUTLINE,
)

# The overfitted arm: the rig's own tabletop laid down once with no texture DR,
# and object and marker colours narrowed to what its camera measures. Everything
# else matches the pair above, so the difference is the scene alone.
_REAL_TABLE = {**_OUTLINE, "real_goal_colour": True, "scene": "real_table"}
_register(
  "Mjlab-PushT-RealTable-DinoV2ViTS14-Afa6-TrossenIdentified",
  "DinoV2ViTS14-Afa6",
  goal_observation_noise=_CALIBRATION_NOISE,
  **_REAL_TABLE,
)
_register(
  "Mjlab-PushT-RealTablePixel-DinoV2ViTS14-Afa6-TrossenIdentified",
  "DinoV2ViTS14-Afa6",
  goal_in_observation=False,
  **_REAL_TABLE,
)
