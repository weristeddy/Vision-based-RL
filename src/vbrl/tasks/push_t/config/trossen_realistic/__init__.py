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
  scene: str = "real_texture",
  episode_length_s: float = 5.0,
  goal_outline: bool = False,
  real_goal_colour: bool = False,
  goal_observation_noise: tuple[float, float] = (0.0, 0.0),
) -> None:
  env = functools.partial(
    trossen_realistic_push_t_rgb_env_cfg,
    action_delta=action_delta,
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

# Two independent ways to keep the goal from reading as a second copy of the
# object, each goal-conditioned and pixel-only. Both run at the 16 s episode the
# 0.03 cap needs: 0.03 x 800 steps is 24 rad of joint travel, against 12 at 8 s
# and the 25 the 0.1-cap generation reached 0.501 success with. An iteration
# costs num_steps_per_env x num_envs whatever the episode length, so the longer
# episode is free.
#
# The goal-conditioned pair carries a per-episode bias on the observed goal --
# 3 mm and 1.5 deg -- which is the rig's measured calibration chain (1.33 mm
# extrinsic repeatability, ~0.4 mm tag detection, +/-1 mm hand-measured margin)
# with headroom. The reward keeps the true goal, so only the actor is misled.

# Separated by shape: the marker is a hollow frame, so colour stays randomized
# over the whole RGB cube for both object and goal, as VisualSlowStep has it.
_OUTLINE = {
  "action_delta": DEPLOYABLE_ACTION_DELTA,
  "episode_length_s": 16.0,
  "goal_outline": True,
}
# Separated by colour instead: filled marker, object and goal pinned near the
# colours the rig actually shows.
_COLOUR = {
  "action_delta": DEPLOYABLE_ACTION_DELTA,
  "episode_length_s": 16.0,
  "real_goal_colour": True,
  "scene": "real_texture_bordeaux",
}
_CALIBRATION_NOISE = (0.003, 0.026)

_register(
  "Mjlab-PushT-GoalOutline-DinoV2ViTS14-Afa6-TrossenRealistic",
  "DinoV2ViTS14-Afa6",
  goal_observation_noise=_CALIBRATION_NOISE,
  **_OUTLINE,
)
_register(
  "Mjlab-PushT-GoalOutlinePixel-DinoV2ViTS14-Afa6-TrossenRealistic",
  "DinoV2ViTS14-Afa6",
  goal_in_observation=False,
  **_OUTLINE,
)
_register(
  "Mjlab-PushT-GoalColour-DinoV2ViTS14-Afa6-TrossenRealistic",
  "DinoV2ViTS14-Afa6",
  goal_observation_noise=_CALIBRATION_NOISE,
  **_COLOUR,
)
_register(
  "Mjlab-PushT-GoalColourPixel-DinoV2ViTS14-Afa6-TrossenRealistic",
  "DinoV2ViTS14-Afa6",
  goal_in_observation=False,
  **_COLOUR,
)
