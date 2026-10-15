import functools

from mjlab.tasks.registry import register_mjlab_task

from vbrl.asset_zoo.robots.definition import CameraView
from vbrl.training.runner import VbrlDistillationRunner, VbrlOnPolicyRunner
from vbrl.vision.architectures import ARCHITECTURES

from .env_cfgs import (
  trossen_realistic_push_t_rgb_env_cfg,
  trossen_realistic_push_t_state_env_cfg,
)
from .rl_cfg import (
  STATE_TASK_ID,
  trossen_realistic_push_t_rgb_distillation_runner_cfg,
  trossen_realistic_push_t_rgb_ppo_runner_cfg,
  trossen_realistic_push_t_state_ppo_runner_cfg,
)


def _register(
  task_id: str,
  architecture: str,
  *,
  goal_in_observation: bool = True,
  scene: str = "real_texture",
  real_goal_colour: bool = False,
  goal_observation_noise: tuple[float, float] = (0.0, 0.0),
  camera_view: CameraView = "external",
  distill: bool = False,
) -> None:
  env = functools.partial(
    trossen_realistic_push_t_rgb_env_cfg,
    goal_in_observation=goal_in_observation,
    scene=scene,
    real_goal_colour=real_goal_colour,
    goal_observation_noise=goal_observation_noise,
    camera_view=camera_view,
    teacher_observations=distill,
  )
  rl_cfg = (
    trossen_realistic_push_t_rgb_distillation_runner_cfg
    if distill
    else trossen_realistic_push_t_rgb_ppo_runner_cfg
  )
  register_mjlab_task(
    task_id,
    env(),
    env(play=True),
    rl_cfg(task_id, ARCHITECTURES[architecture], camera_view),
    VbrlDistillationRunner if distill else VbrlOnPolicyRunner,
  )


register_mjlab_task(
  STATE_TASK_ID,
  trossen_realistic_push_t_state_env_cfg(),
  trossen_realistic_push_t_state_env_cfg(play=True),
  trossen_realistic_push_t_state_ppo_runner_cfg(),
  VbrlOnPolicyRunner,
)
# The goal-conditioned variants carry a per-episode bias on the observed goal --
# 3 mm and 1.5 deg -- the rig's measured calibration chain (1.33 mm extrinsic
# repeatability, ~0.4 mm tag detection, +/-1 mm hand-measured margin) with
# headroom. The reward keeps the true goal, so only the actor is misled.
_CALIBRATION_NOISE = (0.003, 0.026)

_register(
  "Mjlab-PushT-GoalOutline-DinoV2ViTS14-Afa6-TrossenIdentified",
  "DinoV2ViTS14-Afa6",
  goal_observation_noise=_CALIBRATION_NOISE,
)
_register(
  "Mjlab-PushT-GoalOutlinePixel-DinoV2ViTS14-Afa6-TrossenIdentified",
  "DinoV2ViTS14-Afa6",
  goal_in_observation=False,
)

# The overfitted arm: the rig's own tabletop laid down once with no texture DR,
# and object and marker colours narrowed to what its camera measures. Everything
# else matches the pair above, so the difference is the scene alone.
_REAL_TABLE = {"real_goal_colour": True, "scene": "real_table"}
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
_register(
  "Mjlab-PushT-RealTableWrist-DinoV2ViTS14-Afa6-TrossenIdentified",
  "DinoV2ViTS14-Afa6",
  goal_observation_noise=_CALIBRATION_NOISE,
  camera_view="wrist",
  **_REAL_TABLE,
)
_register(
  "Mjlab-PushT-RealTableDistill-DinoV2ViTS14-Afa6-TrossenIdentified",
  "DinoV2ViTS14-Afa6",
  goal_observation_noise=_CALIBRATION_NOISE,
  distill=True,
  **_REAL_TABLE,
)
_register(
  "Mjlab-PushT-RealTableWristDistill-DinoV2ViTS14-Afa6-TrossenIdentified",
  "DinoV2ViTS14-Afa6",
  goal_observation_noise=_CALIBRATION_NOISE,
  camera_view="wrist",
  distill=True,
  **_REAL_TABLE,
)
