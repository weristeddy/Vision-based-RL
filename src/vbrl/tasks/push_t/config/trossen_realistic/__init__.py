"""Register the realistic Trossen Push-T policy contracts, one line per task ID.

``RealTexture`` and ``Default`` differ only in the tabletop -- 1203 photographs
versus one randomized solid colour. Object colour, camera pose and lighting are
randomized identically in both, so the pair isolates how much the tabletop costs
the encoder.
"""

from mjlab.tasks.registry import register_mjlab_task

from vbrl.tasks.push_t.push_t_env_cfg import (
  GOAL_YAW_CURRICULUM_STAGES,
  GOAL_YAW_SLOW_STAGES,
)
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


_REAL_TEXTURE_ENV = trossen_realistic_push_t_rgb_env_cfg(scene="real_texture")
_REAL_TEXTURE_PLAY_ENV = trossen_realistic_push_t_rgb_env_cfg(
  scene="real_texture", play=True
)
_DEFAULT_ENV = trossen_realistic_push_t_rgb_env_cfg(scene="default")
_DEFAULT_PLAY_ENV = trossen_realistic_push_t_rgb_env_cfg(scene="default", play=True)
_FRONT_CAM_ENV = trossen_realistic_push_t_rgb_env_cfg(
  scene="real_texture", camera="external_front"
)
_FRONT_CAM_PLAY_ENV = trossen_realistic_push_t_rgb_env_cfg(
  scene="real_texture", camera="external_front", play=True
)
_CURRICULUM = {
  "scene": "real_texture",
  "camera": "external_front",
  "success_threshold": 0.90,
  "goal_yaw_stages": GOAL_YAW_CURRICULUM_STAGES,
}
_CURRICULUM_ENV = trossen_realistic_push_t_rgb_env_cfg(**_CURRICULUM)
_CURRICULUM_PLAY_ENV = trossen_realistic_push_t_rgb_env_cfg(**_CURRICULUM, play=True)
# `SlowGoal` is the current generation. It keeps the goal-yaw curriculum that
# produced the best result to date and adds the two changes a one-at-a-time
# ablation showed are safe: the tilted camera, which keeps 79% of the object's
# silhouette visible while the gripper is on it rather than 37%, and the fixed
# object colour, which removes the 24% of resets where a uniform RGB sample
# landed within a 1.2:1 luminance ratio of its tabletop.
_SLOW_GOAL = {
  **_CURRICULUM,
  "camera": "external_tilted",
  "scene": "real_texture_red",
  "goal_yaw_stages": GOAL_YAW_SLOW_STAGES,
}
_SLOW_GOAL_ENV = trossen_realistic_push_t_rgb_env_cfg(**_SLOW_GOAL)
_SLOW_GOAL_PLAY_ENV = trossen_realistic_push_t_rgb_env_cfg(**_SLOW_GOAL, play=True)
# --- no curriculum: the goal covers the full circle from the first episode ---
#
# `SlowGoal` pins the goal yaw for 3,000 iterations, which makes the goal
# observation constant and teaches a policy that ignores it. These two variants
# drop the curriculum entirely and differ only in the dense reward's orientation
# factor, because that is the one thing that makes a uniform goal plausible: at
# uniform yaw a quarter of episodes start past 138 degrees, where ManiSkill's
# term has no gradient at all.
_UNIFORM = {k: v for k, v in _SLOW_GOAL.items() if k != "goal_yaw_stages"}
_UNIFORM_ENV = trossen_realistic_push_t_rgb_env_cfg(**_UNIFORM)
_UNIFORM_PLAY_ENV = trossen_realistic_push_t_rgb_env_cfg(**_UNIFORM, play=True)
_UNIFORM_QUAD = {**_UNIFORM, "quadratic_orientation": True}
_UNIFORM_QUAD_ENV = trossen_realistic_push_t_rgb_env_cfg(**_UNIFORM_QUAD)
_UNIFORM_QUAD_PLAY_ENV = trossen_realistic_push_t_rgb_env_cfg(
  **_UNIFORM_QUAD, play=True
)
# `VisualGoal` is `UniformQuad` with the target drawn on the table, which is what
# every published Push-T does: IBC and Diffusion Policy render a green outline,
# ManiSkill3 builds a grey kinematic `goal_Tee` its cameras see. Omitting it was
# VBRL's deviation and it is the largest uncontrolled difference left against the
# benchmark -- without it the policy must estimate the object's absolute yaw from
# pixels and compose it with a goal yaw arriving as two numbers in a separate
# stream, rather than compare two shapes in one image.
#
# It carries no curriculum deliberately. That composition step is the standing
# explanation for why a uniform goal works for only 2 of 15 architectures while
# the curriculum reaches 9, so the drawn target is tested where it is supposed to
# matter. Everything else is `Uniform`'s -- ManiSkill3's own dense reward, the
# one every curriculum generation here was measured with -- so `Uniform` is its
# matched control and the two differ only by the drawn target.
# `FreeStart` removes the artificial separation between where the object starts
# and where the goal is: one shared x range for both, filtered only to keep them
# 1 cm apart. Under `Uniform`'s 15 cm floor an episode never begins near the
# goal, so the fine adjustment that 0.90 overlap requires is only ever met after
# a successful transport, and the sparse at-goal bonus does not fire early. Here
# 50% of episodes start inside 15 cm and 8% inside 5 cm, as a continuum -- no
# schedule and no mixture, so the state distribution stays stationary.
_FREE_START = {**_UNIFORM, "free_start": True}
_FREE_START_ENV = trossen_realistic_push_t_rgb_env_cfg(**_FREE_START)
_FREE_START_PLAY_ENV = trossen_realistic_push_t_rgb_env_cfg(**_FREE_START, play=True)
# `NearGoal` starts a quarter of episodes just short of success -- 6 to 15 mm
# away with 5 to 20 degrees of yaw error -- and leaves the rest as `FreeStart`
# samples them. The band is that tight because overlap only reaches the 0.90
# threshold inside roughly 5 mm *and* 5 degrees together, so a wider one leaves
# the sparse at-goal bonus as unreachable as it is from across the table, and the
# yaw is bounded away from zero because a curriculum that started the object at
# the goal orientation made doing nothing optimal and had to be removed.
#
# A stationary mixture, not a schedule: the state distribution never changes, so
# there is no handover -- the moment that destroyed the single-rung goal-yaw jump
# and the balanced-actor sweep, both of which were healthy right up to it.
_NEAR_GOAL = {**_FREE_START, "near_goal_probability": 0.25}
_NEAR_GOAL_ENV = trossen_realistic_push_t_rgb_env_cfg(**_NEAR_GOAL)
_NEAR_GOAL_PLAY_ENV = trossen_realistic_push_t_rgb_env_cfg(**_NEAR_GOAL, play=True)
# `GrowStart` is the reverse curriculum: every episode starts within 5 cm of its
# goal and the cap grows linearly to the target range's diagonal, past which the
# filter cannot fire. The endpoint is therefore `FreeStart`'s distribution
# exactly, which is what makes the pair attributable.
_GROW_START = {**_FREE_START, "separation_curriculum": True}
_GROW_START_ENV = trossen_realistic_push_t_rgb_env_cfg(**_GROW_START)
_GROW_START_PLAY_ENV = trossen_realistic_push_t_rgb_env_cfg(**_GROW_START, play=True)
# `SlowFree` and `VisualFree` are the two working variants on
# `FreeStart`'s start-state geometry: object and goal drawn from one shared
# rectangle with a 1 cm floor, rather than offset x windows held 15 cm apart.
#
# That floor is not cosmetic. Switching to the shared window alone moves the
# mean separation by 0.7 cm; dropping the floor moves it from 22 cm to 11 cm and
# puts 28% of episodes inside 5 cm. Every variant that has learned to orient the
# T carries the 15 cm floor, and the three that dropped it -- `FreeStart`,
# `NearGoal`, `GrowStart` -- are the three that failed most completely, with the
# last measured doing nothing at all: position error frozen at its reset value
# for 3000 iterations with zero end-effector contact. Whether that floor is
# load-bearing or incidental is the question these two variants ask, holding the
# curriculum and the drawn goal fixed.
_SLOW_GOAL_FREE = {**_SLOW_GOAL, "free_start": True}
_SLOW_GOAL_FREE_ENV = trossen_realistic_push_t_rgb_env_cfg(**_SLOW_GOAL_FREE)
_SLOW_GOAL_FREE_PLAY_ENV = trossen_realistic_push_t_rgb_env_cfg(
  **_SLOW_GOAL_FREE, play=True
)
# `real_texture`, not `real_texture_red`: the object's flat-colour event comes
# back, so the T is resampled over the whole RGB cube every reset, and
# `visual_goal` adds the matching event for the marker. Both are independent and
# unguarded, so on roughly 0.4% of resets they land within an RGB distance of
# 0.1 of each other and on 22% within a 1.2:1 luminance ratio -- episodes where
# the two shapes are hard or impossible to tell apart. That is deliberate: the
# point is a policy that has to find the outline whatever colour it is.
_VISUAL_GOAL = {**_UNIFORM, "visual_goal": True, "scene": "real_texture"}
_VISUAL_GOAL_ENV = trossen_realistic_push_t_rgb_env_cfg(**_VISUAL_GOAL)
_VISUAL_GOAL_PLAY_ENV = trossen_realistic_push_t_rgb_env_cfg(**_VISUAL_GOAL, play=True)
_VISUAL_GOAL_FREE = {**_VISUAL_GOAL, "free_start": True}
_VISUAL_GOAL_FREE_ENV = trossen_realistic_push_t_rgb_env_cfg(**_VISUAL_GOAL_FREE)
_VISUAL_GOAL_FREE_PLAY_ENV = trossen_realistic_push_t_rgb_env_cfg(
  **_VISUAL_GOAL_FREE, play=True
)


def _uniform(task_id: str, architecture: str) -> None:
  """Register one no-curriculum policy on ManiSkill's dense reward.

  The control for `UniformQuad`: if this matches it, the reward change was not
  what mattered and the curriculum was simply unnecessary once the camera and
  the object colour were fixed.
  """
  register_mjlab_task(
    task_id,
    _UNIFORM_ENV,
    _UNIFORM_PLAY_ENV,
    trossen_realistic_push_t_rgb_ppo_runner_cfg(
      task_id,
      ARCHITECTURES[architecture],
      scene="real_texture_red",
      camera="external_tilted_cam",
      success_tag="success_90",
      extra_tags=("no_curriculum",),
    ),
  )


def _uniform_quad(task_id: str, architecture: str) -> None:
  """Register one no-curriculum policy on the quadratic orientation reward."""
  register_mjlab_task(
    task_id,
    _UNIFORM_QUAD_ENV,
    _UNIFORM_QUAD_PLAY_ENV,
    trossen_realistic_push_t_rgb_ppo_runner_cfg(
      task_id,
      ARCHITECTURES[architecture],
      scene="real_texture_red",
      camera="external_tilted_cam",
      success_tag="success_90",
      extra_tags=("no_curriculum", "quadratic_orientation"),
    ),
  )


def _real_texture(task_id: str, architecture: str) -> None:
  """Register one photographic-tabletop policy. ``architecture`` keys ARCHITECTURES."""
  register_mjlab_task(
    task_id,
    _REAL_TEXTURE_ENV,
    _REAL_TEXTURE_PLAY_ENV,
    trossen_realistic_push_t_rgb_ppo_runner_cfg(
      task_id, ARCHITECTURES[architecture], scene="real_texture"
    ),
  )


def _default(task_id: str, architecture: str) -> None:
  """Register one solid-colour-tabletop policy."""
  register_mjlab_task(
    task_id,
    _DEFAULT_ENV,
    _DEFAULT_PLAY_ENV,
    trossen_realistic_push_t_rgb_ppo_runner_cfg(
      task_id, ARCHITECTURES[architecture], scene="default"
    ),
  )


def _curriculum(task_id: str, architecture: str) -> None:
  """Register one goal-yaw-curriculum policy.

  ``Curriculum`` names the whole configuration, as every variant token does:
  photographic tabletop, the front camera, ManiSkill3's 0.90 success threshold,
  and a goal yaw that starts fixed and widens to the full circle.
  """
  register_mjlab_task(
    task_id,
    _CURRICULUM_ENV,
    _CURRICULUM_PLAY_ENV,
    trossen_realistic_push_t_rgb_ppo_runner_cfg(
      task_id,
      ARCHITECTURES[architecture],
      scene="real_texture",
      camera="external_front_cam",
      success_tag="success_90",
      extra_tags=("goal_yaw_curriculum",),
    ),
  )


def _slow_goal(task_id: str, architecture: str) -> None:
  """Register one current-generation policy.

  The goal yaw stays fixed for 3,000 iterations and then widens in 22.5-degree
  rungs, seen through the tilted camera with a fixed-colour object.
  """
  register_mjlab_task(
    task_id,
    _SLOW_GOAL_ENV,
    _SLOW_GOAL_PLAY_ENV,
    trossen_realistic_push_t_rgb_ppo_runner_cfg(
      task_id,
      ARCHITECTURES[architecture],
      scene="real_texture_red",
      camera="external_tilted_cam",
      success_tag="success_90",
      extra_tags=("goal_yaw_curriculum", "slow_goal"),
    ),
  )


def _free_start(task_id: str, architecture: str) -> None:
  """Register one no-curriculum policy whose start and goal share a range.

  Identical to :func:`_uniform` in scene, camera, threshold, reward and the
  absence of a curriculum; the only difference is that the object and the goal
  are drawn from the same x range and separated by 1 cm instead of 15, so an
  episode can begin close to its goal.
  """
  register_mjlab_task(
    task_id,
    _FREE_START_ENV,
    _FREE_START_PLAY_ENV,
    trossen_realistic_push_t_rgb_ppo_runner_cfg(
      task_id,
      ARCHITECTURES[architecture],
      scene="real_texture_red",
      camera="external_tilted_cam",
      success_tag="success_90",
      extra_tags=("no_curriculum", "free_start"),
    ),
  )


def _near_goal(task_id: str, architecture: str) -> None:
  """Register one no-curriculum policy with a quarter of episodes biased close.

  Identical to :func:`_free_start` except that 25% of episodes begin 6-15 mm
  from the goal with 5-20 degrees of yaw error, so the sparse at-goal bonus is
  within reach from the first iteration while three quarters of the data remain
  the unbiased task.
  """
  register_mjlab_task(
    task_id,
    _NEAR_GOAL_ENV,
    _NEAR_GOAL_PLAY_ENV,
    trossen_realistic_push_t_rgb_ppo_runner_cfg(
      task_id,
      ARCHITECTURES[architecture],
      scene="real_texture_red",
      camera="external_tilted_cam",
      success_tag="success_90",
      extra_tags=("no_curriculum", "free_start", "near_goal"),
    ),
  )


def _grow_start(task_id: str, architecture: str) -> None:
  """Register one reverse-curriculum policy.

  The cap on object-goal separation grows linearly from 5 cm to the target
  range's diagonal over 4,000 iterations; beyond that the cap cannot bind and
  the remaining 2,000 iterations run on `FreeStart`'s distribution exactly.
  """
  register_mjlab_task(
    task_id,
    _GROW_START_ENV,
    _GROW_START_PLAY_ENV,
    trossen_realistic_push_t_rgb_ppo_runner_cfg(
      task_id,
      ARCHITECTURES[architecture],
      scene="real_texture_red",
      camera="external_tilted_cam",
      success_tag="success_90",
      extra_tags=("free_start", "separation_curriculum"),
    ),
  )


def _visual_goal(task_id: str, architecture: str) -> None:
  """Register one current-generation policy that can see its target.

  Identical to :func:`_uniform` in scene, camera, threshold and reward --
  and, like it, the goal covers the full circle from the first episode. The only
  difference is a visual-only T posed at the goal each reset, so the target
  reaches the policy through the image as well as through ``target_pose``. The
  numeric channel is deliberately kept: the two are complementary, and dropping
  it would change two things at once.
  """
  register_mjlab_task(
    task_id,
    _VISUAL_GOAL_ENV,
    _VISUAL_GOAL_PLAY_ENV,
    trossen_realistic_push_t_rgb_ppo_runner_cfg(
      task_id,
      ARCHITECTURES[architecture],
      scene="real_texture_red",
      camera="external_tilted_cam",
      success_tag="success_90",
      extra_tags=("no_curriculum", "visual_goal"),
    ),
  )


def _slow_goal_free(task_id: str, architecture: str) -> None:
  """Register one slow goal-yaw curriculum on FreeStart's start-state geometry.

  Differs from :func:`_slow_goal` by the goal-position draw alone: one shared
  rectangle with a 1 cm floor instead of offset windows held 15 cm apart. The
  yaw curriculum, scene, camera, threshold and reward are untouched.
  """
  register_mjlab_task(
    task_id,
    _SLOW_GOAL_FREE_ENV,
    _SLOW_GOAL_FREE_PLAY_ENV,
    trossen_realistic_push_t_rgb_ppo_runner_cfg(
      task_id,
      ARCHITECTURES[architecture],
      scene="real_texture_red",
      camera="external_tilted_cam",
      success_tag="success_90",
      extra_tags=("slow_goal_curriculum", "free_start"),
    ),
  )


def _visual_goal_free(task_id: str, architecture: str) -> None:
  """Register one drawn-goal policy on FreeStart's start-state geometry.

  Differs from :func:`_visual_goal` by the goal-position draw alone.
  """
  register_mjlab_task(
    task_id,
    _VISUAL_GOAL_FREE_ENV,
    _VISUAL_GOAL_FREE_PLAY_ENV,
    trossen_realistic_push_t_rgb_ppo_runner_cfg(
      task_id,
      ARCHITECTURES[architecture],
      scene="real_texture_red",
      camera="external_tilted_cam",
      success_tag="success_90",
      extra_tags=("no_curriculum", "visual_goal", "free_start"),
    ),
  )


def _balanced(task_id: str, architecture: str) -> None:
  """Register one current-generation policy with ManiSkill3's stream balance.

  The environment is byte-for-byte ``SlowGoal``'s -- same scene, camera, success
  threshold and 22.5-degree goal-yaw rungs. What differs is the actor: MJLab
  concatenates the 27 normalised proprioceptive dimensions at their native width
  beside 256 visual ones, so the goal pose is five numbers out of 283.
  ``BalancedVisionModel`` gives the state its own ``Linear(27, 256)`` first, as
  ``ppo_rgb.py`` does, and the streams meet at 256 each.

  The variant token normally names the environment, and here it does not -- the
  environment is identical and only the model changes. It goes in the variant
  anyway because a task ID has to determine the architecture completely and the
  ``<Arch>`` field carries only encoder and adapter.
  """
  register_mjlab_task(
    task_id,
    _SLOW_GOAL_ENV,
    _SLOW_GOAL_PLAY_ENV,
    trossen_realistic_push_t_rgb_ppo_runner_cfg(
      task_id,
      ARCHITECTURES[architecture],
      scene="real_texture_red",
      camera="external_tilted_cam",
      success_tag="success_90",
      extra_tags=("goal_yaw_curriculum", "slow_goal", "balanced_state"),
      actor_class="vbrl.vision.model:BalancedVisionModel",
    ),
  )


def _front_cam(task_id: str, architecture: str) -> None:
  """Register one photographic-tabletop policy seen from the candidate camera."""
  register_mjlab_task(
    task_id,
    _FRONT_CAM_ENV,
    _FRONT_CAM_PLAY_ENV,
    trossen_realistic_push_t_rgb_ppo_runner_cfg(
      task_id,
      ARCHITECTURES[architecture],
      scene="real_texture",
      camera="external_front_cam",
    ),
  )


# --- State: no camera, plain scene -------------------------------------------

register_mjlab_task(
  STATE_TASK_ID,
  trossen_realistic_push_t_state_env_cfg(),
  trossen_realistic_push_t_state_env_cfg(play=True),
  trossen_realistic_push_t_state_ppo_runner_cfg(),
)

# --- RealTexture: 1203 photographic tabletops --------------------------------

_real_texture(
  "Mjlab-PushT-RealTexture-NatureCnn-LocalGrid7-TrossenRealistic",
  "NatureCnn-LocalGrid7",
)
_real_texture(
  "Mjlab-PushT-RealTexture-NatureCnn-SpatialSoftmax-TrossenRealistic",
  "NatureCnn-SpatialSoftmax",
)
_real_texture(
  "Mjlab-PushT-RealTexture-CompactVit-LocalGrid8-TrossenRealistic",
  "CompactVit-LocalGrid8",
)
_real_texture(
  "Mjlab-PushT-RealTexture-CompactVit-SpatialSoftmax-TrossenRealistic",
  "CompactVit-SpatialSoftmax",
)
_real_texture(
  "Mjlab-PushT-RealTexture-DinoV2ViTS14-Linear-TrossenRealistic",
  "DinoV2ViTS14-Linear",
)
_real_texture(
  "Mjlab-PushT-RealTexture-DinoV2ViTS14-LocalGrid7-TrossenRealistic",
  "DinoV2ViTS14-LocalGrid7",
)
_real_texture(
  "Mjlab-PushT-RealTexture-DinoV2ViTS14-SpatialSoftmax-TrossenRealistic",
  "DinoV2ViTS14-SpatialSoftmax",
)
_real_texture(
  "Mjlab-PushT-RealTexture-DinoV2ViTS14-Afa6-TrossenRealistic",
  "DinoV2ViTS14-Afa6",
)
_real_texture(
  "Mjlab-PushT-RealTexture-R3MResNet50-Linear-TrossenRealistic",
  "R3MResNet50-Linear",
)
_real_texture(
  "Mjlab-PushT-RealTexture-R3MResNet50-LocalGrid7-TrossenRealistic",
  "R3MResNet50-LocalGrid7",
)
_real_texture(
  "Mjlab-PushT-RealTexture-R3MResNet50-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50-SpatialSoftmax",
)
_real_texture(
  "Mjlab-PushT-RealTexture-R3MResNet50-Afa32-TrossenRealistic",
  "R3MResNet50-Afa32",
)

# --- Default: one randomized solid tabletop colour ---------------------------

_default(
  "Mjlab-PushT-Default-NatureCnn-LocalGrid7-TrossenRealistic",
  "NatureCnn-LocalGrid7",
)
_default(
  "Mjlab-PushT-Default-NatureCnn-SpatialSoftmax-TrossenRealistic",
  "NatureCnn-SpatialSoftmax",
)
_default(
  "Mjlab-PushT-Default-CompactVit-LocalGrid8-TrossenRealistic",
  "CompactVit-LocalGrid8",
)
_default(
  "Mjlab-PushT-Default-CompactVit-SpatialSoftmax-TrossenRealistic",
  "CompactVit-SpatialSoftmax",
)
_default(
  "Mjlab-PushT-Default-DinoV2ViTS14-Linear-TrossenRealistic",
  "DinoV2ViTS14-Linear",
)
_default(
  "Mjlab-PushT-Default-DinoV2ViTS14-LocalGrid7-TrossenRealistic",
  "DinoV2ViTS14-LocalGrid7",
)
_default(
  "Mjlab-PushT-Default-DinoV2ViTS14-SpatialSoftmax-TrossenRealistic",
  "DinoV2ViTS14-SpatialSoftmax",
)
_default(
  "Mjlab-PushT-Default-DinoV2ViTS14-Afa6-TrossenRealistic",
  "DinoV2ViTS14-Afa6",
)
_default(
  "Mjlab-PushT-Default-R3MResNet50-Linear-TrossenRealistic",
  "R3MResNet50-Linear",
)
_default(
  "Mjlab-PushT-Default-R3MResNet50-LocalGrid7-TrossenRealistic",
  "R3MResNet50-LocalGrid7",
)
_default(
  "Mjlab-PushT-Default-R3MResNet50-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50-SpatialSoftmax",
)
_default(
  "Mjlab-PushT-Default-R3MResNet50-Afa32-TrossenRealistic",
  "R3MResNet50-Afa32",
)

# --- FrontCam: photographic tabletop, candidate camera, native grids ---------
#
# The full current architecture table under the near-overhead front camera. The
# scene is `real_texture`; the variant token names the camera because that is
# the one thing that differs from the RealTexture arm above. Provisional: if the
# camera proves out it becomes `external`, and these fold back into RealTexture.

_front_cam(
  "Mjlab-PushT-FrontCam-NatureCnn-Flatten-TrossenRealistic",
  "NatureCnn-Flatten",
)
_front_cam(
  "Mjlab-PushT-FrontCam-NatureCnn-SpatialSoftmax-TrossenRealistic",
  "NatureCnn-SpatialSoftmax",
)
_front_cam(
  "Mjlab-PushT-FrontCam-NatureCnn-Afa1-TrossenRealistic",
  "NatureCnn-Afa1",
)
_front_cam(
  "Mjlab-PushT-FrontCam-CompactVit-Flatten-TrossenRealistic",
  "CompactVit-Flatten",
)
_front_cam(
  "Mjlab-PushT-FrontCam-CompactVit-SpatialSoftmax-TrossenRealistic",
  "CompactVit-SpatialSoftmax",
)
_front_cam(
  "Mjlab-PushT-FrontCam-CompactVit-Afa2-TrossenRealistic",
  "CompactVit-Afa2",
)
_front_cam(
  "Mjlab-PushT-FrontCam-DinoV2ViTS14-Linear-TrossenRealistic",
  "DinoV2ViTS14-Linear",
)
_front_cam(
  "Mjlab-PushT-FrontCam-DinoV2ViTS14-LocalGrid16-TrossenRealistic",
  "DinoV2ViTS14-LocalGrid16",
)
_front_cam(
  "Mjlab-PushT-FrontCam-DinoV2ViTS14-SpatialSoftmax-TrossenRealistic",
  "DinoV2ViTS14-SpatialSoftmax",
)
_front_cam(
  "Mjlab-PushT-FrontCam-DinoV2ViTS14-Afa6-TrossenRealistic",
  "DinoV2ViTS14-Afa6",
)
_front_cam(
  "Mjlab-PushT-FrontCam-R3MResNet50-Linear-TrossenRealistic",
  "R3MResNet50-Linear",
)
_front_cam(
  "Mjlab-PushT-FrontCam-R3MResNet50-LocalGrid7-TrossenRealistic",
  "R3MResNet50-LocalGrid7",
)
_front_cam(
  "Mjlab-PushT-FrontCam-R3MResNet50-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50-SpatialSoftmax",
)
_front_cam(
  "Mjlab-PushT-FrontCam-R3MResNet50-Afa32-TrossenRealistic",
  "R3MResNet50-Afa32",
)

# --- Curriculum: front camera, 0.90 success, goal yaw fixed then widening ---
#
# The generation that tests whether the goal representation, not perception,
# is what stalls yaw. Probes show seven of these fourteen cannot encode yaw
# even with 150k labels; they are registered so the comparison stays complete.

_curriculum(
  "Mjlab-PushT-Curriculum-NatureCnn-Flatten-TrossenRealistic",
  "NatureCnn-Flatten",
)
_curriculum(
  "Mjlab-PushT-Curriculum-NatureCnn-SpatialSoftmax-TrossenRealistic",
  "NatureCnn-SpatialSoftmax",
)
_curriculum(
  "Mjlab-PushT-Curriculum-NatureCnn-Afa1-TrossenRealistic",
  "NatureCnn-Afa1",
)
_curriculum(
  "Mjlab-PushT-Curriculum-CompactVit-Flatten-TrossenRealistic",
  "CompactVit-Flatten",
)
_curriculum(
  "Mjlab-PushT-Curriculum-CompactVit-SpatialSoftmax-TrossenRealistic",
  "CompactVit-SpatialSoftmax",
)
_curriculum(
  "Mjlab-PushT-Curriculum-CompactVit-Afa2-TrossenRealistic",
  "CompactVit-Afa2",
)
_curriculum(
  "Mjlab-PushT-Curriculum-DinoV2ViTS14-Linear-TrossenRealistic",
  "DinoV2ViTS14-Linear",
)
_curriculum(
  "Mjlab-PushT-Curriculum-DinoV2ViTS14-LocalGrid16-TrossenRealistic",
  "DinoV2ViTS14-LocalGrid16",
)
_curriculum(
  "Mjlab-PushT-Curriculum-DinoV2ViTS14-SpatialSoftmax-TrossenRealistic",
  "DinoV2ViTS14-SpatialSoftmax",
)
_curriculum(
  "Mjlab-PushT-Curriculum-DinoV2ViTS14-Afa6-TrossenRealistic",
  "DinoV2ViTS14-Afa6",
)
_curriculum(
  "Mjlab-PushT-Curriculum-R3MResNet50-Linear-TrossenRealistic",
  "R3MResNet50-Linear",
)
_curriculum(
  "Mjlab-PushT-Curriculum-R3MResNet50-LocalGrid7-TrossenRealistic",
  "R3MResNet50-LocalGrid7",
)
_curriculum(
  "Mjlab-PushT-Curriculum-R3MResNet50-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50-SpatialSoftmax",
)
_curriculum(
  "Mjlab-PushT-Curriculum-R3MResNet50-Afa32-TrossenRealistic",
  "R3MResNet50-Afa32",
)
_curriculum(
  "Mjlab-PushT-Curriculum-R3MResNet50L3-LocalGrid14-TrossenRealistic",
  "R3MResNet50L3-LocalGrid14",
)
_curriculum(
  "Mjlab-PushT-Curriculum-R3MResNet50L3-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50L3-SpatialSoftmax",
)
_curriculum(
  "Mjlab-PushT-Curriculum-R3MResNet50L3-Afa16-TrossenRealistic",
  "R3MResNet50L3-Afa16",
)

# --- SlowGoal: the current generation -------------------------------------
#
# AFA is dropped for the scratch encoders: permutation-invariant pooling over
# position-free CNN features gave 117 mm position error, the predict-the-mean
# baseline. It stays for the frozen backbones, where it is the strongest
# adapter (DINOv2 1.5 deg, R3M layer4 4.3 deg in the supervised probe).

_slow_goal(
  "Mjlab-PushT-SlowGoal-NatureCnn-Flatten-TrossenRealistic",
  "NatureCnn-Flatten",
)
_slow_goal(
  "Mjlab-PushT-SlowGoal-NatureCnn-SpatialSoftmax-TrossenRealistic",
  "NatureCnn-SpatialSoftmax",
)
_slow_goal(
  "Mjlab-PushT-SlowGoal-CompactVit-Flatten-TrossenRealistic",
  "CompactVit-Flatten",
)
_slow_goal(
  "Mjlab-PushT-SlowGoal-CompactVit-SpatialSoftmax-TrossenRealistic",
  "CompactVit-SpatialSoftmax",
)
_slow_goal(
  "Mjlab-PushT-SlowGoal-DinoV2ViTS14-Linear-TrossenRealistic",
  "DinoV2ViTS14-Linear",
)
_slow_goal(
  "Mjlab-PushT-SlowGoal-DinoV2ViTS14-LocalGrid16-TrossenRealistic",
  "DinoV2ViTS14-LocalGrid16",
)
_slow_goal(
  "Mjlab-PushT-SlowGoal-DinoV2ViTS14-SpatialSoftmax-TrossenRealistic",
  "DinoV2ViTS14-SpatialSoftmax",
)
_slow_goal(
  "Mjlab-PushT-SlowGoal-DinoV2ViTS14-Afa6-TrossenRealistic",
  "DinoV2ViTS14-Afa6",
)
_slow_goal(
  "Mjlab-PushT-SlowGoal-R3MResNet50-Linear-TrossenRealistic",
  "R3MResNet50-Linear",
)
_slow_goal(
  "Mjlab-PushT-SlowGoal-R3MResNet50-LocalGrid7-TrossenRealistic",
  "R3MResNet50-LocalGrid7",
)
_slow_goal(
  "Mjlab-PushT-SlowGoal-R3MResNet50-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50-SpatialSoftmax",
)
_slow_goal(
  "Mjlab-PushT-SlowGoal-R3MResNet50-Afa32-TrossenRealistic",
  "R3MResNet50-Afa32",
)
_slow_goal(
  "Mjlab-PushT-SlowGoal-R3MResNet50L3-LocalGrid14-TrossenRealistic",
  "R3MResNet50L3-LocalGrid14",
)
_slow_goal(
  "Mjlab-PushT-SlowGoal-R3MResNet50L3-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50L3-SpatialSoftmax",
)
_slow_goal(
  "Mjlab-PushT-SlowGoal-R3MResNet50L3-Afa16-TrossenRealistic",
  "R3MResNet50L3-Afa16",
)


# --- Uniform: no curriculum, ManiSkill reward (control) ----------------------
_uniform(
  "Mjlab-PushT-Uniform-NatureCnn-Flatten-TrossenRealistic",
  "NatureCnn-Flatten",
)
_uniform(
  "Mjlab-PushT-Uniform-NatureCnn-SpatialSoftmax-TrossenRealistic",
  "NatureCnn-SpatialSoftmax",
)
_uniform(
  "Mjlab-PushT-Uniform-CompactVit-Flatten-TrossenRealistic",
  "CompactVit-Flatten",
)
_uniform(
  "Mjlab-PushT-Uniform-CompactVit-SpatialSoftmax-TrossenRealistic",
  "CompactVit-SpatialSoftmax",
)
_uniform(
  "Mjlab-PushT-Uniform-DinoV2ViTS14-Linear-TrossenRealistic",
  "DinoV2ViTS14-Linear",
)
_uniform(
  "Mjlab-PushT-Uniform-DinoV2ViTS14-LocalGrid16-TrossenRealistic",
  "DinoV2ViTS14-LocalGrid16",
)
_uniform(
  "Mjlab-PushT-Uniform-DinoV2ViTS14-SpatialSoftmax-TrossenRealistic",
  "DinoV2ViTS14-SpatialSoftmax",
)
_uniform(
  "Mjlab-PushT-Uniform-DinoV2ViTS14-Afa6-TrossenRealistic",
  "DinoV2ViTS14-Afa6",
)
_uniform(
  "Mjlab-PushT-Uniform-R3MResNet50-Linear-TrossenRealistic",
  "R3MResNet50-Linear",
)
_uniform(
  "Mjlab-PushT-Uniform-R3MResNet50-LocalGrid7-TrossenRealistic",
  "R3MResNet50-LocalGrid7",
)
_uniform(
  "Mjlab-PushT-Uniform-R3MResNet50-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50-SpatialSoftmax",
)
_uniform(
  "Mjlab-PushT-Uniform-R3MResNet50-Afa32-TrossenRealistic",
  "R3MResNet50-Afa32",
)
_uniform(
  "Mjlab-PushT-Uniform-R3MResNet50L3-LocalGrid14-TrossenRealistic",
  "R3MResNet50L3-LocalGrid14",
)
_uniform(
  "Mjlab-PushT-Uniform-R3MResNet50L3-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50L3-SpatialSoftmax",
)
_uniform(
  "Mjlab-PushT-Uniform-R3MResNet50L3-Afa16-TrossenRealistic",
  "R3MResNet50L3-Afa16",
)

# --- UniformQuad: no curriculum, quadratic orientation reward ----------------
_uniform_quad(
  "Mjlab-PushT-UniformQuad-NatureCnn-Flatten-TrossenRealistic",
  "NatureCnn-Flatten",
)
_uniform_quad(
  "Mjlab-PushT-UniformQuad-NatureCnn-SpatialSoftmax-TrossenRealistic",
  "NatureCnn-SpatialSoftmax",
)
_uniform_quad(
  "Mjlab-PushT-UniformQuad-CompactVit-Flatten-TrossenRealistic",
  "CompactVit-Flatten",
)
_uniform_quad(
  "Mjlab-PushT-UniformQuad-CompactVit-SpatialSoftmax-TrossenRealistic",
  "CompactVit-SpatialSoftmax",
)
_uniform_quad(
  "Mjlab-PushT-UniformQuad-DinoV2ViTS14-Linear-TrossenRealistic",
  "DinoV2ViTS14-Linear",
)
_uniform_quad(
  "Mjlab-PushT-UniformQuad-DinoV2ViTS14-LocalGrid16-TrossenRealistic",
  "DinoV2ViTS14-LocalGrid16",
)
_uniform_quad(
  "Mjlab-PushT-UniformQuad-DinoV2ViTS14-SpatialSoftmax-TrossenRealistic",
  "DinoV2ViTS14-SpatialSoftmax",
)
_uniform_quad(
  "Mjlab-PushT-UniformQuad-DinoV2ViTS14-Afa6-TrossenRealistic",
  "DinoV2ViTS14-Afa6",
)
_uniform_quad(
  "Mjlab-PushT-UniformQuad-R3MResNet50-Linear-TrossenRealistic",
  "R3MResNet50-Linear",
)
_uniform_quad(
  "Mjlab-PushT-UniformQuad-R3MResNet50-LocalGrid7-TrossenRealistic",
  "R3MResNet50-LocalGrid7",
)
_uniform_quad(
  "Mjlab-PushT-UniformQuad-R3MResNet50-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50-SpatialSoftmax",
)
_uniform_quad(
  "Mjlab-PushT-UniformQuad-R3MResNet50-Afa32-TrossenRealistic",
  "R3MResNet50-Afa32",
)
_uniform_quad(
  "Mjlab-PushT-UniformQuad-R3MResNet50L3-LocalGrid14-TrossenRealistic",
  "R3MResNet50L3-LocalGrid14",
)
_uniform_quad(
  "Mjlab-PushT-UniformQuad-R3MResNet50L3-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50L3-SpatialSoftmax",
)
_uniform_quad(
  "Mjlab-PushT-UniformQuad-R3MResNet50L3-Afa16-TrossenRealistic",
  "R3MResNet50L3-Afa16",
)

# --- Balanced: SlowGoal's environment, ManiSkill3's stream balance ----------
#
# One variable against the SlowGoal sweep: the actor projects proprioception to
# 256 before the concat instead of passing 27 raw dimensions. The two scratch
# rows additionally carry ManiSkill's rectified head rather than the layer-normed
# one, so those two differ in two ways and the other thirteen in one.

_balanced(
  "Mjlab-PushT-Balanced-NatureCnn-FlattenRelu-TrossenRealistic",
  "NatureCnn-FlattenRelu",
)
_balanced(
  "Mjlab-PushT-Balanced-NatureCnn-SpatialSoftmax-TrossenRealistic",
  "NatureCnn-SpatialSoftmax",
)
_balanced(
  "Mjlab-PushT-Balanced-CompactVit-FlattenRelu-TrossenRealistic",
  "CompactVit-FlattenRelu",
)
_balanced(
  "Mjlab-PushT-Balanced-CompactVit-SpatialSoftmax-TrossenRealistic",
  "CompactVit-SpatialSoftmax",
)
_balanced(
  "Mjlab-PushT-Balanced-DinoV2ViTS14-Linear-TrossenRealistic",
  "DinoV2ViTS14-Linear",
)
_balanced(
  "Mjlab-PushT-Balanced-DinoV2ViTS14-LocalGrid16-TrossenRealistic",
  "DinoV2ViTS14-LocalGrid16",
)
_balanced(
  "Mjlab-PushT-Balanced-DinoV2ViTS14-SpatialSoftmax-TrossenRealistic",
  "DinoV2ViTS14-SpatialSoftmax",
)
_balanced(
  "Mjlab-PushT-Balanced-DinoV2ViTS14-Afa6-TrossenRealistic",
  "DinoV2ViTS14-Afa6",
)
_balanced(
  "Mjlab-PushT-Balanced-R3MResNet50-Linear-TrossenRealistic",
  "R3MResNet50-Linear",
)
_balanced(
  "Mjlab-PushT-Balanced-R3MResNet50-LocalGrid7-TrossenRealistic",
  "R3MResNet50-LocalGrid7",
)
_balanced(
  "Mjlab-PushT-Balanced-R3MResNet50-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50-SpatialSoftmax",
)
_balanced(
  "Mjlab-PushT-Balanced-R3MResNet50-Afa32-TrossenRealistic",
  "R3MResNet50-Afa32",
)
_balanced(
  "Mjlab-PushT-Balanced-R3MResNet50L3-LocalGrid14-TrossenRealistic",
  "R3MResNet50L3-LocalGrid14",
)
_balanced(
  "Mjlab-PushT-Balanced-R3MResNet50L3-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50L3-SpatialSoftmax",
)
_balanced(
  "Mjlab-PushT-Balanced-R3MResNet50L3-Afa16-TrossenRealistic",
  "R3MResNet50L3-Afa16",
)

# --- VisualGoal: UniformQuad with the target drawn on the table -------------
#
# One variable against the Uniform sweep: ManiSkill3's dense reward, no
# curriculum, goal uniform over the full circle from the first episode. Same
# architectures, so every row is a matched pair.

_visual_goal(
  "Mjlab-PushT-VisualGoal-NatureCnn-Flatten-TrossenRealistic",
  "NatureCnn-Flatten",
)
_visual_goal(
  "Mjlab-PushT-VisualGoal-NatureCnn-SpatialSoftmax-TrossenRealistic",
  "NatureCnn-SpatialSoftmax",
)
_visual_goal(
  "Mjlab-PushT-VisualGoal-CompactVit-Flatten-TrossenRealistic",
  "CompactVit-Flatten",
)
_visual_goal(
  "Mjlab-PushT-VisualGoal-CompactVit-SpatialSoftmax-TrossenRealistic",
  "CompactVit-SpatialSoftmax",
)
_visual_goal(
  "Mjlab-PushT-VisualGoal-DinoV2ViTS14-Linear-TrossenRealistic",
  "DinoV2ViTS14-Linear",
)
_visual_goal(
  "Mjlab-PushT-VisualGoal-DinoV2ViTS14-LocalGrid16-TrossenRealistic",
  "DinoV2ViTS14-LocalGrid16",
)
_visual_goal(
  "Mjlab-PushT-VisualGoal-DinoV2ViTS14-SpatialSoftmax-TrossenRealistic",
  "DinoV2ViTS14-SpatialSoftmax",
)
_visual_goal(
  "Mjlab-PushT-VisualGoal-DinoV2ViTS14-Afa6-TrossenRealistic",
  "DinoV2ViTS14-Afa6",
)
_visual_goal(
  "Mjlab-PushT-VisualGoal-R3MResNet50-Linear-TrossenRealistic",
  "R3MResNet50-Linear",
)
_visual_goal(
  "Mjlab-PushT-VisualGoal-R3MResNet50-LocalGrid7-TrossenRealistic",
  "R3MResNet50-LocalGrid7",
)
_visual_goal(
  "Mjlab-PushT-VisualGoal-R3MResNet50-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50-SpatialSoftmax",
)
_visual_goal(
  "Mjlab-PushT-VisualGoal-R3MResNet50-Afa32-TrossenRealistic",
  "R3MResNet50-Afa32",
)
_visual_goal(
  "Mjlab-PushT-VisualGoal-R3MResNet50L3-LocalGrid14-TrossenRealistic",
  "R3MResNet50L3-LocalGrid14",
)
_visual_goal(
  "Mjlab-PushT-VisualGoal-R3MResNet50L3-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50L3-SpatialSoftmax",
)
_visual_goal(
  "Mjlab-PushT-VisualGoal-R3MResNet50L3-Afa16-TrossenRealistic",
  "R3MResNet50L3-Afa16",
)

# --- FreeStart: no curriculum, start and goal share one range ---------------
#
# One variable against the Uniform sweep: the 15 cm floor between object and goal
# becomes 1 cm, and both are drawn from x (0.25, 0.45).

_free_start(
  "Mjlab-PushT-FreeStart-NatureCnn-Flatten-TrossenRealistic",
  "NatureCnn-Flatten",
)
_free_start(
  "Mjlab-PushT-FreeStart-NatureCnn-SpatialSoftmax-TrossenRealistic",
  "NatureCnn-SpatialSoftmax",
)
_free_start(
  "Mjlab-PushT-FreeStart-CompactVit-Flatten-TrossenRealistic",
  "CompactVit-Flatten",
)
_free_start(
  "Mjlab-PushT-FreeStart-CompactVit-SpatialSoftmax-TrossenRealistic",
  "CompactVit-SpatialSoftmax",
)
_free_start(
  "Mjlab-PushT-FreeStart-DinoV2ViTS14-Linear-TrossenRealistic",
  "DinoV2ViTS14-Linear",
)
_free_start(
  "Mjlab-PushT-FreeStart-DinoV2ViTS14-LocalGrid16-TrossenRealistic",
  "DinoV2ViTS14-LocalGrid16",
)
_free_start(
  "Mjlab-PushT-FreeStart-DinoV2ViTS14-SpatialSoftmax-TrossenRealistic",
  "DinoV2ViTS14-SpatialSoftmax",
)
_free_start(
  "Mjlab-PushT-FreeStart-DinoV2ViTS14-Afa6-TrossenRealistic",
  "DinoV2ViTS14-Afa6",
)
_free_start(
  "Mjlab-PushT-FreeStart-R3MResNet50-Linear-TrossenRealistic",
  "R3MResNet50-Linear",
)
_free_start(
  "Mjlab-PushT-FreeStart-R3MResNet50-LocalGrid7-TrossenRealistic",
  "R3MResNet50-LocalGrid7",
)
_free_start(
  "Mjlab-PushT-FreeStart-R3MResNet50-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50-SpatialSoftmax",
)
_free_start(
  "Mjlab-PushT-FreeStart-R3MResNet50-Afa32-TrossenRealistic",
  "R3MResNet50-Afa32",
)
_free_start(
  "Mjlab-PushT-FreeStart-R3MResNet50L3-LocalGrid14-TrossenRealistic",
  "R3MResNet50L3-LocalGrid14",
)
_free_start(
  "Mjlab-PushT-FreeStart-R3MResNet50L3-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50L3-SpatialSoftmax",
)
_free_start(
  "Mjlab-PushT-FreeStart-R3MResNet50L3-Afa16-TrossenRealistic",
  "R3MResNet50L3-Afa16",
)

# --- NearGoal: FreeStart with a quarter of episodes biased close -------------

_near_goal(
  "Mjlab-PushT-NearGoal-NatureCnn-Flatten-TrossenRealistic",
  "NatureCnn-Flatten",
)
_near_goal(
  "Mjlab-PushT-NearGoal-NatureCnn-SpatialSoftmax-TrossenRealistic",
  "NatureCnn-SpatialSoftmax",
)
_near_goal(
  "Mjlab-PushT-NearGoal-CompactVit-Flatten-TrossenRealistic",
  "CompactVit-Flatten",
)
_near_goal(
  "Mjlab-PushT-NearGoal-CompactVit-SpatialSoftmax-TrossenRealistic",
  "CompactVit-SpatialSoftmax",
)
_near_goal(
  "Mjlab-PushT-NearGoal-DinoV2ViTS14-Linear-TrossenRealistic",
  "DinoV2ViTS14-Linear",
)
_near_goal(
  "Mjlab-PushT-NearGoal-DinoV2ViTS14-LocalGrid16-TrossenRealistic",
  "DinoV2ViTS14-LocalGrid16",
)
_near_goal(
  "Mjlab-PushT-NearGoal-DinoV2ViTS14-SpatialSoftmax-TrossenRealistic",
  "DinoV2ViTS14-SpatialSoftmax",
)
_near_goal(
  "Mjlab-PushT-NearGoal-DinoV2ViTS14-Afa6-TrossenRealistic",
  "DinoV2ViTS14-Afa6",
)
_near_goal(
  "Mjlab-PushT-NearGoal-R3MResNet50-Linear-TrossenRealistic",
  "R3MResNet50-Linear",
)
_near_goal(
  "Mjlab-PushT-NearGoal-R3MResNet50-LocalGrid7-TrossenRealistic",
  "R3MResNet50-LocalGrid7",
)
_near_goal(
  "Mjlab-PushT-NearGoal-R3MResNet50-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50-SpatialSoftmax",
)
_near_goal(
  "Mjlab-PushT-NearGoal-R3MResNet50-Afa32-TrossenRealistic",
  "R3MResNet50-Afa32",
)
_near_goal(
  "Mjlab-PushT-NearGoal-R3MResNet50L3-LocalGrid14-TrossenRealistic",
  "R3MResNet50L3-LocalGrid14",
)
_near_goal(
  "Mjlab-PushT-NearGoal-R3MResNet50L3-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50L3-SpatialSoftmax",
)
_near_goal(
  "Mjlab-PushT-NearGoal-R3MResNet50L3-Afa16-TrossenRealistic",
  "R3MResNet50L3-Afa16",
)

# --- GrowStart: reverse curriculum, the separation cap grows to FreeStart ----

_grow_start(
  "Mjlab-PushT-GrowStart-NatureCnn-Flatten-TrossenRealistic",
  "NatureCnn-Flatten",
)
_grow_start(
  "Mjlab-PushT-GrowStart-NatureCnn-SpatialSoftmax-TrossenRealistic",
  "NatureCnn-SpatialSoftmax",
)
_grow_start(
  "Mjlab-PushT-GrowStart-CompactVit-Flatten-TrossenRealistic",
  "CompactVit-Flatten",
)
_grow_start(
  "Mjlab-PushT-GrowStart-CompactVit-SpatialSoftmax-TrossenRealistic",
  "CompactVit-SpatialSoftmax",
)
_grow_start(
  "Mjlab-PushT-GrowStart-DinoV2ViTS14-Linear-TrossenRealistic",
  "DinoV2ViTS14-Linear",
)
_grow_start(
  "Mjlab-PushT-GrowStart-DinoV2ViTS14-LocalGrid16-TrossenRealistic",
  "DinoV2ViTS14-LocalGrid16",
)
_grow_start(
  "Mjlab-PushT-GrowStart-DinoV2ViTS14-SpatialSoftmax-TrossenRealistic",
  "DinoV2ViTS14-SpatialSoftmax",
)
_grow_start(
  "Mjlab-PushT-GrowStart-DinoV2ViTS14-Afa6-TrossenRealistic",
  "DinoV2ViTS14-Afa6",
)
_grow_start(
  "Mjlab-PushT-GrowStart-R3MResNet50-Linear-TrossenRealistic",
  "R3MResNet50-Linear",
)
_grow_start(
  "Mjlab-PushT-GrowStart-R3MResNet50-LocalGrid7-TrossenRealistic",
  "R3MResNet50-LocalGrid7",
)
_grow_start(
  "Mjlab-PushT-GrowStart-R3MResNet50-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50-SpatialSoftmax",
)
_grow_start(
  "Mjlab-PushT-GrowStart-R3MResNet50-Afa32-TrossenRealistic",
  "R3MResNet50-Afa32",
)
_grow_start(
  "Mjlab-PushT-GrowStart-R3MResNet50L3-LocalGrid14-TrossenRealistic",
  "R3MResNet50L3-LocalGrid14",
)
_grow_start(
  "Mjlab-PushT-GrowStart-R3MResNet50L3-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50L3-SpatialSoftmax",
)
_grow_start(
  "Mjlab-PushT-GrowStart-R3MResNet50L3-Afa16-TrossenRealistic",
  "R3MResNet50L3-Afa16",
)
_slow_goal_free(
  "Mjlab-PushT-SlowFree-NatureCnn-Flatten-TrossenRealistic",
  "NatureCnn-Flatten",
)
_slow_goal_free(
  "Mjlab-PushT-SlowFree-NatureCnn-SpatialSoftmax-TrossenRealistic",
  "NatureCnn-SpatialSoftmax",
)
_slow_goal_free(
  "Mjlab-PushT-SlowFree-CompactVit-Flatten-TrossenRealistic",
  "CompactVit-Flatten",
)
_slow_goal_free(
  "Mjlab-PushT-SlowFree-CompactVit-SpatialSoftmax-TrossenRealistic",
  "CompactVit-SpatialSoftmax",
)
_slow_goal_free(
  "Mjlab-PushT-SlowFree-DinoV2ViTS14-Linear-TrossenRealistic",
  "DinoV2ViTS14-Linear",
)
_slow_goal_free(
  "Mjlab-PushT-SlowFree-DinoV2ViTS14-LocalGrid16-TrossenRealistic",
  "DinoV2ViTS14-LocalGrid16",
)
_slow_goal_free(
  "Mjlab-PushT-SlowFree-DinoV2ViTS14-SpatialSoftmax-TrossenRealistic",
  "DinoV2ViTS14-SpatialSoftmax",
)
_slow_goal_free(
  "Mjlab-PushT-SlowFree-DinoV2ViTS14-Afa6-TrossenRealistic",
  "DinoV2ViTS14-Afa6",
)
_slow_goal_free(
  "Mjlab-PushT-SlowFree-R3MResNet50-Linear-TrossenRealistic",
  "R3MResNet50-Linear",
)
_slow_goal_free(
  "Mjlab-PushT-SlowFree-R3MResNet50-LocalGrid7-TrossenRealistic",
  "R3MResNet50-LocalGrid7",
)
_slow_goal_free(
  "Mjlab-PushT-SlowFree-R3MResNet50-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50-SpatialSoftmax",
)
_slow_goal_free(
  "Mjlab-PushT-SlowFree-R3MResNet50-Afa32-TrossenRealistic",
  "R3MResNet50-Afa32",
)
_slow_goal_free(
  "Mjlab-PushT-SlowFree-R3MResNet50L3-LocalGrid14-TrossenRealistic",
  "R3MResNet50L3-LocalGrid14",
)
_slow_goal_free(
  "Mjlab-PushT-SlowFree-R3MResNet50L3-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50L3-SpatialSoftmax",
)
_slow_goal_free(
  "Mjlab-PushT-SlowFree-R3MResNet50L3-Afa16-TrossenRealistic",
  "R3MResNet50L3-Afa16",
)
_visual_goal_free(
  "Mjlab-PushT-VisualFree-NatureCnn-Flatten-TrossenRealistic",
  "NatureCnn-Flatten",
)
_visual_goal_free(
  "Mjlab-PushT-VisualFree-NatureCnn-SpatialSoftmax-TrossenRealistic",
  "NatureCnn-SpatialSoftmax",
)
_visual_goal_free(
  "Mjlab-PushT-VisualFree-CompactVit-Flatten-TrossenRealistic",
  "CompactVit-Flatten",
)
_visual_goal_free(
  "Mjlab-PushT-VisualFree-CompactVit-SpatialSoftmax-TrossenRealistic",
  "CompactVit-SpatialSoftmax",
)
_visual_goal_free(
  "Mjlab-PushT-VisualFree-DinoV2ViTS14-Linear-TrossenRealistic",
  "DinoV2ViTS14-Linear",
)
_visual_goal_free(
  "Mjlab-PushT-VisualFree-DinoV2ViTS14-LocalGrid16-TrossenRealistic",
  "DinoV2ViTS14-LocalGrid16",
)
_visual_goal_free(
  "Mjlab-PushT-VisualFree-DinoV2ViTS14-SpatialSoftmax-TrossenRealistic",
  "DinoV2ViTS14-SpatialSoftmax",
)
_visual_goal_free(
  "Mjlab-PushT-VisualFree-DinoV2ViTS14-Afa6-TrossenRealistic",
  "DinoV2ViTS14-Afa6",
)
_visual_goal_free(
  "Mjlab-PushT-VisualFree-R3MResNet50-Linear-TrossenRealistic",
  "R3MResNet50-Linear",
)
_visual_goal_free(
  "Mjlab-PushT-VisualFree-R3MResNet50-LocalGrid7-TrossenRealistic",
  "R3MResNet50-LocalGrid7",
)
_visual_goal_free(
  "Mjlab-PushT-VisualFree-R3MResNet50-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50-SpatialSoftmax",
)
_visual_goal_free(
  "Mjlab-PushT-VisualFree-R3MResNet50-Afa32-TrossenRealistic",
  "R3MResNet50-Afa32",
)
_visual_goal_free(
  "Mjlab-PushT-VisualFree-R3MResNet50L3-LocalGrid14-TrossenRealistic",
  "R3MResNet50L3-LocalGrid14",
)
_visual_goal_free(
  "Mjlab-PushT-VisualFree-R3MResNet50L3-SpatialSoftmax-TrossenRealistic",
  "R3MResNet50L3-SpatialSoftmax",
)
_visual_goal_free(
  "Mjlab-PushT-VisualFree-R3MResNet50L3-Afa16-TrossenRealistic",
  "R3MResNet50L3-Afa16",
)
