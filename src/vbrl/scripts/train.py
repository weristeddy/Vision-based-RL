"""Train a registered VBRL task with RSL-RL.

Vendored from ``mjlab.scripts.train`` (mjlab 1.6.0) with two changes: VBRL's
task package is imported so its IDs join the registry, and the worker
environment VBRL needs is added to ``copy_env_vars``. MJLab's motion-tracking
branch is omitted because no VBRL task uses a motion command. Re-diff against
upstream when bumping mjlab.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder

# TorchrunX starts each worker from a bare environment. Beyond MuJoCo's own
# variables these carry the model root, W&B credentials, offline-cache flags,
# and the TLS trust store the compute nodes need.
WORKER_ENV = (
  "MUJOCO*",
  "VBRL*",
  "WANDB*",
  "WARP*",
  "SSL*",
  "REQUESTS*",
  "CURL*",
  "HF*",
  "TRANSFORMERS*",
  "MPLBACKEND",
  "PYOPENGL*",
  "SLURM*",
)


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  log_root: str = "logs/rsl_rl"
  """Root directory under which experiment logs are written."""
  torchrunx_log_dir: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Checkpoint to resume from within the W&B run (e.g. 'model_4000.pt')."""
  label: str = ""
  """Extra W&B tag and run-name prefix, for telling one sweep arm from another.

  Hyperparameters are deliberately not part of the task ID, so two arms that
  differ only in `gamma` register as the same task and would otherwise be
  indistinguishable in W&B except by reading their configs. This travels as a
  command-line argument, which is the only channel that reaches a TorchrunX
  worker through Slurm and Apptainer without extra plumbing.
  """
  min_action_std: float | None = None
  """Override the lower bound of the policy's action-std range.

  `std_range` lives inside `actor.distribution_cfg`, a plain dict, so Tyro does
  not expose it. The RGB actors declare `(0.15, 1.0)` and every vision run so far
  sat pinned at exactly 0.15, so the floor binds. The state actor declares no
  range at all and settles around 0.11 on its own; passing this there installs a
  floor rather than lowering one, which is how the two can be compared at a
  matched action std.

  Note this shapes *training* only. Rollouts use `get_inference_policy`, which is
  the deterministic mean -- measured identical across repeated calls -- so this
  cannot change execution-time jitter, only what the policy learns under noise.
  """
  goal_yaw_pin_iterations: int | None = None
  """Iterations the goal yaw stays pinned before the range starts widening."""
  goal_yaw_rungs: int | None = None
  """How many rungs the goal-yaw range widens in. ``1`` jumps to the full circle.

  Widening a *uniform* range dilutes each new band the moment it appears, so a
  band that arrives late is both late and rare: under the registered eight-rung
  schedule the 0-22.5 degree band gets five times the training of the 157.5-180
  one, and past 112.5 degrees is under 20% of all goals. Measured failures
  concentrate there (mean goal yaw 135 degrees for failed episodes against 78
  for solved), while initial object pose and initial yaw error separate them not
  at all.
  """
  goal_yaw_rung_iterations: int | None = None
  """Iterations between rungs. Ignored when there is only one."""
  separation_pin_iterations: int | None = None
  """Iterations the object-goal separation cap stays at its floor before growing.

  The registered reverse curriculum starts widening from the first iteration, so
  the easiest distances are gone within a few hundred. Pinning holds them, on the
  theory that a policy needs uninterrupted time at short range to acquire the
  skill before the range moves. Measured against that theory: `GrowStart` was at
  a 9 cm cap by iteration 400 and 13 cm by 800 with yaw error flat at chance
  throughout, while the goal-yaw curriculum was already at 0.892 rad by 400 --
  so there was no emerging trend for a pin to protect.
  """
  separation_ramp_iterations: int | None = None
  """Iterations the cap takes to grow from the floor to the full range."""
  separation_start_cm: float | None = None
  """Separation cap the ramp starts from, in centimetres.

  The registered 5 cm is too wide for the mechanism it was built for. Holding
  the goal close is meant to satisfy the position term so orientation is the
  only reward left, but at 5 cm position still offers 0.215 of headroom against
  orientation's 0.312 -- a ratio of 1.5x, and the finished run learned no
  orientation at any cap. The ratio reaches 4.3x at 1.5 cm and 6.4x at 1 cm,
  which is where "only rotation pays" is actually true.
  """
  action_path_weight: float | None = None
  """Weight of the L1 commanded-travel penalty, which is what bounds how far the
  arm moves in total.

  The registered -0.002 is decorative: measured on a solved state policy it costs
  0.21 against a task reward of 14.0, or 1.5%, while that policy travels 1.75 m
  of end-effector path per episode to push the T about 0.2 m and keeps the arm
  moving at 5.5 mm per step through the 71% of steps where the object does not
  move at all. Scaling that share, -0.01 costs about 7.7% and -0.02 about 15%.

  Unlike the top-contact penalty this is not believed to remove a capability --
  excess path is waste rather than function -- but that is an argument, not a
  measurement, and the vertical sweep is a warning that 6% was enough to break
  things there. Must be <= 0.
  """
  action_rate_weight: float | None = None
  """Weight of MJLab's `action_rate_l2`, for oscillation rather than total path.

  Registered at -0.002, a fifth of upstream Lift-Cube's -0.01, because the
  quadratic form charges pure exploration noise as sigma^2 and so bites hardest
  at initialization. Must be <= 0.
  """
  vertical_contact_weight: float | None = None
  """Weight of the top-face contact penalty, which discourages pressing down on
  the object and dragging it instead of pushing a side face.

  A flag rather than a variant because the workable range is narrow. Measured on
  `Episode_Reward` totals, which is the only scale the shares below are in:

  * At the registered -0.05 a solved state policy pays 0.164 against a task
    reward of 14.0 -- 1.2% -- and still puts ~80% of its contacts on the top
    face. Too weak to change the behaviour it exists to change.
  * Scaling that share, -0.25 costs about 5.9% and -0.75 about 17.6%.
  * Both weights collapsed a run to never touching the object, but those were at
    1024 environments where exploration died before transport was learned and the
    policy earned only 6.8; the collapse set in once the penalty reached ~3% of
    that. At 4096 environments the same policy solves the task and earns 14.0, so
    there is roughly twice the headroom -- not more.

  Must be <= 0.
  """
  orientation_reward: (
    Literal["maniskill", "quadratic", "linear", "keypoint"] | None
  ) = None
  """Swap the dense reward's orientation factor.

  ``maniskill`` is ``((cos e + 1) / 2)**2``, flat at 0 and at pi. ``quadratic``
  is ``1 - (|e| / pi)**2``, flat at 0 only. ``linear`` is ``1 - |e| / pi``, flat
  nowhere. ``keypoint`` is the odd one out: it replaces the weighted
  position/orientation split entirely with four tracked points on the T, so
  ``--orientation-weight`` stops having any effect. A flag rather than a variant
  until one earns a task ID; the registered choice stands when this is unset.
  """
  min_xy_separation_cm: float | None = None
  """Closest the goal may be drawn to the object, in centimetres.

  The registered 15 cm is what separates every variant that learned to orient
  the T from the three that did not. Its stated purpose was to keep the sparse
  bonus out of reach early, but it also guarantees every episode begins with a
  transport to perform -- and transport is what makes contact profitable, which
  is what keeps a policy touching the object long enough to discover rotation.
  Dropped to 1 cm, 28% of episodes start inside 5 cm, and the measured result is
  a policy that stops touching the object at all.

  Exposed so the floor can be varied on its own rather than bundled into a task
  ID, since it is the one number in the start-state geometry that has ever
  changed an outcome: shared versus offset x windows moves the mean separation
  by 0.7 cm, this moves it from 22 cm to 11 cm.
  """
  near_goal_separation_cm: tuple[float, float] | None = None
  """Separation band for near-goal episodes, in centimetres.

  The registered ``0.6`` to ``1.5`` was sized around the sparse bonus: at 6 mm
  and perfect alignment overlap is 0.891, one correction short of the 0.90
  threshold. Widening it trades that for position headroom in the *dense* term,
  which is what matters when every episode is near-goal -- the position factor
  sits at 0.941 at 6 mm with only 0.059 left to gain, against 0.903 at 1 cm and
  0.811 at 2 cm. Note what the wider band costs: overlap at perfect alignment is
  0.810 at 1 cm and 0.620 at 2 cm, so the bonus needs real transport rather than
  one nudge, and at 2 cm overlap is flat in yaw below 20 degrees (0.620 at 0
  degrees, 0.633 at 10) so the bonus carries no orientation signal there.
  """
  near_goal_probability: float | None = None
  """Fraction of episodes started a few millimetres from the goal."""
  near_goal_yaw_spread_deg: float | None = None
  """Largest |yaw error| those episodes may start with, in degrees.

  ``180`` is the uniform draw. Deliberately a value rather than a boolean: tyro
  spells a ``bool`` field differently across versions -- the dev environment
  takes a bare ``--flag`` and rejects ``--flag True``, while the container wants
  the value and rejects the bare form -- so a switch here is a submission that
  dies at argument parsing on one side or the other.

  The registered mixture starts them within 5-20 degrees as well as 6-15 mm,
  which leaves them nearly solved: a policy collects that reward without
  rotating anything, and 14 of 15 architectures settled at exactly the
  do-nothing score. Widening the spread keeps the position term 86-94% satisfied
  while leaving orientation 4-10x the remaining reward, so the episode is a
  rotation problem rather than a gift -- and at 180 the do-nothing baseline
  returns to pi/2 for every episode, which makes the yaw metric comparable
  across variants again.
  """
  goal_yaw_levels: int | None = None
  """Quantize the goal yaw to this many angles, doubling each rung.

  A curriculum on the goal's resolution rather than its range: the goal spans
  the full circle throughout and the object's yaw stays uniform, so unlike every
  distance curriculum here it never starts the object near-optimal.
  """
  goal_yaw_levels_rungs: int = 3
  """Rungs before the goal yaw returns to a continuum."""
  goal_yaw_levels_iterations: int = 1500
  """Iterations per rung."""
  success_threshold_start: float | None = None
  """Overlap needed for the sparse bonus at the start of training."""
  success_threshold_iterations: int = 4000
  """Iterations over which it tightens to the registered threshold."""
  orientation_weight: float | None = None
  """Constant share of the shaped reward that scores orientation.

  ``0.5`` is ManiSkill's split. Below roughly 0.58 a push that correctly rotates
  the T is *punished* once the object is near the goal: the position factor's
  gradient peaks there, so its loss from the 1 mm of displacement that rotation
  costs outweighs the orientation gain. Measured at -0.00127 per step at 2 cm
  under the 0.5 split, against +0.00331 at 0.8.

  That is the endgame every episode ends in, so it is not a corner case -- and
  it gets worse the *faster* transport is learned, because the policy arrives in
  the punished regime sooner. Prefer this over
  ``--orientation-weight-start``, which ramps back down to the registered 0.5
  and therefore finishes inside it.
  """
  orientation_weight_start: float | None = None
  """Share of the shaped reward scoring orientation at the start of training.

  ``0.0`` is a position-only reward, which with
  ``--orientation-weight-pin-iterations`` and a raised ``--orientation-weight``
  gives the two-stage schedule: transport first, rotation added afterwards.
  Prefer that over resuming a finished run with a changed weight. Measured on
  the two w=0.8 resumes: the reward changed discontinuously under a critic
  trained on the old one, and the action std ran from 0.250 to the 1.0 ceiling
  by iteration 7000 and stayed pinned, entropy saturating at 8.51 -- overlap
  fell 0.441 to 0.201 while yaw error never moved off chance. The policy was
  re-randomised rather than finetuned. A ramp inside one run has no such
  discontinuity for the critic to lag behind.
  """
  orientation_weight_pin_iterations: int = 0
  """Iterations to hold ``--orientation-weight-start`` before the ramp begins."""
  orientation_weight_iterations: int = 4000
  """Iterations over which it returns to the registered 0.5 split."""
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> TrainConfig:
    return TrainConfig(env=load_env_cfg(task_id), agent=load_rl_cfg(task_id))


def _swap_orientation_reward(cfg: TrainConfig) -> None:
  """Point the dense reward term at a different orientation factor."""
  from vbrl.tasks.push_t import mdp

  if cfg.orientation_reward is None:
    return
  term = (cfg.env.rewards or {}).get("maniskill_dense")
  if term is None:
    raise ValueError("This task has no `maniskill_dense` reward term to swap.")
  term.func = {
    "maniskill": mdp.maniskill_dense_reward,
    "quadratic": mdp.quadratic_orientation_reward,
    "linear": mdp.linear_orientation_reward,
    "keypoint": mdp.keypoint_reward,
  }[cfg.orientation_reward]


def _retune_penalty_weights(cfg: TrainConfig) -> None:
  """Set the weights of the motion and contact penalties from the command line."""
  # (reward term, the flag that sets it) -- the flag name is what a user types,
  # so errors have to quote that rather than the term it happens to write to.
  overrides = (
    ("vertical_contact_force", "--vertical-contact-weight", cfg.vertical_contact_weight),
    ("action_path_length", "--action-path-weight", cfg.action_path_weight),
    ("action_rate_l2", "--action-rate-weight", cfg.action_rate_weight),
  )
  for name, flag, weight in overrides:
    if weight is None:
      continue
    if weight > 0.0:
      raise ValueError(f"{flag} must be <= 0; got {weight}.")
    term = (cfg.env.rewards or {}).get(name)
    if term is None:
      raise ValueError(f"This task has no `{name}` reward term.")
    term.weight = weight


def _install_goal_curricula(cfg: TrainConfig) -> None:
  """Add the opt-in goal-space curricula, each writing to the command config.

  These are not registered on any task ID: they are hypotheses about *why* the
  goal-yaw curriculum works, not settled configurations, so they live behind
  flags until one of them earns a variant.
  """
  from mjlab.managers import CurriculumTermCfg

  from vbrl.tasks.push_t import mdp

  command = (cfg.env.commands or {}).get("push_t_goal")
  requested = (
    cfg.goal_yaw_levels,
    cfg.success_threshold_start,
    cfg.orientation_weight_start,
  )
  if all(value is None for value in requested):
    return
  if command is None:
    raise ValueError("This task has no Push-T goal command to curriculum.")
  per_iteration = cfg.agent.num_steps_per_env

  if cfg.goal_yaw_levels is not None:
    cfg.env.curriculum["goal_yaw_resolution"] = CurriculumTermCfg(
      func=mdp.goal_yaw_resolution_curriculum,
      params={
        "command_name": "push_t_goal",
        "start_levels": cfg.goal_yaw_levels,
        "rungs": cfg.goal_yaw_levels_rungs,
        "rung_iterations": cfg.goal_yaw_levels_iterations,
        "steps_per_iteration": per_iteration,
      },
    )
  if cfg.success_threshold_start is not None:
    cfg.env.curriculum["success_threshold"] = CurriculumTermCfg(
      func=mdp.success_threshold_curriculum,
      params={
        "command_name": "push_t_goal",
        "start": cfg.success_threshold_start,
        "end": command.success_threshold,
        "iterations": cfg.success_threshold_iterations,
        "steps_per_iteration": per_iteration,
      },
    )
  if cfg.orientation_weight_start is not None:
    cfg.env.curriculum["orientation_weight"] = CurriculumTermCfg(
      func=mdp.orientation_weight_curriculum,
      params={
        "command_name": "push_t_goal",
        "start": cfg.orientation_weight_start,
        "end": command.orientation_weight,
        "iterations": cfg.orientation_weight_iterations,
        "steps_per_iteration": per_iteration,
        "pin_iterations": cfg.orientation_weight_pin_iterations,
      },
    )


def _retune_near_goal_mixture(cfg: TrainConfig) -> None:
  """Re-weight the near-goal mixture, and optionally free its orientation.

  ``near_goal_yaw_range`` is read as a magnitude with a random sign, so a range
  of ``(0, pi)`` *is* the uniform draw -- no separate switch is needed in the
  sampler.
  """
  import math

  requested = (
    cfg.near_goal_probability,
    cfg.near_goal_separation_cm,
    cfg.near_goal_yaw_spread_deg,
    cfg.min_xy_separation_cm,
    cfg.orientation_weight,
  )
  if all(value is None for value in requested):
    return
  command = (cfg.env.commands or {}).get("push_t_goal")
  if command is None:
    raise ValueError("This task has no Push-T goal command to re-tune.")
  if cfg.orientation_weight is not None:
    if not 0.0 < cfg.orientation_weight < 1.0:
      raise ValueError(
        f"--orientation-weight must lie in (0, 1); got {cfg.orientation_weight}."
      )
    command.orientation_weight = cfg.orientation_weight
  if cfg.min_xy_separation_cm is not None:
    if not 0.0 < cfg.min_xy_separation_cm <= 40.0:
      raise ValueError(
        f"--min-xy-separation-cm must lie in (0, 40]; got "
        f"{cfg.min_xy_separation_cm}."
      )
    command.min_xy_separation = cfg.min_xy_separation_cm / 100.0
  if cfg.near_goal_separation_cm is not None:
    low, high = cfg.near_goal_separation_cm
    if not 0.0 < low <= high <= 40.0:
      raise ValueError(
        f"--near-goal-separation-cm needs 0 < low <= high <= 40; got "
        f"{cfg.near_goal_separation_cm}."
      )
    command.near_goal_separation_range = (low / 100.0, high / 100.0)
  if cfg.near_goal_probability is not None:
    if not 0.0 <= cfg.near_goal_probability <= 1.0:
      raise ValueError(
        f"--near-goal-probability must lie in [0, 1]; got "
        f"{cfg.near_goal_probability}."
      )
    command.near_goal_probability = cfg.near_goal_probability
  if cfg.near_goal_yaw_spread_deg is not None:
    if not 0.0 < cfg.near_goal_yaw_spread_deg <= 180.0:
      raise ValueError(
        f"--near-goal-yaw-spread-deg must lie in (0, 180]; got "
        f"{cfg.near_goal_yaw_spread_deg}."
      )
    command.near_goal_yaw_range = (
      0.0,
      math.radians(cfg.near_goal_yaw_spread_deg),
    )


def _retime_separation_curriculum(cfg: TrainConfig) -> None:
  """Point the separation ramp at the real rollout length, and optionally pin it.

  The term counts environment steps and converts from iterations with
  ``steps_per_iteration``, which the registration can only fill with a literal.
  Left alone, a run at any other ``num_steps_per_env`` burns through the ramp at
  the wrong rate -- the same trap the goal-yaw stages have, except nothing here
  is derived, so it is silent. Rewriting it from the agent config is a fix, not
  an option, which is why it happens whether or not the flags are given.
  """
  term = (cfg.env.curriculum or {}).get("separation_range")
  if term is None:
    if (
      cfg.separation_pin_iterations is not None
      or cfg.separation_ramp_iterations is not None
    ):
      raise ValueError(
        "This task has no separation curriculum, so --separation-* cannot "
        "apply. The GrowStart variant has one; SlowGoal and FreeStart do not."
      )
    return

  term.params["steps_per_iteration"] = cfg.agent.num_steps_per_env
  if cfg.separation_start_cm is not None:
    start = cfg.separation_start_cm / 100.0
    if not 0.0 < start <= float(term.params["end"]):
      raise ValueError(
        f"--separation-start-cm must be positive and no larger than the full "
        f"range ({float(term.params['end']) * 100:.1f} cm); got "
        f"{cfg.separation_start_cm}."
      )
    term.params["start"] = start
  if cfg.separation_pin_iterations is not None:
    term.params["pin_iterations"] = cfg.separation_pin_iterations
  if cfg.separation_ramp_iterations is not None:
    term.params["iterations"] = cfg.separation_ramp_iterations


def _rebuild_goal_yaw_stages(cfg: TrainConfig) -> None:
  """Re-time the goal-yaw curriculum from the command line.

  The stage list is a plain list of dicts inside a curriculum term's ``params``,
  so Tyro does not expose it. Each unset option keeps whatever the registered
  schedule implies, which is what lets one knob be varied at a time. Stage steps
  count environment steps, so an iteration is ``num_steps_per_env`` of them.
  """
  requested = (
    cfg.goal_yaw_pin_iterations,
    cfg.goal_yaw_rungs,
    cfg.goal_yaw_rung_iterations,
  )
  if all(value is None for value in requested):
    return

  term = (cfg.env.curriculum or {}).get("goal_yaw_range")
  if term is None:
    raise ValueError(
      "This task has no goal-yaw curriculum, so --goal-yaw-* cannot apply. "
      "The SlowGoal and Curriculum variants have one; Uniform does not."
    )

  import math

  per_iteration = cfg.agent.num_steps_per_env
  registered = list(term.params["stages"])
  pin = cfg.goal_yaw_pin_iterations
  if pin is None:
    pin = registered[1]["step"] // per_iteration if len(registered) > 1 else 0
  rungs = cfg.goal_yaw_rungs
  if rungs is None:
    rungs = max(1, len(registered) - 1)
  gap = cfg.goal_yaw_rung_iterations
  if gap is None:
    gap = (
      (registered[2]["step"] - registered[1]["step"]) // per_iteration
      if len(registered) > 2
      else 0
    )

  if pin < 0 or rungs < 1 or gap < 0:
    raise ValueError(
      f"Need pin >= 0, rungs >= 1, gap >= 0; got {pin}, {rungs}, {gap}."
    )

  stages = [{"step": 0, "half_range": 0.0}]
  for rung in range(1, rungs + 1):
    stages.append(
      {
        "step": (pin + (rung - 1) * gap) * per_iteration,
        "half_range": math.pi * rung / rungs,
      }
    )
  term.params["stages"] = stages
  full = stages[-1]["step"] // per_iteration
  print(
    f"[INFO] Goal-yaw curriculum: pinned for {pin} iterations, then {rungs} "
    f"rung(s) every {gap}; full circle at iteration {full} of "
    f"{cfg.agent.max_iterations}."
  )


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    seed = cfg.agent.seed + rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed
  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")
  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  _rebuild_goal_yaw_stages(cfg)
  _retime_separation_curriculum(cfg)
  _retune_near_goal_mixture(cfg)
  _install_goal_curricula(cfg)
  _swap_orientation_reward(cfg)
  _retune_penalty_weights(cfg)

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )
  log_root_path = log_dir.parent

  resume_path: Path | None = None
  if cfg.agent.resume:
    if cfg.wandb_run_path is not None:
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      if rank == 0:
        cached = "cached" if was_cached else "downloaded"
        print(
          f"[INFO]: Loading checkpoint from W&B: {resume_path.name} "
          f"(run: {resume_path.parent.name}, {cached})"
        )
    else:
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  # Only rank 0 records, so parallel workers cannot write the same video files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  # Dump before constructing the runner: the runner mutates agent_cfg in place
  # and injects objects that will not serialize.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  if cfg.min_action_std is not None:
    distribution = agent_cfg["actor"]["distribution_cfg"]
    # The RGB actors declare a range; the state actor does not, so a floor there
    # has to be installed rather than edited. Keep RSL-RL's own open upper bound
    # in that case: substituting 1.0 -- the state actor's `init_std` -- put a
    # ceiling exactly where std starts, and run u4uhvy5k then held std at 0.858
    # for 500 iterations where the unbounded control decayed to 0.109.
    low, high = distribution.get("std_range") or (None, 1e6)
    distribution["std_range"] = (cfg.min_action_std, high)
    print(f"[INFO] Action std range {low} -> {cfg.min_action_std} (upper bound {high}).")

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, agent_cfg, str(log_dir), device)

  add_wandb_tags((*cfg.agent.wandb_tags, cfg.label) if cfg.label else cfg.agent.wandb_tags)
  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )
  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None) -> None:
  args = args or TrainConfig.from_task(task_id)

  # Name the run directory once, before any worker starts, so every rank logs
  # into the same place.
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.label:
    log_dir_name += f"_{args.label}"
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = (Path(args.log_root) / args.agent.experiment_name).resolve() / log_dir_name

  selected_gpus, num_gpus = select_gpus(args.gpu_ids)
  os.environ["CUDA_VISIBLE_DEVICES"] = (
    "" if selected_gpus is None else ",".join(map(str, selected_gpus))
  )
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    run_train(task_id, args, log_dir)
    return

  import torchrunx

  # TorchrunX redirects worker stdout into logging.
  logging.basicConfig(level=logging.INFO)
  if "TORCHRUNX_LOG_DIR" not in os.environ:
    os.environ["TORCHRUNX_LOG_DIR"] = (
      args.torchrunx_log_dir
      if args.torchrunx_log_dir is not None
      else str(log_dir / "torchrunx")
    )

  print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
  torchrunx.Launcher(
    hostnames=["localhost"],
    workers_per_host=num_gpus,
    backend=None,  # rsl_rl initializes the process group itself.
    copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + WORKER_ENV,
  ).run(run_train, task_id, args, log_dir)


def _print_task_overview() -> None:
  """Answer a bare ``--help``, which lands before tyro knows the task."""
  from vbrl.tasks import vbrl_task_ids

  print("usage: vbrl-train <TASK_ID> [OPTIONS]\n")
  print("The task ID fixes the task, robot, scene, camera, and architecture.")
  print("Everything else -- seed, learning rate, iterations, env count, video")
  print("-- is a flag. Run 'vbrl-train <TASK_ID> --help' to see them all.\n")
  print("Registered VBRL tasks:")
  for task_id in vbrl_task_ids():
    print(f"  {task_id}")
  print("\nMJLab's own tasks are selectable here too; see 'list-envs'.")


def main() -> None:
  import mjlab
  import mjlab.tasks  # noqa: F401

  import vbrl.tasks  # noqa: F401

  if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    _print_task_overview()
    return

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=f"{sys.argv[0]} {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()


__all__ = ["WORKER_ENV", "TrainConfig", "launch_training", "main", "run_train"]
