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
  not expose it. The registered floor is 0.15; ManiSkill's PPO has none, and
  every run so far sat pinned at exactly the floor.
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
  orientation_reward: Literal["maniskill", "quadratic", "linear"] | None = None
  """Swap the dense reward's orientation factor.

  ``maniskill`` is ``((cos e + 1) / 2)**2``, flat at 0 and at pi. ``quadratic``
  is ``1 - (|e| / pi)**2``, flat at 0 only. ``linear`` is ``1 - |e| / pi``, flat
  nowhere. A flag rather than a variant until one earns a task ID; the registered
  choice stands when this is unset.
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
  """Share of the shaped reward scoring orientation at the start of training."""
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
  }[cfg.orientation_reward]


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
    low, high = distribution["std_range"]
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
