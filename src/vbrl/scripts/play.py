from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vbrl.runtime import AGENTS, CheckpointRef, build_env, default_device, make_policy
from vbrl.scenes.presets import ood_scenes


# MJLab bakes textures into the meshes it uploads and every hook that decides when to
# redo that watches colour only, so texture swaps never reach the browser.
def track_appearance_randomization() -> None:
  from mjlab.viewer.viser import scene

  textured = {"mat_texid", "geom_matid"}
  scene._VISER_APPEARANCE_HANDLE_FIELDS = frozenset(
    scene._VISER_APPEARANCE_HANDLE_FIELDS | textured
  )
  scene._VISER_BAKED_HANDLE_FIELDS = frozenset(
    scene._VISER_BAKED_HANDLE_FIELDS | textured
  )

  fingerprint = scene.MjlabViserScene._geom_subgroup_visual_fingerprint
  if getattr(fingerprint, "_vbrl_splits_by_texture", False):
    return

  def with_textures(mj_model: Any, geom_ids: Any, is_mocap: bool) -> tuple:
    textures = tuple(
      None if matid < 0 else tuple(mj_model.mat_texid[matid].tolist())
      for matid in (int(mj_model.geom_matid[geom]) for geom in geom_ids)
    )
    return (fingerprint(mj_model, geom_ids, is_mocap), textures)

  with_textures._vbrl_splits_by_texture = True  # type: ignore[attr-defined]
  scene.MjlabViserScene._geom_subgroup_visual_fingerprint = staticmethod(with_textures)


def run_viser(
  env: Any,
  policy: Any,
  *,
  host: str,
  port: int,
  frame_rate: float,
  max_steps: int | None,
) -> None:

  import viser
  from mjlab.viewer import ViserPlayViewer

  track_appearance_randomization()

  server = viser.ViserServer(
    host=host,
    port=port,
    label="vision-based-rl",
    verbose=False,
  )
  try:
    actual_port = int(server.get_port())
    if actual_port != port:
      raise RuntimeError(
        f"Requested Viser port {port}, but the server bound {actual_port}."
      )
    ViserPlayViewer(
      env,
      policy,
      frame_rate=frame_rate,
      viser_server=server,
    ).run(num_steps=max_steps)
  finally:
    server.stop()


def record_rollout(wrapped: Any, policy: Any, args: Any) -> Path:
  from vbrl.evaluation.recording import default_output, record
  from vbrl.paths import artifact_path

  output = artifact_path(args.record)
  (width, height), fps = default_output(output)
  return record(
    wrapped,
    policy,
    path=output,
    steps=args.record_steps,
    width=args.record_width or width,
    height=args.record_height or height,
    fps=args.record_fps or fps or round(1.0 / wrapped.unwrapped.step_dt),
  )


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Inspect a registered MJLab task or deploy a checkpoint."
  )
  parser.add_argument("task_id", help="Exact registered MJLab task ID.")
  parser.add_argument(
    "--agent",
    choices=AGENTS,
    default="trained",
  )
  parser.add_argument("--checkpoint-file", type=Path)
  parser.add_argument("--wandb-run-path")
  parser.add_argument("--wandb-checkpoint-name")
  parser.add_argument(
    "--scene",
    choices=ood_scenes(),
    help="Optional OOD texture replacement for the registered play scene.",
  )
  parser.add_argument("--eval-dr", choices=("fixed", "matched"))
  parser.add_argument("--device")
  parser.add_argument("--num-envs", type=int, default=1)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--no-terminations", action="store_true")
  parser.add_argument("--host", default="0.0.0.0")
  parser.add_argument("--port", type=int, default=8080)
  parser.add_argument("--frame-rate", type=float, default=60.0)
  parser.add_argument(
    "--record",
    type=Path,
    help="Write a video (.mp4) or GIF below artifacts/ instead of serving "
    "Viser, framed on every env at once.",
  )
  parser.add_argument("--record-steps", type=int, default=300)
  parser.add_argument("--record-width", type=int)
  parser.add_argument("--record-height", type=int)
  parser.add_argument("--record-fps", type=int)
  parser.add_argument(
    "--max-steps",
    type=int,
    help="Exit cleanly after this many policy steps (for smoke tests).",
  )
  parser.add_argument(
    "--goal",
    type=float,
    nargs=3,
    metavar=("X", "Y", "YAW"),
    help="Pin the Push-T goal to one pose in the base frame, yaw in radians. "
    "The object still randomizes and the separation floor still applies, so "
    "this is the deployment goal against a fresh start every episode.",
  )
  return parser


def _validate(
  parser: argparse.ArgumentParser,
  args: argparse.Namespace,
) -> CheckpointRef:
  if args.num_envs <= 0:
    parser.error("--num-envs must be positive")
  if not 1 <= args.port <= 65535:
    parser.error("--port must be between 1 and 65535")
  if args.frame_rate <= 0:
    parser.error("--frame-rate must be positive")
  if args.max_steps is not None and args.max_steps <= 0:
    parser.error("--max-steps must be positive")
  if args.eval_dr is not None and args.scene is None:
    parser.error("--eval-dr requires --scene")
  overrides = ("record_steps", "record_width", "record_height", "record_fps")
  if args.record is None:
    named = [
      f"--{name.replace('_', '-')}"
      for name in overrides
      if getattr(args, name) != _parser().get_default(name)
    ]
    if named:
      parser.error(f"{', '.join(named)} require --record")
  for name in overrides:
    value = getattr(args, name)
    if value is not None and value <= 0:
      parser.error(f"--{name.replace('_', '-')} must be positive")
  ref = CheckpointRef.from_args(args)
  try:
    if args.agent == "trained":
      ref.validate(prefix="--agent trained: ")
    elif not ref.is_empty:
      raise ValueError("Checkpoint fields require --agent trained")
  except ValueError as exc:
    parser.error(str(exc))
  return ref


# Not degenerate target ranges: the goal is drawn on a ring and rejected
# outside the rectangle, so a rectangle collapsed to a point never converges.
def _pin_goal(
  parser: argparse.ArgumentParser, env: Any, goal: Sequence[float]
) -> None:
  import torch
  from mjlab.utils.lab_api.math import quat_from_euler_xyz

  from vbrl.tasks.push_t.geometry import HALF_HEIGHT

  command = (env.unwrapped.command_manager._terms or {}).get("push_t_goal")
  if command is None:
    parser.error("--goal needs a Push-T task; this one has no push_t_goal command.")
  x, y, yaw = goal
  origins = env.unwrapped.scene.env_origins
  resample = command._resample_command

  def pinned(env_ids: Any) -> None:
    resample(env_ids)
    offset = origins.new_tensor([x, y, HALF_HEIGHT])
    command.target_pos[env_ids] = origins[env_ids] + offset
    command.target_yaw[env_ids] = yaw
    marker = command._goal_marker
    if marker is not None:
      pos = command.target_pos[env_ids].clone()
      pos[:, 2] = origins[env_ids, 2]
      zeros = torch.zeros(len(env_ids), device=command.device)
      marker.write_mocap_pose_to_sim(
        torch.cat((pos, quat_from_euler_xyz(zeros, zeros, command.target_yaw[env_ids])), dim=-1),
        env_ids=env_ids,
      )

  command._resample_command = pinned
  print(f"[INFO] Goal pinned to x={x:.5f} y={y:.5f} yaw={yaw:.5f} rad")


def main(argv: Sequence[str] | None = None) -> int:
  parser = _parser()
  args = parser.parse_args(argv)
  ref = _validate(parser, args)

  try:
    device = args.device or default_device()
    env = build_env(
      args.task_id,
      device=device,
      num_envs=args.num_envs,
      seed=args.seed,
      scene=args.scene,
      eval_dr=args.eval_dr or "fixed",
      drop_terminations=args.no_terminations,
      # One shot of many envs is lit by whichever env the recorder makes primary, so a
      # randomized sun colour would tint every recording differently.
      fixed_lighting=args.record is not None,
    )
    if args.goal is not None:
      _pin_goal(parser, env, args.goal)
    try:
      wrapped, _, policy, _ = make_policy(
        env,
        task_id=args.task_id,
        agent=args.agent,
        ref=ref,
        device=device,
      )
      if args.record is not None:
        print(f"[INFO] Wrote {record_rollout(wrapped, policy, args)}")
      else:
        run_viser(
          wrapped,
          policy,
          host=args.host,
          port=args.port,
          frame_rate=args.frame_rate,
          max_steps=args.max_steps,
        )
    finally:
      env.close()
  except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
    parser.error(str(exc))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
