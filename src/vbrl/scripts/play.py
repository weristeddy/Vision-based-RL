"""Inspect a registered task or deploy one checkpoint in Viser."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from vbrl.runtime import AGENTS, CheckpointRef, build_env, default_device, make_policy
from vbrl.scenes.presets import ood_scenes


def track_appearance_randomization() -> None:
  """Let Viser follow appearance swaps in time and across envs, not just colour.

  MJLab bakes the texture into the meshes it uploads to the browser, and two
  places decide when that happens. Both compare colour and ignore textures, so
  the photographic table banks -- which vary ``mat_texid`` -- and the procedural
  banks -- which vary ``geom_matid`` -- are invisible to the viewer while the
  camera observation, rendered from the per-world model, has them:

  * ``_VISER_APPEARANCE_HANDLE_FIELDS`` decides when a frame's appearance is
    stale enough to rebuild, so without ``mat_texid`` the view keeps whichever
    texture was current when the viewer was built, for the whole session.
  * ``_VISER_BAKED_HANDLE_FIELDS`` decides whether MJLab's per-env variant path
    runs at all. It is a module constant composed at import time, so widening
    the appearance set does not reach it; left alone, the scene falls through
    to mjviser's plain path, which uploads one mesh per body and instances it
    across every env.
  * ``_geom_subgroup_visual_fingerprint`` then decides how many variants that
    path uploads, and envs sharing a fingerprint share one instanced mesh. Two
    envs whose tables differ only by texture collapse into one variant, so N
    envs render the same tabletop no matter how the model was randomized.

  Textures are appended to the fingerprint rather than replacing it, so
  upstream's own fields keep counting. All three patches are process-global and
  idempotent.

  This costs one uploaded mesh per distinct texture, so a Viser session showing
  many envs pays for many tabletops. That is the intended trade for a handful
  of envs; it is not a reason to point the viewer at a training-sized batch.

  Raises:
    AttributeError: if MJLab renames either hook, so an upgrade surfaces here
      rather than silently restoring the collapsed view.
  """
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
  """Run MJLab's viewer on an explicitly bound Viser server."""

  import viser
  from mjlab.viewer import ViserPlayViewer

  # Before the viewer builds its scene: the widened set is read in
  # MjlabViserScene.__init__.
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
  """Write one framed shot of every env, at the size the container deserves."""

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
    # A video keeps the simulated rate, so playback runs at wall-clock speed.
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
      # One shot of many envs is lit by whichever env the recorder makes
      # primary, so a randomized sun colour would tint every recording
      # differently.
      fixed_lighting=args.record is not None,
    )
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
