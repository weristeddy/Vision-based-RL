from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from vbrl.asset_zoo.objects import PUSH_CHAIN_XML

SCENES = {"real_table": "real_table", "wood": "real_texture_wood"}
POSITION_STD = 0.03
ANGLE_STD = math.radians(30.0)
SUCCESS_THRESHOLD = 0.9
CHAIN_COLOUR, GOAL_COLOUR = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def make_env(scene: str, chain_xml: Path, num_envs: int, fovy: float, seed: int) -> Any:
  import dataclasses

  from mjlab.envs import ManagerBasedRlEnv

  from vbrl.analysis.push_t_measurement import HEIGHT, WIDTH
  from vbrl.deployment.push_chain.silhouette import num_links
  from vbrl.tasks.push_chain.config.trossen_realistic.env_cfgs import (
    trossen_realistic_push_chain_rgb_env_cfg,
  )

  cfg = trossen_realistic_push_chain_rgb_env_cfg(
    scene=SCENES[scene],
    num_links=num_links(chain_xml),
    object_xml=chain_xml,
    play=True,
  )
  cfg.seed = seed
  cfg.scene.num_envs = num_envs
  for name in ("camera_position", "camera_orientation"):
    cfg.events.pop(name)
  # The rig has one chain and one tape: their measured colour, not the training spread.
  colour = cfg.events["object_color"].params
  colour["ranges"] = {
    axis: ((lo + hi) / 2,) * 2 for axis, (lo, hi) in colour["ranges"].items()
  }
  tape = cfg.events["goal_marker_color"].params
  tape["ranges"] = tuple(((lo + hi) / 2,) * 2 for lo, hi in tape["ranges"])
  # The rig's room light casts almost no shadow.
  cfg.scene.sensors = tuple(
    dataclasses.replace(s, width=WIDTH, height=HEIGHT, fovy=fovy, use_shadows=False)
    if s.name == "external_cam"
    else s
    for s in cfg.scene.sensors
  )
  return ManagerBasedRlEnv(cfg=cfg, device="cuda:0")


def render(env: Any, chain: Any, goal: Any) -> Any:
  import torch

  command = env.command_manager.get_term("push_chain_goal")
  ids = torch.arange(env.num_envs, device=env.device)
  origins = env.scene.env_origins
  command._set_goal(
    ids, torch.as_tensor(goal, dtype=torch.float, device=env.device), origins
  )
  command._write_chain(
    ids, torch.as_tensor(chain, dtype=torch.float, device=env.device), origins
  )
  env.sim.forward()
  env.scene.update(dt=env.physics_dt)
  env.sim.sense()
  return env.scene["external_cam"].data.rgb.cpu().numpy().astype(np.uint8)


def _worst_landmark_mm(a: Any, b: Any) -> float:
  import torch

  from vbrl.deployment.push_chain.silhouette import links
  from vbrl.tasks.push_chain.geometry import landmark_errors

  pair = [torch.tensor(links(pose), dtype=torch.float)[None] for pose in (a, b)]
  return float(landmark_errors(*pair).amin(-1)[0]) * 1000.0


def _measure(
  view: Any, clean: Any, episode: Any, goal: Any, chain: Any, chain_xml: Path
) -> dict[str, Any]:
  import cv2

  from vbrl.deployment.push_chain.silhouette import measure_chain, measure_goal, overlap

  cv2.setNumThreads(1)
  count = len(goal) - 2
  fitted_goal = measure_goal(view, clean, count)
  fitted_chain = measure_chain(view, episode, chain_xml)
  goal_pose = fitted_goal.pose[:-1]
  true_coverage, true_link = overlap(chain, goal)
  coverage, link = overlap(fitted_chain.pose, goal_pose)
  names = ("x", "y", "yaw", *(f"hinge{i}" for i in range(count - 1)))
  poses = {
    f"{prefix}_{kind}_{name}": float(value)
    for prefix, truth, fitted in (
      ("goal", goal, goal_pose),
      ("chain", chain, fitted_chain.pose),
    )
    for kind, pose in (("true", truth), ("fit", fitted))
    for name, value in zip(names, pose, strict=True)
  }
  return {
    **poses,
    "goal_error_mm": _worst_landmark_mm(goal_pose, goal),
    "goal_iou": fitted_goal.iou,
    "goal_reliable": fitted_goal.reliable,
    "tape_offset_mm": fitted_goal.pose[-1] * 1000.0,
    "chain_error_mm": _worst_landmark_mm(fitted_chain.pose, chain),
    "chain_iou": fitted_chain.iou,
    "chain_reliable": fitted_chain.reliable,
    "overlap_true": true_coverage,
    "overlap_measured": coverage,
    "link_overlap_true": true_link,
    "link_overlap_measured": link,
  }


def simulate(
  scene: str, chain_xml: Path, samples: int, num_envs: int, seed: int, workers: int
) -> list[dict[str, Any]]:
  import multiprocessing
  from concurrent.futures import ProcessPoolExecutor

  from vbrl.analysis.push_t_measurement import ideal_view
  from vbrl.tasks.push_chain.geometry import HINGE_LIMIT

  view, fovy = ideal_view()
  env = make_env(scene, chain_xml, num_envs, fovy, seed)
  command = env.command_manager.get_term("push_chain_goal")
  generator = np.random.default_rng(seed)
  rows = []
  context = multiprocessing.get_context("spawn")
  with ProcessPoolExecutor(workers, mp_context=context) as pool:
    for _ in range(math.ceil(samples / num_envs)):
      env.reset()
      goal = command.sample_chains(num_envs).cpu().numpy()
      scale = generator.uniform(0.0, 1.0, (num_envs, 1))
      std = (POSITION_STD, POSITION_STD) + (ANGLE_STD,) * (goal.shape[1] - 2)
      chain = goal + scale * generator.normal(0.0, std, goal.shape)
      chain[:, 3:] = chain[:, 3:].clip(-HINGE_LIMIT, HINGE_LIMIT)
      hidden = chain.copy()
      hidden[:, 1] += 5.0
      clean, episode = render(env, hidden, goal), render(env, chain, goal)
      models = [chain_xml] * num_envs
      rows += pool.map(_measure, [view] * num_envs, clean, episode, goal, chain, models)
  env.close()
  return rows[:samples]


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
  both = [r for r in rows if r["goal_reliable"] and r["chain_reliable"]]
  overlap_error = [abs(r["overlap_measured"] - r["overlap_true"]) * 100.0 for r in both]
  flips = sum(
    (r["overlap_measured"] >= SUCCESS_THRESHOLD)
    != (r["overlap_true"] >= SUCCESS_THRESHOLD)
    for r in both
  )

  def spread(key: str, accepted: str) -> str:
    values = [r[key] for r in rows if r[accepted]]
    return f"{statistics.mean(values):.2f} ± {statistics.stdev(values):.2f} (median {statistics.median(values):.2f}, max {max(values):.2f})"

  return {
    "samples": len(rows),
    "goal accepted": sum(r["goal_reliable"] for r in rows),
    "chain accepted": sum(r["chain_reliable"] for r in rows),
    "goal worst landmark mm": spread("goal_error_mm", "goal_reliable"),
    "chain worst landmark mm": spread("chain_error_mm", "chain_reliable"),
    "tape offset mm": spread("tape_offset_mm", "goal_reliable"),
    "overlap abs error pp": f"{statistics.mean(overlap_error):.2f} (median {statistics.median(overlap_error):.2f}, max {max(overlap_error):.2f})",
    "success flips": f"{flips} of {len(both)}",
  }


def plot(rows: list[dict[str, Any]], scene: str, output: Path) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  figure, (errors, overlaps) = plt.subplots(1, 2, figsize=(10.0, 4.2))
  for k, (name, colour) in enumerate((("goal", GOAL_COLOUR), ("chain", CHAIN_COLOUR))):
    values = np.sort([r[f"{name}_error_mm"] for r in rows if r[f"{name}_reliable"]])
    share = np.arange(1, len(values) + 1) / len(rows)
    errors.step(values, share, where="post", color=colour, lw=2, label=name)
    errors.text(
      0.97,
      0.42 - 0.08 * k,
      f"{name}: median {np.median(values):.2f} mm, {len(values)}/{len(rows)} found",
      transform=errors.transAxes,
      ha="right",
      fontsize=9,
      color=INK,
    )
  errors.set_xlim(0, 5)
  errors.set_ylim(0, 1.02)
  errors.set_xlabel("worst-landmark error [mm]")
  errors.set_ylabel("share of samples")
  errors.set_title("Pose error", loc="left", fontsize=11, color=INK)
  errors.legend(frameon=False, loc="lower right")

  both = [r for r in rows if r["goal_reliable"] and r["chain_reliable"]]
  true = np.array([r["overlap_true"] for r in both])
  measured = np.array([r["overlap_measured"] for r in both])
  flipped = (true >= SUCCESS_THRESHOLD) != (measured >= SUCCESS_THRESHOLD)
  overlaps.plot([0, 1], [0, 1], color=MUTED, lw=1)
  for axis in (overlaps.axvline, overlaps.axhline):
    axis(SUCCESS_THRESHOLD, color=MUTED, lw=1, ls="--")
  overlaps.scatter(true[~flipped], measured[~flipped], s=10, color=CHAIN_COLOUR)
  overlaps.scatter(
    true[flipped], measured[flipped], s=24, color=GOAL_COLOUR, label="success flipped"
  )
  error = np.abs(measured - true) * 100.0
  overlaps.set_title("Chain overlap", loc="left", fontsize=11, color=INK)
  overlaps.text(
    0.97,
    0.06,
    f"error {error.mean():.2f} pp (median {np.median(error):.2f})\n"
    f"success flipped in {int(flipped.sum())} of {len(both)}",
    transform=overlaps.transAxes,
    ha="right",
    fontsize=9,
    color=INK,
  )
  overlaps.set_xlabel("true overlap")
  overlaps.set_ylabel("measured overlap")
  overlaps.legend(frameon=False, loc="upper left")
  for axis in (errors, overlaps):
    axis.grid(color=GRID, lw=0.8)
    axis.spines[["top", "right"]].set_visible(False)
  figure.suptitle(f"Push-Chain rig measurement on {scene}", fontsize=12, color=INK)
  figure.tight_layout()
  for suffix in ("png", "pdf"):
    figure.savefig(output / f"measurement.{suffix}", dpi=200)
  plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, Any]]:
  with path.open() as handle:
    return [
      {k: v == "True" if v in ("True", "False") else float(v) for k, v in row.items()}
      for row in csv.DictReader(handle)
    ]


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(
    description="Benchmark the Push-Chain rig measurement against simulator ground truth."
  )
  parser.add_argument("--scene", choices=sorted(SCENES), default="real_table")
  parser.add_argument(
    "--chain-xml",
    type=Path,
    default=PUSH_CHAIN_XML,
    help="Chain model to render and measure (default: the original three-link chain).",
  )
  parser.add_argument(
    "--output",
    type=Path,
    help="default reports/push_chain/measurement/[<model>/]<scene>",
  )
  parser.add_argument("--samples", type=int, default=512)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--workers", type=int, default=12)
  parser.add_argument(
    "--summarize", action="store_true", help="summarize the existing samples.csv"
  )
  args = parser.parse_args(argv)

  folder = Path("reports/push_chain/measurement")
  if args.chain_xml != PUSH_CHAIN_XML:
    folder = folder / args.chain_xml.stem
  output = args.output or folder / args.scene
  samples = output / "samples.csv"
  if args.summarize:
    rows = _read_csv(samples)
  else:
    rows = simulate(
      args.scene,
      args.chain_xml,
      args.samples,
      args.num_envs,
      args.seed,
      args.workers,
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(samples, rows)
  for key, value in summary(rows).items():
    print(f"{key:24s} {value}")
  plot(rows, args.scene, output)


__all__ = ["main", "plot", "render", "simulate", "summary"]


if __name__ == "__main__":
  main()
