from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np

TASK = "Mjlab-PushChain-WoodTable-DinoV2ViTS14-AfaFull6-Trossen"
SCENES = ("real_table", "wood")
CHAIN_OFFSET_STD = (
  0.03,
  0.03,
  math.radians(30.0),
  math.radians(30.0),
  math.radians(30.0),
)
SUCCESS_THRESHOLD = 0.9


def make_env(scene: str, num_envs: int, fovy: float, seed: int) -> Any:
  import dataclasses

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  import vbrl.tasks  # noqa: F401
  from vbrl.analysis.push_t_measurement import HEIGHT, WIDTH
  from vbrl.scenes.builder import replace_scene

  cfg = load_env_cfg(TASK, play=True)
  if scene != "wood":
    replace_scene(cfg, scene=scene)
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
  view: Any, clean: Any, episode: Any, goal: Any, chain: Any
) -> dict[str, Any]:
  import cv2

  from vbrl.deployment.push_chain.silhouette import measure_chain, measure_goal, overlap

  cv2.setNumThreads(1)
  fitted_goal, fitted_chain = measure_goal(view, clean), measure_chain(view, episode)
  true_coverage, true_link = overlap(chain, goal)
  coverage, link = overlap(fitted_chain.pose, fitted_goal.pose)
  return {
    "goal_error_mm": _worst_landmark_mm(fitted_goal.pose, goal),
    "goal_iou": fitted_goal.iou,
    "goal_reliable": fitted_goal.reliable,
    "tape_offset_mm": fitted_goal.pose[5] * 1000.0,
    "chain_error_mm": _worst_landmark_mm(fitted_chain.pose, chain),
    "chain_iou": fitted_chain.iou,
    "chain_reliable": fitted_chain.reliable,
    "overlap_true": true_coverage,
    "overlap_measured": coverage,
    "link_overlap_true": true_link,
    "link_overlap_measured": link,
  }


def simulate(
  scene: str, samples: int, num_envs: int, seed: int, workers: int
) -> list[dict[str, Any]]:
  import multiprocessing
  from concurrent.futures import ProcessPoolExecutor

  from vbrl.analysis.push_t_measurement import ideal_view
  from vbrl.tasks.push_chain.geometry import HINGE_LIMIT

  view, fovy = ideal_view()
  env = make_env(scene, num_envs, fovy, seed)
  command = env.command_manager.get_term("push_chain_goal")
  generator = np.random.default_rng(seed)
  rows = []
  context = multiprocessing.get_context("spawn")
  with ProcessPoolExecutor(workers, mp_context=context) as pool:
    for _ in range(math.ceil(samples / num_envs)):
      env.reset()
      goal = command.sample_chains(num_envs).cpu().numpy()
      scale = generator.uniform(0.0, 1.0, (num_envs, 1))
      chain = goal + scale * generator.normal(0.0, CHAIN_OFFSET_STD, (num_envs, 5))
      chain[:, 3:] = chain[:, 3:].clip(-HINGE_LIMIT, HINGE_LIMIT)
      hidden = chain.copy()
      hidden[:, 1] += 5.0
      clean, episode = render(env, hidden, goal), render(env, chain, goal)
      rows += pool.map(_measure, [view] * num_envs, clean, episode, goal, chain)
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
  parser.add_argument("--scene", choices=SCENES, default="real_table")
  parser.add_argument(
    "--output", type=Path, help="default reports/push_chain/measurement/<scene>"
  )
  parser.add_argument("--samples", type=int, default=512)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--workers", type=int, default=12)
  parser.add_argument(
    "--summarize", action="store_true", help="summarize the existing samples.csv"
  )
  args = parser.parse_args(argv)

  output = args.output or Path("reports/push_chain/measurement") / args.scene
  samples = output / "samples.csv"
  if args.summarize:
    rows = _read_csv(samples)
  else:
    rows = simulate(args.scene, args.samples, args.num_envs, args.seed, args.workers)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(samples, rows)
  for key, value in summary(rows).items():
    print(f"{key:24s} {value}")


__all__ = ["main", "render", "simulate", "summary"]


if __name__ == "__main__":
  main()
