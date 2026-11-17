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
CHAIN_COLOUR, GOAL_COLOUR, INK = "#2a78d6", "#eb6834", "#0b0b0b"


def make_env(scene: str, chain_xml: Path, num_envs: int, fovy: float, seed: int) -> Any:
  import dataclasses

  from mjlab.envs import ManagerBasedRlEnv

  from vbrl.analysis.push_t_measurement import HEIGHT, WIDTH
  from vbrl.deployment.push_chain.silhouette import num_links
  from vbrl.tasks.push_chain.config.trossen_realistic.env_cfgs import (
    trossen_realistic_push_chain_rgb_env_cfg,
  )
  from vbrl.tasks.push_chain.geometry import LINK_HALF_HEIGHT

  cfg = trossen_realistic_push_chain_rgb_env_cfg(
    scene=SCENES[scene],
    num_links=num_links(chain_xml),
    object_xml=chain_xml,
    play=True,
  )
  cfg.seed = seed
  # Rendered at rest on the table like the rig's chain, not at the start height the
  # simulation drops it from: 1 mm of float showed as a 1 mm bias at 45 degrees.
  rest = (LINK_HALF_HEIGHT, LINK_HALF_HEIGHT)
  cfg.commands["push_chain_goal"].object_pose_range.z = rest
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


def _aligned(fit: Any, truth: Any) -> Any:
  import torch

  from vbrl.tasks.push_chain.geometry import (
    centre_link,
    chain_from_pose,
    chain_landmarks,
    landmark_errors,
    reversed_links,
  )

  fitted, true = (chain_from_pose(torch.tensor(p))[None] for p in (fit, truth))
  if landmark_errors(fitted, true)[0].argmin() == 0:
    return np.asarray(fit)
  links = reversed_links(fitted)[0]
  count = len(links)
  middle = chain_landmarks(links)[count]
  yaws = links[:, 2]
  yaw = yaws[centre_link(count)].item()
  return np.r_[middle.numpy(), yaw, torch.diff(yaws).numpy()]


def _pose_errors(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
  accepted = [row for row in rows if row[f"{prefix}_reliable"]]
  hinges = sorted(
    key.removeprefix(f"{prefix}_true_")
    for key in rows[0]
    if key.startswith(f"{prefix}_true_hinge")
  )
  names = ("x", "y", "yaw", *hinges)
  true = np.array([[row[f"{prefix}_true_{n}"] for n in names] for row in accepted])
  fit = np.array([[row[f"{prefix}_fit_{n}"] for n in names] for row in accepted])
  fit = np.array([_aligned(f, t) for f, t in zip(fit, true, strict=True)])
  angles = np.degrees(np.remainder(fit[:, 2:] - true[:, 2:] + np.pi, 2 * np.pi) - np.pi)
  return {
    "true_x": true[:, 0],
    "x": (fit[:, 0] - true[:, 0]) * 1000.0,
    "y": (fit[:, 1] - true[:, 1]) * 1000.0,
    "yaw": angles[:, 0],
    "hinges": angles[:, 1:],
    "position": np.hypot(fit[:, 0] - true[:, 0], fit[:, 1] - true[:, 1]) * 1000.0,
  }


def _pose_row(axes: Any, rows: list, prefix: str, colour: str, name: str) -> None:
  from vbrl.analysis.push_t_measurement import _error_panel, _header, _limit

  errors = _pose_errors(rows, prefix)
  millimetres = _limit(errors["x"], errors["y"])
  degrees = _limit(errors["yaw"], errors["hinges"].ravel())
  labels = "abcd" if prefix == "goal" else "efgh"
  panels = (("x", " mm"), ("y", " mm"), ("yaw", "°"), ("hinges", "°"))
  for axis, (key, unit), label in zip(axes, panels, labels, strict=True):
    error = errors[key]
    x = errors["true_x"]
    if key == "hinges":
      x, error = np.repeat(x, error.shape[1]), error.ravel()
    limit = millimetres if unit == " mm" else degrees
    _error_panel(axis, x, error, colour, limit, unit)
    _header(axis, f"({label}) {name}, {key}", error, limit, unit)
    axis.set_ylabel(f"Δ{key} ({unit.strip()})")
    axis.set_xlabel("True x (m)")


def _overlap_panel(axis: Any, rows: list) -> None:
  from vbrl.analysis.push_t_measurement import (
    REFERENCE_COLOUR,
    _error_panel,
    _header,
    _limit,
  )

  accepted = [row for row in rows if row["chain_reliable"] and row["goal_reliable"]]
  true = np.array([row["overlap_true"] for row in accepted]) * 100.0
  error = np.array([row["overlap_measured"] for row in accepted]) * 100.0 - true
  threshold = SUCCESS_THRESHOLD * 100.0
  disagree = (true >= threshold) != (true + error >= threshold)
  limit = _limit(error)
  _error_panel(axis, true[~disagree], error[~disagree], CHAIN_COLOUR, limit, " pp")
  axis.axvline(threshold, color=REFERENCE_COLOUR, linewidth=0.8, linestyle=(0, (4, 3)))
  axis.scatter(
    true[disagree],
    error[disagree],
    s=12,
    color=GOAL_COLOUR,
    linewidths=0,
    label=f"success at {threshold:.0f} % flipped ({disagree.sum()})",
  )
  axis.legend(loc="lower left", frameon=False, fontsize=6.5, handletextpad=0.2)
  _header(axis, "(i) Chain overlap", error, limit, " pp")
  axis.set_xlabel("True overlap (%)")
  axis.set_ylabel("Δ overlap (pp)")
  axis.set_xlim(0, 100)


def _figure(height: float) -> Any:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  plt.rcParams.update(
    {"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6}
  )
  return plt.figure(figsize=(8.0, height), layout="constrained")


def _save(figure: Any, output: Path, name: str) -> None:
  import matplotlib.pyplot as plt

  output.mkdir(parents=True, exist_ok=True)
  for suffix in ("pdf", "png"):
    figure.savefig(output / f"{name}.{suffix}", dpi=300)
  plt.close(figure)


def plot(rows: list[dict[str, Any]], output: Path) -> None:
  figure = _figure(7.0)
  grid = figure.add_gridspec(3, 4)
  goal = [figure.add_subplot(grid[0, k]) for k in range(4)]
  chain = [figure.add_subplot(grid[1, k]) for k in range(4)]
  _pose_row(goal, rows, "goal", GOAL_COLOUR, "Goal")
  _pose_row(chain, rows, "chain", CHAIN_COLOUR, "Chain")
  _overlap_panel(figure.add_subplot(grid[2, :]), rows)
  _save(figure, output, "measurement_benchmark")


def _bar_panel(axis: Any, rows: list, key: str, unit: str) -> None:
  from matplotlib.colors import to_rgba

  from vbrl.analysis.push_t_measurement import _style

  bars = []
  for prefix, name, colour in (
    ("goal", "Goal", GOAL_COLOUR),
    ("chain", "Chain", CHAIN_COLOUR),
  ):
    errors = np.abs(_pose_errors(rows, prefix)[key]).ravel()
    count = sum(row[f"{prefix}_reliable"] for row in rows)
    bars.append((name, colour, errors.mean(), errors.std(), count))
  top = 1.3 * max(mean + spread for _, _, mean, spread, _ in bars)
  for position, (_, colour, mean, spread, _) in enumerate(bars):
    axis.bar(
      position,
      mean,
      width=0.6,
      color=to_rgba(colour, 0.25),
      edgecolor=colour,
      linewidth=1.4,
    )
    axis.errorbar(
      position, mean, yerr=spread, color=colour, capsize=5, elinewidth=1.2, capthick=1.2
    )
    axis.text(
      position,
      mean + spread + 0.03 * top,
      f"{mean:.2f} ± {spread:.2f}",
      ha="center",
      va="bottom",
      color=INK,
      fontsize=7,
    )
  axis.set_xticks(
    range(len(bars)), [f"{name}\nn = {count}" for name, _, _, _, count in bars]
  )
  axis.tick_params(axis="x", length=0)
  axis.set_xlim(-0.6, len(bars) - 0.4)
  axis.set_ylim(0.0, top)
  axis.set_ylabel(f"Error ({unit})")
  _style(axis)
  axis.grid(False, axis="x")


def plot_summary(rows: list[dict[str, Any]], output: Path) -> None:
  from vbrl.analysis.push_t_measurement import _subtitle

  figure = _figure(3.3)
  figure.suptitle(
    f"Push-Chain pose measurement against ground truth ({len(rows)} rendered scenes)",
    x=0.01,
    ha="left",
    fontsize=10,
    fontweight="bold",
    color=INK,
  )
  grid = figure.add_gridspec(1, 3)
  for k, (key, unit, title, note) in enumerate(
    (
      ("position", "mm", "(a) Position", "middle point, xy distance, mean ± sd"),
      ("yaw", "°", "(b) Yaw", "absolute angle, mean ± sd"),
      ("hinges", "°", "(c) Hinges", "absolute angle per hinge, mean ± sd"),
    )
  ):
    axis = figure.add_subplot(grid[0, k])
    _bar_panel(axis, rows, key, unit)
    _subtitle(axis, title, note)
  _save(figure, output, "measurement_summary")


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
  plot(rows, output)
  plot_summary(rows, output)


__all__ = ["main", "plot", "plot_summary", "render", "simulate", "summary"]


if __name__ == "__main__":
  main()
