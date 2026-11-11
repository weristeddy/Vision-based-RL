from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

TASKS = {
  "real_table": "Mjlab-PushT-RealTable-DinoV2ViTS14-AfaFull6-Trossen",
  "wood": "Mjlab-PushT-WoodTable-DinoV2ViTS14-AfaFull6-Trossen",
}
WIDTH, HEIGHT = 1280, 720
GOAL_X, GOAL_Y = (0.2, 0.5), (-0.2, 0.2)
T_OFFSET_STD = (0.03, 0.03, math.radians(30.0))
# A printed sheet like the rig's, not the simulator's 2.4 mm slab.
MARKER_TOP_M = 0.0002
SUCCESS_THRESHOLD = 0.9
T_COLOUR, GOAL_COLOUR, REFERENCE_COLOUR = "#2a78d6", "#eb6834", "#52514e"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def ideal_view() -> tuple[Any, float]:
  from vbrl.deployment.push_t.view import ExternalView

  rig = ExternalView.load(WIDTH, HEIGHT)
  focal = rig.matrix[1, 1]
  matrix = np.array(
    [[focal, 0.0, (WIDTH - 1) / 2], [0.0, focal, (HEIGHT - 1) / 2], [0.0, 0.0, 1.0]]
  )
  view = ExternalView(matrix, np.zeros(5), rig.pose, WIDTH, HEIGHT)
  return view, math.degrees(2.0 * math.atan(HEIGHT / 2 / focal))


def make_env(task: str, num_envs: int, fovy: float, seed: int) -> Any:
  import dataclasses

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  import vbrl.tasks  # noqa: F401

  cfg = load_env_cfg(task, play=True)
  cfg.seed = seed
  cfg.scene.num_envs = num_envs
  for name in ("camera_position", "camera_orientation"):
    cfg.events.pop(name)
  # The rig has one T and one marker: their measured colour, not the training spread.
  for name in ("object_color", "goal_color"):
    ranges = cfg.events[name].params["ranges"]
    cfg.events[name].params["ranges"] = {
      axis: ((low + high) / 2,) * 2 for axis, (low, high) in ranges.items()
    }
  # The rig's room light casts almost no shadow.
  cfg.scene.sensors = tuple(
    dataclasses.replace(
      sensor, width=WIDTH, height=HEIGHT, fovy=fovy, use_shadows=False
    )
    if sensor.name == "external_cam"
    else sensor
    for sensor in cfg.scene.sensors
  )
  return ManagerBasedRlEnv(cfg=cfg, device="cuda:0")


def render(env: Any, t: Any, goal: Any) -> Any:
  import torch
  from mjlab.utils.lab_api.math import quat_from_euler_xyz

  from vbrl.tasks.push_t.geometry import HALF_HEIGHT
  from vbrl.tasks.push_t.goal_marker import GOAL_HALF_THICKNESS

  origins = env.scene.env_origins
  ids = torch.arange(env.num_envs, device=env.device)

  def pose(values: Any, z: float) -> Any:
    values = torch.as_tensor(values, dtype=torch.float32, device=env.device)
    zero = torch.zeros(len(values), device=env.device)
    position = origins + torch.stack((values[:, 0], values[:, 1], zero + z), 1)
    return torch.cat((position, quat_from_euler_xyz(zero, zero, values[:, 2])), 1)

  hidden = t is None
  t = np.zeros_like(goal) if hidden else t
  obj = env.scene["object"]
  obj.write_root_link_pose_to_sim(pose(t, -1.0 if hidden else HALF_HEIGHT), env_ids=ids)
  obj.write_root_link_velocity_to_sim(torch.zeros(len(ids), 6, device=env.device), ids)
  marker = env.command_manager.get_term("push_t_goal")._goal_marker
  marker_z = MARKER_TOP_M - 2.0 * GOAL_HALF_THICKNESS
  marker.write_mocap_pose_to_sim(pose(goal, marker_z), env_ids=ids)
  env.sim.forward()
  env.scene.update(dt=env.physics_dt)
  env.sim.sense()
  return env.scene["external_cam"].data.rgb.cpu().numpy().astype(np.uint8)


def _errors(fitted: Any, truth: Any) -> tuple[float, float]:
  position = math.dist(fitted.pose[:2], truth[:2]) * 1000.0
  yaw = abs(math.degrees(math.remainder(fitted.pose[2] - truth[2], 2.0 * math.pi)))
  return position, yaw


def _poses(prefix: str, truth: Any, fitted: Any) -> dict[str, float]:
  names = ("x", "y", "yaw")
  return {
    **{f"{prefix}_true_{n}": float(v) for n, v in zip(names, truth, strict=True)},
    **{f"{prefix}_fit_{n}": v for n, v in zip(names, fitted.pose, strict=True)},
  }


def simulate(
  task: str, samples: int, num_envs: int, seed: int
) -> list[dict[str, float]]:
  from vbrl.deployment.push_t.silhouette import measure_goal, measure_t, overlap

  view, fovy = ideal_view()
  env = make_env(task, num_envs, fovy, seed)
  generator = np.random.default_rng(seed)
  rows = []
  for _ in range(math.ceil(samples / num_envs)):
    env.reset()
    goal = np.c_[
      generator.uniform(*GOAL_X, num_envs),
      generator.uniform(*GOAL_Y, num_envs),
      generator.uniform(-math.pi, math.pi, num_envs),
    ]
    scale = generator.uniform(0.0, 1.0, (num_envs, 1))
    t = goal + scale * generator.normal(0.0, T_OFFSET_STD, (num_envs, 3))
    clean, episode = render(env, None, goal), render(env, t, goal)
    for k in range(num_envs):
      fitted_goal, fitted_t = measure_goal(view, clean[k]), measure_t(view, episode[k])
      rows.append({
        **_poses("goal", goal[k], fitted_goal),
        **_poses("t", t[k], fitted_t),
        "goal_position_error_mm": _errors(fitted_goal, goal[k])[0],
        "goal_yaw_error_deg": _errors(fitted_goal, goal[k])[1],
        "goal_iou": fitted_goal.iou,
        "goal_reliable": fitted_goal.reliable,
        "t_position_error_mm": _errors(fitted_t, t[k])[0],
        "t_yaw_error_deg": _errors(fitted_t, t[k])[1],
        "t_iou": fitted_t.iou,
        "t_reliable": fitted_t.reliable,
        "overlap_true": overlap(t[k], goal[k]),
        "overlap_measured": overlap(fitted_t.pose, fitted_goal.pose),
      })
  return rows[:samples]


def real_example(log: Path) -> dict[str, Any]:
  from vbrl.deployment.push_t.silhouette import (
    measure_goal,
    measure_t,
    render_footprint,
    render_t,
  )
  from vbrl.deployment.push_t.view import ExternalView

  data = np.load(log)
  frames, reference = data["frames"], tuple(float(v) for v in data["goal"])
  view = ExternalView.load(424, 240).cropped(frames.shape[1])
  poses = np.array([measure_t(view, frame).pose for frame in frames])
  t, goal = measure_t(view, frames[0]), measure_goal(view, frames[0])
  return {
    "image": frames[0],
    "t": t,
    "goal": goal,
    "t_mask": render_t(view, t.pose),
    "goal_mask": render_footprint(view, goal.pose),
    "reference_mask": render_footprint(view, reference),
    "reference": reference,
    "frames": len(frames),
    "t_std_mm": poses[:, :2].std(axis=0) * 1000.0,
    "t_std_deg": math.degrees(np.std(np.unwrap(poses[:, 2]))),
  }


def _style(axis: Any) -> None:
  axis.grid(True, color=GRID, linewidth=0.6)
  axis.set_axisbelow(True)
  for side in ("top", "right"):
    axis.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    axis.spines[side].set_color(MUTED)
  axis.tick_params(colors=MUTED, labelcolor=INK)


def _signed_errors(rows: list, prefix: str) -> dict[str, Any]:
  accepted = [row for row in rows if row[f"{prefix}_reliable"]]
  values = {
    key: np.array([row[f"{prefix}_{key}"] for row in accepted])
    for key in ("true_x", "true_y", "true_yaw", "fit_x", "fit_y", "fit_yaw")
  }
  yaw = np.remainder(values["fit_yaw"] - values["true_yaw"] + np.pi, 2 * np.pi)
  return {
    "true_x": values["true_x"],
    "x": (values["fit_x"] - values["true_x"]) * 1000.0,
    "y": (values["fit_y"] - values["true_y"]) * 1000.0,
    "yaw": np.degrees(yaw - np.pi),
    "rejected": len(rows) - len(accepted),
    "total": len(rows),
  }


def _limit(*errors: Any) -> float:
  return 1.25 * max(np.percentile(np.abs(error), 99.5) for error in errors)


def _subtitle(axis: Any, title: str, text: str) -> None:
  axis.set_title(title, loc="left", color=INK, fontsize=9, pad=14)
  note = axis.text(
    0.0, 1.02, text, transform=axis.transAxes, ha="left", va="bottom",
    color=MUTED, fontsize=6.5,
  )
  note.set_in_layout(False)


def _header(axis: Any, title: str, error: Any, limit: float, unit: str) -> None:
  outside = int((np.abs(error) > limit).sum())
  stats = f"bias {error.mean():+.2f}{unit}, σ {error.std():.2f}{unit}"
  _subtitle(axis, title, stats + (f", {outside} off scale" if outside else ""))


def _error_panel(
  axis: Any, x: Any, error: Any, colour: str, limit: float, unit: str
) -> None:
  bias, spread = error.mean(), error.std()
  axis.axhspan(bias - 2 * spread, bias + 2 * spread, color=colour, alpha=0.12, lw=0)
  axis.axhline(0.0, color=REFERENCE_COLOUR, linewidth=0.8)
  axis.scatter(x, error, s=6, color=colour, alpha=0.6, linewidths=0)
  axis.set_ylim(-limit, limit)
  _style(axis)


def _pose_row(axes: Any, rows: list, prefix: str, colour: str, name: str) -> None:
  errors = _signed_errors(rows, prefix)
  millimetres = _limit(errors["x"], errors["y"])
  labels = "abc" if prefix == "goal" else "def"
  for axis, key, label in zip(axes, ("x", "y", "yaw"), labels, strict=True):
    unit = "°" if key == "yaw" else " mm"
    limit = _limit(errors["yaw"]) if key == "yaw" else millimetres
    _error_panel(axis, errors["true_x"], errors[key], colour, limit, unit)
    _header(axis, f"({label}) {name}, {key}", errors[key], limit, unit)
    axis.set_ylabel(f"Δ{key} ({unit.strip()})")
    axis.set_xlabel("True x (m)")


def _overlap_panel(axis: Any, rows: list) -> None:
  accepted = [row for row in rows if row["t_reliable"] and row["goal_reliable"]]
  true = np.array([row["overlap_true"] for row in accepted]) * 100.0
  error = np.array([row["overlap_measured"] for row in accepted]) * 100.0 - true
  threshold = SUCCESS_THRESHOLD * 100.0
  disagree = (true >= threshold) != (true + error >= threshold)
  limit = _limit(error)
  _error_panel(axis, true[~disagree], error[~disagree], T_COLOUR, limit, " pp")
  axis.axvline(threshold, color=REFERENCE_COLOUR, linewidth=0.8, linestyle=(0, (4, 3)))
  axis.scatter(
    true[disagree], error[disagree], s=12, color=GOAL_COLOUR, linewidths=0,
    label=f"success at {threshold:.0f} % flipped ({disagree.sum()})",
  )
  axis.legend(loc="lower left", frameon=False, fontsize=6.5, handletextpad=0.2)
  _header(axis, "(g) Overlap", error, limit, " pp")
  axis.set_xlabel("True overlap (%)")
  axis.set_ylabel("Δ overlap (pp)")
  axis.set_xlim(0, 100)


def _real_panel(axis: Any, real: dict[str, Any], title: str) -> None:
  from matplotlib.lines import Line2D

  axis.imshow(real["image"])
  for mask, colour, style in (
    (real["reference_mask"], REFERENCE_COLOUR, (0, (3, 2))),
    (real["goal_mask"], GOAL_COLOUR, "solid"),
    (real["t_mask"], T_COLOUR, "solid"),
  ):
    axis.contour(mask, levels=(0.5,), colors=colour, linewidths=1.0, linestyles=[style])
  std = real["t_std_mm"]
  offset = math.dist(real["goal"].pose[:2], real["reference"][:2]) * 1000.0
  size = real["image"].shape[0]
  _subtitle(axis, title, f"{size} × {size} policy crop")
  axis.text(
    0.5, -0.11,
    f"T at rest, {real['frames']} frames:\nσ {std[0]:.2f} / {std[1]:.2f} mm, "
    f"{real['t_std_deg']:.2f}°\nAprilTag goal {offset:.1f} mm from goal fit",
    transform=axis.transAxes, ha="center", va="top", color=MUTED, fontsize=6.5,
  )
  handles = [
    Line2D([], [], color=T_COLOUR, label="T fit"),
    Line2D([], [], color=GOAL_COLOUR, label="goal fit"),
    Line2D(
      [], [], color=REFERENCE_COLOUR, linestyle=(0, (3, 2)), label="AprilTag goal"
    ),
  ]
  axis.legend(
    handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3,
    frameon=False, fontsize=6.5, handlelength=1.4, columnspacing=0.8,
    handletextpad=0.4,
  )
  axis.set_axis_off()


def plot(rows: list[dict[str, float]], real: dict[str, Any], output: Path) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  plt.rcParams.update(
    {"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6}
  )
  figure = plt.figure(figsize=(6.3, 7.0), layout="constrained")
  grid = figure.add_gridspec(3, 3)
  goal = [figure.add_subplot(grid[0, k]) for k in range(3)]
  t = [figure.add_subplot(grid[1, k]) for k in range(3)]
  _pose_row(goal, rows, "goal", GOAL_COLOUR, "Goal")
  _pose_row(t, rows, "t", T_COLOUR, "T")
  _overlap_panel(figure.add_subplot(grid[2, :2]), rows)
  _real_panel(figure.add_subplot(grid[2, 2]), real, "(h) Rig frame")
  output.mkdir(parents=True, exist_ok=True)
  for suffix in ("pdf", "png"):
    figure.savefig(output / f"measurement_benchmark.{suffix}", dpi=300)
  plt.close(figure)


def _bar_panel(axis: Any, rows: list, key: str, unit: str) -> None:
  from matplotlib.colors import to_rgba

  bars = []
  for prefix, name, colour in (
    ("goal", "Goal", GOAL_COLOUR), ("t", "T", T_COLOUR)
  ):
    accepted = [row for row in rows if row[f"{prefix}_reliable"]]
    errors = np.array([row[f"{prefix}_{key}"] for row in accepted])
    bars.append((name, colour, errors.mean(), errors.std(), len(errors)))
  top = 1.3 * max(mean + spread for _, _, mean, spread, _ in bars)
  for position, (_, colour, mean, spread, _) in enumerate(bars):
    axis.bar(
      position, mean, width=0.6, color=to_rgba(colour, 0.25), edgecolor=colour,
      linewidth=1.4,
    )
    axis.errorbar(
      position, mean, yerr=spread, color=colour, capsize=5, elinewidth=1.2,
      capthick=1.2,
    )
    axis.text(
      position, mean + spread + 0.03 * top, f"{mean:.2f} ± {spread:.2f}",
      ha="center", va="bottom", color=INK, fontsize=7,
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


def plot_summary(
  rows: list[dict[str, float]], real: dict[str, Any], output: Path
) -> None:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  plt.rcParams.update(
    {"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6}
  )
  figure = plt.figure(figsize=(6.3, 3.3), layout="constrained")
  figure.suptitle(
    f"Push-T pose measurement against ground truth ({len(rows)} rendered scenes)",
    x=0.01, ha="left", fontsize=10, fontweight="bold", color=INK,
  )
  grid = figure.add_gridspec(1, 4, width_ratios=(1, 1, 0.12, 1.5))
  position, yaw = figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])
  _bar_panel(position, rows, "position_error_mm", "mm")
  _subtitle(position, "(a) Position", "xy distance, mean ± sd")
  _bar_panel(yaw, rows, "yaw_error_deg", "°")
  _subtitle(yaw, "(b) Yaw", "absolute angle, mean ± sd")
  _real_panel(figure.add_subplot(grid[0, 3]), real, "(c) Rig frame")
  output.mkdir(parents=True, exist_ok=True)
  for suffix in ("pdf", "png"):
    figure.savefig(output / f"measurement_summary.{suffix}", dpi=300)
  plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
  with path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, float]]:
  with path.open() as handle:
    return [
      {key: value == "True" if value in ("True", "False") else float(value)
       for key, value in row.items()}
      for row in csv.DictReader(handle)
    ]


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(
    description="Benchmark the Push-T rig measurement against simulator ground truth."
  )
  parser.add_argument("--scene", choices=sorted(TASKS), default="real_table")
  parser.add_argument(
    "--output", type=Path, help="default reports/push_t/measurement/<scene>"
  )
  parser.add_argument("--samples", type=int, default=512)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--real-log", type=Path, default=Path("artifacts/deployment/wood_table_cnn_1.npz")
  )
  parser.add_argument(
    "--replot", action="store_true", help="plot the existing samples.csv again"
  )
  args = parser.parse_args(argv)

  output = args.output or Path("reports/push_t/measurement") / args.scene
  samples = output / "samples.csv"
  if args.replot:
    rows = _read_csv(samples)
  else:
    rows = simulate(TASKS[args.scene], args.samples, args.num_envs, args.seed)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(samples, rows)
  real = real_example(args.real_log)
  plot(rows, real, output)
  plot_summary(rows, real, output)


__all__ = [
  "main",
  "plot",
  "plot_summary",
  "real_example",
  "simulate",
]


if __name__ == "__main__":
  main()
