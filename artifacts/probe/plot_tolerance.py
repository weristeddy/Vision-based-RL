"""How much pose error the Push-T overlap metric tolerates, at each threshold.

Uses the task's own rasterizer, so the contours are the success rule itself
rather than an approximation of it.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from vbrl.tasks.push_t.geometry import FOOTPRINT_PARTS, FootprintRasterizer

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b6ad"
BLUE, ORANGE = "#2a78d6", "#eb6834"

POS = np.linspace(0.0, 0.030, 61)        # metres
YAW = np.linspace(0.0, np.pi / 2, 73)    # radians -- spans where the runs land
SAMPLES = 256


def main():
  r = FootprintRasterizer(FOOTPRINT_PARTS, device="cpu", dtype=torch.float64)
  g = torch.Generator().manual_seed(0)
  grid = np.empty((len(YAW), len(POS)))

  # Average over the direction of the position offset and the sign of the yaw
  # offset; magnitude is what the axes hold fixed.
  ang = torch.rand(SAMPLES, generator=g, dtype=torch.float64) * 2 * torch.pi
  sign = torch.where(
    torch.rand(SAMPLES, generator=g, dtype=torch.float64) > 0.5, 1.0, -1.0
  )
  zeros2 = torch.zeros(SAMPLES, 2, dtype=torch.float64)
  zeros1 = torch.zeros(SAMPLES, dtype=torch.float64)
  for i, ye in enumerate(YAW):
    for j, pe in enumerate(POS):
      xy = torch.stack((torch.cos(ang), torch.sin(ang)), -1) * float(pe)
      grid[i, j] = r.overlap(
        object_xy=xy,
        object_yaw=sign * float(ye),
        target_xy=zeros2,
        target_yaw=zeros1,
      ).mean()

  fig, ax = plt.subplots(figsize=(7.2, 5.0), facecolor=SURFACE)
  ax.set_facecolor(SURFACE)
  mesh = ax.pcolormesh(
    POS * 1000, np.degrees(YAW), grid, cmap="Blues", vmin=0.0, vmax=1.0,
    shading="gouraud", rasterized=True,
  )
  cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
  cbar.set_label("mean overlap", color=INK2, fontsize=10)
  cbar.ax.tick_params(colors=INK2, labelsize=9)
  cbar.outline.set_visible(False)

  for level, color in ((0.90, ORANGE), (0.98, BLUE)):
    ax.contour(
      POS * 1000, np.degrees(YAW), grid, levels=[level], colors=[color],
      linewidths=2.0,
    )
  # The contours are short and crowd the origin, so identify them in a legend
  # rather than with leader lines that collide with each other.
  handles = [
    plt.Line2D([], [], color=ORANGE, linewidth=2.0,
               label="overlap 0.90  (≤ 5.0 mm or ≤ 5.0°)"),
    plt.Line2D([], [], color=BLUE, linewidth=2.0,
               label="overlap 0.98  (≤ 2.0 mm or ≤ 2.5°)"),
  ]
  legend = ax.legend(
    handles=handles, loc="upper left", frameon=True, fontsize=9.5,
    facecolor=SURFACE, edgecolor=MUTED, framealpha=0.92, borderpad=0.7,
  )
  for text in legend.get_texts():
    text.set_color(INK)

  # Where the trained policies actually land.
  for x, y, label, dx, dy, ha in (
    (19, 76, "best RGB run (DinoV2-Afa6)", -1.5, -8, "right"),
    (13, 5.8, "state policy", 3.0, 16, "left"),
  ):
    ax.scatter([x], [y], s=90, color=INK, zorder=5, edgecolor=SURFACE, linewidth=2)
    ax.annotate(
      label, xy=(x, y), xytext=(x + dx, y + dy), fontsize=9.5, color=INK,
      ha=ha, va="center",
      arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=1.2),
    )

  ax.set_xlabel("position error (mm)", color=INK2, fontsize=10)
  ax.set_ylabel("yaw error (degrees)", color=INK2, fontsize=10)
  ax.set_title(
    "Push-T overlap tolerance: what each success threshold demands",
    color=INK, fontsize=12, pad=12, loc="left",
  )
  ax.tick_params(colors=INK2, labelsize=9)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color(MUTED)

  fig.tight_layout()
  fig.savefig("artifacts/probe/tolerance.png", dpi=170, facecolor=SURFACE)
  print("wrote artifacts/probe/tolerance.png")

  # The numbers behind the contours.
  for level in (0.90, 0.98):
    pure_pos = POS[grid[0] >= level].max() * 1000
    pure_yaw = np.degrees(YAW[grid[:, 0] >= level].max())
    print(f"overlap >= {level}: at most {pure_pos:.1f} mm with perfect yaw, "
          f"or {pure_yaw:.1f} deg with perfect position")


if __name__ == "__main__":
  main()
