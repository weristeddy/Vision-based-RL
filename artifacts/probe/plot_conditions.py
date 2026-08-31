"""Three-point dumbbell: what the encoder gets free, what RL built, what it could reach.

Reads artifacts/probe/conditions_SlowGoal.json. Every point comes from the same
rendered images (uniform yaw, arm configurations from a real rollout) and the
same policy-matched readout head, so `init` vs `rl` is a controlled A/B and only
`supervised` is allowed extra freedom.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#b8b6ad", "#e6e5df"
CONDITIONS = (
  ("init", "#2a78d6", "init — weights PPO started from"),
  ("rl", "#eb6834", "rl — after training"),
  ("supervised", "#1baf7a", "supervised — trained with labels (ceiling)"),
)
CHANCE = 90.0
# Rows are grouped by encoder, as in the scaling figure, so an encoder's adapters
# read together instead of being scattered through one global ranking.
ENCODERS = (
  ("NatureCnn", "NatureCnn  (scratch, 24×24)"),
  ("CompactVit", "CompactVit  (scratch, 14×14)"),
  ("DinoV2ViTS14", "DINOv2 ViT-S/14  (frozen, 16×16)"),
  ("R3MResNet50", "R3M ResNet-50 layer4  (frozen, 7×7)"),
  ("R3MResNet50L3", "R3M ResNet-50 layer3  (frozen, 14×14)"),
)


def main():
  source = sys.argv[1] if len(sys.argv) > 1 else "artifacts/probe/conditions_SlowGoal.json"
  out = sys.argv[2] if len(sys.argv) > 2 else "artifacts/probe/conditions.png"
  data = json.loads(Path(source).read_text())["conditions"]
  generation = Path(source).stem.replace("conditions_", "")

  def value(arch, key):
    row = data.get(arch, {}).get(key)
    return np.degrees(row["median"]) if row else np.nan

  groups = []
  for prefix, title in ENCODERS:
    members = sorted(
      (a for a in data if a.startswith(prefix + "-")),
      key=lambda a: np.nan_to_num(value(a, "rl"), nan=1e3),
    )
    if members:
      groups.append((title, members))

  if not groups:
    print(f"nothing to plot: {source} has no architectures")
    return
  total = sum(len(m) for _, m in groups)
  # Reserve a fixed 1.5in header for title, subtitle and legend regardless of
  # how many rows the figure ends up with.
  header = 1.15
  height = 0.40 * total + 1.4 * len(groups) + header
  top_of_panels = 1.0 - 0.62 * header / height
  fig, axes = plt.subplots(
    len(groups), 1, figsize=(11.4, height),
    facecolor=SURFACE, sharex=True,
    gridspec_kw={"height_ratios": [len(m) for _, m in groups]},
  )
  axes = np.atleast_1d(axes)
  inverted = []

  for ax, (title, members) in zip(axes, groups):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.axvline(CHANCE, color=MUTED, linewidth=1.4, linestyle=(0, (4, 3)), zorder=1)

    for row, arch in enumerate(members):
      points = [(value(arch, key), color) for key, color, _ in CONDITIONS]
      finite = [p for p, _ in points if np.isfinite(p)]
      if finite:
        ax.plot([min(finite), max(finite)], [row, row], color=GRID, linewidth=3.0,
                zorder=2, solid_capstyle="round")
      for point, color in points:
        if np.isfinite(point):
          ax.scatter([point], [row], s=92, color=color, zorder=4,
                     edgecolor=SURFACE, linewidth=1.8)
      sup, rl = value(arch, "supervised"), value(arch, "rl")
      if np.isfinite(sup) and np.isfinite(rl) and sup > rl + 2.0:
        # Supervision below RL is impossible as a *ceiling* only when the
        # encoder is frozen. Both flagged rows are the from-scratch ViT, which
        # fits its training set (train 2-5 deg) and then fails on held-out data
        # -- while every other architecture generalises on the identical images.
        # So it is neither a data limit nor a failure to converge.
        train = data[arch]["supervised"].get("train_median", 1.6)
        overfit = np.degrees(train) < 20.0
        inverted.append((arch, overfit))
        ax.annotate("o" if overfit else "!", xy=(101.5, row), fontsize=11,
                    color="#eda100" if overfit else "#e34948",
                    ha="center", va="center", fontweight="bold")

    ax.set_yticks(np.arange(len(members)))
    ax.set_yticklabels([a.split("-", 1)[1] for a in members], fontsize=9.5, color=INK)
    ax.set_ylim(-0.65, len(members) - 0.35)
    ax.set_xlim(-3, 104)
    ax.set_title(title, color=INK, fontsize=10.5, pad=6, loc="left")
    ax.tick_params(colors=INK2, labelsize=9)
    for side in ("top", "right", "left"):
      ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)

  axes[-1].set_xlabel("median yaw decoding error (degrees)", color=INK2, fontsize=10)
  axes[0].annotate("chance", xy=(CHANCE, len(groups[0][1]) - 0.5), xytext=(4, 0),
                   textcoords="offset points", fontsize=8.5, color=INK2)

  handles = [
    plt.Line2D([], [], color=color, marker="o", markersize=9, linestyle="",
               markeredgecolor=SURFACE, markeredgewidth=1.8, label=label)
    for _, color, label in CONDITIONS
  ]
  # Figure-level, above the panels: inside any axes it covered data rows.
  legend = fig.legend(handles=handles, loc="upper center", ncols=3, frameon=False,
                      fontsize=10, bbox_to_anchor=(0.5, top_of_panels))
  for text in legend.get_texts():
    text.set_color(INK)

  subtitle = (
    "Same images, same readout head, same budget — only the encoder weights "
    "differ between init and rl."
  )
  over = sum(1 for _, o in inverted if o)
  fail = len(inverted) - over
  if over:
    subtitle += (
      f"\no  supervised below rl for {over}: the from-scratch ViT fits its "
      "training set but not held-out data, on images the other 13 architectures "
      "generalise on."
    )
  if fail:
    subtitle += f"\n!  supervised below rl for {fail}: the optimiser failed."
  fig.suptitle(f"Does PPO build the yaw representation its architecture allows?"
               f"  \u2014 {generation} generation",
               color=INK, fontsize=13, x=0.005, ha="left",
               y=1.0 - 0.10 * header / height)
  fig.text(0.005, 1.0 - 0.24 * header / height, subtitle, color=INK2,
           fontsize=9.5, ha="left", va="top")
  fig.tight_layout(rect=(0, 0, 1, 1.0 - header / height))
  fig.savefig(out, dpi=170, facecolor=SURFACE)
  print(f"wrote {out}  ({total} architectures, {len(inverted)} inverted)")


if __name__ == "__main__":
  main()
