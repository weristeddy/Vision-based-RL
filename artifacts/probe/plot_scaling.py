"""Yaw decodability against supervision budget, faceted by encoder.

Usage: plot_scaling.py [results.json] [output.png]
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#b8b6ad", "#e6e5df"
# Colour follows the adapter, held fixed across every panel.
ADAPTER_COLOR = {
  "Flatten": "#2a78d6",
  "SpatialSoftmax": "#eb6834",
  "Afa": "#1baf7a",
  "Linear": "#eda100",
  "LocalGrid": "#e87ba4",
}
ENCODERS = ("NatureCnn", "CompactVit", "DinoV2ViTS14", "R3MResNet50", "R3MResNet50L3")
TITLES = {
  "NatureCnn": "NatureCnn  (scratch, 24×24)",
  "CompactVit": "CompactVit  (scratch, 14×14)",
  "DinoV2ViTS14": "DINOv2 ViT-S/14  (frozen, 16×16)",
  "R3MResNet50": "R3M ResNet-50 layer4  (frozen, 7×7)",
  "R3MResNet50L3": "R3M ResNet-50 layer3  (frozen, 14×14)",
}
CHANCE, SUCCESS = 90.0, 2.5   # degrees


def adapter_of(token):
  tail = token.split("-", 1)[1]
  for key in ADAPTER_COLOR:
    if tail.startswith(key):
      return key, tail
  raise KeyError(token)


def main():  # noqa: C901
  source = sys.argv[1] if len(sys.argv) > 1 else "artifacts/probe/scaling.json"
  out = sys.argv[2] if len(sys.argv) > 2 else "artifacts/probe/scaling.png"
  data = json.loads(open(source).read())
  sizes = np.array(data["sizes"])

  fig, axes = plt.subplots(
    1, len(ENCODERS), figsize=(18.5, 4.3), facecolor=SURFACE, sharey=True
  )
  for ax, encoder in zip(axes, ENCODERS):
    ax.set_facecolor(SURFACE)
    ax.axhline(CHANCE, color=MUTED, linewidth=1.4, linestyle=(0, (4, 3)), zorder=1)
    ax.axhline(SUCCESS, color=INK2, linewidth=1.4, linestyle=(0, (1, 2)), zorder=1)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    ends = []
    for token, runs in data["yaw"].items():
      if not token.startswith(encoder + "-"):
        continue
      key, _ = adapter_of(token)
      y = np.degrees([r["median"] for r in runs])
      # A row whose TRAIN error is at chance never fit its own data: that is an
      # optimiser failure, not a capability measurement, so it is drawn dashed
      # and flagged rather than read as a result.
      failed = bool(runs[-1].get("collapsed", False))
      ax.plot(sizes, y, color=ADAPTER_COLOR[key], linewidth=2.0,
              linestyle=(0, (2, 2)) if failed else "-",
              marker="x" if failed else "o", markersize=7 if failed else 6.5,
              markeredgecolor=SURFACE if not failed else ADAPTER_COLOR[key],
              markeredgewidth=1.5, alpha=0.55 if failed else 1.0, zorder=3)
      ends.append((y[-1], key, failed))

    # Stagger end labels that would print on top of each other.
    ends.sort()
    offsets, previous, level = [], -1e9, 0
    for value, _, _ in ends:
      level = 0 if value - previous > 12 else level + 11
      offsets.append(level)
      previous = value
    for (value, key, failed), shift in zip(ends, offsets):
      ax.annotate(
        "did not fit" if failed else f"{value:.0f}°",
        xy=(sizes[-1], value), xytext=(7, shift), textcoords="offset points",
        fontsize=8.5 if failed else 9.5, color=ADAPTER_COLOR[key],
        va="center", fontweight="normal" if failed else "bold",
        alpha=0.75 if failed else 1.0,
      )

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=10, linscale=0.9)
    ax.set_yticks([0, 2.5, 10, 30, 90])
    ax.set_yticklabels(["0", "2.5", "10", "30", "90"])
    ax.set_ylim(0, 130)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{s // 1000}k" for s in sizes])
    ax.set_xlim(sizes[0] * 0.8, sizes[-1] * 1.6)
    ax.set_title(TITLES[encoder], color=INK, fontsize=10.5, pad=8, loc="left")
    ax.tick_params(colors=INK2, labelsize=9)
    for side in ("top", "right"):
      ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
      ax.spines[side].set_color(MUTED)
  axes[0].set_ylabel("median yaw error (degrees)", color=INK2, fontsize=10)
  axes[0].annotate("chance", xy=(sizes[0] * 0.85, CHANCE), xytext=(0, 6),
                   textcoords="offset points", fontsize=8.5, color=INK2)
  axes[-1].annotate("needed for 0.98 success", xy=(sizes[0] * 0.85, SUCCESS),
                   xytext=(0, 6), textcoords="offset points", fontsize=8.5,
                   color=INK2)

  # One legend for the whole figure: colour identifies the adapter family, and
  # the head count (Afa1 / Afa6 / …) follows from the encoder's channel width.
  handles = [
    plt.Line2D([], [], color=color, linewidth=2.0, marker="o", markersize=6.5,
               markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
    for name, color in ADAPTER_COLOR.items()
  ]
  legend = fig.legend(
    handles=handles, loc="lower center", ncols=5, frameon=False, fontsize=10,
    handlelength=1.8, bbox_to_anchor=(0.5, 0.055),
  )
  for text in legend.get_texts():
    text.set_color(INK)
  fig.supxlabel("labelled training images (equal gradient-step budget)",
                color=INK2, fontsize=10, y=0.005)
  fig.suptitle(
    "Can the policy's vision stack read the T's yaw? Supervised upper bound.",
    color=INK, fontsize=13, x=0.005, ha="left", y=0.995,
  )
  fig.text(
    0.005, 0.95,
    "Arm held at the home pose. With arm configurations from real rollouts the same "
    "probes are worse \u2014 the gripper occludes the T while pushing\n"
    "(DINOv2-SpatialSoftmax 1.4\u00b0 \u2192 3.7\u00b0, NatureCnn-Flatten 1.3\u00b0 "
    "\u2192 7.8\u00b0). CompactVit rows diverged at 40k+ (optimiser failure, not a "
    "capability limit).",
    color=INK2, fontsize=9.5, ha="left", va="top",
  )
  fig.tight_layout(rect=(0, 0.13, 1, 0.865))
  fig.savefig(out, dpi=170, facecolor=SURFACE)
  print(f"wrote {out}")


if __name__ == "__main__":
  main()
