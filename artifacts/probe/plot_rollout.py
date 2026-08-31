"""What the trained policies do, and what their encoders still encode."""
from __future__ import annotations

import sys
from pathlib import Path


import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#b8b6ad", "#e6e5df"
ADAPTER_COLOR = {
  "Flatten": "#2a78d6",
  "SpatialSoftmax": "#eb6834",
  "Afa": "#1baf7a",
  "Linear": "#eda100",
  "LocalGrid": "#e87ba4",
}
ENCODERS = ("NatureCnn", "CompactVit", "DinoV2ViTS14", "R3MResNet50",
            "R3MResNet50L3")
TITLES = {
  "NatureCnn": "NatureCnn (scratch)",
  "CompactVit": "CompactVit (scratch)",
  "DinoV2ViTS14": "DINOv2 ViT-S/14 (frozen)",
  "R3MResNet50": "R3M ResNet-50 layer4 (frozen)",
  "R3MResNet50L3": "R3M ResNet-50 layer3 (frozen)",
}
CHANCE = 90.0


def adapter_of(token):
  tail = token.split("-", 1)[1]
  for key in ADAPTER_COLOR:
    if tail.startswith(key):
      return key
  raise KeyError(token)


def style(ax):
  ax.set_facecolor(SURFACE)
  ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
  ax.set_axisbelow(True)
  ax.tick_params(colors=INK2, labelsize=9)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color(MUTED)


def traces(data, arches, out, generation):
  fig, axes = plt.subplots(
    2, len(ENCODERS), figsize=(18.5, 6.6), facecolor=SURFACE,
    sharex=True, sharey="row"
  )
  for col, encoder in enumerate(ENCODERS):
    for row, (key, label) in enumerate(
      ((("yaw_err_trace"), "yaw error (deg)"), ("overlap_trace", "overlap"))
    ):
      ax = axes[row, col]
      style(ax)
      if row == 0:
        ax.axhline(CHANCE, color=MUTED, linewidth=1.4, linestyle=(0, (4, 3)))
      for arch in arches:
        if not arch.startswith(encoder + "-"):
          continue
        colour = ADAPTER_COLOR[adapter_of(arch)]
        band_key = f"{arch}/{key.replace('_trace', '_bands')}"
        if band_key in data.files:
          # p25/p50/p75 across every episode; the shaded band is the spread,
          # the line is the median (not the mean, which one outlier can drag).
          band = np.degrees(data[band_key]) if row == 0 else data[band_key]
          if np.all(np.isnan(band[:, -1])):     # the universal time-out step
            band = band[:, :-1]
          p25, p50, p75 = band
          steps = np.arange(p50.shape[0])
          ax.fill_between(steps, p25, p75, color=colour, alpha=0.16,
                          linewidth=0, zorder=2)
          ax.plot(steps, p50, color=colour, linewidth=2.0, zorder=3)
        else:
          y = data[f"{arch}/{key}"]
          if row == 0:
            y = np.degrees(y)
          ax.plot(np.arange(len(y)), y, color=colour, linewidth=2.0, zorder=3)
      if row == 0:
        ax.set_ylim(0, 185)
        ax.set_title(TITLES[encoder], color=INK, fontsize=10.5, pad=8, loc="left")
        # How many episodes are still running: a trace averaged over survivors
        # means something different once the T has been shoved off the table.
        for arch in arches:
          if not arch.startswith(encoder + "-"):
            continue
          if f"{arch}/alive_frac" in data.files:
            alive = data[f"{arch}/alive_frac"]
            reached = alive[-2] if len(alive) > 1 else alive[-1]
            if reached < 0.98:
              ax.annotate(f"{reached*100:.0f}% of episodes reach the time limit",
                          xy=(0.02, 0.04), xycoords="axes fraction",
                          fontsize=8, color="#e34948")
            break
      else:
        ax.set_ylim(0, 0.55)
        ax.set_xlabel("step within episode", color=INK2, fontsize=9.5)
      if col == 0:
        ax.set_ylabel(label, color=INK2, fontsize=10)

  axes[0, 0].annotate("chance", xy=(4, CHANCE), xytext=(0, 5),
                      textcoords="offset points", fontsize=8.5, color=INK2)
  handles = [
    plt.Line2D([], [], color=c, linewidth=2.2, label=n)
    for n, c in ADAPTER_COLOR.items()
  ]
  legend = fig.legend(handles=handles, loc="lower center", ncols=5, frameon=False,
                      fontsize=10, bbox_to_anchor=(0.5, 0.005))
  for text in legend.get_texts():
    text.set_color(INK)
  fig.suptitle(
    f"Within-episode behaviour — {generation} generation.  Median over "
    "768 episodes with the interquartile band; chance yaw is 90°.",
    color=INK, fontsize=13, x=0.005, ha="left", y=0.99,
  )
  fig.tight_layout(rect=(0, 0.06, 1, 0.94))
  fig.savefig(out, dpi=170, facecolor=SURFACE)
  print(f"wrote {out}")


def main():
  source = sys.argv[1] if len(sys.argv) > 1 else "artifacts/probe/rollout_SlowGoal.npz"
  out = sys.argv[2] if len(sys.argv) > 2 else "artifacts/probe/rollout_traces.png"
  data = np.load(source)
  arches = sorted({k.split("/")[0] for k in data.files})
  generation = Path(source).stem.replace("rollout_", "")
  print(f"{len(arches)} architectures from {source}")
  traces(data, arches, out, generation)


if __name__ == "__main__":
  main()
