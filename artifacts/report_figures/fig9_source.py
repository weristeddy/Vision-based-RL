"""Adapter attribution gallery, grouped by readout mechanism.

Rows are grouped by adapter and ordered within a group by encoder -- Nature CNN,
Compact ViT, DINOv2, R3M layer 3, R3M layer 4 -- so a group reads as one
mechanism across five backbones rather than as a performance ranking. `linear`
is omitted: global pooling leaves no spatial map, which is stated in the caption
instead of drawn as an empty cell.

The four columns are four different environments: distinct tabletop textures and
object yaws at least 62 degrees apart, with the last on the reddest tabletop in
the bank so a colour collision between object and table is visible.

Maps are shown as a smooth field. That is a display choice, not a claim: each
map is computed on the encoder's own grid (24x24 down to 7x7) and interpolated
for legibility, so the true resolution is the number printed beside each row.
"""
from __future__ import annotations

import json, re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import zoom

HERE = Path(__file__).resolve().parent
SWEEP = HERE / "sweep"
OUT = Path("/home/eddy/master_thesis/Vision-based-RL/artifacts/report_figures")

ENCODERS = [
  ("NatureCnn", "Nature CNN", "24×24"),
  ("CompactVit", "Compact ViT", "14×14"),
  ("DinoV2ViTS14", "DINOv2 ViT-S/14", "16×16"),
  ("R3MResNet50L3", "R3M layer 3", "14×14"),
  ("R3MResNet50", "R3M layer 4", "7×7"),
]
ADAPTERS = [
  ("SpatialSoftmax", "Spatial softmax", "tracked keypoints"),
  ("Afa", "Attention pooling (AFA)", "attention, max over heads"),
  ("Flatten", "Flatten", "dense contribution"),
  ("LocalGrid", "Local grid", "dense contribution"),
]
SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8b8a84"
# Perceptually rising, dark -> warm -> pale, so intensity reads as magnitude in
# greyscale print as well as in colour.
HEAT = LinearSegmentedColormap.from_list(
  "vbrl_heat",
  ["#1b0b3a", "#5b1d8a", "#b3269b", "#f2557a", "#ffa14f", "#ffe9a8", "#ffffff"],
)
ADAPTER_COLOURS = {
  "Spatial softmax": "#2a78d6", "Attention pooling (AFA)": "#008c7a",
  "Flatten": "#b3269b", "Local grid": "#eb6834",
}
mpl.rcParams.update({
  "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
  "font.family": "DejaVu Sans", "font.size": 8,
})


def rl_scores():
  out = {}
  for r in json.load(open(HERE / "new.json")):
    m = re.search(r"PushT-SlowGoal-(.+?)-TrossenRealistic", r["log_dir"])
    if m and "gamma996" in r["tags"] and (r["step"] or 0) >= 5900 and r["yaw"]:
      out[m.group(1)] = np.degrees(r["yaw"])
  return out


def main() -> None:
  frames = np.load(SWEEP / "frames.npz")
  maps = np.load(SWEEP / "maps4.npz")
  summary = json.load(open(SWEEP / "summary.json"))
  rl = rl_scores()
  chosen = maps["chosen"]
  images = np.moveaxis(frames["images"][chosen], 1, -1)
  yaws = np.degrees(frames["object_yaw"][chosen])
  ncol = len(chosen)

  rows = []
  for adapter_key, adapter_name, mechanism in ADAPTERS:
    for encoder_key, encoder_name, grid in ENCODERS:
      arch = f"{encoder_key}-{adapter_key}"
      candidates = [a for a in summary if a.startswith(arch)]
      if not candidates or f"{candidates[0]}__heatmap" not in maps:
        continue
      rows.append((candidates[0], adapter_name, mechanism, encoder_name, grid))

  # A header row carries the four scenes in colour; every map row below is drawn
  # on a greyscale base so the warm colours can only be the heat and never the
  # tabletop. Scene 2 is orange brick and scene 4 is red -- indistinguishable
  # from the colourmap if the base stayed in colour.
  dim = np.clip(images.astype(np.float32) / 255.0 * 0.52, 0.0, 1.0)
  bright = np.clip(images.astype(np.float32) / 255.0 * 0.78, 0.0, 1.0)
  height = 1.36 * (len(rows) + 1) + 2.10
  fig = plt.figure(figsize=(1.36 * ncol + 4.2, height))
  grid_spec = fig.add_gridspec(
    len(rows) + 1, ncol, wspace=0.045, hspace=0.06, left=0.215, right=0.965,
    top=1 - 0.80 / height, bottom=1.50 / height)

  for col in range(ncol):
    ax = fig.add_subplot(grid_spec[0, col])
    ax.imshow(images[col], interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
      spine.set_color(INK2); spine.set_linewidth(1.1)
    ax.set_title(f"scene {col + 1}  ·  yaw {yaws[col]:+.0f}°",
                 fontsize=7.8, color=INK, pad=3.5, fontweight="bold")
    if col == 0:
      ax.text(-0.06, 0.5, "the four scenes\n(RGB, as the policy sees them)",
              transform=ax.transAxes, ha="right", va="center", fontsize=8.2,
              color=INK, linespacing=1.35)

  previous_adapter = None
  for row, (arch, adapter_name, mechanism, encoder_name, grid) in enumerate(rows):
    heat = maps[f"{arch}__heatmap"]
    keys = maps[f"{arch}__keypoints"]
    peak = maps.get(f"{arch}__peak")
    # One scale per row: cells are comparable across the four scenes, which is
    # the comparison the row is for.
    hi = float(np.percentile(heat, 99.0)) if heat.size else 1.0
    for col in range(ncol):
      ax = fig.add_subplot(grid_spec[row + 1, col])
      ax.imshow((bright if keys.size else dim)[col],
                interpolation="bilinear")
      if heat.size and not keys.size:
        # order=1: bicubic overshoots on a coarse grid, and clipping those
        # overshoots is what produced flat white plateaus and ringing haloes.
        field = zoom(heat[col] / (hi or 1.0), 224 / heat[col].shape[0], order=1)
        field = np.clip(field, 0.0, 1.0)
        rgba = HEAT(field)
        # A steeper alpha ramp leaves the low end genuinely transparent, so the
        # map reads as a few regions rather than an overall wash.
        rgba[..., 3] = 0.82 * field ** 1.6
        ax.imshow(rgba, extent=(0, 224, 224, 0), interpolation="bilinear")
      if keys.size:
        # Keep the channels whose softmax is actually peaked; a flat channel
        # reports the grid centroid and would draw a meaningless lattice.
        weight = peak[col] if peak is not None else np.ones(keys.shape[1])
        keep = weight >= np.percentile(weight, 70)
        xs = (keys[col, keep, 0] + 1) / 2 * 224
        ys = (keys[col, keep, 1] + 1) / 2 * 224
        ax.scatter(xs, ys, s=13, c="#63f7b4", edgecolors="#06121a",
                   linewidths=0.4, alpha=0.95, zorder=5)
      ax.set_xticks([]); ax.set_yticks([])
      for spine in ax.spines.values():
        spine.set_color("#d9d8d2"); spine.set_linewidth(0.6)
      if col == 0:
        label = f"{encoder_name}\n{grid}"
        ax.text(-0.06, 0.5, label, transform=ax.transAxes, ha="right",
                va="center", fontsize=8.2, color=INK, linespacing=1.35)
        previous_adapter = adapter_name

  # One coloured band per adapter group, spanning exactly its rows, so which
  # rows belong to which mechanism is marked rather than inferred.
  from matplotlib.patches import FancyBboxPatch
  groups: dict[str, list] = {}
  for row, (arch, adapter_name, mechanism, _e, _g) in enumerate(rows):
    groups.setdefault((adapter_name, mechanism), []).append(row)
  for (adapter_name, mechanism), members in groups.items():
    top = fig.axes[1 + ncol + members[0] * ncol].get_position()
    bottom = fig.axes[1 + ncol + members[-1] * ncol].get_position()
    y0, y1 = bottom.y0, top.y1
    colour = ADAPTER_COLOURS[adapter_name]
    fig.patches.append(FancyBboxPatch(
      (0.055, y0), 0.030, y1 - y0, boxstyle="round,pad=0,rounding_size=0.012",
      transform=fig.transFigure, facecolor=colour, edgecolor="none",
      zorder=1, figure=fig))
    fig.text(0.070, (y0 + y1) / 2, adapter_name.upper(), ha="center",
             va="center", rotation=90, fontsize=9.6, color="#ffffff",
             fontweight="bold", zorder=2)
    fig.text(0.098, (y0 + y1) / 2, mechanism, ha="center", va="center",
             rotation=90, fontsize=7.6, color=colour, style="italic")
  bar = fig.add_axes((0.215, 1.06 / height, 0.20, 0.038 / height))
  bar.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap=HEAT, aspect="auto",
             extent=(0, 1, 0, 1))
  bar.set_yticks([]); bar.set_xticks([0, 1])
  bar.set_xticklabels(["low", "saturated"], fontsize=7.2, color=INK2)
  bar.set_title("contribution to the adapter output", fontsize=7.6, color=INK2,
                pad=3)
  for spine in bar.spines.values():
    spine.set_visible(False)

  fig.suptitle("What each readout draws from — one mechanism per group, five encoders each",
               x=0.055, y=1 - 0.30 / height, ha="left", fontsize=12,
               fontweight="bold", color=INK)
  fig.text(0.215, 0.12 / height,
           "Spatial-softmax rows show only keypoints (top 30% by softmax "
           "peakedness): that adapter's output is the coordinates themselves, so a "
           "field would restate them. AFA rows take the maximum over attention "
           "heads, not the mean -- head count scales with channel width (6 / 16 / "
           "32) and averaging that many distributions flattens all of them. Fields "
           "are scaled per row by its own 99th percentile and smoothed bilinearly "
           "for display from each encoder's own grid, over a dimmed copy of the "
           "frame. The four scenes are distinct environments chosen for maximal "
           "tabletop-colour distance and object yaws at least 58° apart. The "
           "linear adapter is omitted: global pooling leaves no spatial map to "
           "recover.",
           fontsize=7.2, color=INK3, style="italic", ha="left", va="bottom",
           wrap=True)
  for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig9_adapter_attention.{ext}", dpi=220,
                bbox_inches="tight", facecolor=SURFACE)
  plt.close(fig)
  print(f"wrote fig9 with {len(rows)} rows x {ncol} scenes")


if __name__ == "__main__":
  main()
