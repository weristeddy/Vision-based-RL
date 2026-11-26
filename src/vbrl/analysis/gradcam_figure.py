from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from vbrl.analysis.dataset import ROOT
from vbrl.analysis.final_evaluation import ENCODERS, INK, _setup

TITLES = {
  "wood049": "Reference",
  "marble019": "Marble",
  "ood_light": "Light",
  "object_color_sweep": "Object color",
  "real": "Real",
  "real_sim": "Sim twin",
}
FIGURES = {
  "sim": [("wood049", i) for i in range(4)],
  "scenes": [
    (s, 0) for s in ("wood049", "marble019", "ood_light", "object_color_sweep")
  ],
  "real": [(s, i) for i in range(2) for s in ("real", "real_sim")],
}


def overlay(image: Any, heat: Any) -> Any:
  import matplotlib

  heat = (heat - heat.min()) / max(heat.max() - heat.min(), 1e-12)
  colored = matplotlib.colormaps["jet"](heat)[..., :3]
  return 0.5 * image / 255.0 + 0.5 * colored


def plot(task: str, name: str, columns: list[tuple[str, int]], output: Path) -> Path:
  plt = _setup()
  folder = ROOT / task
  policies = [p for p in ENCODERS if (folder / "gradcam" / f"{p}.npz").is_file()]
  maps = {p: np.load(folder / "gradcam" / f"{p}.npz") for p in policies}
  data = {s: np.load(folder / f"{s}.npz") for s in {s for s, _ in columns}}
  figure, axes = plt.subplots(
    len(policies) + 1,
    len(columns),
    figsize=(1.25 * len(columns) + 1.0, 1.25 * (len(policies) + 1)),
    squeeze=False,
  )
  for column, (scene, index) in enumerate(columns):
    image = data[scene]["obs_camera"][index].transpose(1, 2, 0)
    mask_scene = "real_sim" if scene == "real" else scene
    mask = np.load(folder / f"{mask_scene}.npz")["masks"][index]
    axes[0, column].imshow(image)
    axes[0, column].contour(mask, levels=[0.5], colors="#ff00ff", linewidths=0.6)
    axes[0, column].set_title(TITLES[scene], fontsize=8, color=INK)
    for row, policy in enumerate(policies, 1):
      axes[row, column].imshow(overlay(image, maps[policy][scene][index]))
  for row, label in enumerate(["Input"] + [ENCODERS[p][0] for p in policies]):
    axes[row, 0].set_ylabel(
      label, fontsize=8, color=INK, rotation=0, ha="right", va="center"
    )
  for axis in axes.flat:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
      spine.set_visible(False)
  figure.subplots_adjust(wspace=0.04, hspace=0.04, left=0.2)
  output.mkdir(parents=True, exist_ok=True)
  path = output / f"{task}_gradcam_{name}"
  for suffix in (".pdf", ".png"):
    figure.savefig(path.with_suffix(suffix), dpi=200, bbox_inches="tight")
  plt.close(figure)
  return path


def main(argv: Any = None) -> None:
  parser = argparse.ArgumentParser(
    description="Grad-CAM maps of every student on the same frames."
  )
  parser.add_argument("task")
  parser.add_argument("--output", type=Path, default=Path("reports/features"))
  args = parser.parse_args(argv)
  for name, columns in FIGURES.items():
    if all((ROOT / args.task / f"{s}.npz").is_file() for s, _ in columns):
      print(plot(args.task, name, columns, args.output))


if __name__ == "__main__":
  main()


__all__ = ["FIGURES", "main", "overlay", "plot"]
