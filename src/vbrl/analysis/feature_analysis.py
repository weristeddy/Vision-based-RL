from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from vbrl.analysis.dataset import HELD_OUT, REFERENCE, ROOT
from vbrl.analysis.distance import cosine_distance, ratios
from vbrl.analysis.probes import errors, fit, targets

PROJECT = "eduard-nicolae-robot-learning/mjlab"
TASK_NAMES = {"lift_cube": "LiftCube", "push_t": "PushT", "push_chain": "PushChain3"}
BATCH = 64
GRADCAM_BATCH = 16
GRADCAM_FRAMES = 500
FIGURE_FRAMES = 8
FIGURE_SETS = (
  REFERENCE,
  "marble019",
  "ood_light",
  "object_color_sweep",
  "real",
  "real_sim",
)


def load_student(task_id: str, run: str) -> Any:
  from vbrl.analysis.pairs import make_env
  from vbrl.runtime import CheckpointRef, load_trained_policy

  env = make_env(task_id, REFERENCE, 1)
  ref = CheckpointRef(wandb_run_path=f"{PROJECT}/{run}")
  _, runner, _, _ = load_trained_policy(
    env, task_id=task_id, device=env.device, ref=ref
  )
  env.close()
  return runner.alg.student.eval()


def features(model: Any, images: Any) -> Any:
  import torch

  encoder, out = model.cnns["camera"], []
  with torch.no_grad():
    for start in range(0, len(images), BATCH):
      batch = torch.as_tensor(images[start : start + BATCH], device="cuda:0")
      out.append(encoder(batch).float().cpu().numpy())
  return np.concatenate(out)


def _batches(data: Any, count: int):
  import torch
  from tensordict import TensorDict

  groups = [key for key in data.files if key.startswith("obs_")]
  for start in range(0, min(len(data["obs_camera"]), count), GRADCAM_BATCH):
    part = slice(start, min(start + GRADCAM_BATCH, count))
    obs = {g[4:]: torch.as_tensor(data[g][part], device="cuda:0") for g in groups}
    yield TensorDict(obs, batch_size=[len(obs["camera"])])


def gradcam_maps(model: Any, data: Any, config: Any, count: int) -> Any:
  import torch

  from vbrl.analysis.gradcam import gradcam

  return torch.cat([gradcam(model, obs, config).cpu() for obs in _batches(data, count)])


def deletion(model: Any, data: Any, maps: Any) -> tuple[float, float]:
  import torch

  from vbrl.analysis.deletion import deletion_auc

  generator = torch.Generator().manual_seed(0)
  saliency, random, start = [], [], 0
  for obs in _batches(data, len(maps)):
    part = maps[start : start + len(obs)]
    noise = torch.rand(part.shape, generator=generator)
    saliency.append(deletion_auc(model, obs, part).cpu())
    random.append(deletion_auc(model, obs, noise).cpu())
    start += len(obs)
  return float(torch.cat(saliency).mean()), float(torch.cat(random).mean())


def analyse(task: str, policy: str, run: str) -> list[dict[str, Any]]:
  from vbrl.vision.architectures import ARCHITECTURES

  folder = ROOT / task
  names = ["train", "test", REFERENCE, *HELD_OUT, "state", "real", "real_sim"]
  data = {
    n: np.load(folder / f"{n}.npz") for n in names if (folder / f"{n}.npz").is_file()
  }
  model = load_student(f"Mjlab-{TASK_NAMES[task]}-WoodTable-{policy}-Trossen", run)
  feats = {name: features(model, d["obs_camera"]) for name, d in data.items()}
  rows = []

  def row(measure: str, where: str, value: float) -> None:
    rows.append(
      {
        "task": task,
        "policy": policy,
        "run": run,
        "measure": measure,
        "set": where,
        "value": value,
      }
    )

  for state in ("objects", "goals") if task == "push_chain" else ("objects",):
    probe = fit(feats["train"], targets(task, data["train"][state]))
    for name in ("test", REFERENCE, *HELD_OUT, "real"):
      if name in feats:
        for key, value in errors(
          task, probe.predict(feats[name]), data[name][state]
        ).items():
          row(f"probe_{state}_{key}", name, value)
  import torch

  from vbrl.analysis.gradcam import jaccard

  config = ARCHITECTURES[policy]
  for name in (REFERENCE, "real"):
    if name in data:
      maps = gradcam_maps(model, data[name], config, GRADCAM_FRAMES)
      masks = data["real_sim" if name == "real" else name]["masks"][: len(maps)]
      row("gradcam_jaccard", name, float(jaccard(maps, torch.as_tensor(masks)).mean()))
      if name == REFERENCE:
        faithful, random = deletion(model, data[name], maps)
        row("gradcam_deletion_auc", name, faithful)
        row("random_deletion_auc", name, random)
  figure = {
    n: gradcam_maps(model, data[n], config, FIGURE_FRAMES).numpy()
    for n in FIGURE_SETS
    if n in data
  }
  (folder / "gradcam").mkdir(exist_ok=True)
  np.savez_compressed(folder / "gradcam" / f"{policy}.npz", **figure)
  for name, value in ratios(
    feats[REFERENCE], {s: feats[s] for s in HELD_OUT}, feats["state"]
  ).items():
    row("distance_ratio", name, value)
  if "real" in feats:
    d_state = cosine_distance(feats[REFERENCE], feats["state"]).mean()
    row(
      "distance_ratio",
      "real",
      float(cosine_distance(feats["real_sim"], feats["real"]).mean() / d_state),
    )
  return rows


def main(argv: Any = None) -> None:
  parser = argparse.ArgumentParser(
    description="Probes, Grad-CAM and feature distance per student."
  )
  parser.add_argument("task", choices=sorted(TASK_NAMES))
  parser.add_argument(
    "--policy", action="append", help="NAME=RUN; default the deployed best seeds"
  )
  args = parser.parse_args(argv)
  if args.policy:
    policies = dict(item.split("=") for item in args.policy)
  else:
    policies = json.loads(
      Path("ckpts/deployment", args.task, "policies.json").read_text()
    )
  rows = [
    r for policy, run in policies.items() for r in analyse(args.task, policy, run)
  ]
  path = ROOT / args.task / "measures.csv"
  with path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  for r in rows:
    print(f"{r['policy']:24s} {r['measure']:30s} {r['set']:20s} {r['value']:.3f}")
  print(path)


if __name__ == "__main__":
  main()


__all__ = ["analyse", "features", "load_student", "main"]
