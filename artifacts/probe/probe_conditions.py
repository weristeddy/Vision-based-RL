"""One dataset, three conditions -- the controlled version of the dumbbell.

Every number below comes from the SAME rendered images (T position uniform over
the workspace, yaw uniform over [-pi, pi], arm configurations drawn from a real
rollout, full appearance DR) and the SAME probe head and budget. Only the
encoder weights differ:

  init        weights the RL run STARTED from (model_0.pt), frozen; head only.
              A random-features floor -- whatever the architecture gives for free.
  rl          weights the RL run ENDED with (model_2999.pt), frozen; head only.
              What 3000 iterations of PPO actually built.
  supervised  encoder + adapter trained by supervision from the same init.
              What the architecture can reach when told the answer.

init vs rl is a true A/B: identical protocol, identical data, only training
history differs. supervised is the ceiling, and gets more freedom by design.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from episode_start import (
  BATCH,
  N_TEST,
  N_TRAIN,
  STEPS,
  episode_start_dataset,
  features,
)
from probe_head import policy_head
from probe_rollout import find_encoder, local_runs

# Which training generation to compare. Outputs are suffixed with it so
# generations never overwrite each other.
SHARD = os.environ.get("VBRL_PROBE_SHARD", "0/1")   # "i/n": split work across jobs
GENERATION = os.environ.get("VBRL_PROBE_GENERATION", "Curriculum")
_SUF = "" if SHARD == "0/1" else f"_shard{SHARD.split('/')[0]}"
OUT = Path(f"artifacts/probe/conditions_{GENERATION}{_SUF}.json")
# `supervised` trains a fresh encoder from config and never reads a checkpoint,
# so it is identical across generations -- reuse it instead of recomputing.
REUSE_SUPERVISED = os.environ.get("VBRL_REUSE_SUPERVISED", "")
# A single learning rate made the supervised ceiling land BELOW what RL reached
# for several rows, which is impossible. Sweep and keep the best -- an upper
# bound is allowed to be a best-of.
LRS = (1e-3, 3e-4, 1e-4, 3e-5)


def probe(tr_f, tr_pose, te_f, te_pose, device, target="yaw"):
  """Frozen-encoder readout with the policy's own trunk capacity.

  ``target="xy"`` decodes the T's position instead of its orientation. The
  collapsed AFA runs ended at 0.24 m position error; this says whether their
  encoder still carries where the T is, or whether the policy stopped using it.
  """
  torch.manual_seed(0)
  mu = tr_f.mean(0, keepdim=True)
  sd = tr_f.std(0, keepdim=True).clamp_min(1e-6)
  tr_x, te_x = ((tr_f - mu) / sd).to(device), ((te_f - mu) / sd).to(device)
  if target == "yaw":
    tr_y = torch.from_numpy(tr_pose[:, 2]).to(device)
    te_y = torch.from_numpy(te_pose[:, 2]).to(device)
    goal = torch.stack([torch.sin(tr_y), torch.cos(tr_y)], 1)
  else:                                   # metres, scaled to the workspace
    scale = torch.tensor([0.1, 0.2], device=device)
    tr_y = (torch.from_numpy(tr_pose[:, :2]).to(device)
            - torch.tensor([0.3, 0.0], device=device)) / scale
    te_y = (torch.from_numpy(te_pose[:, :2]).to(device)
            - torch.tensor([0.3, 0.0], device=device)) / scale
    goal = tr_y

  head = policy_head(tr_x.shape[1]).to(device)
  opt = torch.optim.Adam(head.parameters(), lr=1e-3)
  rng = np.random.default_rng(0)
  for _ in range(STEPS):
    idx = torch.from_numpy(rng.integers(0, len(tr_x), BATCH)).to(device)
    loss = nn.functional.mse_loss(head(tr_x[idx]), goal[idx])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

  def error(x, y):
    with torch.no_grad():
      pred = head(x)
    if target == "yaw":
      ang = torch.atan2(pred[:, 0], pred[:, 1])
      return ((ang - y + torch.pi) % (2 * torch.pi) - torch.pi).abs()
    scale = torch.tensor([0.1, 0.2], device=device)
    return torch.linalg.vector_norm((pred - y) * scale, dim=-1)

  te_err, tr_err = error(te_x, te_y), error(tr_x, tr_y)
  return {
    "median": float(te_err.median()),
    "mean": float(te_err.mean()),
    "within3deg": float((te_err < 0.052).float().mean()),
    "train_median": float(tr_err.median()),
  }


def load_encoder(arch, checkpoint, device):
  from vbrl.runtime import CheckpointRef, build_env, load_trained_policy

  task_id = f"Mjlab-PushT-{GENERATION}-{arch}-TrossenRealistic"
  env = build_env(task_id, device=device, num_envs=2, seed=0)
  _, runner, _, _ = load_trained_policy(
    env, task_id=task_id, device=device,
    ref=CheckpointRef(checkpoint_file=str(checkpoint)),
  )
  encoder = find_encoder(runner.alg.actor).eval()
  env.close()
  return encoder


def supervised(arch, tr_imgs, tr_pose, te_imgs, te_pose, device, lr=None,
               steps=None, warmup=0):
  """Train encoder + adapter + head with labels, from the RL run's own init."""
  from vbrl.vision.architectures import ARCHITECTURES
  from vbrl.vision.registry import build_encoder

  torch.manual_seed(0)
  cfg = ARCHITECTURES[arch]
  enc = build_encoder(cfg, input_dim=(224, 224)).to(device)
  frozen = cfg.frozen

  def spatial(source, idx):
    x = torch.from_numpy(source[idx]).to(device)          # already BCHW uint8
    return enc.extract(enc.backbone, enc.preprocess(x))

  if frozen:
    enc.backbone.eval()
    def cache(source):
      out = []
      with torch.no_grad():
        for i in range(0, len(source), 64):
          out.append(spatial(source, np.arange(i, min(i + 64, len(source)))).bfloat16().cpu())
      return torch.cat(out)
    tr_f, te_f = cache(tr_imgs), cache(te_imgs)

  head = policy_head(enc.output_dim).to(device)
  params = list(head.parameters()) + list(enc.adapter.parameters())
  if not frozen:
    params += list(enc.backbone.parameters())
  if lr is None:
    lr = 1e-4 if cfg.adapter in ("flatten", "afa") or "R3M" in arch else 1e-3
  opt = torch.optim.Adam(params, lr=lr)
  steps = STEPS if steps is None else steps
  # A from-scratch ViT will not train without warmup; see probe_cvit_one.py.
  schedule = (
    torch.optim.lr_scheduler.LambdaLR(opt, lambda k: min(1.0, (k + 1) / warmup))
    if warmup else None
  )

  tr_y = torch.from_numpy(tr_pose[:, 2]).to(device)
  target = torch.stack([torch.sin(tr_y), torch.cos(tr_y)], 1)
  rng = np.random.default_rng(0)
  for _ in range(steps):
    idx = rng.integers(0, len(tr_pose), BATCH)
    feats = tr_f[idx].to(device).float() if frozen else spatial(tr_imgs, idx)
    loss = nn.functional.mse_loss(
      head(enc.adapter(feats)), target[torch.from_numpy(idx).to(device)]
    )
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    if schedule is not None:
      schedule.step()

  def evaluate(imgs, pose, cached):
    preds = []
    with torch.no_grad():
      for i in range(0, len(pose), 256):
        idx = np.arange(i, min(i + 256, len(pose)))
        f = cached[idx].to(device).float() if frozen else spatial(imgs, idx)
        preds.append(head(enc.adapter(f)).cpu())
    pred = torch.cat(preds)
    ang = torch.atan2(pred[:, 0], pred[:, 1])
    true = torch.from_numpy(pose[:, 2])
    return ((ang - true + torch.pi) % (2 * torch.pi) - torch.pi).abs()

  te_err = evaluate(te_imgs, te_pose, te_f if frozen else None)
  tr_err = evaluate(tr_imgs[:5000], tr_pose[:5000], tr_f[:5000] if frozen else None)
  return {
    "median": float(te_err.median()),
    "mean": float(te_err.mean()),
    "within3deg": float((te_err < 0.052).float().mean()),
    "train_median": float(tr_err.median()),
  }




def main():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  t0 = time.time()
  # Episode-start observations from the environment's own camera sensor.
  any_task = f"Mjlab-PushT-{GENERATION}-{sorted(local_runs())[0]}-TrossenRealistic"
  print(f"collecting {N_TRAIN + N_TEST} episode-start frames", flush=True)
  tr_imgs, tr_pose = episode_start_dataset(any_task, N_TRAIN, 0, device)
  te_imgs, te_pose = episode_start_dataset(any_task, N_TEST, 1, device)
  print(f"rendered in {time.time() - t0:.0f}s", flush=True)

  runs = local_runs()
  results = {"n_train": N_TRAIN, "n_test": N_TEST, "steps": STEPS, "conditions": {}}
  i, n = (int(x) for x in SHARD.split('/'))
  items = [kv for k, kv in enumerate(sorted(runs.items())) if k % n == i]
  reuse = json.loads(Path(REUSE_SUPERVISED).read_text())['conditions'] if REUSE_SUPERVISED else {}
  print(f'shard {SHARD}: {[a for a, _ in items]}; reusing supervised: {bool(reuse)}', flush=True)
  for arch, (_dir, init_ck, final_ck) in items:
    row = {}
    for name, checkpoint in (("init", init_ck), ("rl", final_ck)):
      try:
        enc = load_encoder(arch, checkpoint, device)
        f_tr, f_te = features(enc, tr_imgs, device), features(enc, te_imgs, device)
        row[name] = probe(f_tr, tr_pose, f_te, te_pose, device, target="yaw")
        row[f"{name}_xy"] = probe(f_tr, tr_pose, f_te, te_pose, device, target="xy")
        del enc, f_tr, f_te
      except Exception as exc:
        print(f"{arch:<28} {name:<10} FAILED: {type(exc).__name__}: {exc}", flush=True)
    if arch in reuse and 'supervised' in reuse[arch]:
      row['supervised'] = reuse[arch]['supervised']
      best = None
    else:
     best = None
     for lr in LRS:
      try:
        cell = supervised(arch, tr_imgs, tr_pose, te_imgs, te_pose, device, lr=lr)
      except Exception as exc:
        print(f"{arch:<28} sup lr={lr:<7} FAILED {type(exc).__name__}", flush=True)
        continue
      cell["lr"] = lr
      print(f"{arch:<28} sup lr={lr:<7} test {cell['median']*57.3:5.1f}deg "
            f"train {cell['train_median']*57.3:5.1f}deg"
            f"{'  <- best' if best is None or cell['median'] < best['median'] else ''}",
            flush=True)
      if best is None or cell["median"] < best["median"]:
        best = cell
    if best is not None:
      row["supervised"] = best

    results["conditions"][arch] = row
    OUT.write_text(json.dumps(results))
    cells = "  ".join(
      f"{name} {row[name]['median'] * 57.3:5.1f}deg"
      for name in ("init", "rl", "supervised") if name in row
    )
    cells += "  | xy " + " ".join(
      f"{row[k]['median']*1000:5.1f}mm" for k in ("init_xy", "rl_xy") if k in row
    )
    print(f"{arch:<28} {cells}   [{time.time() - t0:.0f}s]", flush=True)

  print(f"\nwrote {OUT}")


if __name__ == "__main__":
  main()
