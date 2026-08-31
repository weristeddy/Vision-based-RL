"""Upper bound on yaw decodability, as a function of how much data it is given.

Same idea as the first upper-bound probe, with the confound removed: every
(architecture, dataset size) pair gets the SAME number of gradient steps, so the
curve isolates data quantity rather than optimisation budget. Saves JSON for
plotting.

Frozen backbone features are cached in bfloat16 -- float16 overflows on R3M's
ResNet-50 activations, which is what produced the NaNs in the first pass.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from episode_start import episode_start_dataset
from probe_head import policy_head

N_TEST = 5000
SIZES = (10_000, 40_000, 150_000)
STEPS, BATCH = 12_000, 256
TASK = "Mjlab-PushT-FrontCam-DinoV2ViTS14-Afa6-TrossenRealistic"
CAMERA = "robot/external_front_cam"
_SH = os.environ.get("VBRL_PROBE_SHARD", "0/1")
OUT = Path("artifacts/probe/scaling.json" if _SH == "0/1"
           else f"artifacts/probe/scaling_shard{_SH.split(chr(47))[0]}.json")

# Heaviest trainable heads first: they are the ones 35k labels under-served, so
# they must land even if the job runs out of wall clock.
ORDER = (
  "R3MResNet50-Afa32",          # 13.2M trainable
  "NatureCnn-Flatten",          #  9.6M
  "CompactVit-Flatten",         #  7.0M
  "DinoV2ViTS14-LocalGrid16",   #  4.3M
  "R3MResNet50-LocalGrid7",     #  1.0M
)

def _arches():
  """Every architecture a new registration crosses -- all 14, no subset."""
  from vbrl.vision.architectures import CURRENT_ARCHITECTURES

  rest = [t for t in CURRENT_ARCHITECTURES if t not in ORDER]
  all_a = tuple(ORDER) + tuple(rest)
  i, n = (int(x) for x in os.environ.get('VBRL_PROBE_SHARD', '0/1').split('/'))
  return tuple(a for k, a in enumerate(all_a) if k % n == i)


# A single fixed learning rate diverged for several rows -- CompactVit got
# WORSE with more data, and some supervised ceilings landed below what RL
# achieved, which is impossible. The step size is now selected per architecture
# at the largest budget and then held FIXED across sizes, so the scaling curve
# measures data and not per-point tuning.
LRS = (1e-3, 3e-4, 1e-4, 3e-5)
CHANCE_RAD = 1.4  # a train error above this means the run collapsed
RECIPE = json.loads(Path("artifacts/probe/recipe.json").read_text())


def render(n, seed):
  """Episode-start observations from the environment's own camera sensor.

  Was an offline `mujoco.Renderer` pass with the arm pinned at home, which
  produced a dimmer, lower-contrast image than the policy ever sees (mean 134
  vs 171) and a fixed arm. Sharing one collector with the dumbbell means the two
  figures finally describe the same world.
  """
  import torch

  device = "cuda" if torch.cuda.is_available() else "cpu"
  return episode_start_dataset(TASK, n, seed, device)


def make_cache(arch, imgs, te_imgs, device):
  """Frozen backbone features for the FULL train set, computed once per
  architecture and sliced per dataset size. bfloat16 keeps float32's exponent
  range -- float16 overflows on R3M's ResNet-50 activations."""
  from vbrl.vision.architectures import ARCHITECTURES
  from vbrl.vision.registry import build_encoder

  cfg = ARCHITECTURES[arch]
  if not cfg.frozen:
    return None
  enc = build_encoder(cfg, input_dim=(224, 224)).to(device).eval()

  def run_all(source):
    out = []
    with torch.no_grad():
      for i in range(0, len(source), 64):
        x = torch.from_numpy(source[i : i + 64]).to(device)
        out.append(enc.extract(enc.backbone, enc.preprocess(x)).bfloat16().cpu())
    return torch.cat(out)

  return run_all(imgs), run_all(te_imgs)


def run(arch, target, n_train, imgs, pose, te_imgs, te_pose, device, cache, lr,
        steps=None, warmup=0):
  from vbrl.vision.architectures import ARCHITECTURES
  from vbrl.vision.registry import build_encoder

  torch.manual_seed(0)
  cfg = ARCHITECTURES[arch]
  enc = build_encoder(cfg, input_dim=(224, 224)).to(device)
  frozen = cfg.frozen

  def batch(source, idx):
    return torch.from_numpy(source[idx]).to(device)

  def spatial(source, idx):
    return enc.extract(enc.backbone, enc.preprocess(batch(source, idx)))

  if frozen:
    enc.backbone.eval()
    tr_f, te_f = cache

  head = policy_head(enc.output_dim).to(device)
  params = list(head.parameters()) + list(enc.adapter.parameters())
  if not frozen:
    params += list(enc.backbone.parameters())
  opt = torch.optim.Adam(params, lr=lr)
  steps = STEPS if steps is None else steps
  # A from-scratch ViT does not train without warmup: every CompactVit row
  # collapsed at every learning rate, unable to fit even its training set.
  schedule = (
    torch.optim.lr_scheduler.LambdaLR(
      opt, lambda step: min(1.0, (step + 1) / warmup)
    )
    if warmup
    else None
  )

  def labels(p):
    p = torch.from_numpy(p)
    if target == "yaw":
      return torch.stack([torch.sin(p[:, 2]), torch.cos(p[:, 2])], 1)
    return torch.stack([(p[:, 0] - 0.3) / 0.1, p[:, 1] / 0.2], 1)

  tr_y = labels(pose[:n_train]).to(device)
  rng = np.random.default_rng(0)
  for _ in range(steps):
    idx = rng.integers(0, n_train, BATCH)
    feats = (
      tr_f[idx].to(device).float() if frozen else spatial(imgs, idx)
    )
    loss = nn.functional.mse_loss(head(enc.adapter(feats)), tr_y[idx])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    if schedule is not None:
      schedule.step()

  def _median_error(pose_source, image_source, cached, limit):
    out = []
    with torch.no_grad():
      for i in range(0, limit, 256):
        idx = np.arange(i, min(i + 256, limit))
        f = cached[idx].to(device).float() if frozen else spatial(image_source, idx)
        out.append(head(enc.adapter(f)).cpu())
    pred = torch.cat(out)
    ang = torch.atan2(pred[:, 0], pred[:, 1])
    true = torch.from_numpy(pose_source[:limit, 2])
    return float(((ang - true + torch.pi) % (2 * torch.pi) - torch.pi).abs().median())

  train_median = (
    _median_error(pose, imgs, tr_f if frozen else None, min(4000, n_train))
    if target == "yaw" else 0.0
  )
  preds = []
  with torch.no_grad():
    for i in range(0, len(te_pose), 256):
      idx = np.arange(i, min(i + 256, len(te_pose)))
      feats = te_f[idx].to(device).float() if frozen else spatial(te_imgs, idx)
      preds.append(head(enc.adapter(feats)).cpu())
  pred = torch.cat(preds)

  if target == "yaw":
    ang = torch.atan2(pred[:, 0], pred[:, 1])
    true = torch.from_numpy(te_pose[:, 2])
    err = ((ang - true + torch.pi) % (2 * torch.pi) - torch.pi).abs()
    return {
      "median": err.median().item(),
      "mean": err.mean().item(),
      "within3deg": (err < 0.052).float().mean().item(),
      "train_median": train_median,
      "collapsed": bool(train_median > CHANCE_RAD),
      "pred": ang[:1500].tolist(),
      "true": true[:1500].tolist(),
    }
  err = torch.linalg.vector_norm(
    (pred - labels(te_pose)) * torch.tensor([0.1, 0.2]), dim=-1
  )
  return {"median": err.median().item(), "mean": err.mean().item(),
          "within5mm": (err < 0.005).float().mean().item()}


def main():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  t0 = time.time()
  print(f"device={device} sizes={SIZES} steps={STEPS} (equal for every size)")
  tr_imgs, tr_pose = render(max(SIZES), 0)
  te_imgs, te_pose = render(N_TEST, 1)
  print(f"rendered in {time.time()-t0:.0f}s", flush=True)

  arches = _arches()
  print(f"{len(arches)} architectures: {', '.join(arches)}\n", flush=True)
  results = {"sizes": list(SIZES), "steps": STEPS, "yaw": {}, "xy": {}}
  # (sharded runs write partial files; merge_and_plot combines them)
  for arch in arches:
    cache = make_cache(arch, tr_imgs, te_imgs, device)

    # Recipe already selected on the previous data; reuse it so the re-run costs
    # one pass instead of four. Collapse detection still guards it.
    fixed = RECIPE.get(arch)
    best_lr, best = None, None
    for lr in ([fixed["lr"]] if fixed else LRS):
      r = run(arch, "yaw", max(SIZES), tr_imgs, tr_pose, te_imgs, te_pose,
              device, cache, lr,
              steps=(fixed or {}).get("steps"), warmup=(fixed or {}).get("warmup", 0))
      print(f"{arch:<30} lr={lr:<7} n={max(SIZES):<6} "
            f"yaw {r['median']*57.3:5.1f}deg (train {r['train_median']*57.3:5.1f})"
            f"{'  <- best' if best is None or r['median'] < best['median'] else ''}",
            flush=True)
      if best is None or r["median"] < best["median"]:
        best, best_lr = r, lr
    if best["train_median"] > CHANCE_RAD:
      print(f"{arch:<30} COLLAPSED at every lr (train error at chance)", flush=True)

    # Stage 2: the remaining sizes at that same step size.
    results["yaw"][arch] = []
    for n in SIZES:
      r = best if n == max(SIZES) else run(
        arch, "yaw", n, tr_imgs, tr_pose, te_imgs, te_pose, device, cache, best_lr,
        steps=(fixed or {}).get("steps"), warmup=(fixed or {}).get("warmup", 0)
      )
      r["lr"] = best_lr
      results["yaw"][arch].append(r)
      print(f"{arch:<30} n={n:<6} yaw median {r['median']:.3f} rad "
            f"({r['median']*57.3:.1f} deg)  <3deg {r['within3deg']:.2f}  "
            f"lr={best_lr}  [{time.time()-t0:.0f}s]", flush=True)
    results["xy"][arch] = run(
      arch, "xy", max(SIZES), tr_imgs, tr_pose, te_imgs, te_pose, device, cache,
      best_lr, steps=(fixed or {}).get("steps"), warmup=(fixed or {}).get("warmup", 0)
    )
    del cache
    OUT.write_text(json.dumps(results))  # checkpoint after every architecture
    print(f"{arch:<30} n={max(SIZES):<6} xy  median "
          f"{results['xy'][arch]['median']*1000:.1f} mm", flush=True)

  OUT.write_text(json.dumps(results))
  print(f"\nwrote {OUT}")


if __name__ == "__main__":
  main()
