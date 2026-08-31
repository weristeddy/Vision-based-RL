"""Roll out every trained policy of one generation and record what it saw and did.

One pass produces three things:

  traces  -- yaw error and overlap against timestep, so we can see whether the
             policy ever *attempts* rotation within an episode
  probe   -- can the TRAINED encoder's 256-d output still decode the T's yaw?
             Compared against the fresh-adapter upper bound, this separates
             "the representation never formed" from "it formed, actor ignores it"
  scatter -- predicted vs true yaw for the probe

Weights come from W&B (the local copies are symlinks the uploader cleaned up).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from probe_head import policy_head

# Which training generation to roll out. Set VBRL_PROBE_GENERATION to switch;
# every output is suffixed with it so generations never overwrite each other.
SHARD = os.environ.get("VBRL_PROBE_SHARD", "0/1")   # "i/n": split work across jobs
GENERATION = os.environ.get("VBRL_PROBE_GENERATION", "FrontCam")
# The T barely rotates within an episode, so the number of DISTINCT yaw values
# the probe sees is the environment count, not the sample count. 64 envs gave it
# ~48 effective training points and it overfitted; 256 envs is the fix.
NUM_ENVS, STEPS, EPISODE_BATCHES = 256, 250, 3
_SUF = "" if SHARD == "0/1" else f"_shard{SHARD.split('/')[0]}"
OUT = Path(f"artifacts/probe/rollout_{GENERATION}{_SUF}.npz")
PROBE_STEPS, PROBE_BATCH = 3000, 256


LOG_ROOT = Path("logs/rsl_rl/push_t_rgb_trossen_realistic_d435")


def local_runs():
  """Architecture -> (run directory, first checkpoint, last checkpoint).

  Read off the filesystem rather than W&B: the API stopped listing whole
  generations (FrontCam and SlowGoal both returned 0 while their runs plainly
  existed), and the checkpoints are sitting right here anyway.
  """
  marker = f"PushT-{GENERATION}-"
  found = {}
  for run_dir in sorted(LOG_ROOT.glob(f"*{marker}*")):
    numbers = sorted(
      int(m.group(1))
      for m in (re.fullmatch(r"model_(\d+)\.pt", f.name) for f in run_dir.iterdir())
      if m
    )
    if len(numbers) < 2:
      continue
    arch = run_dir.name.split(marker)[1].removesuffix("-TrossenRealistic")
    found[arch] = (
      run_dir,
      run_dir / f"model_{numbers[0]}.pt",
      run_dir / f"model_{numbers[-1]}.pt",
    )
  return found


def find_encoder(module):
  from vbrl.vision.encoder import VisualEncoder

  found = [m for m in module.modules() if isinstance(m, VisualEncoder)]
  if len(found) != 1:
    raise RuntimeError(f"Expected exactly one VisualEncoder, found {len(found)}.")
  return found[0]


def rollout(arch, checkpoint, device):
  from tensordict import TensorDict

  from vbrl.runtime import CheckpointRef, build_env, load_trained_policy

  task_id = f"Mjlab-PushT-{GENERATION}-{arch}-TrossenRealistic"
  # Terminations are KEPT: with them dropped, an episode that shoves the T off
  # the table keeps contributing to the average with the object on the floor,
  # which is exactly the failure mode of the collapsed AFA runs.
  env = build_env(task_id, device=device, num_envs=NUM_ENVS, seed=0)
  wrapped, runner, policy, _ = load_trained_policy(
    env,
    task_id=task_id,
    device=device,
    ref=CheckpointRef(checkpoint_file=str(checkpoint)),
  )

  encoder = find_encoder(runner.alg.actor)
  captured = {}
  handle = encoder.register_forward_hook(
    lambda _m, _i, out: captured.__setitem__("f", out.detach())
  )

  command = wrapped.unwrapped.command_manager.get_term("push_t_goal")
  obj = wrapped.unwrapped.scene["object"]

  from mjlab.utils.lab_api.math import euler_xyz_from_quat, wrap_to_pi

  # Several independent batches of episodes, all aligned to step 0, so the
  # traces carry a distribution rather than a single mean.
  feats, obj_yaw, yaw_err, overlap = [], [], [], []
  for batch in range(EPISODE_BATCHES):
    wrapped.seed(batch)
    observations, _ = wrapped.reset()
    alive = torch.ones(wrapped.num_envs, dtype=torch.bool, device=device)
    b_f, b_y, b_e, b_o = [], [], [], []
    for _ in range(STEPS):
      with torch.inference_mode():
        actions = policy(observations)
      observations, _, done, _ = wrapped.step(actions)

      # The env auto-resets on termination, so the state read after step() for a
      # done env already belongs to the NEXT episode. Mask the terminating step
      # itself, not just the ones after it.
      alive &= ~done.bool()
      y = euler_xyz_from_quat(obj.data.root_link_quat_w)[2]
      mask = alive.float().cpu().numpy()
      mask[mask == 0] = np.nan
      b_f.append(captured["f"].float().cpu().numpy())
      b_y.append(y.float().cpu().numpy())
      b_e.append(wrap_to_pi(command.target_yaw - y).abs().float().cpu().numpy() * mask)
      b_o.append(command.get_overlap(force_refresh=True).float().cpu().numpy() * mask)

      if bool(done.any()):
        observations = TensorDict(
          wrapped.unwrapped.reset(env_ids=done.nonzero().squeeze(-1))[0],
          batch_size=[wrapped.num_envs],
        )
    feats.append(np.stack(b_f)); obj_yaw.append(np.stack(b_y))
    yaw_err.append(np.stack(b_e)); overlap.append(np.stack(b_o))

  handle.remove()
  wrapped.close()
  return (
    np.concatenate(feats, axis=1),     # [T, batches*N, 256]
    np.concatenate(obj_yaw, axis=1),
    np.concatenate(yaw_err, axis=1),
    np.concatenate(overlap, axis=1),
  )


def probe(feats, target, device, kind="yaw"):
  """Decode the T's pose from the TRAINED encoder's output. Split by environment
  so train and test never share an episode. Reports train error too: a train/test
  gap means the probe overfitted, an equally bad train error means the feature
  genuinely lacks the signal."""
  torch.manual_seed(0)
  obj_yaw = target
  t, n, d = feats.shape
  held_out = max(2, n // 4)
  train_env, test_env = np.arange(0, n - held_out), np.arange(n - held_out, n)
  x = torch.from_numpy(feats).to(device)
  y = torch.from_numpy(np.nan_to_num(obj_yaw, nan=0.0)).to(device)
  tr_x, tr_y = x[:, train_env].reshape(-1, d), y[:, train_env].reshape(-1)
  te_x, te_y = x[:, test_env].reshape(-1, d), y[:, test_env].reshape(-1)

  mu, sd = tr_x.mean(0, keepdim=True), tr_x.std(0, keepdim=True).clamp_min(1e-6)
  tr_x, te_x = (tr_x - mu) / sd, (te_x - mu) / sd
  target = torch.stack([torch.sin(tr_y), torch.cos(tr_y)], 1)

  head = policy_head(d).to(device)
  opt = torch.optim.Adam(head.parameters(), lr=1e-3)
  rng = np.random.default_rng(0)
  for _ in range(PROBE_STEPS):
    idx = torch.from_numpy(rng.integers(0, len(tr_x), PROBE_BATCH)).to(device)
    loss = nn.functional.mse_loss(head(tr_x[idx]), target[idx])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

  def angular_error(x, y):
    with torch.no_grad():
      pred = head(x)
    ang = torch.atan2(pred[:, 0], pred[:, 1])
    return ((ang - y + torch.pi) % (2 * torch.pi) - torch.pi).abs(), ang

  te_err, te_ang = angular_error(te_x, te_y)
  tr_err, _ = angular_error(tr_x, tr_y)
  return (
    te_err.cpu().numpy(),
    te_ang.cpu().numpy(),
    te_y.cpu().numpy(),
    float(tr_err.median()),
  )


def main():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  runs = local_runs()
  print(f"device={device}  {len(runs)} local checkpoints for {GENERATION}\n", flush=True)
  if not runs:
    raise SystemExit(f"no {GENERATION} runs under {LOG_ROOT}")

  store, t0 = {}, time.time()
  i, n = (int(x) for x in SHARD.split('/'))
  items = [kv for k, kv in enumerate(sorted(runs.items())) if k % n == i]
  print(f'shard {SHARD}: {[a for a, _ in items]}', flush=True)
  for arch, (_dir, _first, checkpoint) in items:
    try:
      feats, obj_yaw, yaw_err, overlap = rollout(arch, checkpoint, device)
    except Exception as exc:  # keep going; report honestly at the end
      print(f"{arch:<30} ROLLOUT FAILED: {type(exc).__name__}: {exc}", flush=True)
      continue
    err, pred, true, train_err = probe(feats, obj_yaw, device)
    def bands(a):
      return np.stack([np.nanpercentile(a, q, axis=1) for q in (25, 50, 75)])
    store[f"{arch}/yaw_err_trace"] = np.nanmean(yaw_err, axis=1)
    store[f"{arch}/overlap_trace"] = np.nanmean(overlap, axis=1)
    store[f"{arch}/yaw_err_bands"] = bands(yaw_err)      # [3, T] = p25/p50/p75
    store[f"{arch}/overlap_bands"] = bands(overlap)
    store[f"{arch}/alive_frac"] = np.mean(~np.isnan(yaw_err), axis=1)
    store[f"{arch}/probe_err"] = err
    store[f"{arch}/probe_train_err"] = np.array([train_err])
    store[f"{arch}/probe_pred"] = pred[:6000]
    store[f"{arch}/probe_true"] = true[:6000]
    np.savez_compressed(OUT, **store)
    print(
      f"{arch:<30} rollout yaw {np.nanmean(yaw_err):.3f} "
      f"(start {np.nanmean(yaw_err[0]):.3f} end {np.nanmean(yaw_err[-1]):.3f}) "
      f"alive {np.mean(~np.isnan(yaw_err[-1])):.2f} "
      f"overlap {np.nanmean(overlap[-1]):.3f} | probe {np.median(err)*57.3:5.1f}deg "
      f"train {train_err*57.3:5.1f}deg  [{time.time()-t0:.0f}s]",
      flush=True,
    )

  print(f"\nwrote {OUT} with {len(store)//5} architectures")


if __name__ == "__main__":
  main()
