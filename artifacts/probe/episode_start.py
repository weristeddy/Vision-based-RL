"""Probe dataset built from episode-start observations.

Replaces the offline re-render. Three problems went away at once:

  renderer   frames come from the environment's own camera sensor (MuJoCo Warp),
             the exact path the policy trains on, instead of `mujoco.Renderer`
             which produced a dimmer, lower-contrast image (mean 134 vs 171).
  joint pose arm and T are whatever the reset produced together, rather than
             sampled independently -- the independent version put them >20cm
             apart in 49% of images, a configuration that never occurs.
  DR         texture, colour and lighting come from the real reset events.

The reset already draws the T uniformly over x, y and yaw, so the yaw probe
stays unbiased without any extra sampling.
"""
from __future__ import annotations

import numpy as np
import torch

CAMERA_KEY = "camera"

# Shared probe budget. One place, so every figure uses the same numbers.
N_TRAIN, N_TEST = 40_000, 5_000
STEPS, BATCH = 8_000, 256


def episode_start_dataset(task_id, n, seed, device, num_envs=256):
  """Return (uint8 [n,3,224,224] observations, float32 [n,3] object x/y/yaw)."""
  from mjlab.utils.lab_api.math import euler_xyz_from_quat

  from vbrl.runtime import build_env

  env = build_env(task_id, device=device, num_envs=num_envs, seed=seed)
  obj = env.scene["object"]
  images, poses = [], []
  batch = 0
  while sum(len(x) for x in images) < n:
    env.seed(seed * 10_000 + batch)
    obs, _ = env.reset()
    cam = obs[CAMERA_KEY]
    if not torch.is_tensor(cam):
      cam = next(iter(cam.values()))
    pos = obj.data.root_link_pos_w
    yaw = euler_xyz_from_quat(obj.data.root_link_quat_w)[2]
    images.append(cam.cpu().numpy())
    poses.append(
      torch.stack((pos[:, 0], pos[:, 1], yaw), dim=-1).float().cpu().numpy()
    )
    batch += 1
  env.close()
  return (
    np.concatenate(images)[:n],
    np.concatenate(poses)[:n],
  )


def features(encoder, imgs, device, chunk=64):
  """Encoder output for a batch of episode-start observations (uint8 BCHW)."""
  import torch

  out = []
  with torch.no_grad():
    for i in range(0, len(imgs), chunk):
      x = torch.from_numpy(imgs[i : i + chunk]).to(device)
      out.append(encoder(x).float().cpu())
  return torch.cat(out)


__all__ = ["BATCH", "N_TEST", "N_TRAIN", "STEPS", "episode_start_dataset", "features"]
