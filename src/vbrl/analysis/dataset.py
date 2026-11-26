from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from vbrl.analysis.pairs import (
  OBJECT,
  States,
  make_env,
  real_frames,
  render,
  rollout_states,
  task_of,
)

ROOT = Path("artifacts/features")
TRAINING = "real_texture_wood"
REFERENCE = "wood049"
HELD_OUT = ("real_table", "marble019", "ood_light", "object_color_sweep")
GRIPPER = (
  "robot/link_6",
  "robot/carriage_",
  "robot/gripper_",
  "robot/camera_mount_d405",
  "robot/camera_link",
)
RELEVANT = (*GRIPPER, "goal_marker")
BATCH = 256
TEST_FRACTION = 0.2


def _relevant_geoms(env: Any, task: str) -> Any:
  model = env.sim.mj_model
  prefixes = (*RELEVANT, f"{OBJECT[task]}/")
  names = [model.body(model.geom_bodyid[g]).name for g in range(model.ngeom)]
  return np.array([name.startswith(prefixes) for name in names])


def render_set(
  task_id: str, scene: str, states: States, shuffle: bool = False
) -> dict[str, Any]:
  import mujoco
  import torch

  task = task_of(task_id)
  size = min(BATCH, len(states))
  env = make_env(task_id, scene, size, segmentation=True)
  relevant = torch.tensor(_relevant_geoms(env, task), device=env.device)
  parts: dict[str, list] = {}
  for start in range(0, len(states), size):
    index = np.arange(start, min(start + size, len(states)))
    if shuffle:
      index = np.roll(index, 1)
    out = render(env, task, states.take(index))
    count = len(index)
    ids, kinds = out["segmentation"][:count].unbind(-1)
    mask = (kinds == int(mujoco.mjtObj.mjOBJ_GEOM)) & relevant[ids.clamp(min=0)]
    parts.setdefault("masks", []).append(mask.cpu().numpy())
    for group, value in out["obs"].items():
      parts.setdefault(f"obs_{group}", []).append(value[:count].cpu().numpy())
  env.close()
  data = {key: np.concatenate(value) for key, value in parts.items()}
  return {
    **data,
    "joints": states.joints,
    "objects": states.objects,
    "goals": states.goals,
  }


def real_set(task: str, every: int) -> tuple[Any, States]:
  images, states = [], []
  for run in sorted(Path("artifacts/real_eval", task).glob("goal*/*/run*")):
    frames, run_states, reliable = real_frames(run, every)
    images.append(frames[reliable])
    states.append(run_states.take(reliable))
  if not images:
    return None, None
  joined = States(
    *(
      np.concatenate([getattr(s, f) for s in states])
      for f in ("joints", "objects", "goals")
    )
  )
  return np.concatenate(images), joined


def build(task_id: str, run: str, count: int, every: int, real_every: int) -> Path:
  task = task_of(task_id)
  folder = ROOT / task
  folder.mkdir(parents=True, exist_ok=True)
  states = rollout_states(task_id, run, count, every)
  order = np.random.default_rng(0).permutation(len(states))
  test = order[: int(len(order) * TEST_FRACTION)]
  train = order[len(test) :]
  sets = {
    "train": (TRAINING, train),
    "test": (TRAINING, test),
    REFERENCE: (REFERENCE, test),
  }
  sets.update({scene: (scene, test) for scene in HELD_OUT})
  for name, (scene, index) in sets.items():
    np.savez_compressed(
      folder / f"{name}.npz", **render_set(task_id, scene, states.take(index))
    )
  np.savez_compressed(
    folder / "state.npz",
    **render_set(task_id, REFERENCE, states.take(test), shuffle=True),
  )
  images, real = real_set(task, real_every)
  if images is not None:
    paired = render_set(task_id, REFERENCE, real)
    np.savez_compressed(
      folder / "real.npz", **{**paired, "obs_camera": images.transpose(0, 3, 1, 2)}
    )
    np.savez_compressed(folder / "real_sim.npz", **paired)
  return folder


def main(argv: Any = None) -> None:
  parser = argparse.ArgumentParser(
    description="Render the feature analysis images of one task."
  )
  parser.add_argument(
    "task_id",
    help="the visual task ID of the student that rolls out, e.g. DINOv2 + AFA",
  )
  parser.add_argument(
    "--run", required=True, help="W&B run of the student that rolls out"
  )
  parser.add_argument("--states", type=int, default=5000)
  parser.add_argument(
    "--every", type=int, default=10, help="keep every n-th rollout step"
  )
  parser.add_argument(
    "--real-every", type=int, default=20, help="keep every n-th real frame"
  )
  args = parser.parse_args(argv)
  print(build(args.task_id, args.run, args.states, args.every, args.real_every))


if __name__ == "__main__":
  main()


__all__ = ["HELD_OUT", "REFERENCE", "ROOT", "TRAINING", "build", "render_set"]
