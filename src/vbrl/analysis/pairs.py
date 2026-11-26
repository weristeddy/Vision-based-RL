from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EXPOSURE_MS = 8.0
PROJECT = "eduard-nicolae-robot-learning/mjlab"
OBJECT = {"lift_cube": "cube", "push_t": "object", "push_chain": "object"}
CAMERA_DR = ("camera_position", "camera_orientation", "camera_fovy")


@dataclass
class States:
  joints: Any
  objects: Any
  goals: Any

  def __len__(self) -> int:
    return len(self.joints)

  def take(self, index: Any) -> States:
    return States(self.joints[index], self.objects[index], self.goals[index])

  def save(self, path: Path) -> None:
    np.savez_compressed(
      path, joints=self.joints, objects=self.objects, goals=self.goals
    )

  @classmethod
  def load(cls, path: Path) -> States:
    data = np.load(path)
    return cls(data["joints"], data["objects"], data["goals"])


def task_of(task_id: str) -> str:
  return next(
    t
    for t, n in (
      ("lift_cube", "LiftCube"),
      ("push_t", "PushT"),
      ("push_chain", "PushChain"),
    )
    if n in task_id
  )


def frame_joints(trace: Any) -> Any:
  read = trace["time"] - trace["compute_ms"] / 1000.0
  shot = read - (trace["camera_age_ms"] + EXPOSURE_MS / 2.0) / 1000.0
  joints = trace["joint_pos"]
  return np.stack(
    [np.interp(shot, read, joints[:, j]) for j in range(joints.shape[1])], -1
  )


def sim_joints(real: Any) -> Any:
  return np.concatenate([real[:, :7], real[:, 6:7]], axis=1)


def real_frames(run: Path, every: int = 1) -> tuple[Any, States, Any]:
  from vbrl.rig.measure import measure_object, view_for

  episode = json.loads((run / "episode.json").read_text())
  task, camera = episode["task"], episode.get("camera", episode.get("encoder_camera"))
  trace = np.load(run / "trace.npz")
  steps = np.arange(0, len(trace["time"]), every)
  images = trace[f"{camera}_frames"][steps]
  external = trace["external_frames"][steps]
  view = view_for(external[0])
  fits = [measure_object(view, image, task) for image in external]
  objects = np.array([fit.pose for fit in fits])
  goal = np.asarray(episode["goal"], dtype=float)
  states = States(
    sim_joints(frame_joints(trace)[steps]),
    objects,
    np.repeat(goal[None], len(steps), 0),
  )
  return images, states, np.array([fit.reliable for fit in fits])


def make_env(
  task_id: str, scene: str, num_envs: int, segmentation: bool = False
) -> Any:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  import vbrl.tasks  # noqa: F401
  from vbrl.scenes.builder import replace_scene

  cfg = replace_scene(load_env_cfg(task_id, play=True), scene=scene)
  cfg.scene.num_envs, cfg.seed = num_envs, 0
  for name in CAMERA_DR:
    cfg.events.pop(name, None)
  if segmentation:
    camera = next(s for s in cfg.scene.sensors if hasattr(s, "data_types"))
    camera.data_types = ("rgb", "segmentation")
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  env.seed(0)
  env.reset()
  return env


def _yaw_quat(yaw: Any) -> Any:
  import torch
  from mjlab.utils.lab_api.math import quat_from_euler_xyz

  zero = torch.zeros_like(yaw)
  return quat_from_euler_xyz(zero, zero, yaw)


def set_states(env: Any, task: str, states: States) -> None:
  import torch

  from vbrl.tasks.push_t.goal_marker import GOAL_MARKER_Z

  device, n = env.device, len(states)
  ids = torch.arange(n, device=device)
  origins = env.scene.env_origins[:n]
  robot = env.scene["robot"]
  joints = torch.tensor(states.joints, dtype=torch.float32, device=device)
  robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=ids)
  objects = torch.tensor(states.objects, dtype=torch.float32, device=device)
  goals = torch.tensor(states.goals, dtype=torch.float32, device=device)
  command = env.command_manager.get_term(env.command_manager.active_terms[0])
  if task == "push_chain":
    command._write_chain(ids, objects, origins)
    command._set_goal(ids, goals, origins)
    return
  entity = env.scene[OBJECT[task]]
  z = origins[:, 2:3] + command.cfg.object_pose_range.z[0]
  position = torch.cat((origins[:, :2] + objects[:, :2], z), -1)
  entity.write_root_link_pose_to_sim(
    torch.cat((position, _yaw_quat(objects[:, 2])), -1), env_ids=ids
  )
  entity.write_root_link_velocity_to_sim(torch.zeros(n, 6, device=device), env_ids=ids)
  if task == "push_t":
    command.target_pos[:n, :2] = origins[:, :2] + goals[:, :2]
    command.target_yaw[:n] = goals[:, 2]
    marker = torch.cat(
      (origins[:, :2] + goals[:, :2], origins[:, 2:3] + GOAL_MARKER_Z), -1
    )
    command._goal_marker.write_mocap_pose_to_sim(
      torch.cat((marker, _yaw_quat(goals[:, 2])), -1), env_ids=ids
    )


def render(env: Any, task: str, states: States) -> dict[str, Any]:
  set_states(env, task, states)
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.sim.sense()
  obs = env.observation_manager.compute(update_history=True)
  camera = next(s for s in env.scene.sensors.values() if hasattr(s.data, "rgb"))
  return {"obs": obs, "segmentation": camera.data.segmentation}


def rollout_states(task_id: str, run: str, states: int, every: int = 10) -> States:
  import torch
  from mjlab.tasks.registry import load_env_cfg

  from vbrl.runtime import CheckpointRef, build_env, load_trained_policy
  from vbrl.tasks.push_chain.geometry import centre_link
  from vbrl.tasks.push_t.geometry import yaw_from_quat

  task = task_of(task_id)
  cfg = load_env_cfg(task_id, play=True)
  steps = round(cfg.episode_length_s / (cfg.decimation * cfg.sim.mujoco.timestep))
  count = -(-states * every // steps)
  env = build_env(task_id, device="cuda:0", num_envs=count, seed=0)
  ref = CheckpointRef(wandb_run_path=f"{PROJECT}/{run}")
  env, _, policy, _ = load_trained_policy(
    env, task_id=task_id, device="cuda:0", ref=ref
  )
  raw = env.unwrapped
  command = raw.command_manager.get_term(raw.command_manager.active_terms[0])
  origins = raw.scene.env_origins
  records = []
  observations = env.get_observations()
  for step in range(raw.max_episode_length):
    with torch.inference_mode():
      observations, *_ = env.step(policy(observations))
    if step % every:
      continue
    joints = raw.scene["robot"].data.joint_pos
    if task == "push_chain":
      links = command.chain_links()[:, centre_link(command.cfg.num_links)]
      objects = torch.cat(
        (links[:, :2] - origins[:, :2], links[:, 2:], command.chain_hinges()), -1
      )
      goals = command.command().clone()
      goals[:, :2] -= origins[:, :2]
    else:
      entity = raw.scene[OBJECT[task]]
      xy = entity.data.root_link_pos_w[:, :2] - origins[:, :2]
      objects = torch.cat(
        (xy, yaw_from_quat(entity.data.root_link_quat_w)[:, None]), -1
      )
      goal_xy = command.target_pos[:, :2] - origins[:, :2]
      yaw = getattr(command, "target_yaw", torch.zeros(count, device=raw.device))
      goals = torch.cat((goal_xy, yaw[:, None]), -1)
    records.append([t.cpu().numpy() for t in (joints, objects, goals)])
  env.close()
  joints, objects, goals = (
    np.concatenate(parts) for parts in zip(*records, strict=True)
  )
  return States(joints, objects, goals)


def main(argv: Any = None) -> None:
  from PIL import Image

  from vbrl.scenes.presets import EVALUATION_SCENES

  parser = argparse.ArgumentParser(
    description="Render a real frame's state in simulated scenes."
  )
  parser.add_argument(
    "run", type=Path, help="artifacts/real_eval/<task>/goal<n>/<policy>/run<k>"
  )
  parser.add_argument("--step", type=int, required=True)
  parser.add_argument(
    "--task-id", required=True, help="a visual task ID of the run's task"
  )
  parser.add_argument("--scenes", nargs="*", default=list(EVALUATION_SCENES[:4]))
  parser.add_argument("--out", type=Path, default=Path("artifacts/pairs"))
  args = parser.parse_args(argv)
  images, states, reliable = real_frames(args.run)
  if not reliable[args.step]:
    raise SystemExit(f"The object fit at step {args.step} is unreliable.")
  args.out.mkdir(parents=True, exist_ok=True)
  Image.fromarray(images[args.step]).save(args.out / "real.png")
  state = states.take([args.step])
  for scene in args.scenes:
    env = make_env(args.task_id, scene, 1)
    image = render(env, task_of(args.task_id), state)["obs"]["camera"][0]
    env.close()
    Image.fromarray(image.permute(1, 2, 0).cpu().numpy()).save(
      args.out / f"{scene}.png"
    )
  print(f"{args.out}: real.png and {', '.join(args.scenes)}")


if __name__ == "__main__":
  main()


__all__ = [
  "PROJECT",
  "States",
  "frame_joints",
  "make_env",
  "real_frames",
  "render",
  "rollout_states",
  "set_states",
  "task_of",
]
