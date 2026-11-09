from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.tasks.manipulation.mdp import LiftingCommand, LiftingCommandCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, sample_uniform

from vbrl.tasks.push_t.geometry import yaw_from_quat

from ..geometry import (
  LINK_HALF_EXTENTS,
  LINK_HALF_HEIGHT,
  LinkOverlap,
  end_links_gap,
  gap_to_boxes,
  link_poses,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

LINK_NAMES = ("push_chain_link0", "push_chain_link1", "push_chain_link2")
HINGE_NAMES = ("push_chain_hinge0", "push_chain_hinge1")


class PushChainCommand(LiftingCommand):
  cfg: PushChainCommandCfg

  def __init__(self, cfg: PushChainCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    n = self.num_envs
    self.target_yaw = torch.zeros(n, device=self.device)
    self.target_hinges = torch.zeros(n, 2, device=self.device)
    self.target_links = torch.zeros(n, 3, 3, device=self.device)
    self.observed_links = torch.zeros(n, 3, 3, device=self.device)
    self.link_ids = self.object.find_bodies(LINK_NAMES, preserve_order=True)[0]
    self._hinge_ids = torch.tensor(
      self.object.find_joints(HINGE_NAMES, preserve_order=True)[0], device=self.device
    )
    self._overlap = LinkOverlap(device=self.device, dtype=torch.float)
    self._overlap_cache = torch.zeros(n, 3, device=self.device)
    self._overlap_cache_step: int | None = None
    self._robot_boxes = self._robot_footprint(env)
    self._goal_markers = [env.scene[name] for name in cfg.goal_marker_names]
    for name in ("object_height", "at_goal", "position_error"):
      self.metrics.pop(name)
    for i in range(3):
      self.metrics[f"link{i}_overlap"] = torch.zeros(n, device=self.device)
    self.metrics["min_link_overlap"] = torch.zeros(n, device=self.device)

  def _robot_footprint(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    import mujoco

    model = env.sim.mj_model
    data = mujoco.MjData(model)
    robot = env.scene["robot"]
    for name, position in zip(
      robot.joint_names, robot.data.default_joint_pos[0].tolist(), strict=True
    ):
      data.joint(f"robot/{name}").qpos = position
    mujoco.mj_kinematics(model, data)
    top = 2.0 * LINK_HALF_HEIGHT + self.cfg.robot_clearance
    boxes = []
    for geom in range(model.ngeom):
      body = model.body(model.geom_bodyid[geom]).name
      collides = model.geom_contype[geom] or model.geom_conaffinity[geom]
      if not body.startswith("robot/") or not collides:
        continue
      rotation = data.geom_xmat[geom].reshape(3, 3)
      centre = data.geom_xpos[geom] + rotation @ model.geom_aabb[geom, :3]
      half = np.abs(rotation) @ model.geom_aabb[geom, 3:]
      if centre[2] - half[2] < top:
        boxes.append((*centre[:2], *half[:2]))
    return torch.tensor(boxes, dtype=torch.float, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat(
      (self.target_pos[:, :2], self.target_yaw[:, None], self.target_hinges), dim=-1
    )

  def chain_links(self) -> torch.Tensor:
    position = self.object.data.body_link_pos_w[:, self.link_ids, :2]
    quat = self.object.data.body_link_quat_w[:, self.link_ids]
    yaw = yaw_from_quat(quat.reshape(-1, 4)).view(-1, 3, 1)
    return torch.cat((position, yaw), dim=-1)

  def chain_hinges(self) -> torch.Tensor:
    return self.object.data.joint_pos[:, self._hinge_ids]

  def get_overlap(self, *, force_refresh: bool = False) -> torch.Tensor:
    step = int(getattr(self._env, "common_step_counter", -1))
    if force_refresh or self._overlap_cache_step != step:
      self._overlap_cache = self._overlap.best(self.chain_links(), self.target_links)
      self._overlap_cache_step = step
    return self._overlap_cache

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    extras = {
      name: value[env_ids].nanmean().item() for name, value in self.metrics.items()
    }
    super().reset(env_ids)
    return extras

  def _record_success(self, overlap: torch.Tensor) -> torch.Tensor:
    lowest = overlap.amin(-1)
    at_goal = lowest >= self.cfg.success_threshold
    self.episode_success = torch.maximum(
      self.episode_success, at_goal.to(self.episode_success.dtype)
    )
    for i in range(3):
      self.metrics[f"link{i}_overlap"] = overlap[:, i]
    self.metrics["min_link_overlap"] = lowest
    self.metrics["episode_success"] = self.episode_success
    return at_goal

  def get_at_goal(self) -> torch.Tensor:
    return self._record_success(self.get_overlap())

  def _update_metrics(self) -> None:
    self._record_success(self.get_overlap())

  def compute_success(self) -> torch.Tensor:
    return self.get_at_goal()

  def sample_chains(self, n: int) -> torch.Tensor:
    r = self.cfg.object_pose_range
    assert r is not None
    hinge = self.cfg.hinge_range
    low = torch.tensor((r.x[0], r.y[0], r.yaw[0], -hinge, -hinge), device=self.device)
    high = torch.tensor((r.x[1], r.y[1], r.yaw[1], hinge, hinge), device=self.device)
    samples = sample_uniform(low, high, (n, self.cfg.sampling_attempts, 5), self.device)
    links = link_poses(samples[..., :2], samples[..., 2], samples[..., 3:])
    valid = (end_links_gap(links) >= self.cfg.end_link_gap) & (
      gap_to_boxes(links, self._robot_boxes) >= self.cfg.robot_clearance
    )
    chosen = samples[torch.arange(n, device=self.device), valid.int().argmax(-1)]
    fallback = ~valid.any(-1)
    chosen[fallback, 0] = sum(r.x) / 2.0
    chosen[fallback, 1] = sum(r.y) / 2.0
    chosen[fallback, 3:] = 0.0
    return chosen

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self.episode_success[env_ids] = 0.0
    self._overlap_cache_step = None
    origins = self._env.scene.env_origins[env_ids]
    self._write_chain(env_ids, self.sample_chains(len(env_ids)), origins)
    self._set_goal(env_ids, self.sample_chains(len(env_ids)), origins)

  def _write_chain(
    self, env_ids: torch.Tensor, chain: torch.Tensor, origins: torch.Tensor
  ) -> None:
    r = self.cfg.object_pose_range
    assert r is not None
    zeros = torch.zeros(len(env_ids), device=self.device)
    position = torch.cat((chain[:, :2], torch.full_like(zeros, r.z[0])[:, None]), -1)
    pose = torch.cat(
      (position + origins, quat_from_euler_xyz(zeros, zeros, chain[:, 2])), dim=-1
    )
    self.object.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    self.object.write_root_link_velocity_to_sim(
      torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids
    )
    self.object.write_joint_state_to_sim(
      chain[:, 3:],
      torch.zeros_like(chain[:, 3:]),
      joint_ids=self._hinge_ids,
      env_ids=env_ids,
    )

  def _set_goal(
    self, env_ids: torch.Tensor, goal: torch.Tensor, origins: torch.Tensor
  ) -> None:
    n = len(env_ids)
    xy = goal[:, :2] + origins[:, :2]
    self.target_pos[env_ids] = torch.cat(
      (xy, origins[:, 2:] + LINK_HALF_HEIGHT), dim=-1
    )
    self.target_yaw[env_ids] = goal[:, 2]
    self.target_hinges[env_ids] = goal[:, 3:]
    self.target_links[env_ids] = link_poses(xy, goal[:, 2], goal[:, 3:])
    shift = self.cfg.observation_position_noise
    turn = self.cfg.observation_yaw_noise
    offset = sample_uniform(-shift, shift, (n, 2), self.device)
    yaw_offset = sample_uniform(-turn, turn, (n,), self.device)
    self.observed_links[env_ids] = link_poses(
      xy + offset, goal[:, 2] + yaw_offset, goal[:, 3:]
    )
    zeros = torch.zeros(n, device=self.device)
    for link, marker in enumerate(self._goal_markers):
      links = self.target_links[env_ids, link]
      pose = torch.cat(
        (
          links[:, :2],
          origins[:, 2:],
          quat_from_euler_xyz(zeros, zeros, links[:, 2]),
        ),
        dim=-1,
      )
      marker.write_mocap_pose_to_sim(pose, env_ids=env_ids)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    del env_ids

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    for batch in visualizer.get_env_indices(self.num_envs):
      for index in torch.as_tensor(batch).flatten().tolist():
        height = float(self.target_pos[index, 2])
        for link, (x, y, yaw) in enumerate(self.target_links[index].tolist()):
          cos, sin = math.cos(yaw), math.sin(yaw)
          visualizer.add_box(
            center=np.array((x, y, height)),
            size=np.array((*LINK_HALF_EXTENTS[link], LINK_HALF_HEIGHT)),
            mat=np.array(((cos, -sin, 0.0), (sin, cos, 0.0), (0.0, 0.0, 1.0))),
            color=self.cfg.viz.target_color,
            label=f"push_chain_goal_link{link}_{index}",
          )


@dataclass(kw_only=True)
class PushChainCommandCfg(LiftingCommandCfg):
  observation_position_noise: float = 0.0
  observation_yaw_noise: float = 0.0
  hinge_range: float = math.radians(85.0)
  orientation_weight: float = 0.5
  end_link_gap: float = 0.005
  robot_clearance: float = 0.02
  sampling_attempts: int = 32
  goal_marker_names: tuple[str, ...] = ()

  def build(self, env: ManagerBasedRlEnv) -> PushChainCommand:
    return PushChainCommand(self, env)


def push_chain_command(env, name: str) -> PushChainCommand:
  command = env.command_manager.get_term(name)
  if not isinstance(command, PushChainCommand):
    raise TypeError(f"Command {name!r} must be a PushChainCommand.")
  return command


__all__ = [
  "HINGE_NAMES",
  "LINK_NAMES",
  "PushChainCommand",
  "PushChainCommandCfg",
  "push_chain_command",
]
