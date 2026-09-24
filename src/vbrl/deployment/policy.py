from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vbrl.deployment.config import TARGET_Z_BASE_M
from vbrl.deployment.kinematics import Kinematics

TERMS = (
  "joint_pos",
  "joint_vel",
  "joint_target",
  "actions",
  "goal_position",
  "target_pose",
)

ARM_JOINTS = (
  "joint_0",
  "joint_1",
  "joint_2",
  "joint_3",
  "joint_4",
  "joint_5",
  "left_carriage_joint",
)

_REQUIRED = ("joint_names", "default_joint_pos", "observation_names", "action_scale")


@dataclass(frozen=True)
class PolicyMetadata:
  joint_names: tuple[str, ...]
  default_joint_pos: Any
  action_offset: Any
  action_scale: Any
  observation_terms: tuple[str, ...]
  action_dim: int
  action_clip: Any
  action_type: str
  target_limits: Any
  clip_actions: float | None
  needs_camera: bool
  source_run: str

  @classmethod
  def from_onnx(cls, onnx_session: Any) -> PolicyMetadata:
    meta = onnx_session.get_modelmeta().custom_metadata_map
    missing = set(_REQUIRED) - set(meta)
    if missing:
      raise ValueError(
        f"The ONNX carries no {sorted(missing)} metadata, so its observation "
        "contract is unknown. Re-export it with vbrl-export-onnx."
      )

    joint_names = tuple(meta["joint_names"].split(","))
    default = np.array([float(v) for v in meta["default_joint_pos"].split(",")])
    index = {name: position for position, name in enumerate(joint_names)}
    return cls(
      joint_names=joint_names,
      default_joint_pos=default,
      action_offset=np.array([default[index[name]] for name in ARM_JOINTS]),
      action_scale=np.array([float(v) for v in meta["action_scale"].split(",")]),
      observation_terms=tuple(meta["observation_names"].split(",")),
      action_dim=int(onnx_session.get_outputs()[0].shape[-1]),
      action_clip=(
        (
          np.array([float(v) for v in meta["action_clip_low"].split(",")]),
          np.array([float(v) for v in meta["action_clip_high"].split(",")]),
        )
        if "action_clip_low" in meta
        else None
      ),
      action_type=meta.get("action_type", "absolute"),
      target_limits=(
        (
          np.array([float(v) for v in meta["target_low"].split(",")]),
          np.array([float(v) for v in meta["target_high"].split(",")]),
        )
        if "target_low" in meta
        else None
      ),
      clip_actions=(
        float(meta["clip_actions"]) if "clip_actions" in meta else None
      ),
      needs_camera=any(i.name == "camera" for i in onnx_session.get_inputs()),
      source_run=meta.get("run_path", "unknown"),
    )

  @property
  def home_pose(self) -> Any:
    return self.action_offset


class Policy:
  def __init__(
    self,
    onnx_session: Any,
    *,
    goal: tuple[float, float, float],
    smoothing: float = 1.0,
  ) -> None:
    self.metadata = PolicyMetadata.from_onnx(onnx_session)
    self._onnx = onnx_session
    self._kinematics = Kinematics()
    self._goal = np.asarray(goal, dtype=np.float64)
    self._smoothing = smoothing
    self._last_action = np.zeros(self.metadata.action_dim)
    self._network_action = np.zeros(self.metadata.action_dim)
    self._goal_position = np.full(3, np.inf)
    self._position = self.metadata.action_offset
    self._target = self.metadata.action_offset

    unsupported = set(self.metadata.observation_terms) - set(TERMS)
    if unsupported:
      raise ValueError(
        f"Cannot assemble {sorted(unsupported)} from hardware; this implements "
        f"{sorted(TERMS)}. Push-T, for instance, wants target_pose instead."
      )

  @property
  def provider(self) -> str:
    return self._onnx.get_providers()[0]

  @property
  def goal_distance(self) -> float:
    return float(np.linalg.norm(self._goal_position))

  @property
  def goal(self) -> Any:
    return self._goal.copy()

  @goal.setter
  def goal(self, position: Any) -> None:
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,):
      raise ValueError(f"goal must be 3 values; got {position.shape}.")
    self._goal = position

  def observe(self, *, joint_pos: Any, joint_vel: Any, image: Any) -> dict[str, Any]:
    position = self._position = self._mirror_gripper(joint_pos)
    terms = {
      "joint_pos": position - self.metadata.default_joint_pos,
      "joint_vel": self._mirror_gripper(joint_vel),
      "actions": self._last_action,
    }
    if "joint_target" in self.metadata.observation_terms:
      terms["joint_target"] = self._target[:6] - self.metadata.default_joint_pos[:6]
    if "goal_position" in self.metadata.observation_terms:
      ee_position, ee_quaternion = self._kinematics.ee_pose(position)
      self._goal_position = _rotate_by_inverse(
        ee_quaternion, self._goal - ee_position
      )
      terms["goal_position"] = self._goal_position
    if "target_pose" in self.metadata.observation_terms:
      x, y, yaw = self._goal
      terms["target_pose"] = np.array(
        [x, y, TARGET_Z_BASE_M, np.sin(yaw), np.cos(yaw)]
      )
    observation = {
      "obs": np.concatenate(
        [terms[name] for name in self.metadata.observation_terms]
      ).astype(np.float32)[None]
    }
    if self.metadata.needs_camera:
      observation["camera"] = (
        np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
      )
    return observation

  def act(self, *, joint_pos: Any, joint_vel: Any, image: Any) -> Any:
    observation = self.observe(joint_pos=joint_pos, joint_vel=joint_vel, image=image)
    self._network_action = raw_action = self._infer(observation)
    if self.metadata.clip_actions is not None:
      bound = self.metadata.clip_actions
      raw_action = np.clip(raw_action, -bound, bound)
    self._last_action = (
      self._smoothing * raw_action + (1.0 - self._smoothing) * self._last_action
    )
    return self._last_action

  @property
  def network_action(self) -> Any:
    return self._network_action

  @property
  def has_gripper(self) -> bool:
    return self.metadata.action_dim == len(ARM_JOINTS)

  def joint_targets(self, action: Any) -> Any:
    n = self.metadata.action_dim
    base = {
      "absolute": self.metadata.action_offset,
      "relative": self._position,
      "target": self._target,
    }[self.metadata.action_type]
    targets = self.metadata.action_offset.copy()
    delta = self.metadata.action_scale[:n] * action
    if self.metadata.action_clip is not None:
      low, high = self.metadata.action_clip
      delta = np.clip(delta, low[:n], high[:n])
    targets[:n] = base[:n] + delta
    if self.metadata.target_limits is not None:
      low, high = self.metadata.target_limits
      targets[:n] = np.clip(targets[:n], low[:n], high[:n])
    self._target = targets
    return targets

  def warm_up(
    self, *, joint_pos: Any, joint_vel: Any, image: Any, runs: int = 5
  ) -> None:
    self._target = self._mirror_gripper(joint_pos)[: len(ARM_JOINTS)].copy()
    observation = self.observe(joint_pos=joint_pos, joint_vel=joint_vel, image=image)
    for _ in range(runs):
      self._infer(observation)

  def _infer(self, observation: dict[str, Any]) -> Any:
    return self._onnx.run(None, observation)[0].reshape(-1)

  def _mirror_gripper(self, measured: Any) -> Any:
    if len(measured) == len(self.metadata.joint_names):
      return measured
    return np.concatenate([measured, measured[-1:]])


def load_policy(config: Any) -> Policy:
  import onnxruntime as ort

  onnx_session = ort.InferenceSession(
    config.onnx_file, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
  )
  return Policy(
    onnx_session,
    goal=config.goal,
    smoothing=config.motion.action_smoothing,
  )


def _rotate_by_inverse(quaternion: Any, vector: Any) -> Any:
  w, x, y, z = quaternion
  axis = np.array([-x, -y, -z])
  first = np.cross(axis, vector)
  return vector + 2.0 * (w * first + np.cross(axis, first))


__all__ = ["ARM_JOINTS", "TERMS", "Policy", "PolicyMetadata", "load_policy"]
