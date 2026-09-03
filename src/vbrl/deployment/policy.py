from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vbrl.deployment.kinematics import Kinematics

TERMS = ("joint_pos", "joint_vel", "actions", "goal_position")

# The gripper's two carriage joints mirror one another and the arm reports one,
# so the hardware's seventh value is the left carriage.
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
  """How the trained policy expects to be fed, read out of the ONNX file.

  mjlab attaches all of this at export, so deployment never consults the task
  registry and cannot drift from the run that produced the weights.
  """

  joint_names: tuple[str, ...]
  """The model's joint order: 6 arm joints and both gripper carriages."""
  default_joint_pos: Any
  """The nominal pose. ``joint_pos`` is reported relative to it."""
  action_offset: Any
  action_scale: Any
  """``target = action_offset + action_scale * action``, over the 7 actuated joints."""
  observation_terms: tuple[str, ...]
  """The order the observation vector is concatenated in."""
  needs_camera: bool
  source_run: str
  """The training run the weights came from, for the startup banner."""

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
      needs_camera=any(i.name == "camera" for i in onnx_session.get_inputs()),
      source_run=meta.get("run_path", "unknown"),
    )

  @property
  def home_pose(self) -> Any:
    """The 7 hardware joint targets matching the nominal pose, gripper last."""
    return self.action_offset


class Policy:
  """The exported policy: sensor readings in, an action out.

  ``act`` remembers the action it returns, because the policy's own last action
  is one of its inputs. Keeping that inside means the caller cannot feed back
  something else -- the clamped joint target, say, which stalls the loop.
  """

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
    self._last_action = np.zeros(len(self.metadata.action_scale))
    self._goal_position = np.full(3, np.inf)

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
    """How far the end effector is from the goal, as of the last observation."""
    return float(np.linalg.norm(self._goal_position))

  def observe(self, *, joint_pos: Any, joint_vel: Any, image: Any) -> dict[str, Any]:
    """One step's observation, in the term order the metadata gives."""
    position = self._mirror_gripper(joint_pos)
    ee_position, ee_quaternion = self._kinematics.ee_pose(position)
    self._goal_position = _rotate_by_inverse(ee_quaternion, self._goal - ee_position)

    terms = {
      "joint_pos": position - self.metadata.default_joint_pos,
      "joint_vel": self._mirror_gripper(joint_vel),
      "actions": self._last_action,
      "goal_position": self._goal_position,
    }
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
    """The action to apply, smoothed and remembered as the next `actions` term."""
    observation = self.observe(joint_pos=joint_pos, joint_vel=joint_vel, image=image)
    raw_action = self._infer(observation)
    self._last_action = (
      self._smoothing * raw_action + (1.0 - self._smoothing) * self._last_action
    )
    return self._last_action

  def joint_targets(self, action: Any) -> Any:
    """``offset + scale * action``, the mapping the action term applies in sim."""
    return self.metadata.action_offset + self.metadata.action_scale * action

  def warm_up(
    self, *, joint_pos: Any, joint_vel: Any, image: Any, runs: int = 5
  ) -> None:
    """Pay kernel selection before the first real step, without changing state."""
    observation = self.observe(joint_pos=joint_pos, joint_vel=joint_vel, image=image)
    for _ in range(runs):
      self._infer(observation)

  def _infer(self, observation: dict[str, Any]) -> Any:
    """One forward pass. ``None`` asks onnxruntime for every output."""
    return self._onnx.run(None, observation)[0].reshape(-1)

  def _mirror_gripper(self, measured: Any) -> Any:
    """The arm's 7 values as the model's 8, duplicating the gripper carriage."""
    if len(measured) == len(self.metadata.joint_names):
      return measured
    return np.concatenate([measured, measured[-1:]])


def load_policy(config: Any) -> Policy:
  """Open the exported ONNX and wrap it with the run's goal and smoothing."""
  import onnxruntime as ort

  onnx_session = ort.InferenceSession(
    config.onnx_file, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
  )
  return Policy(
    onnx_session, goal=config.goal, smoothing=config.motion.action_smoothing
  )


def _rotate_by_inverse(quaternion: Any, vector: Any) -> Any:
  """Rotate by the inverse of a ``(w, x, y, z)`` quaternion."""
  w, x, y, z = quaternion
  axis = np.array([-x, -y, -z])
  first = np.cross(axis, vector)
  return vector + 2.0 * (w * first + np.cross(axis, first))


__all__ = ["ARM_JOINTS", "TERMS", "Policy", "PolicyMetadata", "load_policy"]
