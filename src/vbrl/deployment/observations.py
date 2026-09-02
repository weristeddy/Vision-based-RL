"""Rebuild the policy's observation from a real arm and a real camera.

The actor group is ``joint_pos, joint_vel, actions, goal_position`` -- read off
the observation manager rather than written down here, so a re-registered task
cannot leave this file quietly assembling the wrong vector.

Three of the four terms are direct sensor reads. ``goal_position`` is not: it is
the lift target expressed in the end-effector frame, and in training it came
from the command manager. On hardware the target is a configured parameter and
the end-effector frame comes from forward kinematics, which is the only place
this module does arithmetic the simulator would otherwise have done.
"""

from __future__ import annotations

from typing import Any


class ObservationAssembler:
  """Sensor reads in, the policy's observation dict out."""

  def __init__(self, spec: Any, *, goal: tuple[float, float, float], device: str) -> None:
    import numpy as np
    import torch

    self._spec = spec
    self._device = device
    self._torch = torch
    self._goal = np.asarray(goal, dtype=np.float64)
    self._num_actions = int(spec.action_scale.shape[-1])
    self._last_action = np.zeros(self._num_actions, dtype=np.float64)

    unsupported = set(spec.actor_terms) - set(self._TERMS)
    if unsupported:
      raise ValueError(
        f"Cannot assemble actor terms {sorted(unsupported)} from hardware; "
        f"this module implements {sorted(self._TERMS)}. A task needing more "
        "state than the arm reports cannot be deployed as-is."
      )

  # Term name -> method name. Keeps the dispatch explicit and greppable.
  _TERMS = {
    "joint_pos": "_joint_pos",
    "joint_vel": "_joint_vel",
    "actions": "_actions",
    "goal_position": "_goal_position",
  }

  def expand_joints(self, measured: Any) -> Any:
    """Map the arm's 7 reported values onto the model's joint vector.

    The MJCF gives the gripper two mirrored carriage joints while the hardware
    reports one. Duplicating the measurement is what keeps ``joint_pos_rel`` the
    same width the policy trained on.
    """
    import numpy as np

    measured = np.asarray(measured, dtype=np.float64).reshape(-1)
    width = len(self._spec.joint_names)
    if len(measured) == width:
      return measured
    if len(measured) == width - 1:
      return np.concatenate([measured[:-1], measured[-1:], measured[-1:]])
    raise ValueError(
      f"The arm reported {len(measured)} joints; the model has {width} "
      f"({self._spec.joint_names}), so neither a direct nor a mirrored-gripper "
      "mapping applies."
    )

  def _joint_pos(self, state: dict[str, Any]) -> Any:
    return state["joint_pos"] - self._spec.default_joint_pos

  def _joint_vel(self, state: dict[str, Any]) -> Any:
    # mjlab's joint_vel_rel subtracts the default velocity, which is zero.
    return state["joint_vel"]

  def _actions(self, state: dict[str, Any]) -> Any:
    del state
    return self._last_action

  def _goal_position(self, state: dict[str, Any]) -> Any:
    import numpy as np

    ee_position, ee_quaternion = self._spec.ee_pose(state["joint_pos"])
    return _quat_apply_inverse(ee_quaternion, np.asarray(self._goal) - ee_position)

  def build(self, *, joint_pos: Any, joint_vel: Any, rgb: Any | None) -> Any:
    """Assemble the observation dict the policy is called with."""
    import numpy as np
    import torch

    state = {
      "joint_pos": self.expand_joints(joint_pos),
      "joint_vel": self.expand_joints(joint_vel),
    }
    parts = [
      np.asarray(getattr(self, self._TERMS[term])(state), dtype=np.float64).reshape(-1)
      for term in self._spec.actor_terms
    ]
    actor = torch.as_tensor(
      np.concatenate(parts), dtype=torch.float32, device=self._device
    ).unsqueeze(0)

    observations: dict[str, Any] = {"actor": actor}
    if self._spec.camera_group is not None:
      if rgb is None:
        raise ValueError(
          f"The task declares a {self._spec.camera_group!r} observation group, "
          "so a camera frame is required."
        )
      # The camera observation is channel-first: the manager emits
      # (N, 3, H, W), while a RealSense frame arrives as (H, W, 3).
      frame = np.asarray(rgb)
      if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Expected an (H, W, 3) RGB frame; got shape {frame.shape}.")
      observations[self._spec.camera_group] = torch.as_tensor(
        np.ascontiguousarray(frame.transpose(2, 0, 1)),
        dtype=torch.uint8,
        device=self._device,
      ).unsqueeze(0)
    return observations

  def record_action(self, action: Any) -> None:
    """Remember what was commanded; the next observation reports it."""
    import numpy as np

    self._last_action = np.asarray(action, dtype=np.float64).reshape(-1)

  def effective_action(self, sent_targets: Any) -> Any:
    """Invert ``joint_targets``: the action the commanded pose corresponds to.

    The clamps in ``TrossenArm.command`` mean the pose actually commanded is not
    always the one the policy asked for. In simulation the two never diverge --
    the target is applied in full -- so ``last_action`` there is always the
    action that produced the current target. Reporting the raw action on
    hardware breaks that invariant: the policy reads back a move that never
    happened, sees no response, and pushes harder every step. Feeding back the
    commanded pose keeps the observation honest about what the arm did.
    """
    import numpy as np

    sent = np.asarray(sent_targets, dtype=np.float64).reshape(-1)
    return (sent - self._spec.action_offset) / self._spec.action_scale

  def joint_targets(self, action: Any) -> Any:
    """``offset + scale * action``, the mapping the action term applies in sim."""
    import numpy as np

    action = np.asarray(action, dtype=np.float64).reshape(-1)
    return self._spec.action_offset + self._spec.action_scale * action


def _quat_apply_inverse(quaternion: Any, vector: Any) -> Any:
  """Rotate ``vector`` by the inverse of a ``(w, x, y, z)`` quaternion."""
  import numpy as np

  w, x, y, z = (float(v) for v in np.asarray(quaternion).reshape(-1))
  # Inverse of a unit quaternion is its conjugate.
  conjugate = np.array([w, -x, -y, -z])
  return _quat_apply(conjugate, vector)


def _quat_apply(quaternion: Any, vector: Any) -> Any:
  import numpy as np

  w, x, y, z = (float(v) for v in np.asarray(quaternion).reshape(-1))
  vector = np.asarray(vector, dtype=np.float64).reshape(3)
  axis = np.array([x, y, z])
  # v' = v + 2w(a x v) + 2a x (a x v)
  first = np.cross(axis, vector)
  second = np.cross(axis, first)
  return vector + 2.0 * (w * first + second)


__all__ = ["ObservationAssembler"]
