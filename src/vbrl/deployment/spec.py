"""What the simulator knows that the hardware does not.

Deployment builds one simulated environment and never steps it. That sounds
wasteful, and it is deliberate: the constants a real arm has to reproduce --
joint order, the default pose every observation is relative to, the per-joint
action scale, the end-effector site, and the kinematic model behind
``goal_position`` -- are all *derived* in the sim from the same MJCF the policy
trained against. Reading them back out is exact; restating them here would be a
second copy free to drift from the task ID that owns them.

The environment is also what ``load_trained_policy`` needs in order to build the
actor at all, so it is not an extra cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RobotSpec:
  """Sim-derived constants plus a CPU model for forward kinematics."""

  joint_names: tuple[str, ...]
  default_joint_pos: Any
  """``numpy`` array; ``joint_pos_rel`` and the action offset are relative to it."""
  action_scale: Any
  """Per-action scale; ``target = action_offset + action_scale * action``."""
  action_offset: Any
  """What ``use_default_offset`` resolved to: the default pose of the 7 actuated
  joints. Narrower than ``default_joint_pos``, because the gripper's two mirrored
  carriage joints share one action."""
  actor_terms: tuple[str, ...]
  """Actor observation term names, in the order the manager concatenates them."""
  camera_group: str | None
  ee_site_name: str
  _mj_model: Any = None
  _mj_data: Any = None

  @classmethod
  def from_env(cls, env: Any, *, ee_site_name: str) -> RobotSpec:
    import numpy as np

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    joint_names = tuple(robot.joint_names)
    default = robot.data.default_joint_pos[0].detach().cpu().numpy().astype(np.float64)

    action_term = unwrapped.action_manager.get_term("joint_pos")
    scale = getattr(action_term, "_scale", None)
    if scale is None:
      raise RuntimeError(
        "The joint_pos action term exposes no _scale; deployment cannot "
        "reproduce the mapping from policy output to joint target."
      )
      # A missing scale would silently change how far every action moves.
    scale_array = _as_row(scale)
    offset = getattr(action_term, "_offset", None)
    if offset is None:
      raise RuntimeError(
        "The joint_pos action term exposes no _offset; with "
        "use_default_offset=True that offset is the default pose every action "
        "is relative to, and deployment cannot reconstruct the target without it."
      )
    offset_array = _as_row(offset)

    manager = unwrapped.observation_manager
    groups = dict(manager.active_terms)
    if "actor" not in groups:
      raise RuntimeError(
        f"No 'actor' observation group; found {sorted(groups)}. Deployment "
        "assembles that group by name."
      )
    camera_group = "camera" if "camera" in groups else None

    return cls(
      joint_names=joint_names,
      default_joint_pos=default,
      action_scale=scale_array,
      action_offset=offset_array,
      actor_terms=tuple(groups["actor"]),
      camera_group=camera_group,
      ee_site_name=ee_site_name,
      _mj_model=_mj_model_of(unwrapped),
      _mj_data=None,
    )

  def ee_pose(self, joint_pos: Any) -> tuple[Any, Any]:
    """Forward kinematics: end-effector position and quaternion in the base frame.

    Taken from the MJCF rather than the arm SDK's own Cartesian read, because
    ``goal_position`` was defined by *this* site and this quaternion convention.
    A frame or handedness mismatch there is invisible in the numbers and wrong
    in the behaviour.
    """
    import mujoco
    import numpy as np

    if self._mj_model is None:
      raise RuntimeError("No MuJoCo model available for forward kinematics.")
    if self._mj_data is None:
      self._mj_data = mujoco.MjData(self._mj_model)

    data = self._mj_data
    data.qpos[: len(joint_pos)] = np.asarray(joint_pos, dtype=np.float64)
    mujoco.mj_kinematics(self._mj_model, data)

    site_id = _site_id(self._mj_model, self.ee_site_name)

    position = np.array(data.site_xpos[site_id], dtype=np.float64)
    quaternion = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, data.site_xmat[site_id])
    return position, quaternion


def _site_id(mj_model: Any, name: str) -> int:
  """Resolve a site by name, tolerating the scene's attachment prefix.

  The robot is attached into a composed scene, so the MJCF's ``ee_site`` becomes
  something like ``wxai_follower/ee_site``. Never fall through to
  ``mj_name2id``'s -1: indexing ``site_xpos[-1]`` returns the *last* site and
  forward kinematics then silently reports the wrong frame.
  """
  import mujoco

  exact = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
  if exact >= 0:
    return exact

  names = []
  for site in range(mj_model.nsite):
    resolved = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, site)
    if resolved is not None:
      names.append((site, resolved))

  matches = [site for site, resolved in names if resolved.split("/")[-1] == name]
  if len(matches) == 1:
    return matches[0]
  if not matches:
    raise RuntimeError(
      f"No site named {name!r} in the model; it has {[n for _, n in names]}."
    )
  raise RuntimeError(
    f"Site name {name!r} is ambiguous: {[names[m][1] for m in matches]}. "
    "Name it fully to disambiguate."
  )


def _as_row(value: Any) -> Any:
  """A torch tensor or array of shape (1, n) or (n,) as a flat numpy array."""
  import numpy as np

  raw = value.detach().cpu().numpy() if hasattr(value, "detach") else value
  return np.asarray(raw, dtype=np.float64).reshape(-1)


def _mj_model_of(unwrapped: Any) -> Any:
  """Find the CPU MuJoCo model behind an mjlab environment."""
  for holder, attribute in (
    (getattr(unwrapped, "sim", None), "mj_model"),
    (getattr(unwrapped, "sim", None), "model"),
    (unwrapped, "mj_model"),
    (getattr(unwrapped, "scene", None), "mj_model"),
  ):
    candidate = getattr(holder, attribute, None) if holder is not None else None
    if candidate is not None and hasattr(candidate, "nq"):
      return candidate
  return None


__all__ = ["RobotSpec"]
