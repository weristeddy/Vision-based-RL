from __future__ import annotations

from typing import Any

import numpy as np

JOINTS = tuple(f"robot/joint_{i}" for i in range(6))
PADS = ("robot/gripper_left_pad", "robot/gripper_right_pad")


def _rotation(yaw: float) -> np.ndarray:
  down = np.array([0.0, 0.0, -1.0])
  across = np.array([np.cos(yaw), np.sin(yaw), 0.0])
  return np.column_stack((down, across, np.cross(down, across)))


def placement_table(
  model: Any,
  seed: np.ndarray,
  limits: np.ndarray,
  xs: np.ndarray,
  ys: np.ndarray,
  yaws: np.ndarray,
  height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  import mujoco

  data = mujoco.MjData(model)
  joints = [model.joint(name).id for name in JOINTS]
  qpos, dofs = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
  low, high = limits.T
  site = model.site("robot/ee_site").id
  table = model.geom("table/table_top").id
  pads = [model.geom(name).id for name in PADS]
  jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))

  data.qpos[qpos] = seed
  mujoco.mj_forward(model, data)
  base = data.xpos[model.body("robot/base_link").id].copy()
  top = data.geom_xpos[table, 2] + model.geom_size[table, 2]
  pad_mid = data.geom_xpos[pads].mean(0) - data.site_xpos[site]
  offset = data.site_xmat[site].reshape(3, 3).T @ pad_mid

  def touches_table(q: np.ndarray) -> bool:
    data.qpos[qpos] = q
    mujoco.mj_forward(model, data)
    return any(
      table in (c.geom1, c.geom2)
      and model.geom(c.geom1 + c.geom2 - table).name.startswith("robot/")
      for c in data.contact[: data.ncon]
    )

  def solve(q: np.ndarray, position: np.ndarray, rotation: np.ndarray):
    for _ in range(60):
      data.qpos[qpos] = q
      mujoco.mj_kinematics(model, data)
      mujoco.mj_comPos(model, data)
      current = data.site_xmat[site].reshape(3, 3)
      error = np.concatenate(
        (position - data.site_xpos[site], 0.5 * np.cross(current.T, rotation.T).sum(0))
      )
      if np.linalg.norm(error[:3]) < 2e-3 and np.linalg.norm(error[3:]) < 0.035:
        return q
      mujoco.mj_jacSite(model, data, jacp, jacr, site)
      jac = np.vstack((jacp[:, dofs], jacr[:, dofs]))
      step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), error)
      q = np.clip(q + step, low, high)
    return None

  joint_pos = np.full((len(xs), len(ys), len(yaws), 6), np.nan)
  warm = np.tile(seed, (len(yaws), 1))
  for i, x in enumerate(xs):
    for j in range(len(ys)) if i % 2 == 0 else reversed(range(len(ys))):
      for k, yaw in enumerate(yaws):
        rotation = _rotation(yaw)
        pad = np.array([base[0] + x, base[1] + ys[j], top + height])
        q = solve(warm[k].copy(), pad - rotation @ offset, rotation)
        if q is not None and not touches_table(q):
          joint_pos[i, j, k] = warm[k] = q

  grid = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
  joint_pos = joint_pos.reshape(len(grid), len(yaws), 6)
  return grid, joint_pos, ~np.isnan(joint_pos[..., 0])


__all__ = ["placement_table"]
