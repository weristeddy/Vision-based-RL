from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

JOINTS = tuple(f"joint_{i}" for i in range(6))
SENSORS = tuple(f"{joint}_{kind}" for kind in ("pos", "vel") for joint in JOINTS)
TIMESTEP = 0.005
WINDOW_S = 4.0
SENSOR_HZ = 200.0
MAX_DELAY_S = 0.04
SCALE = np.r_[np.full(6, 0.005), np.full(6, 0.1)]


def load(path: Path):
  data = np.load(path)
  return (
    data["command_time"], data["sent"], data["sample_time"],
    data["positions"], data["velocities"], float(data["goal_time"]),
  )


def make_spec():
  import mujoco

  from vbrl.asset_zoo.robots import get_robot

  spec = get_robot("trossen_identified").make_entity_cfg().spec_fn()
  spec.option.timestep = TIMESTEP
  spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
  for name in SENSORS:
    joint, kind = name.rsplit("_", 1)
    spec.add_sensor(
      name=name,
      type=mujoco.mjtSensor.mjSENS_JOINTPOS if kind == "pos" else mujoco.mjtSensor.mjSENS_JOINTVEL,
      objtype=mujoco.mjtObj.mjOBJ_JOINT,
      objname=joint,
    )
  return spec


def windows(path: Path, model):
  from mujoco import sysid

  command_time, sent, sample_time, positions, velocities, goal_time = load(path)
  measured = np.c_[positions[:, :6], velocities[:, :6]]
  per_window = int(round(WINDOW_S / goal_time))
  grid = np.arange(1, int(WINDOW_S * SENSOR_HZ)) / SENSOR_HZ
  gaps = np.diff(command_time)
  for start in range(1, len(command_time) - per_window, per_window):
    stop = start + per_window
    if np.abs(gaps[start - 1 : stop] - goal_time).max() > 0.015:
      continue
    t0 = command_time[start]
    control = sysid.TimeSeries(
      np.r_[0.0, command_time[start:stop] - t0 + goal_time], sent[start - 1 : stop]
    )
    data = np.stack([np.interp(t0 + grid, sample_time, column) for column in measured.T], 1)
    sensors = sysid.TimeSeries.from_names(grid, data, model, names=list(SENSORS))
    qpos, qvel = np.zeros(model.nq), np.zeros(model.nv)
    for i, joint in enumerate(JOINTS):
      qpos[model.jnt_qposadr[model.joint(joint).id]] = np.interp(t0, sample_time, positions[:, i])
      qvel[model.jnt_dofadr[model.joint(joint).id]] = np.interp(t0, sample_time, velocities[:, i])
    for joint in ("left_carriage_joint", "right_carriage_joint"):
      qpos[model.jnt_qposadr[model.joint(joint).id]] = np.interp(t0, sample_time, positions[:, 6])
    yield f"{path.stem}_{start}", sysid.create_initial_state(model, qpos, qvel), control, sensors


def sequences(name: str, spec, paths):
  from mujoco import sysid

  model = spec.compile()
  names, states, controls, sensors = zip(*(w for path in paths for w in windows(path, model)), strict=True)
  return sysid.ModelSequences(name, spec, names, states, controls, sensors)


def parameters(model):
  from mujoco import sysid

  params = sysid.ParameterDict()
  for joint in JOINTS:
    dof = model.jnt_dofadr[model.joint(joint).id]
    kp = float(model.actuator_gainprm[model.actuator(joint).id, 0])
    params.add(sysid.Parameter(
      f"{joint}_kp", kp, 0.2 * kp, 50.0 * kp,
      modifier=lambda spec, p, j=joint: sysid.apply_pgain(spec, j, p.value[0]),
    ))
    params.add(sysid.Parameter(
      f"{joint}_damping", float(model.dof_damping[dof]), 0.01, 50.0,
      modifier=lambda spec, p, j=joint: setattr(spec.joint(j), "damping", [p.value[0], 0.0, 0.0]),
    ))
    params.add(sysid.Parameter(
      f"{joint}_armature", float(model.dof_armature[dof]), 0.0, 2.0,
      modifier=lambda spec, p, j=joint: setattr(spec.joint(j), "armature", p.value[0]),
    ))
    params.add(sysid.Parameter(
      f"{joint}_frictionloss", float(model.dof_frictionloss[dof]), 0.0, 6.0,
      modifier=lambda spec, p, j=joint: setattr(spec.joint(j), "frictionloss", p.value[0]),
    ))
  params.add(sysid.Parameter("delay", 0.005, 0.0, MAX_DELAY_S))
  return params


def delayed_residual(params, predicted, measured, model, return_pred_all, state=None):
  from mujoco import sysid

  measured = sysid.apply_delayed_ts_window(measured, predicted, 0.0, MAX_DELAY_S)
  predicted = predicted.resample(measured.times - params["delay"].value[0])
  return (measured.data - predicted.data) / SCALE, predicted, measured


def tracking_rms(params, model_sequences):
  from mujoco import sysid

  _, predicted, measured = sysid.residual(
    params.as_vector(), params, [model_sequences],
    modify_residual=delayed_residual, return_pred_all=True,
  )
  error = np.concatenate([
    m.data - p.data for ps, ms in zip(predicted, measured, strict=True) for p, m in zip(ps, ms, strict=True)
  ])
  return np.sqrt((error**2).mean(0))


def main() -> None:
  import copy

  from mujoco import sysid

  parser = argparse.ArgumentParser()
  parser.add_argument("--fit", nargs="+", type=Path, default=[
    Path("artifacts/deployment/sysid_run0.npz"),
    Path("artifacts/deployment/sysid_run1.npz"),
    Path("artifacts/deployment/sysid_broadband0.npz"),
  ])
  parser.add_argument("--holdout", nargs="+", type=Path, default=[
    Path("artifacts/deployment/sysid_run2.npz"),
    Path("artifacts/deployment/sysid_task0.npz"),
    Path("artifacts/deployment/sysid_task1.npz"),
    Path("artifacts/deployment/sysid_broadband1.npz"),
  ])
  parser.add_argument("--out", type=Path, default=Path("artifacts/sysid/trossen_identified"))
  parser.add_argument("--max-iters", type=int, default=100)
  arguments = parser.parse_args()

  spec = make_spec()
  nominal = parameters(spec.compile())
  fit = sequences("fit", spec, arguments.fit)
  initial = copy.deepcopy(nominal)
  initial.move_off_bounds()
  low, high = initial.get_bounds()
  residual_fn = sysid.build_residual_fn(models_sequences=[fit], modify_residual=delayed_residual)
  fitted, result = sysid.optimize(
    initial_params=initial, residual_fn=residual_fn,
    optimizer="mujoco", max_iters=arguments.max_iters, x_scale=high - low,
  )
  sysid.save_results(arguments.out, [fit], nominal, fitted, result, residual_fn)
  sysid.default_report(
    [fit], nominal, fitted, residual_fn, result,
    title="Trossen WXAI", generate_videos=False,
  ).save(str(arguments.out / "report.html"))

  print(f"{len(fit.sequence_name)} fit windows")
  print("joint     " + "".join(f"{j:>10}" for j in JOINTS))
  for kind in ("kp", "damping", "armature", "frictionloss"):
    print(f"{kind:<10}" + "".join(f"{fitted[f'{j}_{kind}'].value[0]:>10.3f}" for j in JOINTS))
    print(f"{'  menag.':<10}" + "".join(f"{nominal[f'{j}_{kind}'].value[0]:>10.3f}" for j in JOINTS))
  print(f"delay {fitted['delay'].value[0] * 1000:.1f} ms")
  for path in [*arguments.fit, *arguments.holdout]:
    held = sequences(path.stem, make_spec(), [path])
    before, after = tracking_rms(nominal, held)[:6], tracking_rms(fitted, held)[:6]
    print(f"{path.stem:<14} {len(held.sequence_name):>2} windows  position RMS mrad "
          f"menagerie {np.round(before * 1000, 1).tolist()}  fitted {np.round(after * 1000, 1).tolist()}")


if __name__ == "__main__":
  main()
