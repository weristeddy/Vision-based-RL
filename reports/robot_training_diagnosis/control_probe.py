"""Native MuJoCo control-mode probe; leaves robot XMLs unchanged."""
import json
from pathlib import Path
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
HOME=np.array([0,1.33,1.42,-1.3,0,0,0,0.])
fig,axes=plt.subplots(1,2,figsize=(10,4))
results=[]
for filename in ['wxai_realistic.xml','wxai_identified.xml']:
  m=mujoco.MjModel.from_xml_path(str(ROOT/'src/vbrl/asset_zoo/robots/trossen_wxai/xmls'/filename))
  m.opt.timestep=.005
  m.opt.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
  for mode in ['relative_substep','held_target']:
    d=mujoco.MjData(m); d.qpos[:]=HOME; mujoco.mj_forward(m,d)
    qs=[];vs=[]
    for k in range(400):
      if mode=='relative_substep':
        d.ctrl[:6]=d.qpos[:6]
        if k<20: d.ctrl[2]+=.03
      else:
        d.ctrl[:6]=HOME[:6]; d.ctrl[2]+=.03
      mujoco.mj_step(m,d);qs.append(d.qpos[:6].copy());vs.append(d.qvel[:6].copy())
    qs=np.array(qs);vs=np.array(vs)
    label=('Old' if 'realistic' in filename else 'New')+' / '+mode
    axes[0].plot(np.arange(400)*.005,qs[:,2]-HOME[2],label=label)
    axes[1].plot(np.arange(400)*.005,vs[:,2],label=label)
    results.append(dict(model=filename,mode=mode,elbow_final_displacement_rad=float(qs[-1,2]-HOME[2]),elbow_peak_speed_rad_s=float(np.abs(vs[:,2]).max())))
for ax in axes:ax.set_xlabel('Time (s)');ax.grid(alpha=.2)
axes[0].set_ylabel('Elbow displacement (rad)');axes[1].set_ylabel('Elbow velocity (rad/s)');axes[0].legend(fontsize=7)
fig.suptitle('Same 0.03 rad error: refreshed for 0.1 s vs one held position target')
fig.tight_layout();fig.savefig(Path(__file__).with_name('control_response.png'),dpi=160)
Path(__file__).with_name('control_probe.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
