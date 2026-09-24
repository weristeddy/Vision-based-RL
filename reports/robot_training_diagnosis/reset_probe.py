"""Reproduce startup command-delay kick without modifying training source."""
import contextlib
import io
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import EventTermCfg
from vbrl.asset_zoo.objects import PUSH_T_XML
from vbrl.asset_zoo.robots.trossen_wxai import make_wxai_identified, make_wxai_realistic
from vbrl.scenes.builder import apply_scene
from vbrl.tasks.push_t.push_t_env_cfg import build_env_cfg


def prime_targets(env, env_ids):
  robot = env.scene['robot']
  ids = slice(None) if env_ids is None else env_ids
  robot.set_joint_position_target(robot.data.joint_pos[ids], env_ids=ids)


def run(name, factory, delay=True, prime=False):
  robot_def = factory()
  cfg = build_env_cfg(robot=robot_def, object_name='object', action_scale=.03)
  apply_scene(cfg, scene='default', robot=robot_def, camera_view=None,
              object_xml=PUSH_T_XML, object_name='object')
  cfg.scene.num_envs = 32
  cfg.seed = 0
  if not delay:
    for act in cfg.scene.entities['robot'].articulation.actuators:
      act.delay_max_lag = 0
  if prime:
    cfg.events['prime_position_targets'] = EventTermCfg(func=prime_targets, mode='reset')
  with contextlib.redirect_stdout(io.StringIO()):
    env = ManagerBasedRlEnv(cfg, device='cuda:0')
    env.reset(seed=0)
  robot = env.scene['robot']
  q0 = robot.data.joint_pos.clone()
  target_error = float((robot.data.joint_pos_target[:,:6]-q0[:,:6]).abs().max())
  q = []; v = []
  for _ in range(100):
    env.step(torch.zeros((32,6),device='cuda:0'))
    q.append(robot.data.joint_pos[:,:6].clone())
    v.append(robot.data.joint_vel[:,:6].clone())
  q = torch.stack(q); v = torch.stack(v)
  result = dict(case=name, robot=str(robot_def.xml_path),
    initial_max_target_error_rad=target_error,
    max_drift_rad=float((q-q0[:,:6]).abs().max()),
    first_step_max_velocity_rad_s=float(v[0].abs().max()),
    max_velocity_rad_s=float(v.abs().max()),
    max_drift_per_joint_rad=(q-q0[:,:6]).abs().amax(dim=(0,1)).tolist(),
    worlds_drifting_over_001rad=int(((q-q0[:,:6]).abs().amax(dim=(0,2))>.01).sum()))
  # Repeat explicit partial reset to exercise selective initialization.
  ids=torch.tensor([0,2,4,6],device=env.device)
  env.reset(seed=1,env_ids=ids); start=robot.data.joint_pos[ids,:6].clone()
  for _ in range(50): env.step(torch.zeros((32,6),device=env.device))
  result['partial_reset_drift_rad']=float((robot.data.joint_pos[ids,:6]-start).abs().max())
  env.close()
  print(json.dumps(result),flush=True)
  return result


if __name__ == '__main__':
  results = [run('identified_current_reset',make_wxai_identified),
             run('identified_no_delay',make_wxai_identified,delay=False),
             run('identified_primed_targets',make_wxai_identified,prime=True),
             run('realistic_current_reset',make_wxai_realistic)]
  Path(__file__).with_name('reset_probe.json').write_text(json.dumps(results,indent=2))
