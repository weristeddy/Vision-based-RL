"""The evaluation scenes and the real rig, all showing the same moment.

Run: uv run python reports/eval_scenes/render_eval_scenes.py figures/scenes

One real Push-T frame (the policy's 224 px camera input) is rebuilt in simulation:
the arm at its recorded joint angles, the T at its pose fitted in that frame, the
goal at its measured pose. Each scene is rendered from the policy camera at 224 px,
so every image differs from the reference only in its own factor. The colour sweeps
have their own chart figures.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import vbrl.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz
from vbrl.rig.measure import measure_object
from vbrl.rig.view import ExternalView
from vbrl.scenes.builder import replace_scene
from vbrl.scenes.presets import GOAL_COLOUR_EVENT
from vbrl.tasks.push_t.goal_marker import GOAL_MARKER_Z

TASK = "Mjlab-PushT-WoodTable-DinoV2ViTS14-AfaFull6-Trossen"
GOAL = Path("artifacts/real_eval/push_t/goal1")
RUN, STEP = GOAL / "DinoV2ViTS14-AfaFull6" / "run4", 16
SCENES = {
  "wood049": "reference",
  "real_table": "real_table",
  "marble019": "marble",
  "ood_light": "light",
}
NOMINAL_OBJECT = (0.20, 0.068, 0.011)
NOMINAL_GOAL = (0.12, 0.16, 0.05)


def fixed(rgb):
  return tuple((value, value) for value in rgb)


def real_state():
  trace = np.load(RUN / "trace.npz")
  frame = trace["external_frames"][STEP]
  fit = measure_object(ExternalView.load(424, 240).cropped(224), frame, "push_t")
  if not fit.reliable:
    raise SystemExit(f"The T fit at step {STEP} is unreliable (IoU {fit.iou:.2f}).")
  goal = json.loads((GOAL / "goal.json").read_text())["goal"]
  # The joints are read just after the frame arrives; move them back to the moment the
  # frame was exposed: its age on arrival plus half the 8 ms exposure.
  lag = (trace["camera_age_ms"][STEP] + 4.0) / 1000.0
  joints = trace["joint_pos"][STEP][:6] - trace["joint_vel"][STEP][:6] * lag
  return frame, joints, fit.pose, goal


def render(scene: str, joints, t_pose, goal) -> Image.Image:
  cfg = replace_scene(load_env_cfg(TASK, play=True), scene=scene)
  cfg.scene.num_envs, cfg.seed = 1, 0
  cfg.events["object_color"].params["ranges"] = dict(enumerate(fixed(NOMINAL_OBJECT)))
  cfg.events[GOAL_COLOUR_EVENT].params["ranges"] = fixed(NOMINAL_GOAL)
  for name in ("camera_position", "camera_orientation", "camera_fovy"):
    cfg.events.pop(name, None)
  cfg.events["light_direction"].params["ranges"] = dict(enumerate(fixed((0.18, 0.28, -1.0))))
  cfg.events["light_intensity"].params.update(
    diffuse=(0.85, 0.85),
    specular=(0.08, 0.08),
    ambient=(0.10, 0.10),
    warmth=(0.0, 0.0),
    shadow_probability=1.0,
  )
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  try:
    env.seed(0)
    env.reset()
    device = env.device
    robot, obj = env.scene["robot"], env.scene["object"]
    base = robot.data.root_link_pos_w[0]

    joint_pos = robot.data.joint_pos.clone()
    joint_pos[0, :6] = torch.tensor(joints, device=device)
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))

    def pose(x, y, z, yaw):
      zero = torch.zeros(1, device=device)
      position = torch.tensor([[base[0] + x, base[1] + y, z]], device=device)
      yaw = torch.tensor([yaw], device=device, dtype=torch.float32)
      return torch.cat((position, quat_from_euler_xyz(zero, zero, yaw)), dim=-1)

    z = float(obj.data.root_link_pos_w[0, 2])
    obj.write_root_link_pose_to_sim(pose(t_pose[0], t_pose[1], z, t_pose[2]))
    obj.write_root_link_velocity_to_sim(torch.zeros(1, 6, device=device))
    command = env.command_manager.get_term("push_t_goal")
    command.target_pos[0, :2] = base[:2] + torch.tensor(goal[:2], device=device)
    command.target_yaw[0] = goal[2]
    marker_z = float(env.scene.env_origins[0, 2]) + GOAL_MARKER_Z
    command._goal_marker.write_mocap_pose_to_sim(pose(goal[0], goal[1], marker_z, goal[2]))

    # No physics step: the controller's command delay would pull the arm off its
    # recorded pose (4 degrees in a single step). Render the set state directly.
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.sim.sense()
    joint_error = (robot.data.joint_pos[0, :6] - joint_pos[0, :6]).abs().max()
    assert joint_error < 1e-6, f"arm off its recorded pose by {joint_error} rad"
    image = env.observation_manager.compute(update_history=True)["camera"][0].float()
  finally:
    env.close()
  image = image * (255.0 if image.max() <= 1.0 else 1.0)
  if image.shape[0] == 3:
    image = image.permute(1, 2, 0)
  return Image.fromarray(image.clamp(0, 255).byte().cpu().numpy())


def main(folder: str) -> None:
  torch.set_grad_enabled(False)
  out = Path(folder)
  out.mkdir(parents=True, exist_ok=True)
  frame, joints, t_pose, goal = real_state()
  Image.fromarray(frame).save(out / "real_robot.png")
  for scene, name in SCENES.items():
    render(scene, joints, t_pose, goal).save(out / f"{name}.png")
  print(f"{RUN.name} step {STEP}: T {np.round(t_pose, 3)}, goal {np.round(goal, 3)} -> {out}")


if __name__ == "__main__":
  main(sys.argv[1])
