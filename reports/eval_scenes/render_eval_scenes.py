"""One image per evaluation scene, the same episode in each, for the report.

Run: uv run python reports/eval_scenes/render_eval_scenes.py reports/eval_scenes/eval_scenes.png

The evaluation randomizes every factor not under test as in training. For the
figure those sit at one nominal value, the same in both tasks, so each column
differs from wood049 only in its own factor.
"""

import sys

import torch
from PIL import Image, ImageDraw, ImageFont

import vbrl.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from vbrl.scenes.builder import replace_scene
from vbrl.scenes.presets import (
  EVALUATION_SCENES,
  GOAL_COLOUR_EVENT,
  REAL_TABLE_GOAL_COLOUR_RANGE,
  REAL_TABLE_OBJECT_COLOUR_RANGE,
  get_preset,
)

CAPTIONS = {
  "wood049": "in distribution",
  "real_table": "table: rig photo",
  "marble019": "table: marble",
  "ood_light": "light colour\n(Colosseum-style)",
  "object_blue": "object colour",
  "goal_blue": "goal colour",
  "object_blue_goal_purple": "object and\ngoal colour",
  "colour_swap": "object green,\ngoal bordeaux",
}
TASKS = [
  ("Push-T\nexternal camera", "Mjlab-PushT-WoodTable-DinoV2ViTS14-AfaFull6-Trossen"),
  ("Lift-Cube\nwrist camera", "Mjlab-LiftCube-WoodTable-DinoV2ViTS14-AfaFull6-Trossen"),
]
NOMINAL_OBJECT = (0.20, 0.068, 0.011)
NOMINAL_GOAL = (0.12, 0.16, 0.05)
ENV = 1


def fixed(rgb):
  return tuple((value, value) for value in rgb)


def render(task: str, scene: str):
  preset = get_preset(scene)
  cfg = replace_scene(load_env_cfg(task, play=True), scene=scene)
  cfg.scene.num_envs, cfg.seed = 3, 0
  if preset.object_colour_range == REAL_TABLE_OBJECT_COLOUR_RANGE:
    cfg.events["object_color"].params["ranges"] = dict(enumerate(fixed(NOMINAL_OBJECT)))
  if GOAL_COLOUR_EVENT in cfg.events and preset.goal_colour_range in (
    None,
    REAL_TABLE_GOAL_COLOUR_RANGE,
  ):
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
    actions = torch.zeros(3, env.action_manager.total_action_dim, device=env.device)
    image = env.step(actions)[0]["camera"].float()
    image = image * (255.0 if image.max() <= 1.0 else 1.0)
    if image.shape[1] == 3:
      image = image.permute(0, 2, 3, 1)
    return Image.fromarray(image[ENV].clamp(0, 255).byte().cpu().numpy())
  finally:
    env.close()


def main(path: str) -> None:
  torch.set_grad_enabled(False)
  size, left, top, gap = 224, 130, 58, 8
  font = ImageFont.load_default(size=15)
  small = ImageFont.load_default(size=12)
  sheet = Image.new(
    "RGB",
    (left + len(EVALUATION_SCENES) * (size + gap), top + len(TASKS) * (size + gap)),
    "white",
  )
  draw = ImageDraw.Draw(sheet)
  for column, scene in enumerate(EVALUATION_SCENES):
    x = left + column * (size + gap)
    draw.text((x + 2, 4), scene, font=font, fill=(11, 11, 11))
    draw.multiline_text((x + 2, 24), CAPTIONS[scene], font=small, fill=(82, 81, 78))
  for row, (label, task) in enumerate(TASKS):
    y = top + row * (size + gap)
    draw.multiline_text((6, y + size // 2 - 16), label, font=font, fill=(11, 11, 11))
    for column, scene in enumerate(EVALUATION_SCENES):
      sheet.paste(render(task, scene), (left + column * (size + gap), y))
  sheet.save(path)


if __name__ == "__main__":
  main(sys.argv[1])
