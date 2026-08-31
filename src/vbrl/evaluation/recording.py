"""Record a policy as one shot framed on every environment at once.

MJLab's offscreen renderer draws the tracked env plus its nearest neighbours,
which only reads as a grid because the tabletop scenes lay their envs out on
one -- see :func:`vbrl.tasks.utils.lay_out_envs_on_a_grid`. The framing is
derived from the env origins the scene actually built rather than recomputed
from the env count, so it follows whatever grid MJLab chose.

The renderer here is this module's own, not the one ``render_mode="rgb_array"``
installs: a recording needs a camera pulled back off the tracked robot, and
``ManagerBasedRlEnv`` fixes its renderer's camera at construction.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from vbrl.scenes.presets import TABLE_CENTER

# MuJoCo puts the free camera at ``lookat - distance * (cos e cos a, cos e sin
# a, sin e)``, so azimuth 180 is the +x side: the same side the robot faces and
# its own camera looks from, which is the front of the workspace. Azimuth 0 is
# behind the base, looking over its shoulder.
AZIMUTH_DEG = 180.0
# The floor is an infinite plane MuJoCo draws as a quad sized from
# ``model.stat.extent``, which the offscreen renderer derives from the camera
# distance. Below this the camera sees past its edge and the shot gains black
# bands, so this is as low as a front view can sit.
ELEVATION_DEG = -20.0
# The env origins are the robot bases. Each table reaches 0.75 m in front of
# one and 0.35 m to either side, so the grid of origins is not the whole scene.
SCENE_PADDING_M = 1.4
# A flat fit puts the *centre* of the grid at the right size and clips the near
# row, which perspective projects larger than the far one. This is the smallest
# margin that keeps a 3x3 grid whole from the front, so the shot stays close.
FRAME_MARGIN = 1.0

# A video keeps the simulated frame rate; a GIF is a README asset, where file
# size decides whether it loads at all.
VIDEO_SIZE = (1280, 720)
GIF_SIZE = (640, 360)
GIF_FPS = 25


def grid_camera(env: Any, *, width: int, height: int) -> Any:
  """Frame every env in the scene, keeping the task's own render settings.

  Derived from :attr:`Scene.env_origins`, so it is correct for any env count
  and any spacing. ``geom_group`` is inherited rather than chosen: a
  CollisionCam task must record the collision proxies its policy is fed.
  """
  from mjlab.viewer.viewer_config import ViewerConfig

  origins = env.scene.env_origins
  low, high = origins.amin(dim=0), origins.amax(dim=0)
  extent = (high - low).tolist()
  centre = ((low + high) / 2.0).tolist()

  # The grid is a flat patch seen from above and rotated off-axis, so what has
  # to fit is its diagonal across the *horizontal* field: elevation squashes
  # the depth axis, and fitting the span vertically instead leaves the shot
  # mostly background.
  span = math.hypot(extent[0], extent[1]) + SCENE_PADDING_M
  fovy_deg = float(env.cfg.viewer.fovy or env.sim.mj_model.vis.global_.fovy)
  half_fov_x = math.atan(math.tan(math.radians(fovy_deg) / 2.0) * width / height)
  distance = FRAME_MARGIN * span / (2.0 * math.tan(half_fov_x))

  return replace(
    env.cfg.viewer,
    origin_type=ViewerConfig.OriginType.WORLD,
    lookat=(centre[0] + TABLE_CENTER[0], centre[1] + TABLE_CENTER[1], 0.0),
    distance=distance,
    elevation=ELEVATION_DEG,
    azimuth=AZIMUTH_DEG,
    width=width,
    height=height,
    # Every env, not the two neighbours a training video settles for.
    max_extra_envs=max(0, env.num_envs - 1),
  )


def default_output(path: Path) -> tuple[tuple[int, int], int | None]:
  """Return the ``(size, fps)`` defaults this container is worth writing at."""
  if path.suffix.lower() == ".gif":
    return GIF_SIZE, GIF_FPS
  return VIDEO_SIZE, None


def record(
  env: Any,
  policy: Any,
  *,
  path: Path,
  steps: int,
  width: int,
  height: int,
  fps: int,
) -> Path:
  """Step ``policy`` for ``steps`` policy steps and write ``path``.

  ``env`` is the RSL-RL-wrapped environment ``make_policy`` returns, the same
  object evaluation rolls out, so a recording shows the policy under exactly
  the observations it is scored on.

  Frames are written every ``sim_rate / fps`` steps rather than every step, so
  the result plays at wall-clock speed: at one frame per step a 25 fps GIF of a
  50 Hz simulation runs at half speed, and costs twice the frames to do it.
  """
  import os

  # MuJoCo's offscreen renderer needs a GL platform, and left to auto-detection
  # it raises "an OpenGL platform library has not been loaded into this
  # process". `train.py` pins the same backend before it records.
  os.environ.setdefault("MUJOCO_GL", "egl")

  import imageio.v2 as imageio
  import torch
  from mjlab.viewer.offscreen_renderer import OffscreenRenderer

  base = env.unwrapped
  renderer = OffscreenRenderer(
    model=base.sim.mj_model,
    cfg=grid_camera(base, width=width, height=height),
    scene=base.scene,
    sim_model=base.sim.model,
    expanded_fields=base.sim.expanded_fields,
  )
  renderer.initialize()

  def draw(visualizer: Any) -> None:
    # Every env's goal marker, not just the tracked one: a viewer watching nine
    # robots needs to see what each of them is aiming at.
    visualizer.show_all_envs = True
    base.update_visualizers(visualizer)

  callback = draw if hasattr(base, "update_visualizers") else None
  stride = max(1, round((1.0 / base.step_dt) / fps))
  path.parent.mkdir(parents=True, exist_ok=True)
  try:
    observations, _ = env.reset()
    with imageio.get_writer(path, fps=fps) as writer:
      for step in range(steps):
        if step % stride == 0:
          renderer.update(base.sim.data, debug_vis_callback=callback)
          writer.append_data(renderer.render())
        with torch.inference_mode():
          actions = policy(observations)
        observations = env.step(actions)[0]
  finally:
    renderer.close()
  return path


__all__ = [
  "AZIMUTH_DEG",
  "ELEVATION_DEG",
  "GIF_FPS",
  "GIF_SIZE",
  "VIDEO_SIZE",
  "default_output",
  "grid_camera",
  "record",
]
