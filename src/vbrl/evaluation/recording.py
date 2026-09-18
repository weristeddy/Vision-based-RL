from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from vbrl.scenes.presets import TABLE_CENTER

# Azimuth 180 is the +x side the robot faces. Below -20 elevation the shot gains
# black bands past the edge of MuJoCo's floor quad.
AZIMUTH_DEG = 180.0
ELEVATION_DEG = -20.0
SCENE_PADDING_M = 1.4
FRAME_MARGIN = 1.0

VIDEO_SIZE = (1280, 720)
GIF_SIZE = (640, 360)
GIF_FPS = 25


def grid_camera(env: Any, *, width: int, height: int) -> Any:
  from mjlab.viewer.viewer_config import ViewerConfig

  origins = env.scene.env_origins
  low, high = origins.amin(dim=0), origins.amax(dim=0)
  extent = (high - low).tolist()
  centre = ((low + high) / 2.0).tolist()

  span = math.hypot(extent[0], extent[1]) + SCENE_PADDING_M
  fovy_deg = float(env.cfg.viewer.fovy or env.sim.mj_model.vis.global_.fovy)
  half_fov_x = math.atan(math.tan(math.radians(fovy_deg) / 2.0) * width / height)

  return replace(
    env.cfg.viewer,
    origin_type=ViewerConfig.OriginType.WORLD,
    lookat=(centre[0] + TABLE_CENTER[0], centre[1] + TABLE_CENTER[1], 0.0),
    distance=FRAME_MARGIN * span / (2.0 * math.tan(half_fov_x)),
    elevation=ELEVATION_DEG,
    azimuth=AZIMUTH_DEG,
    width=width,
    height=height,
    max_extra_envs=max(0, env.num_envs - 1),
  )


def default_output(path: Path) -> tuple[tuple[int, int], int | None]:
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
  import os

  # Auto-detection raises "an OpenGL platform library has not been loaded".
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
    visualizer.show_all_envs = True
    base.update_visualizers(visualizer)

  callback = draw if hasattr(base, "update_visualizers") else None
  # One frame per step would play a 25 fps GIF at half speed.
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
