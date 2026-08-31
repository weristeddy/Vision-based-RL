"""Capture reusable RGB observations and aligned task targets."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .io import load_npz, prefixed, save_npz, string_list


TargetGetter = Callable[[Any, Any, int], Any]


@dataclass(frozen=True)
class CaptureBatch:
  images: np.ndarray
  targets: Mapping[str, np.ndarray] = field(default_factory=dict)
  metadata: Mapping[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    for name, values in self.targets.items():
      if len(values) != len(self.images):
        raise ValueError(
          f"Target '{name}' has {len(values)} samples; expected {len(self.images)}."
        )


def _nested_get(value: Any, path: Sequence[str]) -> Any:
  try:
    current = value
    for key in path:
      current = current[key]
    return current
  except (KeyError, TypeError, IndexError) as exc:
    raise KeyError(f"Observation path '{'.'.join(path)}' was not found.") from exc


def _as_nhwc_uint8(images: Any) -> np.ndarray:
  tensor = torch.as_tensor(images).detach().cpu()
  if tensor.ndim == 3:
    tensor = tensor.unsqueeze(0)
  if tensor.ndim != 4:
    raise ValueError(f"Expected a 3D/4D image tensor, got {tuple(tensor.shape)}.")
  channels = (1, 3, 4)
  if tensor.shape[-1] not in channels and tensor.shape[1] in channels:
    tensor = tensor.permute(0, 2, 3, 1)
  if tensor.shape[-1] not in channels:
    raise ValueError(
      f"Could not identify image channels in shape {tuple(tensor.shape)}."
    )
  if tensor.dtype == torch.uint8:
    return tensor.contiguous().numpy()
  tensor = tensor.float()
  if tensor.numel() and float(tensor.max()) <= 1.0:
    tensor = tensor * 255.0
  return tensor.clamp_(0.0, 255.0).round_().to(torch.uint8).contiguous().numpy()


def capture_rollout(
  env: Any,
  policy: Callable[[Any], torch.Tensor],
  *,
  num_frames: int,
  image_path: Sequence[str] = ("camera",),
  target_getters: Mapping[str, TargetGetter] | None = None,
  seed: int | None = None,
  metadata: Mapping[str, Any] | None = None,
) -> CaptureBatch:
  """Capture one environment without coupling analysis to a task implementation."""
  if num_frames <= 0:
    raise ValueError("num_frames must be positive.")

  getters = target_getters or {}
  if seed is not None and callable(getattr(env, "seed", None)):
    env.seed(seed)
  reset_result = env.reset()
  observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
  images: list[np.ndarray] = []
  targets: dict[str, list[np.ndarray]] = {name: [] for name in getters}

  for frame in range(num_frames):
    if frame:
      with torch.inference_mode():
        action = policy(observation)
      step_result = env.step(action)
      observation = step_result[0] if isinstance(step_result, tuple) else step_result
    images.append(_as_nhwc_uint8(_nested_get(observation, image_path))[0])
    for name, getter in getters.items():
      value = torch.as_tensor(getter(env, observation, 0)).detach().cpu()
      targets[name].append(np.asarray(value))

  return CaptureBatch(
    images=np.stack(images),
    targets={name: np.stack(values) for name, values in targets.items()},
    metadata={"seed": seed, **(metadata or {})},
  )


def save_capture(batch: CaptureBatch, path: str | Path) -> Path:
  arrays = {
    "images": batch.images,
    **{f"target__{name}": values for name, values in batch.targets.items()},
  }
  return save_npz(path, arrays, batch.metadata)


def load_capture(path: str | Path) -> CaptureBatch:
  arrays, metadata = load_npz(path)
  return CaptureBatch(
    images=arrays["images"],
    targets=prefixed(arrays, "target__"),
    metadata=metadata,
  )


def _cube_position(env: Any, _observation: Any, env_index: int) -> Any:
  base = getattr(env, "unwrapped", env)
  position = base.scene["cube"].data.root_link_pos_w[env_index]
  origins = getattr(base.scene, "env_origins", None)
  return position if origins is None else position - origins[env_index]


def _goal_position(env: Any, _observation: Any, env_index: int) -> Any:
  base = getattr(env, "unwrapped", env)
  position = base.command_manager.get_term("lift_height").target_pos[env_index]
  origins = getattr(base.scene, "env_origins", None)
  return position if origins is None else position - origins[env_index]


# --- Push-T ------------------------------------------------------------------
#
# Yaw is recorded as (sin, cos) rather than an angle: a probe fitted against a
# quantity that wraps at +/-pi would be scored on a discontinuity the encoder
# cannot represent, and a 179-degree error would read as small. The pair is
# continuous everywhere and its two components are independently decodable.


def _push_t_command(env: Any) -> Any:
  base = getattr(env, "unwrapped", env)
  return base.command_manager.get_term("push_t_goal")


def _object_yaw(env: Any, _observation: Any, env_index: int) -> Any:
  from vbrl.tasks.push_t.geometry import yaw_from_quat

  base = getattr(env, "unwrapped", env)
  import torch

  yaw = yaw_from_quat(base.scene["object"].data.root_link_quat_w)[env_index]
  return torch.stack((torch.sin(yaw), torch.cos(yaw)))


def _goal_yaw(env: Any, _observation: Any, env_index: int) -> Any:
  import torch

  yaw = _push_t_command(env).target_yaw[env_index]
  return torch.stack((torch.sin(yaw), torch.cos(yaw)))


def _relative_yaw(env: Any, _observation: Any, env_index: int) -> Any:
  """The quantity the task is scored on: goal yaw minus object yaw."""
  from mjlab.utils.lab_api.math import wrap_to_pi

  from vbrl.tasks.push_t.geometry import yaw_from_quat

  import torch

  base = getattr(env, "unwrapped", env)
  command = _push_t_command(env)
  error = wrap_to_pi(
    command.target_yaw - yaw_from_quat(base.scene["object"].data.root_link_quat_w)
  )[env_index]
  return torch.stack((torch.sin(error), torch.cos(error)))


def _object_position(env: Any, _observation: Any, env_index: int) -> Any:
  base = getattr(env, "unwrapped", env)
  position = base.scene["object"].data.root_link_pos_w[env_index]
  origins = getattr(base.scene, "env_origins", None)
  return position if origins is None else position - origins[env_index]


def _object_to_goal(env: Any, _observation: Any, env_index: int) -> Any:
  base = getattr(env, "unwrapped", env)
  command = _push_t_command(env)
  return (
    command.target_pos[env_index]
    - base.scene["object"].data.root_link_pos_w[env_index]
  )


def _overlap(env: Any, _observation: Any, env_index: int) -> Any:
  return _push_t_command(env).get_overlap(force_refresh=True)[env_index].reshape(1)


TARGET_GETTERS: Mapping[str, TargetGetter] = {
  "cube_position": _cube_position,
  "goal_position": _goal_position,
  "object_yaw": _object_yaw,
  "goal_yaw": _goal_yaw,
  "relative_yaw": _relative_yaw,
  "object_position": _object_position,
  "object_to_goal": _object_to_goal,
  "overlap": _overlap,
}


def run(
  context: Any,
  *,
  output: str,
  num_frames: int = 256,
  num_envs: int | None = None,
  seed: int | None = 0,
  image_path: Sequence[str] = ("camera",),
  targets: Sequence[str] = (),
) -> Path:
  """Capture observations from the pipeline's one shared native runtime."""
  del num_envs  # Read by the entry point while it sizes the shared environment.
  names = string_list(targets, "capture.targets")
  try:
    target_getters = {name: TARGET_GETTERS[name] for name in names}
  except KeyError as exc:
    raise ValueError(
      f"Unknown capture target {exc.args[0]!r}; choose from {tuple(TARGET_GETTERS)}."
    ) from exc

  batch = capture_rollout(
    context.env,
    context.policy,
    num_frames=num_frames,
    image_path=string_list(image_path, "capture.image_path"),
    target_getters=target_getters,
    seed=seed,
    metadata={**context.provenance(), "agent": context.agent},
  )
  return save_capture(batch, context.output(output))
