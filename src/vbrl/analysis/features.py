"""Extract and persist shared visual-encoder features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from vbrl.vision.preprocessing import prepare_images

from .io import load_npz, prefixed, save_npz


@dataclass(frozen=True)
class FeatureBatch:
  features: Mapping[str, np.ndarray]
  metadata: Mapping[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    lengths = {len(value) for value in self.features.values()}
    if len(lengths) > 1:
      raise ValueError(f"Feature arrays are not aligned: lengths={sorted(lengths)}.")


def encode_stages(
  encoder: Any,
  images: torch.Tensor,
  stages: Sequence[Literal["backbone", "adapter"]],
) -> dict[str, torch.Tensor]:
  """Run the encoder once and return only the requested stages.

  The adapter consumes the backbone output, so a request for both costs one
  forward pass, not two.
  """
  backbone = encoder.encode_features(images)
  produced: dict[str, torch.Tensor] = {}
  if "backbone" in stages:
    produced["backbone"] = backbone
  if "adapter" in stages:
    produced["adapter"] = encoder.project_features(backbone)
  return produced


def flatten_features(features: np.ndarray) -> np.ndarray:
  """Flatten any batched global/spatial feature contract to ``(N, D)``."""
  if features.ndim < 2:
    raise ValueError(f"Expected batched features, got shape {features.shape}.")
  return features.reshape(len(features), -1).astype(np.float32, copy=False)


def extract_features(
  encoder: Any,
  images: np.ndarray | torch.Tensor,
  *,
  stages: Sequence[Literal["backbone", "adapter"]] = ("backbone", "adapter"),
  batch_size: int = 64,
  device: str | torch.device,
  metadata: Mapping[str, Any] | None = None,
) -> FeatureBatch:
  if batch_size <= 0:
    raise ValueError("batch_size must be positive.")
  if not stages:
    raise ValueError("At least one feature stage is required.")
  unknown = sorted(set(stages) - {"backbone", "adapter"})
  if unknown:
    raise ValueError(f"Unknown feature stages: {unknown}.")

  encoder = encoder.to(torch.device(device)).eval()
  outputs: dict[str, list[np.ndarray]] = {stage: [] for stage in stages}

  with torch.inference_mode():
    for images_batch in torch.as_tensor(images).split(batch_size):
      batch = prepare_images(images_batch).to(device)
      for stage, values in encode_stages(encoder, batch, stages).items():
        outputs[stage].append(values.float().cpu().numpy())

  return FeatureBatch(
    {stage: np.concatenate(chunks) for stage, chunks in outputs.items()},
    dict(metadata or {}),
  )


def save_features(batch: FeatureBatch, path: str | Path) -> Path:
  arrays = {f"feature__{name}": values for name, values in batch.features.items()}
  return save_npz(path, arrays, batch.metadata)


def load_features(path: str | Path) -> FeatureBatch:
  arrays, metadata = load_npz(path)
  return FeatureBatch(features=prefixed(arrays, "feature__"), metadata=metadata)


def camera_encoder(context: Any) -> Any:
  """Return the camera encoder from the already strict-loaded actor."""
  policy = context.policy
  if context.agent != "trained" or policy is None:
    raise ValueError("Feature analysis requires agent: trained.")
  encoders = getattr(policy, "cnns", None)
  if encoders is None or "camera" not in encoders:
    raise ValueError(f"Task {context.task_id!r} has no actor camera encoder.")
  return encoders["camera"]


def run(
  context: Any,
  *,
  capture: str,
  output: str,
  stages: Sequence[Literal["backbone", "adapter"]] = ("backbone", "adapter"),
  batch_size: int = 64,
) -> Path:
  """Extract stages from a capture using the already-loaded trained actor."""
  from .capture import load_capture

  source = context.input(capture)
  captured = load_capture(source)
  metadata = {**context.provenance(), "capture": str(source)}
  result = extract_features(
    camera_encoder(context),
    captured.images,
    stages=stages,
    batch_size=batch_size,
    device=context.device,
    metadata=metadata,
  )
  return save_features(result, context.output(output))
