"""Patch-occlusion sensitivity artifacts and heatmaps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F

from vbrl.vision.preprocessing import prepare_images

from .features import encode_stages
from .io import load_npz, resolve_outputs, save_figure, save_npz, string_list


OcclusionStage = Literal["backbone", "adapter"]
OcclusionFill = Literal["zero", "mean"]


@dataclass(frozen=True)
class OcclusionResult:
  """Per-image cosine-distance scores over an occlusion patch grid."""

  scores: np.ndarray
  metadata: Mapping[str, Any] = field(default_factory=dict)


def _encode(encoder: Any, images: torch.Tensor, stage: OcclusionStage) -> torch.Tensor:
  return encode_stages(encoder, images, (stage,))[stage]


def occlusion_sensitivity(
  encoder: Any,
  images: np.ndarray | torch.Tensor,
  *,
  stage: OcclusionStage = "adapter",
  patch_size: int = 32,
  batch_size: int = 64,
  fill: OcclusionFill = "mean",
  device: str | torch.device,
  metadata: Mapping[str, Any] | None = None,
) -> OcclusionResult:
  """Measure feature cosine distance after occluding each image patch."""
  if patch_size <= 0 or batch_size <= 0:
    raise ValueError("patch_size and batch_size must be positive.")
  if fill not in {"zero", "mean"}:
    raise ValueError("fill must be 'zero' or 'mean'.")

  prepared = prepare_images(torch.as_tensor(images))
  if len(prepared) == 0:
    raise ValueError("At least one image is required.")

  encoder = encoder.to(torch.device(device)).eval()
  prepared = prepared.to(device)
  rows = (prepared.shape[-2] + patch_size - 1) // patch_size
  columns = (prepared.shape[-1] + patch_size - 1) // patch_size
  scores = np.empty((len(prepared), rows, columns), dtype=np.float32)

  with torch.inference_mode():
    baseline_batches = []
    for batch in prepared.split(batch_size):
      baseline_batches.append(_encode(encoder, batch, stage).flatten(start_dim=1))
    baseline = torch.cat(baseline_batches)

    for row in range(rows):
      y = slice(row * patch_size, min((row + 1) * patch_size, prepared.shape[-2]))
      for column in range(columns):
        x = slice(
          column * patch_size,
          min((column + 1) * patch_size, prepared.shape[-1]),
        )
        for start in range(0, len(prepared), batch_size):
          stop = min(start + batch_size, len(prepared))
          batch = prepared[start:stop].clone()
          replacement = (
            batch.mean(dim=(-2, -1), keepdim=True)
            if fill == "mean"
            else 0.0
          )
          batch[:, :, y, x] = replacement
          changed = _encode(encoder, batch, stage).flatten(start_dim=1)
          scores[start:stop, row, column] = (
            1.0
            - F.cosine_similarity(baseline[start:stop], changed, dim=1)
          ).float().cpu().numpy()

  return OcclusionResult(
    scores=scores,
    metadata={
      "stage": stage,
      "patch_size": patch_size,
      "fill": fill,
      "image_height": int(prepared.shape[-2]),
      "image_width": int(prepared.shape[-1]),
      **dict(metadata or {}),
    },
  )


def save_occlusion(result: OcclusionResult, path: str | Path) -> Path:
  """Persist an occlusion result as a compressed, non-pickle NPZ."""
  return save_npz(path, {"scores": result.scores}, result.metadata)


def load_occlusion(path: str | Path) -> OcclusionResult:
  """Restore a saved occlusion result."""
  arrays, metadata = load_npz(path)
  return OcclusionResult(scores=arrays["scores"], metadata=metadata)


def plot_occlusion_heatmap(
  result: OcclusionResult,
  *,
  output: str | Path,
  title: str = "Mean patch occlusion sensitivity",
) -> Path:
  """Plot the mean patch sensitivity over all analyzed images."""
  import matplotlib.pyplot as plt

  figure, axis = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
  image = axis.imshow(result.scores.mean(axis=0), cmap="magma", interpolation="nearest")
  figure.colorbar(image, ax=axis, label="Cosine distance")
  axis.set(title=title, xlabel="Patch column", ylabel="Patch row")
  return save_figure(figure, output)


def run(
  context: Any,
  *,
  capture: str,
  output: str,
  plot: str,
  stages: Sequence[str],
  patch_size: int = 32,
  batch_size: int = 64,
  num_images: int = 64,
  fill: str = "mean",
) -> tuple[Path, ...]:
  """Compute sensitivity with the already-loaded actor camera encoder."""
  from .capture import load_capture
  from .features import camera_encoder

  limit = int(num_images)
  if limit <= 0:
    raise ValueError("occlusion num_images must be positive.")
  names = string_list(stages, "occlusion.stages")
  jobs = resolve_outputs(
    context,
    {"output": output, "plot": plot},
    ({"stage": stage} for stage in names),
    label="Occlusion",
  )

  source = context.input(capture)
  capture_batch = load_capture(source)
  encoder = camera_encoder(context)
  metadata = {**context.provenance(), "capture": str(source)}

  generated: list[Path] = []
  for values, paths in jobs:
    stage = cast(OcclusionStage, values["stage"])
    result = occlusion_sensitivity(
      encoder,
      capture_batch.images[:limit],
      stage=stage,
      patch_size=int(patch_size),
      batch_size=int(batch_size),
      fill=cast(OcclusionFill, fill),
      device=context.device,
      metadata=metadata,
    )
    generated.append(save_occlusion(result, paths["output"]))
    generated.append(
      plot_occlusion_heatmap(
        result,
        output=paths["plot"],
        title=f"{context.task_id} / {stage} occlusion",
      )
    )
  return tuple(generated)
