"""Where on the feature grid each adapter draws its output from.

Every registered adapter reduces a spatial map to one vector, and each does it
through a different mechanism -- so each admits a different, *exact* window onto
what it used. None of these is an occlusion estimate; they are the quantities the
adapter actually computes, decomposed over the grid:

``spatial_softmax``
  The softmax over positions is already a per-channel spatial distribution, and
  the output *is* the expected coordinate. Both come out directly, so the map
  and a set of trackable keypoints are read rather than inferred.

``afa``
  The latent query's attention over spatial tokens, reduced over heads by the
  maximum rather than the mean. Head count follows the channel width -- 6 for
  DINOv2, 16 for R3M layer 3, 32 for layer 4 -- and averaging that many
  independent distributions flattens every one of them, which makes a wide
  encoder look inattentive purely as an artifact of the reduction. The maximum
  reports what the most confident head selected at each position.

``flatten`` / ``flatten_relu`` / ``local_grid``
  These end in a dense layer over a flattened grid, so ``W`` reshaped to
  ``[out, channels, H, W]`` contracted with the features gives each cell's
  contribution to the output vector. That is the same quantity patch occlusion
  approximates, computed in closed form and without one forward pass per patch.

``linear`` / ``global``
  Global pooling is permutation-invariant over positions, so there is no map to
  recover -- ``None`` here is the finding, not a gap.

The maps live on the encoder's own grid (24x24 for Nature CNN, 16x16 for DINOv2,
7x7 for R3M layer 4), which is the resolution at which the adapter sees anything
at all. Upsample for display with nearest-neighbour: interpolation would imply a
spatial precision the grid does not have.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  import torch

__all__ = [
  "AdapterAttribution",
  "adapter_family",
  "attribution_is_available",
  "spatial_attribution",
]


@dataclass(frozen=True)
class AdapterAttribution:
  """One spatial map per frame, plus keypoints where the adapter emits them."""

  family: str
  heatmap: "torch.Tensor | None"
  """``[B, H, W]``, non-negative, or ``None`` for a pooling adapter."""
  keypoints: "torch.Tensor | None"
  """``[B, K, 2]`` in normalized ``[-1, 1]`` (x, y), or ``None``."""

  def __post_init__(self) -> None:
    if self.heatmap is not None and self.heatmap.ndim != 3:
      raise ValueError(f"heatmap must be [B,H,W], got {tuple(self.heatmap.shape)}.")
    if self.keypoints is not None and (
      self.keypoints.ndim != 3 or self.keypoints.shape[-1] != 2
    ):
      raise ValueError(
        f"keypoints must be [B,K,2], got {tuple(self.keypoints.shape)}."
      )


_DENSE_FAMILIES = {"flatten", "flatten_relu", "local_grid"}


def adapter_family(adapter: Any) -> str:
  """Classify an adapter module by the window it admits."""
  name = type(adapter).__name__
  return {
    "SpatialSoftmaxAdapter": "spatial_softmax",
    "AFAAdapter": "afa",
    "FlattenAdapter": "flatten",
    "FlattenReluAdapter": "flatten_relu",
    "LocalGridAdapter": "local_grid",
  }.get(name, "pooled")


def attribution_is_available(adapter: Any) -> bool:
  """Whether this adapter retains any positional information to recover."""
  return adapter_family(adapter) != "pooled"


def _dense_contribution(
  weight: "torch.Tensor",
  grid: "torch.Tensor",
  *,
  centre: bool,
) -> "torch.Tensor":
  """Per-cell contribution of a flattened-grid dense layer to its output.

  ``weight`` is ``[out, channels * H * W]`` over a grid flattened in
  ``(channels, H, W)`` order -- the order both ``FlattenAdapter`` and
  ``LocalGridAdapter`` produce. Contracting it with the features gives
  ``[B, out, H, W]``; the norm over the output axis is how much each cell moved
  the adapter's vector.
  """
  import torch

  batch, channels, height, width = grid.shape
  out = weight.shape[0]
  if weight.shape[1] != channels * height * width:
    raise ValueError(
      f"weight expects {weight.shape[1]} inputs but the grid flattens to "
      f"{channels * height * width}."
    )
  kernel = weight.reshape(out, channels, height, width)
  # Centring needs more than one frame: with a batch of one the mean *is* that
  # frame, so subtracting it annihilates the map entirely.
  if centre and batch > 1:
    # Subtract the batch mean so the map shows what *varies* with the scene
    # rather than the constant offset every frame shares.
    grid = grid - grid.mean(dim=0, keepdim=True)
  contribution = torch.einsum("ochw,bchw->bohw", kernel.float(), grid.float())
  return contribution.norm(dim=1)


def spatial_attribution(
  encoder: Any,
  images: "torch.Tensor",
  *,
  centre: bool = True,
) -> AdapterAttribution:
  """Run one batch through ``encoder`` and recover its adapter's spatial map.

  ``images`` is ``[B, C, H, W]`` in whatever dtype the encoder expects. Nothing
  is perturbed and nothing is fitted: a forward hook captures the tensor handed
  to the adapter, and the map is computed from the adapter's own parameters.
  """
  import torch

  adapter = encoder.adapter
  family = adapter_family(adapter)
  captured: dict[str, torch.Tensor] = {}

  def grab(key: str):
    def hook(_module, args):
      # Cloned out of inference mode: these tensors are fed back through the
      # adapter's own layers below, which inference-mode tensors forbid.
      captured[key] = args[0].detach().clone()
      return None  # A pre-hook that returns replaces the arguments.

    return hook

  handles = [adapter.register_forward_pre_hook(grab("features"))]
  if family in _DENSE_FAMILIES:
    dense = adapter.flatten_proj if family == "local_grid" else adapter.proj
    handles.append(dense.register_forward_pre_hook(grab("dense_input")))

  try:
    device = next(encoder.parameters()).device
    with torch.no_grad():
      encoder(images.to(device))
  finally:
    for handle in handles:
      handle.remove()

  features = captured["features"]
  if family == "pooled":
    return AdapterAttribution("pooled", None, None)

  if family == "spatial_softmax":
    with torch.no_grad():
      logits = adapter.proj(features.float())
    batch, channels, height, width = logits.shape
    probabilities = logits.reshape(batch, channels, -1).softmax(dim=-1)
    heatmap = probabilities.mean(dim=1).reshape(batch, height, width)
    axis_x = torch.linspace(-1.0, 1.0, width, device=logits.device)
    axis_y = torch.linspace(-1.0, 1.0, height, device=logits.device)
    grid = probabilities.reshape(batch, channels, height, width)
    keypoints = torch.stack(
      (
        (grid.sum(dim=2) * axis_x).sum(-1),
        (grid.sum(dim=3) * axis_y).sum(-1),
      ),
      dim=-1,
    )
    return AdapterAttribution(family, heatmap.cpu(), keypoints.cpu())

  if family == "afa":
    pool = adapter.pool
    with torch.no_grad():
      tokens = features.float().flatten(2).transpose(1, 2)
      tokens = tokens / tokens.norm(dim=-1, keepdim=True).clamp(min=1e-6)
      batch, count, channels = tokens.shape
      query = pool.q(pool.latent.expand(batch, -1, -1))
      query = query.reshape(batch, 1, pool.num_heads, pool.head_dim).transpose(1, 2)
      key = (
        pool.kv(tokens)
        .reshape(batch, count, 2, pool.num_heads, pool.head_dim)
        .permute(2, 0, 3, 1, 4)[0]
      )
      attention = (
        (pool.q_norm(query) * pool.scale) @ pool.k_norm(key).transpose(-2, -1)
      ).softmax(-1)
    height, width = features.shape[-2:]
    # Max, not mean: see the module docstring. `attention` is [B, heads, 1, N].
    heatmap = attention.squeeze(2).max(dim=1).values.reshape(batch, height, width)
    return AdapterAttribution(family, heatmap.cpu(), None)

  # Dense readouts: reshape the flattened input back onto the grid it came from.
  dense_input = captured["dense_input"]
  if family == "local_grid":
    channels = adapter.proj.out_channels
    height = width = adapter.target_grid_size
    weight = adapter.flatten_proj.weight
  else:
    channels = adapter.input_channels
    height = width = adapter.grid_size
    weight = adapter.proj.weight
  grid = dense_input.reshape(dense_input.shape[0], channels, height, width)
  with torch.no_grad():
    heatmap = _dense_contribution(weight, grid, centre=centre)
  return AdapterAttribution(family, heatmap.cpu(), None)


def _overlay(
  images: Any,
  heatmap: Any,
  keypoints: Any,
  family: str,
  destination: Any,
) -> Any:
  """Draw each frame with its adapter map on top, and keypoints where they exist."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  import numpy as np

  count = len(images)
  fig, axes = plt.subplots(1, count, figsize=(3.1 * count, 3.4), squeeze=False)
  scale = float(np.percentile(heatmap, 99)) if heatmap is not None else 1.0
  for index in range(count):
    ax = axes[0][index]
    ax.imshow(images[index])
    if heatmap is not None:
      # Per-cell alpha rather than a flat wash: cells the adapter barely uses
      # stay transparent, so the frame underneath is still readable and the
      # bright regions are the claim.
      norm = np.clip(heatmap[index] / (scale or 1.0), 0.0, 1.0)
      rgba = plt.get_cmap("inferno")(norm)
      rgba[..., 3] = 0.85 * norm
      # Nearest-neighbour: the map lives on the feature grid and smoothing it
      # would imply a spatial precision that grid does not have.
      ax.imshow(
        rgba,
        extent=(0, images.shape[2], images.shape[1], 0),
        interpolation="nearest",
      )
    if keypoints is not None:
      # Normalized [-1, 1] back to pixels.
      xs = (keypoints[index, :, 0] + 1.0) / 2.0 * images.shape[2]
      ys = (keypoints[index, :, 1] + 1.0) / 2.0 * images.shape[1]
      ax.scatter(xs, ys, s=7, c="#39ff9e", edgecolors="none", alpha=0.75)
    ax.set_xticks([])
    ax.set_yticks([])
  fig.suptitle(f"adapter: {family}", fontsize=11, fontweight="bold")
  fig.tight_layout()
  fig.savefig(destination, dpi=170, bbox_inches="tight")
  plt.close(fig)
  return destination


def run(
  context: Any,
  *,
  capture: str,
  output: str,
  figure: str | None = None,
  frames: int = 6,
  centre: bool = True,
) -> Any:
  """Recover the trained adapter's spatial map over a saved capture.

  Mirrors :func:`vbrl.analysis.features.run`: it reads a capture written earlier
  in the manifest and uses the already-loaded trained actor, so the map comes
  from the policy that was actually trained rather than a fresh encoder.
  """
  import numpy as np
  import torch

  from .capture import load_capture
  from .features import camera_encoder
  from .io import save_npz

  captured = load_capture(context.input(capture))
  # Spread across the capture: consecutive frames of one rollout differ by a
  # single control step and would show the same picture six times.
  chosen = np.linspace(0, len(captured.images) - 1, num=frames).round().astype(int)
  images = captured.images[chosen]
  batch = torch.from_numpy(np.moveaxis(images, -1, 1).copy()).to(context.device)
  result = spatial_attribution(camera_encoder(context), batch.float(), centre=centre)

  arrays: dict[str, Any] = {"family": np.array(result.family)}
  if result.heatmap is not None:
    arrays["heatmap"] = result.heatmap.numpy()
  if result.keypoints is not None:
    arrays["keypoints"] = result.keypoints.numpy()
  written = [save_npz(context.output(output), arrays, context.provenance())]
  if figure is not None:
    written.append(
      _overlay(
        images,
        arrays.get("heatmap"),
        arrays.get("keypoints"),
        result.family,
        context.output(figure),
      )
    )
  return written
