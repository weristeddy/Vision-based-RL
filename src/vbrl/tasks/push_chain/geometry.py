from __future__ import annotations

import math

import torch

from vbrl.tasks.push_t.geometry import FootprintPart, FootprintRasterizer

LINK_HALF_HEIGHT = 0.015
REST_HEIGHT = 0.016
LINK_HALF_WIDTH = 0.015
END_LINK_HALF_LENGTH = 0.035
MIDDLE_LINK_HALF_LENGTH = 0.04375
LINK_SPACING = MIDDLE_LINK_HALF_LENGTH + END_LINK_HALF_LENGTH
PIN_RADIUS = 0.015
HINGE_LIMIT = math.radians(97.0)
LINK_HALF_EXTENTS = (
  (END_LINK_HALF_LENGTH, LINK_HALF_WIDTH),
  (MIDDLE_LINK_HALF_LENGTH, LINK_HALF_WIDTH),
  (END_LINK_HALF_LENGTH, LINK_HALF_WIDTH),
)
OVERLAP_PIXEL = 0.001


def _along(yaw: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
  return torch.stack((length * torch.cos(yaw), length * torch.sin(yaw)), dim=-1)


def link_poses(
  xy: torch.Tensor, yaw: torch.Tensor, hinges: torch.Tensor
) -> torch.Tensor:
  yaws = torch.stack((yaw - hinges[..., 0], yaw, yaw + hinges[..., 1]), dim=-1)
  pins = xy.unsqueeze(-2) + _along(
    yaw.unsqueeze(-1),
    xy.new_tensor((-MIDDLE_LINK_HALF_LENGTH, MIDDLE_LINK_HALF_LENGTH)),
  )
  ends = pins + _along(
    yaws[..., ::2], xy.new_tensor((-END_LINK_HALF_LENGTH, END_LINK_HALF_LENGTH))
  )
  centres = torch.stack((ends[..., 0, :], xy, ends[..., 1, :]), dim=-2)
  return torch.cat((centres, yaws.unsqueeze(-1)), dim=-1)


def reversed_links(links: torch.Tensor) -> torch.Tensor:
  flipped = links.flip(-2)
  return torch.cat((flipped[..., :2], flipped[..., 2:] + math.pi), dim=-1)


def pin_positions(links: torch.Tensor) -> torch.Tensor:
  ends = links[..., ::2, :]
  return ends[..., :2] + _along(
    ends[..., 2], links.new_tensor((END_LINK_HALF_LENGTH, -END_LINK_HALF_LENGTH))
  )


def _box_distance(
  points: torch.Tensor, centres: torch.Tensor, yaws: torch.Tensor, half: torch.Tensor
) -> torch.Tensor:
  dx, dy = (points - centres).unbind(-1)
  cos, sin = torch.cos(yaws), torch.sin(yaws)
  q = torch.stack((cos * dx + sin * dy, cos * dy - sin * dx), dim=-1).abs() - half
  return torch.linalg.vector_norm(q.clamp_min(0.0), dim=-1) + q.amax(-1).clamp_max(0.0)


def chain_signed_distance(
  points: torch.Tensor, links: torch.Tensor, *, pins: bool = False
) -> torch.Tensor:
  half = links.new_tensor(LINK_HALF_EXTENTS)
  distance = _box_distance(points.unsqueeze(-2), links[..., :2], links[..., 2], half)
  if pins:
    to_pins = points.unsqueeze(-2) - pin_positions(links)
    pin_distance = torch.linalg.vector_norm(to_pins, dim=-1) - PIN_RADIUS
    distance = torch.cat((distance, pin_distance), dim=-1)
  return distance.amin(-1)


def _axes(yaw: torch.Tensor) -> torch.Tensor:
  cos, sin = torch.cos(yaw), torch.sin(yaw)
  return torch.stack((torch.stack((cos, sin), -1), torch.stack((-sin, cos), -1)), -2)


def _rectangle_gap(
  a: torch.Tensor, a_half: torch.Tensor, b: torch.Tensor, b_half: torch.Tensor
) -> torch.Tensor:
  a, b = torch.broadcast_tensors(a, b)
  a_axes, b_axes = _axes(a[..., 2]), _axes(b[..., 2])
  axes = torch.cat((a_axes, b_axes), dim=-2)

  def radius(own: torch.Tensor, half: torch.Tensor) -> torch.Tensor:
    return ((axes @ own.mT).abs() * half.unsqueeze(-2)).sum(-1)

  centre = (axes @ (b[..., :2] - a[..., :2]).unsqueeze(-1)).squeeze(-1).abs()
  return (centre - radius(a_axes, a_half) - radius(b_axes, b_half)).amax(-1)


def end_links_gap(links: torch.Tensor) -> torch.Tensor:
  half = links.new_tensor(LINK_HALF_EXTENTS[0])
  first, last = links[..., 0, :], links[..., 2, :]
  pins = pin_positions(links)
  return torch.stack(
    (
      _rectangle_gap(first, half, last, half),
      _box_distance(pins[..., 0, :], last[..., :2], last[..., 2], half) - PIN_RADIUS,
      _box_distance(pins[..., 1, :], first[..., :2], first[..., 2], half) - PIN_RADIUS,
      torch.linalg.vector_norm(pins[..., 0, :] - pins[..., 1, :], dim=-1)
      - 2.0 * PIN_RADIUS,
    ),
    dim=-1,
  ).amin(-1)


def gap_to_boxes(links: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
  half = links.new_tensor(LINK_HALF_EXTENTS).unsqueeze(-2)
  box_poses = torch.cat((boxes[:, :2], boxes.new_zeros(len(boxes), 1)), dim=-1)
  rectangles = _rectangle_gap(links.unsqueeze(-2), half, box_poses, boxes[:, 2:])
  pins = _box_distance(
    pin_positions(links).unsqueeze(-2),
    boxes[:, :2],
    boxes.new_zeros(len(boxes)),
    boxes[:, 2:],
  )
  return torch.cat((rectangles, pins - PIN_RADIUS), dim=-2).amin((-2, -1))


class LinkOverlap:
  def __init__(self, *, device: torch.device | str, dtype: torch.dtype) -> None:
    self._rasterizers = [
      FootprintRasterizer(
        (FootprintPart(center_xy=(0.0, 0.0), half_extents_xy=half),),
        device=device,
        dtype=dtype,
        resolution=round(2.0 * half[0] / OVERLAP_PIXEL) + 2,
        half_width=half[0] + OVERLAP_PIXEL,
      )
      for half in LINK_HALF_EXTENTS
    ]

  def __call__(self, chain: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    return torch.stack(
      [
        rasterizer.overlap(
          object_xy=chain[:, i, :2],
          object_yaw=chain[:, i, 2],
          target_xy=goal[:, i, :2],
          target_yaw=goal[:, i, 2],
        )
        for i, rasterizer in enumerate(self._rasterizers)
      ],
      dim=-1,
    )

  def best(self, chain: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    overlaps = torch.stack((self(chain, goal), self(chain, reversed_links(goal))), 1)
    best = overlaps.amin(-1).argmax(-1)
    return overlaps[torch.arange(len(chain), device=chain.device), best]


__all__ = [
  "END_LINK_HALF_LENGTH",
  "HINGE_LIMIT",
  "LINK_HALF_EXTENTS",
  "LINK_HALF_HEIGHT",
  "LINK_HALF_WIDTH",
  "LINK_SPACING",
  "LinkOverlap",
  "MIDDLE_LINK_HALF_LENGTH",
  "PIN_RADIUS",
  "REST_HEIGHT",
  "chain_signed_distance",
  "end_links_gap",
  "gap_to_boxes",
  "link_poses",
  "pin_positions",
  "reversed_links",
]
