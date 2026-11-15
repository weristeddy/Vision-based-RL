from __future__ import annotations

import math

import torch

from vbrl.tasks.push_t.geometry import FootprintPart, FootprintRasterizer

LINK_HALF_HEIGHT = 0.015
REST_HEIGHT = 0.016
LINK_HALF_WIDTH = 0.015
END_LINK_HALF_LENGTH = 0.035
MIDDLE_LINK_HALF_LENGTH = 0.04375
PIN_RADIUS = 0.015
HINGE_LIMIT = math.radians(97.0)
OVERLAP_PIXEL = 0.001
CHAIN_OVERLAP_PIXEL = 0.002


def half_lengths(num_links: int) -> tuple[float, ...]:
  if num_links < 3:
    raise ValueError(f"A chain has at least 3 links; got {num_links}.")
  middle = (MIDDLE_LINK_HALF_LENGTH,) * (num_links - 2)
  return (END_LINK_HALF_LENGTH, *middle, END_LINK_HALF_LENGTH)


def half_extents(num_links: int) -> tuple[tuple[float, float], ...]:
  return tuple((length, LINK_HALF_WIDTH) for length in half_lengths(num_links))


def centre_link(num_links: int) -> int:
  return num_links // 2


def _along(yaw: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
  return torch.stack((length * torch.cos(yaw), length * torch.sin(yaw)), dim=-1)


def _half(links: torch.Tensor) -> torch.Tensor:
  return links.new_tensor(half_lengths(links.shape[-2]))


def link_poses(
  xy: torch.Tensor, yaw: torch.Tensor, hinges: torch.Tensor
) -> torch.Tensor:
  yaws = torch.cat((yaw[..., None], yaw[..., None] + hinges.cumsum(-1)), dim=-1)
  steps = _along(yaws, xy.new_tensor(half_lengths(yaws.shape[-1])))
  offsets = (steps[..., :-1, :] + steps[..., 1:, :]).cumsum(-2)
  start = torch.zeros_like(offsets[..., :1, :])
  centres = xy[..., None, :] + torch.cat((start, offsets), dim=-2)
  return torch.cat((centres, yaws[..., None]), dim=-1)


def chain_from_pose(pose: torch.Tensor) -> torch.Tensor:
  hinges = pose[..., 3:]
  zero = torch.zeros_like(pose[..., 2])
  shape = link_poses(torch.zeros_like(pose[..., :2]), zero, hinges)
  count = hinges.shape[-1] + 1
  link = shape[..., centre_link(count), :]
  # The pose sits halfway along the chain: the centre link's centre for odd N, the
  # pin in front of it for even N.
  back = 0.0 if count % 2 else half_lengths(count)[centre_link(count)]
  middle = link[..., :2] - _along(link[..., 2], link.new_tensor(back))
  turn = pose[..., 2] - link[..., 2]
  cos, sin = torch.cos(turn)[..., None], torch.sin(turn)[..., None]
  x, y = (shape[..., :2] - middle[..., None, :]).unbind(-1)
  centres = pose[..., None, :2] + torch.stack(
    (cos * x - sin * y, sin * x + cos * y), -1
  )
  return torch.cat((centres, shape[..., 2:] + turn[..., None, None]), dim=-1)


def reversed_links(links: torch.Tensor) -> torch.Tensor:
  flipped = links.flip(-2)
  return torch.cat((flipped[..., :2], flipped[..., 2:] + math.pi), dim=-1)


def _ends(links: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  step = _along(links[..., 2], _half(links))
  return links[..., :2] - step, links[..., :2] + step


def pin_positions(links: torch.Tensor) -> torch.Tensor:
  return _ends(links)[1][..., :-1, :]


def chain_landmarks(links: torch.Tensor) -> torch.Tensor:
  back, front = _ends(links)
  hinges = (front[..., :-1, :] + back[..., 1:, :]) / 2.0
  last = links.shape[-2] - 1
  points = [back[..., 0, :]]
  for i in range(last + 1):
    points += [links[..., i, :2], front[..., i, :] if i == last else hinges[..., i, :]]
  return torch.stack(points, dim=-2)


def landmark_errors(chain: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
  current, target = chain_landmarks(chain), chain_landmarks(goal)
  return torch.stack(
    (
      torch.linalg.vector_norm(current - target, dim=-1).amax(-1),
      torch.linalg.vector_norm(current - target.flip(-2), dim=-1).amax(-1),
    ),
    dim=-1,
  )


def _box_distance(
  points: torch.Tensor, centres: torch.Tensor, yaws: torch.Tensor, half: torch.Tensor
) -> torch.Tensor:
  dx, dy = (points - centres).unbind(-1)
  cos, sin = torch.cos(yaws), torch.sin(yaws)
  q = torch.stack((cos * dx + sin * dy, cos * dy - sin * dx), dim=-1).abs() - half
  return torch.linalg.vector_norm(q.clamp_min(0.0), dim=-1) + q.amax(-1).clamp_max(0.0)


def _part_distances(points: torch.Tensor, links: torch.Tensor) -> torch.Tensor:
  half = links.new_tensor(half_extents(links.shape[-2]))
  boxes = _box_distance(points.unsqueeze(-2), links[..., :2], links[..., 2], half)
  to_pins = points.unsqueeze(-2) - pin_positions(links)
  pins = torch.linalg.vector_norm(to_pins, dim=-1) - PIN_RADIUS
  return torch.cat((boxes, pins), dim=-1)


def chain_signed_distance(
  points: torch.Tensor, links: torch.Tensor, *, pins: bool = False
) -> torch.Tensor:
  parts = 2 * links.shape[-2] - 1 if pins else links.shape[-2]
  return _part_distances(points, links)[..., :parts].amin(-1)


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


def self_gap(links: torch.Tensor) -> torch.Tensor:
  count = links.shape[-2]
  half = links.new_tensor(half_extents(count))
  pins = pin_positions(links)
  gaps = []
  for i in range(count):
    for j in range(i + 2, count):
      gaps.append(_rectangle_gap(links[..., i, :], half[i], links[..., j, :], half[j]))
  for k in range(count - 1):
    for j in set(range(count)) - {k, k + 1}:
      box = _box_distance(pins[..., k, :], links[..., j, :2], links[..., j, 2], half[j])
      gaps.append(box - PIN_RADIUS)
    for m in range(k + 1, count - 1):
      distance = torch.linalg.vector_norm(pins[..., k, :] - pins[..., m, :], dim=-1)
      gaps.append(distance - 2.0 * PIN_RADIUS)
  return torch.stack(gaps, dim=-1).amin(-1)


def gap_to_boxes(links: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
  half = links.new_tensor(half_extents(links.shape[-2])).unsqueeze(-2)
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
  def __init__(
    self, num_links: int, *, device: torch.device | str, dtype: torch.dtype
  ) -> None:
    self._rasterizers = [
      FootprintRasterizer(
        (FootprintPart(center_xy=(0.0, 0.0), half_extents_xy=half),),
        device=device,
        dtype=dtype,
        resolution=round(2.0 * half[0] / OVERLAP_PIXEL) + 2,
        half_width=half[0] + OVERLAP_PIXEL,
      )
      for half in half_extents(num_links)
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


class ChainOverlap:
  def __init__(
    self, num_links: int, *, device: torch.device | str, dtype: torch.dtype
  ) -> None:
    disc = _pixels(PIN_RADIUS, PIN_RADIUS)
    disc = disc[torch.linalg.vector_norm(disc, dim=-1) <= PIN_RADIUS]
    parts = [_pixels(*half) for half in half_extents(num_links)]
    parts += [disc] * (num_links - 1)
    self._parts = [part.to(device=device, dtype=dtype) for part in parts]
    order = torch.cat([torch.full((len(part),), i) for i, part in enumerate(parts)])
    self._earlier = (torch.arange(len(parts)) < order[:, None]).to(device)

  def goal_pixels(self, goal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pins = pin_positions(goal)
    poses = torch.cat(
      (goal, torch.cat((pins, torch.zeros_like(pins[..., :1])), -1)), -2
    )
    points = torch.cat(
      [_place(part, poses[:, i]) for i, part in enumerate(self._parts)], dim=1
    )
    inside_goal = _part_distances(points, goal[:, None]) <= 0.0
    return points, ~(inside_goal & self._earlier).any(-1)

  def coverage(
    self, chain: torch.Tensor, points: torch.Tensor, counted: torch.Tensor
  ) -> torch.Tensor:
    covered = chain_signed_distance(points, chain[:, None], pins=True) <= 0.0
    return (counted & covered).sum(-1) / counted.sum(-1)

  def __call__(self, chain: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    return self.coverage(chain, *self.goal_pixels(goal))


def _pixels(half_x: float, half_y: float) -> torch.Tensor:
  axes = (
    torch.arange(-half + CHAIN_OVERLAP_PIXEL / 2.0, half, CHAIN_OVERLAP_PIXEL)
    for half in (half_x, half_y)
  )
  return torch.cartesian_prod(*axes)


def _place(points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
  cos, sin = torch.cos(pose[:, None, 2]), torch.sin(pose[:, None, 2])
  x, y = points.unbind(-1)
  return pose[:, None, :2] + torch.stack((cos * x - sin * y, sin * x + cos * y), -1)


__all__ = [
  "CHAIN_OVERLAP_PIXEL",
  "ChainOverlap",
  "END_LINK_HALF_LENGTH",
  "HINGE_LIMIT",
  "LINK_HALF_HEIGHT",
  "LINK_HALF_WIDTH",
  "LinkOverlap",
  "MIDDLE_LINK_HALF_LENGTH",
  "PIN_RADIUS",
  "REST_HEIGHT",
  "centre_link",
  "chain_from_pose",
  "chain_landmarks",
  "chain_signed_distance",
  "gap_to_boxes",
  "half_extents",
  "half_lengths",
  "landmark_errors",
  "link_poses",
  "pin_positions",
  "reversed_links",
  "self_gap",
]
