from collections.abc import Callable, Mapping

from .definition import (
  CameraGeometry,
  CameraView,
  RobotCameraDefinition,
  RobotDefinition,
)
from .trossen_wxai import make_wxai, make_wxai_identified, make_wxai_realistic

ROBOTS: Mapping[str, Callable[[], RobotDefinition]] = {
  "trossen": make_wxai,
  "trossen_realistic": make_wxai_realistic,
  "trossen_identified": make_wxai_identified,
}


def list_robots() -> tuple[str, ...]:
  return tuple(sorted(ROBOTS))


def get_robot(name: str) -> RobotDefinition:
  try:
    factory = ROBOTS[name]
  except KeyError as exc:
    choices = ", ".join(list_robots())
    raise ValueError(f"Unknown robot {name!r}. Choose one of: {choices}.") from exc
  return factory()


__all__ = [
  "ROBOTS",
  "CameraGeometry",
  "CameraView",
  "RobotCameraDefinition",
  "RobotDefinition",
  "get_robot",
  "list_robots",
  "make_wxai",
  "make_wxai_identified",
  "make_wxai_realistic",
]
