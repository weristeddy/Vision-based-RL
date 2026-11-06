from collections.abc import Callable, Mapping

from .definition import (
  CameraGeometry,
  CameraView,
  RobotCameraDefinition,
  RobotDefinition,
)
from .trossen_wxai import make_wxai_measured

ROBOTS: Mapping[str, Callable[[], RobotDefinition]] = {
  "trossen_measured": make_wxai_measured,
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
  "make_wxai_measured",
]
