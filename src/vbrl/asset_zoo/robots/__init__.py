"""Robot definitions and their packaged MJCF assets."""

from collections.abc import Callable, Mapping

from .definition import (
  CameraGeometry,
  CameraView,
  RobotCameraDefinition,
  RobotDefinition,
)
from .i2rt_yam import make_yam
from .trossen_wxai import make_wxai, make_wxai_realistic


# The canonical selector for every robot this repository can build. Task
# configs import the factory they need directly; this mapping exists so
# consumers that iterate over robots have one place to look.
ROBOTS: Mapping[str, Callable[[], RobotDefinition]] = {
  "trossen": make_wxai,
  "trossen_realistic": make_wxai_realistic,
  "yam": make_yam,
}


def list_robots() -> tuple[str, ...]:
  """Return canonical robot selector names."""
  return tuple(sorted(ROBOTS))


def get_robot(name: str) -> RobotDefinition:
  """Resolve a canonical robot selector to a fresh definition."""
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
  "make_wxai_realistic",
  "make_yam",
]
