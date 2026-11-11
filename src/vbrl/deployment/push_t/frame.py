from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

RESOLUTION = (1280, 720)


def parser(description: str) -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument("manifest", type=Path, help="Push-T deployment manifest.")
  parser.add_argument("--image", type=Path, help="Measure this frame, not a capture.")
  parser.add_argument("--save", type=Path, help="Write the captured frame here.")
  parser.add_argument("--annotate", type=Path, help="Write the fit drawn here.")
  return parser


def read(path: Path) -> Any:
  import cv2

  image = cv2.imread(str(path))
  if image is None:
    raise SystemExit(f"Could not read {path}.")
  return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write(path: Path, image: Any) -> None:
  import cv2

  path.parent.mkdir(parents=True, exist_ok=True)
  cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def load(arguments: argparse.Namespace) -> tuple[Any, Any, Any]:
  from vbrl.deployment.camera import still
  from vbrl.deployment.config import load_config
  from vbrl.deployment.push_t.view import ExternalView

  config = load_config(arguments.manifest)
  if config.goal_space != "push_t_xy_yaw":
    raise SystemExit(f"{arguments.manifest} is not a Push-T manifest.")
  if arguments.image is None:
    image = still(config, *RESOLUTION)
  else:
    image = read(arguments.image)
  if arguments.save is not None:
    write(arguments.save, image)
  return config, ExternalView.load(image.shape[1], image.shape[0]), image


def report(
  arguments: argparse.Namespace, config: Any, image: Any, name: str, fit: Any, mask: Any
) -> None:
  import cv2

  if arguments.annotate is not None:
    annotated = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(annotated, contours, -1, (255, 0, 255), 1)
    write(arguments.annotate, annotated)
  print(f"{name:13s} {pose(fit.pose)}  IoU {fit.iou:.3f}")
  print(f"{'manifest goal':13s} {pose(config.goal)}")
  if not fit.reliable:
    raise SystemExit("Unreliable fit: something in view, or the exposure is off.")


def pose(values: Any) -> str:
  x, y, yaw = values
  return f"x {x:.4f}  y {y:+.4f}  yaw {math.degrees(yaw):+7.2f} deg"


__all__ = ["RESOLUTION", "load", "parser", "pose", "read", "report", "write"]
