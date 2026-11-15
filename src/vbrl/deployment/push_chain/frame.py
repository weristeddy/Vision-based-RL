from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

GOAL_FILE = Path("artifacts/push_chain/goal.json")


def parser(description: str) -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=description)
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument("--image", type=Path, help="Measure this frame.")
  source.add_argument(
    "--manifest",
    type=Path,
    help="Capture a frame with this manifest's external-camera settings.",
  )
  parser.add_argument("--save", type=Path, help="Write the captured frame here.")
  parser.add_argument("--annotate", type=Path, help="Write the fit drawn here.")
  parser.add_argument("--goal-file", type=Path, default=GOAL_FILE)
  return parser


def load(arguments: argparse.Namespace) -> tuple[Any, Any]:
  from vbrl.deployment.push_t import frame
  from vbrl.deployment.push_t.view import ExternalView

  if arguments.image is not None:
    image = frame.read(arguments.image)
  else:
    from vbrl.deployment.camera import still
    from vbrl.deployment.config import load_config

    image = still(load_config(arguments.manifest), *frame.RESOLUTION)
  if arguments.save is not None:
    frame.write(arguments.save, image)
  return ExternalView.load(image.shape[1], image.shape[0]), image


def report(
  arguments: argparse.Namespace,
  image: Any,
  name: str,
  values: Any,
  fit: Any,
  mask: Any,
) -> None:
  import cv2

  from vbrl.deployment.push_t import frame

  if arguments.annotate is not None:
    annotated = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(annotated, contours, -1, (255, 0, 255), 1)
    frame.write(arguments.annotate, annotated)
  print(f"{name:6s} {pose(values)}  IoU {fit.iou:.3f}")
  if not fit.reliable:
    raise SystemExit("Unreliable fit: something in view, or the exposure is off.")


def pose(values: Any) -> str:
  x, y, yaw, *hinges = values
  angles = " ".join(f"{math.degrees(v):+7.2f}" for v in hinges)
  return f"x {x:.4f}  y {y:+.4f}  yaw {math.degrees(yaw):+7.2f}  hinges {angles} deg"


def read_goal(path: Path) -> list[float]:
  if not path.is_file():
    raise SystemExit(
      f"No goal at {path}: run vbrl.deployment.push_chain.measure_goal first."
    )
  return json.loads(path.read_text())["goal"]


def write_goal(path: Path, goal: Any) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps({"goal": [round(float(v), 5) for v in goal]}) + "\n")


__all__ = ["GOAL_FILE", "load", "parser", "pose", "read_goal", "report", "write_goal"]
