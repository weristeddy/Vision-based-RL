from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vbrl.deployment.config import DeploymentConfig, Motion

DATA_ROOT = Path("artifacts/real_eval")
RIG = Path("configs/deployment/rig.yaml")
CAMERAS = ("external", "wrist")
TASKS = ("push_t", "lift_cube")
FIELDS = {"task", "camera", "steps", "runs", "goals", "policies"}
RIG_FIELDS = {"control_hz", "command_goal_time_s", "goal_space", "motion"}


@dataclass(frozen=True)
class Session:
  task: str
  camera: str
  steps: int
  runs: int
  goals: tuple[tuple[float, ...], ...]
  policies: dict[str, str]
  cameras: dict[str, dict[str, Any]]
  rig: dict[str, Any]

  @property
  def other_camera(self) -> str:
    return next(name for name in CAMERAS if name != self.camera)

  def config(
    self, policy: str | None = None, goal: Any = None, camera: str = ""
  ) -> DeploymentConfig:
    policy = policy or next(iter(self.policies))
    if policy not in self.policies:
      raise SystemExit(f"No policy {policy!r}; choose from {list(self.policies)}.")
    config = DeploymentConfig(
      onnx_file=self.policies[policy],
      **({} if goal is None else {"goal": tuple(float(v) for v in goal)}),
      **self.rig,
      **self.cameras[camera or self.camera],
    )
    config.validate()
    return config

  def goal_file(self, number: int) -> Path:
    return DATA_ROOT / self.task / f"goal{number}" / "goal.json"

  def goal(self, value: str) -> list[float]:
    if value.endswith(".json"):
      return json.loads(Path(value).read_text())["goal"]
    if "," in value:
      return [float(v) for v in value.split(",")]
    number = int(value)
    if self.task == "lift_cube":
      return list(self.goals[number - 1])
    path = self.goal_file(number)
    if not path.is_file():
      raise SystemExit(
        f"{path} is missing: measure it with vbrl-measure {self.task} "
        f"configs/deployment/{self.task}.yaml --save {path}"
      )
    return json.loads(path.read_text())["goal"]


def _yaml(path: Path) -> dict[str, Any]:
  import yaml

  document = yaml.safe_load(path.read_text())
  if document.pop("version", None) != 1:
    raise ValueError(f"{path}: expected 'version: 1'.")
  return document


def load_session(path: str | Path, rig: str | Path = RIG) -> Session:
  path = Path(path)
  document, shared = _yaml(path), _yaml(Path(rig))
  unknown = set(document) - FIELDS - RIG_FIELDS
  if unknown:
    raise ValueError(f"{path}: unknown fields {sorted(unknown)}.")
  if document["task"] not in TASKS:
    raise ValueError(f"{path}: task must be one of {TASKS}.")
  if document["camera"] not in CAMERAS or set(shared["cameras"]) != set(CAMERAS):
    raise ValueError(f"{path}: camera and {rig}'s cameras must name {CAMERAS}.")
  settings = {name: document[name] for name in RIG_FIELDS if name in document}
  settings["motion"] = Motion(**settings.get("motion", {}))
  return Session(
    task=document["task"],
    camera=document["camera"],
    steps=int(document["steps"]),
    runs=int(document["runs"]),
    goals=tuple(tuple(goal) for goal in document.get("goals") or ()),
    policies=document["policies"],
    cameras=shared["cameras"],
    rig={"arm_ip": shared["arm_ip"], **settings},
  )


__all__ = ["CAMERAS", "DATA_ROOT", "Session", "load_session"]
