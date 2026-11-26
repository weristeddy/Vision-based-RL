from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from vbrl.deployment.config import DeploymentConfig
from vbrl.deployment.session import CAMERAS, DATA_ROOT, Session, load_session

STILL = (1280, 720)
PUSH_T_SUCCESS = 0.90
LIFT_SUCCESS_M = 0.05


def stills(session: Session, config: DeploymentConfig) -> dict[str, Any]:
  from vbrl.deployment.camera import still

  return {
    name: still(dataclasses.replace(config, **session.cameras[name]), *STILL)
    for name in CAMERAS
  }


def episode(arm: Any, session: Session, encoder: str, goal: Any) -> dict[str, Any]:
  from vbrl.deployment.camera import RealSenseCamera
  from vbrl.deployment.loop import execute, go_home, report_exposure
  from vbrl.deployment.policy import load_policy

  config = session.config(encoder, goal)
  policy = load_policy(config)
  home, seconds = policy.metadata.home_pose, config.motion.home_seconds
  go_home(arm, home, seconds)
  start = stills(session, config)
  camera = RealSenseCamera(config)
  other = RealSenseCamera(session.config(encoder, goal, session.other_camera))
  report_exposure(camera, config)
  joint_pos, joint_vel = arm.read()
  policy.warm_up(joint_pos=joint_pos, joint_vel=joint_vel, image=camera.frame())
  started = time.strftime("%Y-%m-%dT%H:%M:%S")
  trace = execute(
    arm,
    policy,
    camera,
    other,
    control_hz=config.control_hz,
    max_arm_action=config.motion.max_arm_action,
    max_steps=session.steps,
  )
  camera.close()
  other.close()
  trace[f"{session.camera}_frames"] = trace.pop("frames")
  trace[f"{session.other_camera}_frames"] = trace.pop("other_frames")
  go_home(arm, home, seconds)
  end = stills(session, config)
  result = measure(session.task, start, end, trace, goal)
  # Ctrl-C ends the episode early; a stopped episode counts as a failure.
  stopped = len(trace["action"]) < session.steps
  result = {"stopped": stopped, **result, "success": result["success"] and not stopped}
  return {
    "start": start,
    "end": end,
    "trace": trace,
    "started": started,
    "source_run": policy.metadata.source_run,
    "result": result,
  }


def measure(task: str, start: Any, end: Any, trace: Any, goal: Any) -> dict[str, Any]:
  if task == "lift_cube":
    holding = bool(trace["holding"][-1])
    error = float(trace["goal_distance"][-1])
    return {
      "success": holding and error < LIFT_SUCCESS_M,
      "holding": holding,
      "tcp_goal_error_m": error,
    }
  from vbrl.rig.measure import measure_object, overlap
  from vbrl.rig.view import ExternalView

  view = ExternalView.load(*STILL)
  result: dict[str, Any] = {}
  for moment, images in (("start", start), ("end", end)):
    fit = measure_object(view, images["external"], "push_t")
    result[f"{moment}_pose"] = list(fit.pose)
    result[f"{moment}_overlap"] = overlap("push_t", fit.pose, goal)
    result[f"{moment}_fit_reliable"] = bool(fit.reliable)
  result["success"] = result["end_overlap"] >= PUSH_T_SUCCESS
  return result


def push_t_goal(arm: Any, session: Session, folder: Path) -> list[float]:
  from vbrl.deployment.loop import go_home
  from vbrl.deployment.policy import load_policy
  from vbrl.rig.measure import measure_goal, write
  from vbrl.rig.view import ExternalView

  path = folder / "goal.json"
  if path.is_file():
    return json.loads(path.read_text())["goal"]
  config = session.config()
  go_home(arm, load_policy(config).metadata.home_pose, config.motion.home_seconds)
  input("Take the T off the table, then Enter to measure the goal outline.")
  image = stills(session, config)["external"]
  fit = measure_goal(ExternalView.load(*STILL), image, "push_t")
  goal = list(fit.pose[:-1])
  print(f"goal      {np.round(goal, 4)}  IoU {fit.iou:.3f}")
  if not fit.reliable:
    raise SystemExit("Unreliable goal fit: something in view, or the exposure is off.")
  write(folder / "goal_external.png", image)
  path.write_text(json.dumps({"goal": goal, "iou": fit.iou}, indent=2) + "\n")
  return goal


def save(folder: Path, record: dict[str, Any], meta: dict[str, Any]) -> None:
  from vbrl.rig.measure import write

  folder.mkdir(parents=True)
  for moment in ("start", "end"):
    for camera, image in record[moment].items():
      write(folder / f"{moment}_{camera}.png", image)
  np.savez_compressed(folder / "trace.npz", **record["trace"])
  (folder / "episode.json").write_text(json.dumps(meta, indent=2) + "\n")


def append_result(path: Path, row: dict[str, Any]) -> None:
  new = not path.is_file()
  with path.open("a", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(row))
    if new:
      writer.writeheader()
    writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
  from vbrl.deployment.arm import TrossenArm

  parser = argparse.ArgumentParser(
    description="Collect one goal block: every start pose, every encoder in order."
  )
  parser.add_argument("session", type=Path, help="configs/deployment/<task>.yaml")
  parser.add_argument("--goal", type=int, required=True, help="goal number, from 1")
  parser.add_argument("--steps", type=int, help="override the session's episode length")
  args = parser.parse_args(argv)

  session = load_session(args.session)
  if args.steps is not None:
    session = dataclasses.replace(session, steps=args.steps)
  root = DATA_ROOT / session.task
  arm = TrossenArm(session.config())
  if session.task == "lift_cube":
    goal = list(session.goals[args.goal - 1])
  else:
    goal = push_t_goal(arm, session, root / f"goal{args.goal}")
  commit = subprocess.run(
    ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
  ).stdout.strip()

  for run in range(1, session.runs + 1):
    for number, encoder in enumerate(session.policies, 1):
      folder = root / f"goal{args.goal}" / encoder / f"run{run}"
      if folder.exists():
        continue
      where = (
        f"goal {args.goal}  run {run}/{session.runs}  "
        f"policy {number}/{len(session.policies)} {encoder}"
      )
      while True:
        print(f"\n=== {where} ===")
        input(f"Place the object at start {run}, then Enter.")
        record = episode(arm, session, encoder, goal)
        print(f"\n=== {where} ===\nresult    {record['result']}")
        if input("[a]ccept or [r]etry? ").strip().lower() == "a":
          break
      meta = {
        "task": session.task,
        "encoder": encoder,
        "goal_number": args.goal,
        "run": run,
        "goal": goal,
        "onnx_file": session.policies[encoder],
        "source_run": record["source_run"],
        "steps": session.steps,
        "steps_executed": len(record["trace"]["action"]),
        "camera": session.camera,
        "cameras": session.cameras,
        "started": record["started"],
        "commit": commit,
        "result": record["result"],
      }
      save(folder, record, meta)
      append_result(
        root / "results.csv",
        {"encoder": encoder, "goal": args.goal, "run": run, **record["result"]},
      )
  print("Goal block complete; the arm holds at home.")
  arm.close()
  return 0


__all__ = ["append_result", "main", "measure"]


if __name__ == "__main__":
  raise SystemExit(main())
