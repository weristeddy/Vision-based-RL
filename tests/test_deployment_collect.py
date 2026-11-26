from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from vbrl.deployment.collect import append_result, measure
from vbrl.deployment.fetch_policies import best_runs
from vbrl.deployment.session import load_session

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("task", ["push_t", "lift_cube"])
def test_session_configs_load(task):
  session = load_session(ROOT / "configs/deployment" / f"{task}.yaml")
  assert session.task == task
  assert len(session.policies) == (6 if task == "lift_cube" else 7)
  assert session.rig["control_hz"] == 20.0
  assert session.other_camera != session.camera
  assert len(session.goals) == (3 if task == "lift_cube" else 0)


def test_session_rejects_unknown_fields(tmp_path):
  text = (ROOT / "configs/deployment/push_t.yaml").read_text() + "\nexposure: 1\n"
  (tmp_path / "s.yaml").write_text(text)
  with pytest.raises(ValueError, match="exposure"):
    load_session(tmp_path / "s.yaml")


def test_lift_success_needs_holding_and_distance():
  trace = {"holding": np.array([False, True]), "goal_distance": np.array([0.2, 0.03])}
  assert measure("lift_cube", None, None, trace, None)["success"]
  trace["holding"][-1] = False
  assert not measure("lift_cube", None, None, trace, None)["success"]
  trace = {"holding": np.array([True]), "goal_distance": np.array([0.06])}
  assert not measure("lift_cube", None, None, trace, None)["success"]


def test_results_csv_writes_one_header(tmp_path):
  path = tmp_path / "results.csv"
  append_result(path, {"encoder": "a", "run": 1, "success": True})
  append_result(path, {"encoder": "b", "run": 2, "success": False})
  rows = list(csv.DictReader(path.open()))
  assert [row["encoder"] for row in rows] == ["a", "b"]


def test_best_run_prefers_success_then_tie_break():
  def rows(run, success, overlap):
    return [
      {
        "task": "PushT",
        "scene": "wood049",
        "encoder": "E",
        "run": run,
        "metric": "success",
        "value": success,
      },
      {
        "task": "PushT",
        "scene": "wood049",
        "encoder": "E",
        "run": run,
        "metric": "overlap",
        "value": overlap,
      },
      {
        "task": "PushT",
        "scene": "marble019",
        "encoder": "E",
        "run": run,
        "metric": "success",
        "value": 1.0,
      },
    ]

  table = rows("a", 0.6, 0.9) + rows("b", 0.7, 0.8) + rows("c", 0.7, 0.85)
  assert best_runs(table, "push_t") == {"E": "c"}



def test_a_task_file_picks_policy_camera_and_goal(tmp_path) -> None:
  session = load_session(ROOT / "configs/deployment/lift_cube.yaml")
  config = session.config("CompactVit-Svea", session.goal("2"))
  assert config.onnx_file.endswith("CompactVit-Svea.onnx")
  assert config.camera_serial == session.cameras["wrist"]["camera_serial"]
  assert config.goal == (0.45, 0.12, 0.25)
  other = session.config(camera=session.other_camera)
  assert other.camera_exposure_us == session.cameras["external"]["camera_exposure_us"]
  assert session.goal("0.4,0.1,0.3") == [0.4, 0.1, 0.3]
  (tmp_path / "goal.json").write_text('{"goal": [0.38, 0.02, 0.1]}')
  assert session.goal(str(tmp_path / "goal.json")) == [0.38, 0.02, 0.1]
  with pytest.raises(SystemExit, match="No policy"):
    session.config("DinoV3ViTS16-AfaFull6")
