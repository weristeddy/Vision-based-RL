from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

TASKS = {"push_t": "PushT", "lift_cube": "LiftCube"}
# Every Lift-Cube student lifts in all 512 episodes, so precision decides there.
TIE_BREAK = {"push_t": ("overlap", max), "lift_cube": ("position_error", min)}


def best_runs(rows: list[dict], task: str) -> dict[str, str]:
  metric, pick = TIE_BREAK[task]
  values: dict[tuple[str, str], dict[str, float]] = {}
  for row in rows:
    if row["task"] == TASKS[task] and row["scene"] == "wood049":
      values.setdefault((row["encoder"], row["run"]), {})[row["metric"]] = row["value"]
  best: dict[str, str] = {}
  for encoder in sorted({encoder for encoder, _ in values}):
    runs = {run: v for (e, run), v in values.items() if e == encoder}
    top = max(v["success"] for v in runs.values())
    tied = {run: v[metric] for run, v in runs.items() if v["success"] == top}
    best[encoder] = pick(tied, key=tied.get)
  return best


def main(argv: list[str] | None = None) -> None:
  import wandb

  from vbrl.analysis.final_evaluation import ENTITY_PROJECT, fetch
  from vbrl.deployment.session import load_session

  parser = argparse.ArgumentParser(
    description="Download the best seed of every policy a session lists, as ONNX."
  )
  parser.add_argument("session", type=Path, help="configs/deployment/<task>.yaml")
  args = parser.parse_args(argv)

  session = load_session(args.session)
  chosen = best_runs(fetch(), session.task)
  missing = set(session.policies) - set(chosen)
  if missing:
    raise SystemExit(f"No evaluated runs for {sorted(missing)}.")
  api = wandb.Api(timeout=120)
  record = {}
  for encoder, path in session.policies.items():
    target, run_id = Path(path), chosen[encoder]
    record[encoder] = run_id
    if target.is_file():
      print(f"{encoder:24s} {run_id}  {target} exists")
      continue
    target.parent.mkdir(parents=True, exist_ok=True)
    run = api.run(f"{ENTITY_PROJECT}/{run_id}")
    onnx = next(file for file in run.files() if file.name.endswith(".onnx"))
    onnx.download(root=str(target.parent / run_id), replace=True)
    (target.parent / run_id / onnx.name).replace(target)
    shutil.rmtree(target.parent / run_id)
    print(f"{encoder:24s} {run_id}  -> {target}")
  (target.parent / "policies.json").write_text(json.dumps(record, indent=2) + "\n")


__all__ = ["best_runs", "main"]


if __name__ == "__main__":
  main()
