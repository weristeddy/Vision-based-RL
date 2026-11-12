from __future__ import annotations

import argparse
import os

STEP = "eval/iteration"


def eval_keys(entity: str, project: str) -> list[str]:
  import wandb

  runs = wandb.Api().runs(
    f"{entity}/{project}", filters={f"summary_metrics.{STEP}": {"$exists": True}}
  )
  keys = {key for run in runs for key in run.summary.keys() if key.startswith("eval/")}
  return sorted(keys - {STEP})


def build(entity: str, project: str, name: str, keys: list[str]):
  import wandb_workspaces.reports.v2 as wr
  import wandb_workspaces.workspaces as ws

  panels = [
    wr.LinePlot(
      title=key.removeprefix("eval/"),
      x=STEP,
      y=[key],
      aggregate=True,
      groupby_aggfunc="mean",
      groupby_rangefunc="stddev",
    )
    for key in keys
  ]
  return ws.Workspace(
    entity=entity,
    project=project,
    name=name,
    sections=[
      ws.Section(name="Evaluation (mean ± std over seeds)", panels=panels, is_open=True)
    ],
    runset_settings=ws.RunsetSettings(
      filters=[ws.Metric("Group").notin([None, ""])],
      groupby=[ws.Metric("Group")],
    ),
  )


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Save a W&B view that groups runs by W&B group (one group per "
    "--label and task, so per seed set) and draws every eval/* metric any run in "
    "the project has logged as the mean with a standard-deviation band. Run it "
    "again after a task logs new metrics. Run with: uv run --with "
    "wandb-workspaces python cluster/wandb_view.py"
  )
  parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
  parser.add_argument("--project", default="mjlab")
  parser.add_argument("--name", default="Seeds: eval mean and std")
  args = parser.parse_args()
  if not args.entity:
    parser.error("--entity is required (or set WANDB_ENTITY).")
  keys = eval_keys(args.entity, args.project)
  workspace = build(args.entity, args.project, args.name, keys).save()
  print(f"{len(keys)} eval metrics: {workspace.url}")


if __name__ == "__main__":
  main()
