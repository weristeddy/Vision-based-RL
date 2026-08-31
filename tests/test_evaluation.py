from __future__ import annotations

import csv
import statistics
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import vbrl.evaluation.report as report
import vbrl.evaluation.suite as suite
from vbrl.evaluation.rollout import run_episodes
from vbrl.runtime import CheckpointRef
from vbrl.evaluation.suite import EvaluationConfig, EvaluationModel, Scene


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = (
  "Mjlab-LiftCube-CollisionCam-DinoV2ViTS14-LocalGrid7-Trossen"
)


def _model(run: str = "run") -> EvaluationModel:
  return EvaluationModel(
    name=run,
    task_id=TASK_ID,
    ref=CheckpointRef(
      wandb_run_path=f"entity/project/{run}",
      wandb_checkpoint_name="model_2999.pt",
    ),
  )


def _config(output: Path) -> EvaluationConfig:
  return EvaluationConfig(
    name="test evaluation",
    models=(_model(),),
    scenes=(Scene("wood_fixed", "wood", "fixed"),),
    episodes=3,
    seeds=(10, 20),
    output=output,
  )


def _episode(
  run: str,
  scene: str,
  seed: int,
  reward: float,
  success: float,
  *,
  episode_index: int = 0,
) -> dict[str, object]:
  return {
    "name": run,
    "task_id": TASK_ID,
    "checkpoint_file": None,
    "wandb_run_path": f"entity/project/{run}",
    "wandb_checkpoint_name": "model_2999.pt",
    "checkpoint_path": "model_2999.pt",
    "architecture": "dinov2_vits14 + local_grid",
    "scene": scene,
    "seed": seed,
    "episode_index": episode_index,
    "worker_env_id": episode_index,
    "reward": reward,
    "length": 2,
    "success": success,
    "terminated": True,
    "timed_out": False,
  }


def _write_config(path: Path, **changes) -> Path:
  document = {
    "version": 1,
    "name": "multi-scene",
    "models": [
      {
        "name": "run",
        "task_id": TASK_ID,
        "wandb_run_path": "entity/project/run",
        "wandb_checkpoint_name": "model_2999.pt",
      }
    ],
    "scenes": ["wood_fixed", "plaster_matched", "peacock"],
    "episodes": 64,
    "seeds": [10, 20, 30],
    "output": "artifacts/evaluation/unit",
  }
  document.update(changes)
  path.write_text(yaml.safe_dump(document), encoding="utf-8")
  return path


def test_load_config_reads_exact_model_scene_and_seed_contract(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  output = tmp_path / "output"
  monkeypatch.setattr(suite, "artifact_path", lambda _path: output)
  config = suite.load_config(_write_config(tmp_path / "evaluation.yaml"))

  assert config.models == (_model(),)
  assert config.scenes == (
    Scene("wood_fixed", "wood", "fixed"),
    Scene("plaster_matched", "plaster", "matched"),
    Scene("peacock", "peacock", "fixed"),
  )
  assert config.episodes == 64
  assert config.seeds == (10, 20, 30)
  assert config.output == output


def _bad_model(**fields) -> dict[str, object]:
  return {"models": [{"name": "run", "task_id": TASK_ID, **fields}]}


@pytest.mark.parametrize(
  ("changes", "message"),
  [
    ({"seeds": [True]}, "seeds must contain integers"),
    ({"seeds": [4, 4]}, "seeds must not contain duplicates"),
    ({"scenes": ["real_texture"]}, "not an OOD texture replacement"),
    ({"models": [{}]}, "missing=.*name.*task_id"),
    (_bad_model(wandb_run_path="bad"), "must be 'entity/project/run_id'"),
    (
      _bad_model(
        wandb_run_path="entity/project/run", wandb_checkpoint_name="latest.pt"
      ),
      "wandb_checkpoint_name must be model_N.pt",
    ),
    (
      _bad_model(wandb_run_path="entity/project/run", encoder="dinov2"),
      "unknown=.*encoder",
    ),
    (
      _bad_model(
        checkpoint_file="ckpts/model.pt", wandb_run_path="entity/project/run"
      ),
      "exactly one of checkpoint_file or wandb_run_path",
    ),
  ],
)
def test_load_config_rejects_invalid_or_duplicated_model_metadata(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  changes: dict[str, object],
  message: str,
) -> None:
  monkeypatch.setattr(suite, "artifact_path", Path)
  path = _write_config(tmp_path / "evaluation.yaml", **changes)
  with pytest.raises(ValueError, match=message):
    suite.load_config(path)


def test_committed_thesis_evaluations_pin_local_structural_references() -> None:
  from mjlab.tasks.registry import list_tasks

  import vbrl.tasks  # noqa: F401

  registered = set(list_tasks())
  references: set[tuple[str, str]] = set()
  for path in sorted((ROOT / "configs/evaluation").glob("*.yaml")):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["models"]
    for model in raw["models"]:
      assert set(model) == {"name", "task_id", "checkpoint_file"}
      assert model["task_id"] in registered
      assert model["checkpoint_file"].startswith("ckpts/")
      references.add((model["task_id"], model["checkpoint_file"]))
    assert len({model["name"] for model in raw["models"]}) == len(raw["models"])
    loaded = suite.load_config(path)
    assert len(loaded.models) == len(raw["models"])

  lift = {
    reference
    for reference in references
    if reference[0].startswith("Mjlab-LiftCube-")
  }
  assert len({checkpoint for _, checkpoint in lift}) == 24
  assert {
    reference
    for reference in references
    if reference[0].startswith("Mjlab-PushT-")
  } == {
    (
      "Mjlab-PushT-State-TrossenRealistic",
      "ckpts/push_t/state.pt",
    ),
    (
      "Mjlab-PushT-RealTexture-DinoV2ViTS14-LocalGrid7-TrossenRealistic",
      "ckpts/push_t/dinov2_vits14_local_grid7_real_texture_success98.pt",
    ),
  }


class _CommandManager:
  active_terms = ("lift",)

  def __init__(self) -> None:
    self.term = SimpleNamespace(metrics={"episode_success": torch.zeros(3)})

  def get_term(self, name: str):
    assert name == "lift"
    return self.term


class _VectorEnv:
  num_envs = 3
  max_episode_length = 3
  device = "cpu"

  def __init__(self) -> None:
    self.unwrapped = self
    self.command_manager = _CommandManager()
    self.ages = torch.zeros(3, dtype=torch.int64)
    self.thresholds = torch.tensor([1, 2, 3])
    self.reset_ids: list[list[int] | None] = []
    self.seeds: list[int] = []

  def seed(self, seed: int) -> None:
    self.seeds.append(seed)

  def reset(self, *, env_ids=None):
    ids = None if env_ids is None else env_ids.tolist()
    self.reset_ids.append(ids)
    if env_ids is None:
      self.ages.zero_()
    else:
      self.ages[env_ids] = 0
    return {"policy": torch.zeros(3, 1)}, {}

  def step(self, actions):
    del actions
    self.ages += 1
    done = self.ages >= self.thresholds
    self.command_manager.term.metrics["episode_success"] = torch.tensor(
      [1.0, 0.25, 0.75]
    )
    time_outs = done & torch.tensor([False, False, True])
    return (
      {"policy": torch.zeros(3, 1)},
      torch.tensor([1.0, 2.0, 3.0]),
      done,
      {"time_outs": time_outs},
    )


def test_rollout_records_first_episode_per_worker_and_resets_manually() -> None:
  env = _VectorEnv()
  rows = run_episodes(env, lambda observations: observations, seed=41)

  assert [row["worker_env_id"] for row in rows] == [0, 1, 2]
  assert [row["episode_index"] for row in rows] == [0, 1, 2]
  assert [row["reward"] for row in rows] == [1.0, 4.0, 9.0]
  assert [row["length"] for row in rows] == [1, 2, 3]
  assert [row["success"] for row in rows] == [1.0, 0.25, 0.75]
  assert [row["terminated"] for row in rows] == [True, True, False]
  assert [row["timed_out"] for row in rows] == [False, False, True]
  assert env.seeds == [41]
  assert env.reset_ids == [None, [0], [0, 1]]


def test_summarize_gives_each_seed_equal_weight() -> None:
  rows = [
    _episode("run", "wood", 1, 0.0, 0.0),
    _episode("run", "wood", 2, 9.0, 1.0),
    _episode("run", "wood", 2, 10.0, 1.0, episode_index=1),
    _episode("run", "wood", 2, 11.0, 1.0, episode_index=2),
  ]
  (summary,) = report.summarize(rows)
  assert summary["seeds"] == 2
  assert summary["episodes"] == 4
  assert summary["mean_reward"] == 5.0
  assert summary["reward_seed_std"] == pytest.approx(
    statistics.stdev([0.0, 10.0])
  )
  assert summary["success_rate"] == 0.5


def test_write_report_handles_generic_model_and_scene_lists(
  tmp_path: Path,
) -> None:
  rows = [
    _episode("real", scene, 1, 2.0, 1.0)
    for scene in ("wood_fixed", "peacock")
  ] + [
    _episode("procedural", scene, 1, 1.0, 0.0)
    for scene in ("wood_fixed", "peacock")
  ]
  output = report.write_report(rows, output=tmp_path, title="evaluation")
  episode_rows = list(
    csv.DictReader((output / "episodes.csv").open(encoding="utf-8"))
  )
  summary_rows = list(
    csv.DictReader((output / "summary.csv").open(encoding="utf-8"))
  )
  assert len(episode_rows) == 4
  assert len(summary_rows) == 4
  assert (output / "evaluation.png").stat().st_size > 0


def test_run_suite_executes_the_complete_model_scene_seed_matrix(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  config = EvaluationConfig(
    name="matrix",
    models=(_model("first"), _model("second")),
    scenes=(
      Scene("wood_fixed", "wood", "fixed"),
      Scene("peacock", "peacock", "fixed"),
    ),
    episodes=3,
    seeds=(10, 20),
    output=tmp_path,
  )
  calls = []
  monkeypatch.setattr(
    suite,
    "_run_case",
    lambda _config, model, scene, seed, device: calls.append(
      (model.name, scene.name, seed, device)
    ) or [_episode(model.name, scene.name, seed, 1.0, 1.0)],
  )
  written = []
  monkeypatch.setattr(
    suite,
    "write_report",
    lambda rows, *, output, title: written.extend(rows) or output,
  )

  assert suite.run_suite(config, "cpu") == tmp_path
  assert calls == [
    (model.name, scene.name, seed, "cpu")
    for model in config.models
    for scene in config.scenes
    for seed in config.seeds
  ]
  assert len(written) == 8
