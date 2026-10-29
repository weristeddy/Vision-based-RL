from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
TASK_ID = "Mjlab-LiftCube-CollisionCam-DinoV2ViTS14-LocalGrid7-Trossen"


def test_scene_and_path_imports_stay_free_of_a_simulator(tmp_path: Path) -> None:
  script = """
import sys
import vbrl
import vbrl.paths
import vbrl.asset_zoo.robots
import vbrl.scenes.presets
import vbrl.runtime
import vbrl.vision
import vbrl.scripts.play
for heavy in ('viser', 'wandb', 'transformers', 'r3m', 'mujoco', 'mjlab'):
  assert heavy not in sys.modules, heavy
"""
  environment = dict(os.environ, PYTHONPATH=str(SOURCE_ROOT))
  subprocess.run(
    [sys.executable, "-c", script], cwd=tmp_path, env=environment, check=True
  )


def test_train_config_exposes_the_flags_sweeps_and_cluster_scripts_use() -> None:
  from vbrl.scripts.train import WORKER_ENV, TrainConfig

  config = TrainConfig.from_task(TASK_ID)
  fields = set(TrainConfig.__dataclass_fields__)

  assert {"env", "agent", "video", "gpu_ids", "log_root"} <= fields
  assert {"label", "evaluate"} <= fields
  assert config.evaluate is True
  assert config.gpu_ids == [0]
  assert config.env.scene.num_envs > 0
  assert {"VBRL*", "WANDB*", "HF*"} <= set(WORKER_ENV)


def test_training_returns_its_run_directory(monkeypatch, tmp_path: Path) -> None:
  import mjlab.scripts.train as upstream
  import mjlab.utils.gpu

  from vbrl.scripts.train import TrainConfig, _launch

  calls = []
  monkeypatch.setattr(upstream, "run_train", lambda *a: calls.append(a))
  monkeypatch.setattr(mjlab.utils.gpu, "select_gpus", lambda ids: ([0], 1))
  args = TrainConfig.from_task(TASK_ID)
  object.__setattr__(args, "log_root", str(tmp_path))

  log_dir = _launch(TASK_ID, args)

  assert calls == [(TASK_ID, args, log_dir)]
  assert log_dir.parent == tmp_path / args.agent.experiment_name
  assert log_dir.name.endswith(args.agent.run_name)


def test_saved_checkpoints_sort_by_iteration(tmp_path: Path) -> None:
  from vbrl.evaluation.checkpoints import saved_checkpoints

  for name in ("model_1000.pt", "model_200.pt", "model_100.pt", "run.onnx", "x.pt"):
    (tmp_path / name).touch()

  saved = saved_checkpoints(tmp_path)

  assert [iteration for iteration, _ in saved] == [100, 200, 1000]
  assert saved[-1][1] == tmp_path / "model_1000.pt"


def test_each_checkpoint_is_averaged_over_its_episodes(monkeypatch) -> None:
  import vbrl.evaluation.rollout as rollout
  import vbrl.runtime as runtime
  from vbrl.evaluation.checkpoints import evaluate_checkpoints

  env = SimpleNamespace(close=lambda: None)
  monkeypatch.setattr(runtime, "build_env", lambda *a, **k: env)
  monkeypatch.setattr(
    runtime, "load_trained_policy", lambda env, **k: (env, None, k["ref"], None)
  )
  episodes = {
    "a.pt": [{"success": 1.0, "seed": 0}, {"success": 0.0, "seed": 0}],
    "b.pt": [{"success": 1.0, "seed": 0}, {"success": 1.0, "seed": 0}],
  }
  monkeypatch.setattr(
    rollout, "run_episodes", lambda env, ref, seed: episodes[ref.checkpoint_file]
  )

  rows = evaluate_checkpoints(
    TASK_ID, [(1, Path("a.pt")), (2, Path("b.pt"))], device="cpu"
  )

  assert rows == [{"iteration": 1, "success": 0.5}, {"iteration": 2, "success": 1.0}]


class _Run:
  def __init__(self) -> None:
    self.rows, self.summary, self.saved, self.finished = [], {}, [], False

  def define_metric(self, *args, **kwargs) -> None:
    pass

  def log(self, row: dict) -> None:
    self.rows.append(row)

  def save(self, path: str, base_path: str) -> None:
    self.saved.append(Path(path).name)

  def finish(self) -> None:
    self.finished = True


@pytest.mark.parametrize("evaluate", (True, False))
def test_after_training_the_run_is_evaluated_published_and_trimmed(
  monkeypatch, tmp_path: Path, evaluate: bool
) -> None:
  import wandb

  import vbrl.evaluation.checkpoints as checkpoints
  from vbrl.scripts.train import TrainConfig, _finish

  for iteration in (100, 200):
    (tmp_path / f"model_{iteration}.pt").touch()
  (tmp_path / f"{tmp_path.name}.onnx").touch()

  def evaluate_checkpoints(task_id, saved, *, device, scene=None):
    return [{"iteration": i, "success": 0.5 if scene else 0.25} for i, _ in saved]

  run = _Run()
  monkeypatch.setattr(checkpoints, "evaluate_checkpoints", evaluate_checkpoints)
  monkeypatch.setattr(wandb, "init", lambda **k: run)
  monkeypatch.setenv("WANDB_RUN_ID", "abc123")
  args = TrainConfig.from_task("Mjlab-PushT-WoodTable-DinoV2ViTS14-AfaFull6-Trossen")
  object.__setattr__(args, "evaluate", evaluate)

  _finish("Mjlab-PushT-WoodTable-DinoV2ViTS14-AfaFull6-Trossen", args, tmp_path)

  assert run.finished
  assert run.saved == ["model_200.pt", f"{tmp_path.name}.onnx"]
  assert (tmp_path / "model_100.pt").exists() is not evaluate
  if evaluate:
    assert (tmp_path / "evaluation.csv").exists()
    assert (tmp_path / "final_evaluation.csv").exists()
    assert {"eval/iteration": 200, "eval/success": 0.25} in run.rows
    assert run.summary["final/real_texture_wood/success"] == 0.5


def test_vbrl_list_prints_every_registry_from_its_own_table(capsys) -> None:
  from vbrl.scripts.list_registries import SECTIONS, main

  assert main([]) == 0

  printed = capsys.readouterr().out
  for section, (source, read) in SECTIONS.items():
    assert section in printed
    assert source in printed
    rows = read()
    assert rows, section
    assert str(rows[0]).split()[0] in printed


def test_vbrl_list_can_print_one_section(capsys) -> None:
  from vbrl.scripts.list_registries import main
  from vbrl.vision.architectures import ARCHITECTURES

  assert main(["architectures"]) == 0

  printed = capsys.readouterr().out
  assert "tasks" not in printed
  for token in ARCHITECTURES:
    assert token in printed


def test_bare_train_help_lists_every_registered_task(capsys) -> None:
  import vbrl.scripts.train as cli
  from vbrl.tasks import vbrl_task_ids

  cli._print_task_overview()

  printed = capsys.readouterr().out
  assert "usage: vbrl-train <TASK_ID> [OPTIONS]" in printed
  for task_id in vbrl_task_ids():
    assert task_id in printed


def test_play_parser_accepts_a_local_file_or_a_wandb_run() -> None:
  import vbrl.scripts.play as cli

  local = cli._parser().parse_args([TASK_ID, "--checkpoint-file", "ckpts/m.pt"])
  assert local.task_id == TASK_ID
  assert local.agent == "trained"
  assert local.checkpoint_file == Path("ckpts/m.pt")

  remote = cli._parser().parse_args(
    [
      TASK_ID,
      "--wandb-run-path",
      "entity/project/run",
      "--wandb-checkpoint-name",
      "model_2999.pt",
    ]
  )
  assert remote.wandb_run_path == "entity/project/run"
  assert remote.wandb_checkpoint_name == "model_2999.pt"

  # Architecture comes from the task ID, never from a flag.
  action_names = {action.dest for action in cli._parser()._actions}
  assert not {"encoder", "adapter", "robot", "hidden_dims"} & action_names


@pytest.mark.parametrize(
  "argv",
  (
    [TASK_ID],
    [TASK_ID, "--agent", "zero", "--wandb-run-path", "entity/project/run"],
    [TASK_ID, "--eval-dr", "matched", "--wandb-run-path", "entity/project/run"],
    [TASK_ID, "--num-envs", "0", "--wandb-run-path", "entity/project/run"],
  ),
)
def test_play_cli_rejects_incomplete_or_incompatible_selection(argv) -> None:
  import vbrl.scripts.play as cli

  with pytest.raises(SystemExit):
    cli.main(argv)


def test_run_viser_uses_the_official_viewer_and_stops_the_server(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import mjlab.viewer
  import viser

  from vbrl.scripts.play import run_viser

  calls: dict[str, object] = {}

  class Server:
    def __init__(self, **kwargs) -> None:
      calls["server"] = kwargs

    def get_port(self) -> int:
      return 8080

    def stop(self) -> None:
      calls["stopped"] = True

  class Viewer:
    def __init__(self, env, policy, **kwargs) -> None:
      calls["viewer"] = (env, policy, kwargs)

    def run(self, *, num_steps) -> None:
      calls["num_steps"] = num_steps

  monkeypatch.setattr(viser, "ViserServer", Server)
  monkeypatch.setattr(mjlab.viewer, "ViserPlayViewer", Viewer)
  env, policy = object(), object()

  run_viser(env, policy, host="0.0.0.0", port=8080, frame_rate=30, max_steps=7)

  assert calls["server"] == {
    "host": "0.0.0.0",
    "port": 8080,
    "label": "vision-based-rl",
    "verbose": False,
  }
  assert calls["viewer"] == (env, policy, {"frame_rate": 30, "viser_server": ANY})
  assert calls["num_steps"] == 7
  assert calls["stopped"] is True


def test_record_options_require_a_destination() -> None:
  import vbrl.scripts.play as cli

  parser = cli._parser()
  for argv in (
    [TASK_ID, "--agent", "zero", "--record-width", "640"],
    [TASK_ID, "--agent", "zero", "--record-steps", "10"],
  ):
    with pytest.raises(SystemExit):
      cli._validate(parser, parser.parse_args(argv))


def test_a_gif_is_written_smaller_than_a_video() -> None:
  from vbrl.evaluation.recording import GIF_FPS, GIF_SIZE, VIDEO_SIZE, default_output

  assert default_output(Path("out.gif")) == (GIF_SIZE, GIF_FPS)
  assert default_output(Path("out.GIF")) == (GIF_SIZE, GIF_FPS)
  assert default_output(Path("out.mp4")) == (VIDEO_SIZE, None)
  assert GIF_SIZE < VIDEO_SIZE


def test_the_recording_camera_frames_every_env_it_is_given() -> None:
  torch = pytest.importorskip("torch")
  from mjlab.viewer.viewer_config import ViewerConfig

  from vbrl.evaluation.recording import (
    FRAME_MARGIN,
    SCENE_PADDING_M,
    grid_camera,
  )
  from vbrl.scenes.presets import TABLE_CENTER

  def env(origins: list[list[float]]):
    return SimpleNamespace(
      scene=SimpleNamespace(env_origins=torch.tensor(origins)),
      cfg=SimpleNamespace(viewer=ViewerConfig(geom_group=(1, 0, 0, 1, 0, 0))),
      sim=SimpleNamespace(
        mj_model=SimpleNamespace(vis=SimpleNamespace(global_=SimpleNamespace(fovy=45.0)))
      ),
      num_envs=len(origins),
    )

  one = grid_camera(env([[0.0, 0.0, 0.0]]), width=1280, height=720)
  assert one.origin_type is ViewerConfig.OriginType.WORLD
  assert one.lookat == (TABLE_CENTER[0], TABLE_CENTER[1], 0.0)
  assert one.geom_group == (1, 0, 0, 1, 0, 0)
  assert one.max_extra_envs == 0
  assert one.distance == pytest.approx(
    FRAME_MARGIN
    * SCENE_PADDING_M
    / (2.0 * math.tan(math.radians(45.0) / 2.0) * 1280 / 720)
  )

  assert grid_camera(env([[0.0, 0.0, 0.0]]), width=720, height=720).distance > (
    one.distance
  )

  grid = grid_camera(
    env([[-1.0, -1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0]]),
    width=640,
    height=360,
  )
  assert grid.lookat == (TABLE_CENTER[0], TABLE_CENTER[1], 0.0)
  assert grid.distance > one.distance
  assert grid.max_extra_envs == 3
  assert (grid.width, grid.height) == (640, 360)

  shifted = grid_camera(env([[4.0, 2.0, 0.0], [6.0, 2.0, 0.0]]), width=64, height=64)
  assert shifted.lookat == (5.0 + TABLE_CENTER[0], 2.0 + TABLE_CENTER[1], 0.0)


def test_viser_follows_the_appearance_fields_the_scene_banks_vary() -> None:
  import numpy as np
  from mjlab.viewer.model_sync import VIEWER_MODEL_FIELDS
  from mjlab.viewer.viser import scene

  from vbrl.scripts.play import track_appearance_randomization

  banked = {"mat_texid", "geom_matid"}
  assert banked <= VIEWER_MODEL_FIELDS

  track_appearance_randomization()
  track_appearance_randomization()  # idempotent: no wrapper on a wrapper

  # 1. a texture swap makes the frame's appearance stale enough to rebuild,
  # 2. and turns MJLab's per-env variant path on, which is gated separately.
  for fields in (
    scene._VISER_APPEARANCE_HANDLE_FIELDS,
    scene._VISER_BAKED_HANDLE_FIELDS,
  ):
    assert banked <= fields
    assert {"geom_rgba", "mat_rgba"} <= fields  # upstream's own still count

  # 3. that path uploads one variant per distinct texture rather than one for
  # every env at once.
  def table(texture: int):
    return SimpleNamespace(
      geom_matid=np.array([0]),
      mat_texid=np.array([[texture] * 10]),
      mat_rgba=np.array([[1.0, 1.0, 1.0, 1.0]]),
      geom_type=np.array([6]),
      geom_dataid=np.array([-1]),
      geom_size=np.zeros((1, 3)),
      geom_rgba=np.ones((1, 4)),
      geom_pos=np.zeros((1, 3)),
      geom_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
    )

  fingerprint = scene.MjlabViserScene._geom_subgroup_visual_fingerprint
  assert fingerprint(table(7), [0], False) == fingerprint(table(7), [0], False)
  assert fingerprint(table(7), [0], False) != fingerprint(table(8), [0], False)


def test_zero_agent_uses_the_registered_clip_actions(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  import mjlab.envs as mjlab_envs
  import mjlab.rl as mjlab_rl
  from mjlab.tasks import registry as task_registry

  import vbrl.scripts.play as cli

  cfg = SimpleNamespace(
    scene=SimpleNamespace(num_envs=1), seed=0, terminations={}
  )
  raw = SimpleNamespace(
    close=lambda: None, action_space=SimpleNamespace(shape=(1, 2)), device="cpu"
  )

  class Env:
    def __init__(self, **kwargs):
      del kwargs
      self.unwrapped = raw

    def close(self):
      pass

  class Wrapper:
    def __init__(self, env, *, clip_actions):
      self.unwrapped = env.unwrapped
      self.clip_actions = clip_actions

  calls: dict[str, object] = {}
  monkeypatch.setattr(task_registry, "load_env_cfg", lambda task, play: cfg)
  monkeypatch.setattr(
    task_registry, "load_rl_cfg", lambda task: SimpleNamespace(clip_actions=0.5)
  )
  monkeypatch.setattr(mjlab_envs, "ManagerBasedRlEnv", Env)
  monkeypatch.setattr(mjlab_rl, "RslRlVecEnvWrapper", Wrapper)
  monkeypatch.setattr(
    cli, "run_viser", lambda env, policy, **kwargs: calls.update(
      env=env, policy=policy
    )
  )

  assert cli.main([TASK_ID, "--agent", "zero", "--device", "cpu"]) == 0
  assert calls["env"].clip_actions == 0.5
  action = calls["policy"](None)
  assert tuple(action.shape) == (1, 2)
  assert action.count_nonzero() == 0
