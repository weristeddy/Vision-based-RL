from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mjlab")


# Restated independently of vision/architectures.py, so a mistake in that table cannot
# silently agree with itself here.
POOLED_ARCHITECTURES = (
  "NatureCnn-LocalGrid7",
  "NatureCnn-SpatialSoftmax",
  "CompactVit-LocalGrid8",
  "CompactVit-SpatialSoftmax",
  "DinoV2ViTS14-Linear",
  "DinoV2ViTS14-LocalGrid7",
  "DinoV2ViTS14-SpatialSoftmax",
  "DinoV2ViTS14-Afa6",
  "R3MResNet50-Linear",
  "R3MResNet50-LocalGrid7",
  "R3MResNet50-SpatialSoftmax",
  "R3MResNet50-Afa32",
)
COLLISION_CAM_ARCHITECTURES = tuple(
  "NatureCnn-LocalGrid16" if arch == "NatureCnn-LocalGrid7" else arch
  for arch in POOLED_ARCHITECTURES
)
# The retained grid, unchanged: this generation moves the camera and nothing
# else, so substituting rows would change the adapter at the same time.
SIM2REAL_ARCHITECTURES = COLLISION_CAM_ARCHITECTURES
EXPECTED_TASK_IDS = frozenset(
  (
    *(
      f"Mjlab-LiftCube-CollisionCam-{arch}-Trossen"
      for arch in COLLISION_CAM_ARCHITECTURES
    ),
    *(f"Mjlab-LiftCube-RealTexture-{arch}-Trossen" for arch in POOLED_ARCHITECTURES),
    *(
      f"Mjlab-LiftCube-Sim2Real-{arch}-TrossenRealistic"
      for arch in SIM2REAL_ARCHITECTURES
    ),
    "Mjlab-PushT-State-TrossenRealistic",
    "Mjlab-PushT-VisualSlowStep-DinoV2ViTS14-Afa6-TrossenRealistic",
    "Mjlab-PushT-PixelGoalFixed-DinoV2ViTS14-Afa6-TrossenRealistic",
    "Mjlab-PushT-GoalOutline-DinoV2ViTS14-Afa6-TrossenRealistic",
    "Mjlab-PushT-GoalOutlinePixel-DinoV2ViTS14-Afa6-TrossenRealistic",
    "Mjlab-PushT-GoalColour-DinoV2ViTS14-Afa6-TrossenRealistic",
    "Mjlab-PushT-GoalColourPixel-DinoV2ViTS14-Afa6-TrossenRealistic",
  )
)


def test_the_registered_id_set_is_exactly_these_43_tasks() -> None:
  from vbrl.tasks import vbrl_task_ids

  assert frozenset(vbrl_task_ids()) == EXPECTED_TASK_IDS
  assert len(EXPECTED_TASK_IDS) == 43


def test_no_id_names_the_default_camera() -> None:
  from vbrl.tasks import vbrl_task_ids

  assert not [t for t in vbrl_task_ids() if "VisualCam" in t]


def test_rl_def_asserts_task_ids_that_actually_exist() -> None:
  import re

  from vbrl.tasks import vbrl_task_ids

  definition = Path(__file__).resolve().parent.parent / "rl.def"
  test_block = definition.read_text().split("%test", 1)[1]
  asserted = set(re.findall(r'"(Mjlab-[A-Za-z0-9-]+)"', test_block))

  assert asserted, "rl.def's %test names no task IDs at all"
  missing = sorted(asserted - set(vbrl_task_ids()))
  assert not missing, f"rl.def's %test requires unregistered task IDs: {missing}"


def test_every_visual_task_sees_the_one_external_camera() -> None:
  from mjlab.tasks.registry import load_env_cfg

  from vbrl.tasks import vbrl_task_ids

  retired = {"external_front_cam", "external_tilted_cam"}
  seen_external = 0
  for task_id in vbrl_task_ids():
    sensors = {s.name for s in (load_env_cfg(task_id).scene.sensors or ())}
    assert not (sensors & retired), f"{task_id} names a retired camera"
    external = {s for s in sensors if s.startswith("external")}
    assert external <= {"external_cam"}, f"{task_id} has {external}"
    seen_external += bool(external)

  # Lift-Cube is a wrist-camera task, so its 36 visual IDs declare `cam` alone.
  assert seen_external == 6


def test_only_the_visual_arms_widen_the_goal_yaw() -> None:
  from mjlab.tasks.registry import load_env_cfg

  from vbrl.tasks import vbrl_task_ids
  from vbrl.tasks.push_t.push_t_env_cfg import GOAL_YAW_STAGES

  seen = 0
  for task_id in vbrl_task_ids():
    if not task_id.startswith("Mjlab-PushT-"):
      continue
    cfg = load_env_cfg(task_id)
    command = cfg.commands["push_t_goal"]
    scheduled = "goal_yaw_range" in cfg.curriculum
    assert scheduled is (task_id != "Mjlab-PushT-State-TrossenRealistic"), task_id
    assert command.success_threshold == pytest.approx(0.90), task_id
    # The registered range is always the full circle; the curriculum narrows it
    # at runtime and hands it back, so evaluation is never made easier.
    assert command.target_yaw_range == pytest.approx((-math.pi, math.pi)), task_id
    seen += scheduled

  assert seen == 6
  assert GOAL_YAW_STAGES[0]["half_range"] == 0.0
  assert GOAL_YAW_STAGES[-1]["half_range"] == pytest.approx(math.pi)
  assert GOAL_YAW_STAGES[1]["step"] == 9_600
  assert len(GOAL_YAW_STAGES) == 9


def test_a_play_environment_never_carries_a_curriculum() -> None:
  from mjlab.tasks.registry import load_env_cfg

  from vbrl.tasks import vbrl_task_ids

  for task_id in vbrl_task_ids():
    assert load_env_cfg(task_id, play=True).curriculum == {}, task_id


def test_every_visual_id_names_a_row_of_the_architecture_table() -> None:
  import re

  from vbrl.tasks import vbrl_task_ids
  from vbrl.vision.architectures import ARCHITECTURES

  registered = vbrl_task_ids()
  for task_id in registered:
    if task_id.endswith(("-State-Trossen", "-State-TrossenRealistic")):
      continue
    token = re.sub(r"^Mjlab-\w+-\w+-|-\w+$", "", task_id)
    assert token in ARCHITECTURES, task_id


def test_registration_is_task_local_and_static() -> None:
  root = Path("src/vbrl/tasks")
  for package in (
    root / "lift_cube/config/trossen",
    root / "push_t/config/trossen_realistic",
  ):
    assert (package / "__init__.py").is_file()
    assert (package / "env_cfgs.py").is_file()
    assert (package / "rl_cfg.py").is_file()


def test_installed_entry_points_include_tasks_and_console_commands() -> None:
  pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

  assert '[project.entry-points."mjlab.tasks"]' in pyproject
  assert 'vbrl = "vbrl.tasks"' in pyproject
  for command, target in {
    "vbrl-list": "vbrl.scripts.list_registries:main",
    "vbrl-train": "vbrl.scripts.train:main",
    "vbrl-evaluate": "vbrl.scripts.evaluate:main",
    "vbrl-analyze": "vbrl.scripts.analyze:main",
    "vbrl-play": "vbrl.scripts.play:main",
    "vbrl-fetch-backbones": "vbrl.scripts.fetch_backbones:main",
  }.items():
    assert f'{command} = "{target}"' in pyproject


def test_task_packages_populate_registry_in_a_fresh_process() -> None:
  source = """
import json
from vbrl.tasks import vbrl_task_ids
print(json.dumps(list(vbrl_task_ids())))
"""
  environment = dict(os.environ, PYTHONPATH=str(Path("src").resolve()))
  result = subprocess.run(
    [sys.executable, "-c", source],
    check=True,
    capture_output=True,
    text=True,
    env=environment,
  )

  registered = json.loads(result.stdout.strip().splitlines()[-1])
  assert frozenset(registered) == EXPECTED_TASK_IDS


def test_native_registry_returns_independent_environment_and_agent_copies() -> None:
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  task_id = "Mjlab-PushT-VisualSlowStep-DinoV2ViTS14-Afa6-TrossenRealistic"
  first_env, second_env = load_env_cfg(task_id), load_env_cfg(task_id)
  first_agent, second_agent = load_rl_cfg(task_id), load_rl_cfg(task_id)

  assert first_env is not second_env
  assert first_env is not load_env_cfg(task_id, play=True)
  assert first_agent is not second_agent

  original_envs = second_env.scene.num_envs
  original_iterations = second_agent.max_iterations
  first_env.scene.num_envs = 7
  first_agent.max_iterations = 9
  assert second_env.scene.num_envs == original_envs
  assert second_agent.max_iterations == original_iterations


def test_every_registered_train_config_is_cloudpickle_serializable() -> None:
  import cloudpickle

  from vbrl.scripts.train import TrainConfig

  for task_id in sorted(EXPECTED_TASK_IDS):
    train_cfg = TrainConfig.from_task(task_id)
    restored = cloudpickle.loads(cloudpickle.dumps(train_cfg))
    assert restored.agent.wandb_tags == train_cfg.agent.wandb_tags
    assert restored.env.scene.num_envs == train_cfg.env.scene.num_envs
