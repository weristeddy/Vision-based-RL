"""The exact set of task IDs this package registers, and how it registers them."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest


pytest.importorskip("mjlab")


# Restated independently of vision/architectures.py so a mistake in that table
# cannot silently agree with itself here.
#
# The pooled generation. Every local grid here is smaller than the grid its
# encoder produces, which is what CURRENT_ARCHITECTURES replaced; these IDs stay
# registered because runs and checkpoints exist against them.
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
# The frozen Lift-Cube generation: NatureCnn keeps the grid its checkpoints were
# trained with.
COLLISION_CAM_ARCHITECTURES = tuple(
  "NatureCnn-LocalGrid16" if arch == "NatureCnn-LocalGrid7" else arch
  for arch in POOLED_ARCHITECTURES
)
# Current: every adapter reads its encoder's native grid, and no scratch encoder
# carries a head heavier than itself.
CURRENT_ARCHITECTURES = (
  "NatureCnn-Flatten",
  "NatureCnn-SpatialSoftmax",
  "NatureCnn-Afa1",
  "CompactVit-Flatten",
  "CompactVit-SpatialSoftmax",
  "CompactVit-Afa2",
  "DinoV2ViTS14-Linear",
  "DinoV2ViTS14-LocalGrid16",
  "DinoV2ViTS14-SpatialSoftmax",
  "DinoV2ViTS14-Afa6",
  "R3MResNet50-Linear",
  "R3MResNet50-LocalGrid7",
  "R3MResNet50-SpatialSoftmax",
  "R3MResNet50-Afa32",
)
# R3M tapped one stage earlier. Added after the FrontCam generation had already
# trained, so only the Curriculum arm crosses these.
# AFA is dropped for the scratch encoders from SlowGoal on: it is
# permutation-invariant over position-free CNN features and scored the
# predict-the-mean baseline. It stays for the frozen backbones.
SLOW_GOAL_ARCHITECTURES = tuple(
  a for a in CURRENT_ARCHITECTURES if a not in ("NatureCnn-Afa1", "CompactVit-Afa2")
)
LAYER3_ARCHITECTURES = (
  "R3MResNet50L3-LocalGrid14",
  "R3MResNet50L3-SpatialSoftmax",
  "R3MResNet50L3-Afa16",
)
# `Balanced` crosses the same architectures as SlowGoal, except that the two
# trainable trunks take ManiSkill3's rectified head instead of the layer-normed
# one. Every other row is unchanged, so the two sweeps differ by the actor's
# stream balance alone.
BALANCED_ARCHITECTURES = tuple(
  a.replace("-Flatten", "-FlattenRelu")
  for a in SLOW_GOAL_ARCHITECTURES + LAYER3_ARCHITECTURES
)
EXPECTED_TASK_IDS = frozenset(
  (
    *(
      f"Mjlab-LiftCube-CollisionCam-{arch}-Trossen"
      for arch in COLLISION_CAM_ARCHITECTURES
    ),
    *(
      f"Mjlab-LiftCube-RealTexture-{arch}-Trossen"
      for arch in POOLED_ARCHITECTURES
    ),
    *(
      f"Mjlab-PushT-{variant}-{arch}-TrossenRealistic"
      for variant in ("RealTexture", "Default")
      for arch in POOLED_ARCHITECTURES
    ),
    *(
      f"Mjlab-PushT-{variant}-{arch}-TrossenRealistic"
      for variant in ("FrontCam", "Curriculum")
      for arch in CURRENT_ARCHITECTURES
    ),
    *(
      f"Mjlab-PushT-Curriculum-{arch}-TrossenRealistic"
      for arch in LAYER3_ARCHITECTURES
    ),
    *(
      f"Mjlab-PushT-SlowGoal-{arch}-TrossenRealistic"
      for arch in SLOW_GOAL_ARCHITECTURES + LAYER3_ARCHITECTURES
    ),
    # No curriculum: the goal covers the full circle from the first episode.
    # `Uniform` keeps ManiSkill's reward and is the control for `UniformQuad`,
    # which swaps the orientation factor for one with a gradient at 180 degrees.
    *(
      f"Mjlab-PushT-{variant}-{arch}-TrossenRealistic"
      for variant in ("Uniform", "UniformQuad")
      for arch in SLOW_GOAL_ARCHITECTURES + LAYER3_ARCHITECTURES
    ),
    *(
      f"Mjlab-PushT-Balanced-{arch}-TrossenRealistic"
      for arch in BALANCED_ARCHITECTURES
    ),
    # SlowGoal with the target drawn on the table, matched row for row.
    *(
      f"Mjlab-PushT-VisualGoal-{arch}-TrossenRealistic"
      for arch in SLOW_GOAL_ARCHITECTURES + LAYER3_ARCHITECTURES
    ),
    # The object and the goal share one sampling range: unbiased, a quarter of
    # episodes biased close, and a cap that grows from close to unbiased.
    *(
      f"Mjlab-PushT-{variant}-{arch}-TrossenRealistic"
      for variant in ("FreeStart", "NearGoal", "GrowStart")
      for arch in SLOW_GOAL_ARCHITECTURES + LAYER3_ARCHITECTURES
    ),
    # The two arms that learned to orient the T, on that same shared range.
    # `FreeStart`, `NearGoal` and `GrowStart` drop the 15 cm floor *and* the
    # curriculum or drawn goal at once, so which of the two mattered is not
    # separable from them; these hold the second fixed and move only the floor.
    *(
      f"Mjlab-PushT-{variant}-{arch}-TrossenRealistic"
      for variant in ("SlowFree", "VisualFree")
      for arch in SLOW_GOAL_ARCHITECTURES + LAYER3_ARCHITECTURES
    ),
    "Mjlab-PushCube-State-Trossen",
    "Mjlab-PushT-State-TrossenRealistic",
  )
)


def test_the_registered_id_set_is_exactly_these_231_tasks() -> None:
  from vbrl.tasks import vbrl_task_ids

  assert frozenset(vbrl_task_ids()) == EXPECTED_TASK_IDS
  assert len(EXPECTED_TASK_IDS) == 231


def test_no_id_names_the_default_camera() -> None:
  """The visual camera is the unnamed default; only the legacy set is marked."""
  from vbrl.tasks import vbrl_task_ids

  assert not [t for t in vbrl_task_ids() if "VisualCam" in t]


def test_rl_def_asserts_task_ids_that_actually_exist() -> None:
  """``rl.def``'s %test names one task ID per family; they have to be real.

  That block is the only gate on the image build, and it runs *after* twenty
  minutes of installing. A renamed variant silently turns it into a build that
  always fails -- which is exactly what happened to ``Mjlab-LiftCube-VisualCam-``
  after the variant became ``RealTexture``: the suite already asserted no ID
  named ``VisualCam`` while ``rl.def`` still required one. Checking the two
  against each other here costs a second and moves that failure off the cluster.
  """
  import re

  from vbrl.tasks import vbrl_task_ids

  definition = Path(__file__).resolve().parent.parent / "rl.def"
  test_block = definition.read_text().split("%test", 1)[1]
  asserted = set(re.findall(r'"(Mjlab-[A-Za-z0-9-]+)"', test_block))

  assert asserted, "rl.def's %test names no task IDs at all"
  missing = sorted(asserted - set(vbrl_task_ids()))
  assert not missing, f"rl.def's %test requires unregistered task IDs: {missing}"


def test_the_candidate_camera_reaches_only_the_arms_evaluating_it() -> None:
  """Each non-default camera reaches only the generations that use it.

  ``external`` is still the MJCF default. ``external_front`` belongs to the two
  generations that evaluated it, and ``external_tilted`` -- which keeps twice as
  much of the object visible during contact -- to the current one. Promoting one
  means moving its pose onto ``external``, not spreading a third name through the
  registry.
  """
  from mjlab.tasks.registry import load_env_cfg

  from vbrl.tasks import vbrl_task_ids

  candidate_arms = ("-FrontCam-", "-Curriculum-")
  tilted_arms = (
    "-SlowGoal-", "-Uniform-", "-UniformQuad-", "-Balanced-", "-VisualGoal-",
    "-SlowFree-", "-VisualFree-",
    "-FreeStart-", "-NearGoal-", "-GrowStart-",
  )
  for task_id in vbrl_task_ids():
    sensors = {s.name for s in (load_env_cfg(task_id).scene.sensors or ())}
    assert ("external_front_cam" in sensors) is any(
      arm in task_id for arm in candidate_arms
    ), task_id
    assert ("external_tilted_cam" in sensors) is any(
      arm in task_id for arm in tilted_arms
    ), task_id


def test_only_the_scheduled_arms_widen_the_goal_yaw() -> None:
  """Which variants schedule the goal yaw, and which use ManiSkill's threshold.

  The two are no longer the same set. `Curriculum` and `SlowGoal` schedule the
  goal; `Uniform` and `UniformQuad` deliberately do not, but share the 0.90
  threshold because they are meant to be compared against `SlowGoal`. Every
  other generation keeps the 0.98 threshold its results were measured against.
  """
  from mjlab.tasks.registry import load_env_cfg

  from vbrl.tasks import vbrl_task_ids
  from vbrl.tasks.push_t.push_t_env_cfg import (
    GOAL_YAW_CURRICULUM_STAGES,
    GOAL_YAW_SLOW_STAGES,
  )

  seen = 0
  for task_id in vbrl_task_ids():
    if not task_id.startswith("Mjlab-PushT-"):
      continue
    cfg = load_env_cfg(task_id)
    command = cfg.commands["push_t_goal"]
    scheduled = "goal_yaw_range" in cfg.curriculum
    arm = any(
      m in task_id
      for m in ("-Curriculum-", "-SlowGoal-", "-Balanced-", "-SlowFree-")
    )
    assert scheduled is arm, task_id
    lenient = arm or any(
      m in task_id
      for m in (
        "-Uniform-", "-UniformQuad-", "-VisualGoal-", "-VisualFree-",
        "-FreeStart-",
        "-NearGoal-", "-GrowStart-",
      )
    )
    assert command.success_threshold == (0.90 if lenient else 0.98), task_id
    # The registered range is always the full circle; the curriculum narrows it
    # at runtime and hands it back, so evaluation is never made easier.
    assert command.target_yaw_range == pytest.approx((-math.pi, math.pi)), task_id
    seen += scheduled

  assert seen == 62
  # Starts fixed, ends at the full circle -- strictly harder than ManiSkill3,
  # whose goal pose stays fixed for the whole of training.
  for stages in (GOAL_YAW_CURRICULUM_STAGES, GOAL_YAW_SLOW_STAGES):
    assert stages[0]["half_range"] == 0.0
    assert stages[-1]["half_range"] == pytest.approx(math.pi)
  # SlowGoal exists to hold the goal fixed for far longer before widening.
  assert GOAL_YAW_SLOW_STAGES[1]["step"] > 5 * GOAL_YAW_CURRICULUM_STAGES[1]["step"]
  # SlowGoal's rungs are half the size of Curriculum's, so each transition is a
  # smaller distribution shift. Across the 15 runs trained on the coarse version,
  # 45-degree rungs made yaw error worse in 8 of them.
  assert len(GOAL_YAW_SLOW_STAGES) > len(GOAL_YAW_CURRICULUM_STAGES)


def test_a_play_environment_never_carries_a_curriculum() -> None:
  """Evaluation must not inherit a partially-widened goal range."""
  from mjlab.tasks.registry import load_env_cfg

  from vbrl.tasks import vbrl_task_ids

  for task_id in vbrl_task_ids():
    assert load_env_cfg(task_id, play=True).curriculum == {}, task_id


def test_the_architecture_table_and_the_registry_cannot_drift() -> None:
  """Every table row reaches a task ID, and every visual ID uses a table row."""
  import re

  from vbrl.tasks import vbrl_task_ids
  from vbrl.vision.architectures import ARCHITECTURES

  registered = vbrl_task_ids()
  for token in ARCHITECTURES:
    assert any(f"-{token}-" in task_id for task_id in registered), token

  # Mjlab-<Task>-<Variant>-<Arch>-<Robot>; State tasks carry no architecture.
  for task_id in registered:
    if task_id.endswith(("-State-Trossen", "-State-TrossenRealistic")):
      continue
    token = re.sub(r"^Mjlab-\w+-\w+-|-\w+$", "", task_id)
    assert token in ARCHITECTURES, task_id


def test_registration_is_task_local_and_static() -> None:
  """Each ``config/<robot>/`` package registers its own IDs; nothing generates them."""
  root = Path("src/vbrl/tasks")
  for package in (
    root / "lift_cube/config/trossen",
    root / "push_cube/config/trossen",
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
    "vbrl-visualize": "vbrl.scripts.play:main",
    "vbrl-fetch-backbones": "vbrl.scripts.fetch_backbones:main",
  }.items():
    assert f'{command} = "{target}"' in pyproject


def test_task_packages_populate_registry_in_a_fresh_process() -> None:
  """TorchrunX workers import from scratch, so registration cannot rely on state."""
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

  task_id = "Mjlab-PushT-RealTexture-DinoV2ViTS14-LocalGrid7-TrossenRealistic"
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
  """TorchrunX cloudpickles the whole TrainConfig out to every worker."""
  import cloudpickle

  from vbrl.scripts.train import TrainConfig

  for task_id in sorted(EXPECTED_TASK_IDS):
    train_cfg = TrainConfig.from_task(task_id)
    restored = cloudpickle.loads(cloudpickle.dumps(train_cfg))
    assert restored.agent.wandb_tags == train_cfg.agent.wandb_tags
    assert restored.env.scene.num_envs == train_cfg.env.scene.num_envs
