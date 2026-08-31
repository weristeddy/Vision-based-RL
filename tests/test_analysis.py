"""Manifest validation, step dispatch, and the numeric analysis primitives."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import vbrl.analysis.features as feature_module
import vbrl.scripts.analyze as analysis
from vbrl.analysis.capture import (
  CaptureBatch,
  capture_rollout,
  load_capture,
  save_capture,
)
from vbrl.analysis.comparison import compare_features
from vbrl.analysis.features import extract_features, load_features, save_features
from vbrl.analysis.occlusion import (
  load_occlusion,
  occlusion_sensitivity,
  save_occlusion,
)
from vbrl.analysis.probe import load_probe, ridge_probe, save_probe
from vbrl.runtime import CheckpointRef
from vbrl.scripts.analyze import Context


TASK_ID = "Mjlab-LiftCube-CollisionCam-DinoV2ViTS14-LocalGrid7-Trossen"


class _DummyEnv:
  def __init__(self) -> None:
    self.step_index = 0

  def _observation(self):
    return {"camera": torch.full((2, 3, 4, 5), self.step_index / 10.0)}

  def reset(self, seed=None):
    self.step_index = 0
    return self._observation(), {"seed": seed}

  def step(self, action):
    assert action.shape == (2, 1)
    self.step_index += 1
    return self._observation(), torch.zeros(2), torch.zeros(2), torch.zeros(2), {}


class _DummyEncoder(torch.nn.Module):
  def encode_features(self, images):
    return images.mean(dim=(-2, -1), keepdim=True)

  def project_features(self, features):
    return features.flatten(1)


def _context(tmp_path: Path, *, agent: str = "trained") -> Context:
  return Context(
    path=tmp_path / "analysis.yaml",
    task_id=TASK_ID,
    agent=agent,
    ref=CheckpointRef(
      checkpoint_file="ckpts/model.pt" if agent == "trained" else None
    ),
    output_dir=tmp_path / "output",
    device="cpu",
  )


def _write(path: Path, **changes) -> Path:
  document = {
    "version": 1,
    "task_id": TASK_ID,
    "checkpoint_file": "ckpts/model.pt",
    "output": "artifacts/analysis/test",
    "steps": [{"script": "capture", "args": {"output": "capture.npz"}}],
  }
  document.update(changes)
  path.write_text(yaml.safe_dump(document), encoding="utf-8")
  return path


# --- manifest and dispatch ---------------------------------------------------


def test_load_reads_only_task_checkpoint_output_and_steps(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  output = tmp_path / "output"
  monkeypatch.setattr(analysis, "artifact_path", lambda _path: output)
  context, steps = analysis.load(_write(tmp_path / "analysis.yaml"), "cpu")

  assert context.task_id == TASK_ID
  assert context.agent == "trained"
  assert context.ref == CheckpointRef(checkpoint_file="ckpts/model.pt")
  assert context.output_dir == output
  assert steps == [{"script": "capture", "args": {"output": "capture.npz"}}]


@pytest.mark.parametrize(
  ("changes", "message"),
  [
    ({"checkpoint_file": None}, "exactly one of checkpoint_file"),
    (
      {"checkpoint_file": None, "wandb_run_path": "run"},
      "must be 'entity/project/run_id'",
    ),
    (
      {
        "checkpoint_file": None,
        "wandb_run_path": "entity/project/run",
        "wandb_checkpoint_name": "latest.pt",
      },
      "wandb_checkpoint_name must be model_N.pt",
    ),
    (
      {"wandb_run_path": "entity/project/run"},
      "exactly one of checkpoint_file or wandb_run_path",
    ),
    (
      {"agent": "zero", "checkpoint_file": "ckpts/model.pt"},
      "Checkpoint fields require agent: trained",
    ),
    ({"agent": "invalid"}, "agent must be trained, zero, or random"),
    ({"encoder": "dinov2"}, "Unknown analysis fields:.*encoder"),
    ({"steps": []}, "steps must be a non-empty list"),
  ],
)
def test_load_rejects_invalid_manifests(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  changes: dict[str, object],
  message: str,
) -> None:
  monkeypatch.setattr(analysis, "artifact_path", Path)
  with pytest.raises(ValueError, match=message):
    analysis.load(_write(tmp_path / "analysis.yaml", **changes), "cpu")


def test_zero_and_random_analysis_require_no_checkpoint(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr(analysis, "artifact_path", Path)
  for agent in ("zero", "random"):
    path = _write(tmp_path / f"{agent}.yaml", agent=agent, checkpoint_file=None)
    context, _ = analysis.load(path, "cpu")
    assert context.agent == agent
    assert context.ref.is_empty


def test_context_resolves_inputs_and_keeps_outputs_below_root(
  tmp_path: Path,
) -> None:
  context = _context(tmp_path)

  assert context.input("capture.npz") == (
    context.output_dir / "capture.npz"
  ).resolve()
  assert context.output("features.npz") == (
    context.output_dir / "features.npz"
  ).resolve()
  with pytest.raises(ValueError, match="Output must stay below"):
    context.output("../escape.npz")


def test_required_num_envs_uses_capture_indices_and_validates_bounds() -> None:
  assert (
    analysis._required_num_envs(
      [
        {"script": "capture", "args": {"num_envs": 8, "env_index": 7}},
        {"script": "features", "args": {}},
      ]
    )
    == 8
  )
  with pytest.raises(ValueError, match="env_index must be inside num_envs"):
    analysis._required_num_envs(
      [{"script": "capture", "args": {"num_envs": 2, "env_index": 2}}]
    )


def test_steps_table_covers_every_analysis_module() -> None:
  from vbrl.analysis import (
    attribution,
    capture,
    comparison,
    features,
    occlusion,
    pca,
    probe,
    report,
  )

  assert analysis.STEPS == {
    "capture": capture.run,
    "features": features.run,
    "probe": probe.run,
    "pca": pca.run,
    "occlusion": occlusion.run,
    "comparison": comparison.run,
    "attribution": attribution.run,
    "report": report.run,
  }
  assert analysis._RUNTIME_STEPS <= set(analysis.STEPS)


def test_execute_builds_one_runtime_runs_steps_in_order_and_closes(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  context = _context(tmp_path)
  context.output_dir.mkdir()
  calls: list[tuple[str, dict[str, object]]] = []

  class RawEnv:
    def close(self):
      calls.append(("close", {}))

  def prepare(selected, steps):
    assert selected is context
    assert len(steps) == 2
    selected.raw_env = RawEnv()

  def step(name, result):
    def run(selected, **kwargs):
      assert selected is context
      calls.append((name, kwargs))
      return result

    return run

  monkeypatch.setattr(analysis, "_prepare_runtime", prepare)
  monkeypatch.setitem(analysis.STEPS, "capture", step("capture", "capture.npz"))
  monkeypatch.setitem(
    analysis.STEPS, "features", step("features", "features.npz")
  )

  generated = analysis.execute(
    context,
    [
      {"script": "capture", "args": {"frames": 2}},
      {"script": "features", "args": {"batch_size": 8}},
    ],
  )

  assert generated == [
    (context.output_dir / "capture.npz").resolve(),
    (context.output_dir / "features.npz").resolve(),
  ]
  assert calls == [
    ("capture", {"frames": 2}),
    ("features", {"batch_size": 8}),
    ("close", {}),
  ]


def test_execute_rejects_an_unknown_step(tmp_path: Path) -> None:
  context = _context(tmp_path)
  with pytest.raises(ValueError, match="Unknown analysis step 'nope'"):
    analysis.execute(context, [{"script": "nope", "args": {}}])


def test_feature_analysis_uses_camera_encoder_from_loaded_actor(
  tmp_path: Path,
) -> None:
  from vbrl.analysis.features import camera_encoder

  encoder = object()
  context = _context(tmp_path)
  context.policy = SimpleNamespace(cnns={"camera": encoder})
  assert camera_encoder(context) is encoder

  context.agent = "zero"
  with pytest.raises(ValueError, match="requires agent: trained"):
    camera_encoder(context)

  context.agent = "trained"
  context.policy = SimpleNamespace(cnns={})
  with pytest.raises(ValueError, match="has no actor camera encoder"):
    camera_encoder(context)


# --- numeric primitives ------------------------------------------------------


def test_capture_is_reusable_and_keeps_targets_aligned(tmp_path) -> None:
  env = _DummyEnv()
  batch = capture_rollout(
    env,
    lambda _: torch.zeros((2, 1)),
    num_frames=3,
    target_getters={"step": lambda fake_env, _obs, _index: fake_env.step_index},
  )

  assert batch.images.shape == (3, 4, 5, 3)
  assert batch.images.dtype == np.uint8
  np.testing.assert_array_equal(batch.targets["step"], [0, 1, 2])
  # Stepping happens *between* frames: N frames cost N-1 environment steps.
  assert env.step_index == 2

  restored = load_capture(save_capture(batch, tmp_path / "capture.npz"))
  np.testing.assert_array_equal(restored.images, batch.images)
  np.testing.assert_array_equal(restored.targets["step"], batch.targets["step"])


def test_capture_batch_rejects_misaligned_targets() -> None:
  with pytest.raises(ValueError, match="bad"):
    CaptureBatch(np.zeros((2, 4, 5, 3), np.uint8), {"bad": np.zeros((1,))})


def test_feature_extraction_preprocesses_one_batch_at_a_time(
  monkeypatch, tmp_path
) -> None:
  batch_sizes: list[int] = []
  original = feature_module.prepare_images

  def tracked(batch):
    batch_sizes.append(len(batch))
    return original(batch)

  monkeypatch.setattr(feature_module, "prepare_images", tracked)
  result = extract_features(
    _DummyEncoder(),
    np.full((5, 4, 6, 3), 255, dtype=np.uint8),
    batch_size=2,
    device="cpu",
  )

  assert batch_sizes == [2, 2, 1]
  assert result.features["backbone"].shape == (5, 3, 1, 1)
  assert result.features["adapter"].shape == (5, 3)
  restored = load_features(save_features(result, tmp_path / "features.npz"))
  np.testing.assert_allclose(restored.features["adapter"], 1.0)


def test_occlusion_scores_every_patch_and_survives_a_round_trip(tmp_path) -> None:
  images = np.random.default_rng(0).integers(0, 256, (3, 16, 16, 3), dtype=np.uint8)
  result = occlusion_sensitivity(
    _DummyEncoder(),
    images,
    stage="adapter",
    patch_size=8,
    batch_size=2,
    device="cpu",
    metadata={"task_id": "test"},
  )

  # 16x16 images with 8px patches tile into a 2x2 grid, one score per patch.
  assert result.scores.shape == (3, 2, 2)
  assert np.isfinite(result.scores).all()
  assert result.metadata["patch_size"] == 8

  restored = load_occlusion(save_occlusion(result, tmp_path / "occlusion.npz"))
  np.testing.assert_array_equal(restored.scores, result.scores)
  assert restored.metadata == result.metadata

  with pytest.raises(ValueError, match="fill must be"):
    occlusion_sensitivity(
      _DummyEncoder(),
      np.zeros((1, 8, 8, 3), np.uint8),
      fill="median",  # type: ignore[arg-type]
      device="cpu",
    )


def test_compare_features_is_exact_for_identical_and_orthogonal_inputs() -> None:
  left = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)

  identical = compare_features(left, left)
  np.testing.assert_allclose(identical.cosine_similarity, [1.0, 1.0], atol=1e-6)
  np.testing.assert_allclose(identical.euclidean_distance, [0.0, 0.0], atol=1e-6)

  orthogonal = compare_features(left, np.array([[0.0, 1.0], [2.0, 0.0]], np.float32))
  np.testing.assert_allclose(orthogonal.cosine_similarity, [0.0, 0.0], atol=1e-6)
  np.testing.assert_allclose(
    orthogonal.euclidean_distance, [np.sqrt(2.0), np.sqrt(8.0)], atol=1e-6
  )

  with pytest.raises(ValueError, match="shapes must match"):
    compare_features(np.zeros((2, 3), np.float32), np.zeros((2, 4), np.float32))


def test_probe_fits_and_survives_a_save_load_round_trip(tmp_path) -> None:
  rng = np.random.default_rng(0)
  features = rng.normal(size=(40, 6)).astype(np.float32)
  # A learnable target, so the probe reports a meaningful fit as object-pose
  # probing does.
  targets = (features[:, :3] * 2.0 - 1.0).astype(np.float32)
  result = ridge_probe(features, targets, train_fraction=0.7, alpha=1.0, seed=0)

  assert result.train_size + result.test_size == 40
  assert result.r2 > 0.9
  assert result.predictions.shape == result.targets.shape

  restored = load_probe(
    save_probe(result, tmp_path / "probe.npz", metadata={"stage": "adapter"})
  )
  assert restored.r2 == pytest.approx(result.r2)
  assert restored.mean_absolute_error == pytest.approx(result.mean_absolute_error)
  np.testing.assert_allclose(restored.predictions, result.predictions)
  assert restored.metadata["stage"] == "adapter"
