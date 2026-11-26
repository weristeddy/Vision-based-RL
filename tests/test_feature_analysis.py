from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from vbrl.analysis.distance import cosine_distance, ratios  # noqa: E402
from vbrl.analysis.gradcam import jaccard  # noqa: E402
from vbrl.analysis.pairs import EXPOSURE_MS, frame_joints, sim_joints  # noqa: E402
from vbrl.analysis.probes import errors, fit, targets  # noqa: E402


def test_joints_are_read_at_the_moment_the_frame_was_exposed() -> None:
  time = np.arange(5) * 0.05
  trace = {
    "time": time,
    "compute_ms": np.zeros(5),
    "camera_age_ms": np.full(5, 50.0 - EXPOSURE_MS / 2.0),
    "joint_pos": np.repeat(time[:, None], 7, axis=1),
  }
  joints = frame_joints(trace)
  np.testing.assert_allclose(joints[1:, 0], time[:-1])
  assert sim_joints(joints).shape == (5, 8)


def test_probe_targets_encode_angles_and_errors_wrap_them() -> None:
  states = np.array([[0.3, 0.1, math.pi - 0.01], [0.4, -0.1, 0.2]])
  values = targets("push_t", states)
  assert values.shape == (2, 4)
  shifted = values.copy()
  shifted[:, 2:] = targets("push_t", states + [0.0, 0.0, 0.02])[:, 2:]
  result = errors("push_t", shifted, states)
  assert result["position_mm"] == pytest.approx(0.0)
  assert result["angle_deg"] == pytest.approx(math.degrees(0.02), abs=1e-6)
  assert set(errors("lift_cube", values[:, :2], states)) == {"position_mm"}


def test_a_linear_probe_recovers_a_linear_state() -> None:
  rng = np.random.default_rng(0)
  features = rng.normal(size=(400, 8))
  states = np.c_[features[:, :2] @ [[0.01, 0], [0, 0.01]], np.zeros(400)]
  probe = fit(features, targets("lift_cube", states))
  assert errors("lift_cube", probe.predict(features), states)["position_mm"] < 1.0


def test_jaccard_keeps_as_many_top_pixels_as_the_mask_has() -> None:
  maps = torch.zeros(2, 4, 4)
  masks = torch.zeros(2, 4, 4, dtype=torch.bool)
  masks[:, 0, :2] = True
  maps[0, 0, :2] = 1.0
  maps[1, 3, :2] = 1.0
  np.testing.assert_allclose(jaccard(maps, masks).numpy(), [1.0, 0.0])


def test_distance_ratio_is_relative_to_a_state_change() -> None:
  reference = np.array([[1.0, 0.0], [0.0, 1.0]])
  same = reference.copy()
  other = reference[::-1].copy()
  assert cosine_distance(reference, same) == pytest.approx([0.0, 0.0])
  result = ratios(reference, {"same": same, "half": (reference + other) / 2}, other)
  assert result["same"] == pytest.approx(0.0)
  assert 0.0 < result["half"] < 1.0


def test_deleting_the_used_pixels_first_changes_the_action_sooner() -> None:
  from tensordict import TensorDict

  from vbrl.analysis.deletion import deletion_auc

  class Model(torch.nn.Module):
    def forward(self, obs):
      return obs["camera"][:, :, :8, :8].float().mean((1, 2, 3))[:, None]

  images = torch.full((1, 3, 32, 32), 255, dtype=torch.uint8)
  images[:, :, 16:] = 0
  obs = TensorDict({"camera": images}, batch_size=[1])
  used = torch.zeros(1, 32, 32)
  used[:, :8, :8] = 1.0
  unused = torch.zeros(1, 32, 32)
  unused[:, 24:, 24:] = 1.0
  assert float(deletion_auc(Model(), obs, used)) > float(
    deletion_auc(Model(), obs, unused)
  )
