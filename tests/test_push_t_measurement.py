from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("torch")
pytest.importorskip("cv2")

from vbrl.deployment.push_t.silhouette import (  # noqa: E402
  measure_goal,
  measure_t,
  overlap,
  render_footprint,
  render_t,
)
from vbrl.deployment.push_t.view import ExternalView  # noqa: E402
from vbrl.tasks.push_t.goal_marker import GOAL_OUTLINE_THICKNESS  # noqa: E402

# Rig colours at deployment exposure.
TABLE_RGB = (187, 133, 103)
T_RGB = (125, 52, 38)
MARKER_RGB = (47, 110, 40)
GOAL = (0.38442, 0.01567, 0.0124)


@pytest.fixture(scope="module")
def view() -> ExternalView:
  return ExternalView.load(1280, 720)


def _frame(
  view: ExternalView, t: tuple[float, float, float] | None, slack: float = 0.0
) -> np.ndarray:
  image = np.empty((view.height, view.width, 3), np.uint8)
  image[:] = TABLE_RGB
  outer = render_footprint(view, GOAL, slack + GOAL_OUTLINE_THICKNESS)
  image[(outer > 0) & (render_footprint(view, GOAL, slack) == 0)] = MARKER_RGB
  if t is not None:
    image[render_t(view, t) > 0] = T_RGB
  return image


def _assert_close(fitted, truth) -> None:
  assert fitted.reliable
  assert math.dist(fitted.pose[:2], truth[:2]) < 5e-4
  assert abs(math.remainder(fitted.pose[2] - truth[2], 2 * math.pi)) < 0.01


@pytest.mark.parametrize(
  "t", [(0.30, 0.12, -1.14), (0.25, -0.15, 2.9), (0.386, 0.018, 0.05), GOAL]
)
def test_the_t_fit_recovers_the_pose_and_its_overlap(view, t) -> None:
  fitted = measure_t(view, _frame(view, t))
  _assert_close(fitted, t)
  assert overlap(fitted.pose, GOAL) == pytest.approx(overlap(t, GOAL), abs=0.03)


@pytest.mark.parametrize("slack", [-0.002, 0.0, 0.002])
def test_the_goal_fit_centres_the_t_in_a_hole_taped_too_wide_or_narrow(
  view, slack
) -> None:
  _assert_close(measure_goal(view, _frame(view, None, slack)), GOAL)


def test_an_empty_table_is_unreliable(view) -> None:
  image = np.empty((view.height, view.width, 3), np.uint8)
  image[:] = TABLE_RGB
  assert not measure_t(view, image).reliable
  assert not measure_goal(view, image).reliable


def test_the_crop_view_matches_the_policy_crop() -> None:
  from vbrl.deployment.push_t.view import ExternalView

  full = ExternalView.load(424, 240)
  crop = full.cropped(224)
  point = [[0.35, 0.0, -0.005]]
  np.testing.assert_allclose(crop.project(point), full.project(point) - (100, 8))
