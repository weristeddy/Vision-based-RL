from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from vbrl.deployment.push_chain.silhouette import (  # noqa: E402
  links,
  measure_chain,
  measure_goal,
  overlap,
  render_chain,
  render_goal,
)
from vbrl.deployment.push_t.view import ExternalView  # noqa: E402
from vbrl.tasks.push_chain.geometry import landmark_errors  # noqa: E402
from vbrl.tasks.push_t.goal_marker import GOAL_OUTLINE_THICKNESS  # noqa: E402

# Rig colours at deployment exposure, as in the Push-T measurement tests.
TABLE_RGB = (187, 133, 103)
CHAIN_RGB = (125, 52, 38)
MARKER_RGB = (47, 110, 40)
GOAL = (0.36, 0.02, 0.4, math.radians(60.0), math.radians(-30.0))


@pytest.fixture(scope="module")
def view() -> ExternalView:
  return ExternalView.load(1280, 720)


def _frame(view: ExternalView, chain, slack: float = 0.0) -> np.ndarray:
  image = np.empty((view.height, view.width, 3), np.uint8)
  image[:] = TABLE_RGB
  outer = render_goal(view, (*GOAL, slack + GOAL_OUTLINE_THICKNESS))
  image[(outer > 0) & (render_goal(view, (*GOAL, slack)) == 0)] = MARKER_RGB
  if chain is not None:
    image[render_chain(view, chain) > 0] = CHAIN_RGB
  return image


def _worst_landmark(fitted, truth) -> float:
  pair = [
    torch.tensor(links(pose), dtype=torch.float)[None] for pose in (fitted, truth)
  ]
  return float(landmark_errors(*pair).amin(-1)[0])


@pytest.mark.parametrize(
  "chain",
  [
    (0.30, 0.12, -1.14, 0.9, 0.2),
    (0.25, -0.15, 2.9, -1.4, -1.4),
    (0.37, 0.03, 0.45, math.radians(55.0), math.radians(-25.0)),
  ],
)
def test_the_chain_fit_recovers_the_pose_and_its_overlap(view, chain) -> None:
  fitted = measure_chain(view, _frame(view, chain))
  assert fitted.reliable
  assert _worst_landmark(fitted.pose, chain) < 2e-3
  measured, truth = overlap(fitted.pose, GOAL), overlap(chain, GOAL)
  assert measured == pytest.approx(truth, abs=0.03)


@pytest.mark.parametrize("slack", [-0.002, 0.0, 0.002])
def test_the_goal_fit_finds_the_chain_in_a_hole_taped_too_wide_or_narrow(
  view, slack
) -> None:
  fitted = measure_goal(view, _frame(view, None, slack))
  assert fitted.reliable
  assert _worst_landmark(fitted.pose, GOAL) < 1e-3
  assert fitted.pose[5] == pytest.approx(slack, abs=1e-3)


def test_the_overlap_ignores_which_end_is_which() -> None:
  x, y, yaw, first, second = GOAL
  reversed_goal = (x, y, yaw + math.pi, -second, -first)
  assert overlap(reversed_goal, GOAL)[0] == pytest.approx(1.0)


def test_an_empty_table_is_unreliable(view) -> None:
  image = np.empty((view.height, view.width, 3), np.uint8)
  image[:] = TABLE_RGB
  assert not measure_chain(view, image).reliable
  assert not measure_goal(view, image).reliable
