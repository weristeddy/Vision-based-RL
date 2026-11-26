from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("mujoco")
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from vbrl.rig.measure import (  # noqa: E402
  links,
  measure_goal,
  measure_object,
  overlap,
  render_goal,
  render_object,
)
from vbrl.rig.view import ExternalView  # noqa: E402
from vbrl.tasks.push_chain.geometry import landmark_errors  # noqa: E402
from vbrl.tasks.push_t.goal_marker import GOAL_OUTLINE_THICKNESS  # noqa: E402

# Rig colours at deployment exposure.
TABLE_RGB = (187, 133, 103)
OBJECT_RGB = (125, 52, 38)
MARKER_RGB = (47, 110, 40)
T_GOAL = (0.38442, 0.01567, 0.0124)
CHAIN_GOAL = (0.36, 0.02, 0.4, math.radians(60.0), math.radians(-30.0))


@pytest.fixture(scope="module")
def view() -> ExternalView:
  return ExternalView.load(1280, 720)


def _frame(view, task, pose=None, goal=None, slack=0.0, num_links=3) -> np.ndarray:
  image = np.empty((view.height, view.width, 3), np.uint8)
  image[:] = TABLE_RGB
  if goal is not None:
    outer = render_goal(view, task, (*goal, slack + GOAL_OUTLINE_THICKNESS), num_links)
    inner = render_goal(view, task, (*goal, slack), num_links)
    image[(outer > 0) & (inner == 0)] = MARKER_RGB
  if pose is not None:
    image[render_object(view, task, pose, num_links) > 0] = OBJECT_RGB
  return image


def _close(fit, truth, period=2 * math.pi) -> None:
  assert fit.reliable
  assert math.dist(fit.pose[:2], truth[:2]) < 5e-4
  assert abs(math.remainder(fit.pose[2] - truth[2], period)) < 0.01


def _worst_landmark(fitted, truth) -> float:
  pair = [torch.tensor(links(p), dtype=torch.float)[None] for p in (fitted, truth)]
  return float(landmark_errors(*pair).amin(-1)[0])


@pytest.mark.parametrize("t", [(0.30, 0.12, -1.14), (0.25, -0.15, 2.9), T_GOAL])
def test_the_t_is_recovered_beside_and_on_its_goal(view, t) -> None:
  fit = measure_object(view, _frame(view, "push_t", t, T_GOAL), "push_t")
  _close(fit, t)
  measured = overlap("push_t", fit.pose, T_GOAL)
  assert measured == pytest.approx(overlap("push_t", t, T_GOAL), abs=0.03)


@pytest.mark.parametrize("slack", [-0.002, 0.002])
def test_a_goal_taped_too_wide_or_narrow_is_centred(view, slack) -> None:
  fit = measure_goal(view, _frame(view, "push_t", None, T_GOAL, slack), "push_t")
  _close(fit, T_GOAL)
  assert fit.pose[3] == pytest.approx(slack, abs=1e-3)


@pytest.mark.parametrize("cube", [(0.32, 0.05, 0.3), (0.40, -0.12, -0.6)])
def test_the_cube_is_recovered_up_to_its_symmetry(view, cube) -> None:
  _close(
    measure_object(view, _frame(view, "lift_cube", cube), "lift_cube"),
    cube,
    math.pi / 2,
  )


@pytest.mark.parametrize(
  "chain", [(0.30, 0.12, -1.14, 0.9, 0.2), (0.33, -0.04, 0.7, 0.6, -0.9, 0.4)]
)
def test_chains_of_three_and_four_links_are_recovered(view, chain) -> None:
  num_links = len(chain) - 2
  image = _frame(view, "push_chain", chain, num_links=num_links)
  fit = measure_object(view, image, "push_chain", num_links)
  assert fit.reliable
  assert _worst_landmark(fit.pose, chain) < 2e-3


def test_the_chain_goal_is_recovered(view) -> None:
  fit = measure_goal(view, _frame(view, "push_chain", None, CHAIN_GOAL), "push_chain")
  assert fit.reliable
  assert _worst_landmark(fit.pose[:-1], CHAIN_GOAL) < 1e-3


def test_the_chain_overlap_ignores_which_end_is_which() -> None:
  x, y, yaw, first, second = CHAIN_GOAL
  reversed_goal = (x, y, yaw + math.pi, -second, -first)
  assert overlap("push_chain", reversed_goal, CHAIN_GOAL) == pytest.approx(1.0)


def test_an_empty_table_is_unreliable(view) -> None:
  image = _frame(view, "push_t")
  assert not measure_object(view, image, "push_t").reliable
  assert not measure_goal(view, image, "push_t").reliable


def test_the_crop_view_matches_the_policy_crop() -> None:
  full = ExternalView.load(424, 240)
  point = [[0.35, 0.0, -0.005]]
  np.testing.assert_allclose(
    full.cropped(224).project(point), full.project(point) - (100, 8)
  )
