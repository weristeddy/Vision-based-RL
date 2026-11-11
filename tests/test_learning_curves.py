from __future__ import annotations

import numpy as np
import pytest

from vbrl.analysis.learning_curves import aggregate, parse_curve


def test_curves_parse_from_name_and_run_ids() -> None:
  curve = parse_curve("State teacher=a, b,c")
  assert (curve.name, curve.runs) == ("State teacher", ("a", "b", "c"))
  with pytest.raises(ValueError, match="Name=run"):
    parse_curve("no-runs")


def test_seeds_aggregate_on_the_iterations_every_seed_evaluated() -> None:
  histories = [
    {0: {"success": 0.0}, 100: {"success": 0.2}, 200: {"success": 0.9}},
    {0: {"success": 0.0}, 100: {"success": 0.6}},
  ]
  iterations, mean, low, high = aggregate(histories, "success")
  np.testing.assert_array_equal(iterations, [0, 100])
  np.testing.assert_allclose(mean, [0.0, 0.4])
  np.testing.assert_allclose(low, [0.0, 0.2])
  np.testing.assert_allclose(high, [0.0, 0.6])
