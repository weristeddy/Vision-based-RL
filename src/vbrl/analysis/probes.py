from __future__ import annotations

from typing import Any

import numpy as np

ALPHA = 1.0
ANGLES = {"lift_cube": 0, "push_t": 1, "push_chain": 3}


def targets(task: str, states: Any) -> Any:
  angles = states[:, 2 : 2 + ANGLES[task]]
  return np.concatenate([states[:, :2], np.sin(angles), np.cos(angles)], axis=1)


def fit(features: Any, values: Any) -> Any:
  from sklearn.linear_model import Ridge
  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import StandardScaler

  return make_pipeline(StandardScaler(), Ridge(alpha=ALPHA)).fit(features, values)


def errors(task: str, predicted: Any, states: Any) -> dict[str, float]:
  result = {
    "position_mm": float(
      np.linalg.norm(predicted[:, :2] - states[:, :2], axis=1).mean() * 1000
    )
  }
  count = ANGLES[task]
  if count:
    sin, cos = predicted[:, 2 : 2 + count], predicted[:, 2 + count : 2 + 2 * count]
    difference = np.arctan2(sin, cos) - states[:, 2 : 2 + count]
    result["angle_deg"] = float(
      np.degrees(np.abs(np.angle(np.exp(1j * difference)))).mean()
    )
  return result


__all__ = ["ALPHA", "errors", "fit", "targets"]
