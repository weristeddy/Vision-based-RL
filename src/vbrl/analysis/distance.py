from __future__ import annotations

from typing import Any

import numpy as np


def cosine_distance(a: Any, b: Any) -> Any:
  a = a / np.linalg.norm(a, axis=1, keepdims=True)
  b = b / np.linalg.norm(b, axis=1, keepdims=True)
  return 1.0 - (a * b).sum(1)


def ratios(reference: Any, changed: dict[str, Any], state: Any) -> dict[str, float]:
  d_state = cosine_distance(reference, state).mean()
  return {
    name: float(cosine_distance(reference, f).mean() / d_state)
    for name, f in changed.items()
  }


__all__ = ["cosine_distance", "ratios"]
