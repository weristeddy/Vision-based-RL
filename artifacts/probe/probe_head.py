"""The readout head every probe uses.

Shaped exactly like the policy actor's trunk -- hidden_dims (256, 256, 128) with
ReLU -- so a probe result cannot be dismissed as "the readout was weaker than the
policy". It is not identical to the policy: the actor concatenates ~28
proprioception dims onto the visual 256 and emits action means plus distribution
parameters, whereas a probe reads the visual 256 alone and emits (sin, cos).

Capacity is not a free choice. On the RL-trained encoders, widening the head from
one 256-wide hidden layer to this trunk moved NatureCnn-SpatialSoftmax from 46.8
to 34.0 degrees, so two probes using different heads are not comparable.
"""
from __future__ import annotations

import torch.nn as nn

HEAD_DIMS = (256, 256, 128)


def policy_head(in_dim: int, out_dim: int = 2) -> nn.Sequential:
  layers: list[nn.Module] = []
  size = in_dim
  for width in HEAD_DIMS:
    layers += [nn.Linear(size, width), nn.ReLU()]
    size = width
  layers.append(nn.Linear(size, out_dim))
  return nn.Sequential(*layers)


__all__ = ["HEAD_DIMS", "policy_head"]
