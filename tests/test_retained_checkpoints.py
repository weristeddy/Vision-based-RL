"""Explicit GPU audit for all thesis checkpoints retained under ``ckpts/``.

This module is intentionally skipped during normal pytest runs.  Set
``VBRL_AUDIT_RETAINED_CHECKPOINTS=1`` from the repository root after copying
the 26 checkpoints into ``ckpts/`` to construct every registered policy and
strict-load its actor weights.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest


pytestmark = [
  pytest.mark.gpu,
  pytest.mark.integrity,
]

if os.environ.get("VBRL_AUDIT_RETAINED_CHECKPOINTS") != "1":
  pytest.skip(
    "set VBRL_AUDIT_RETAINED_CHECKPOINTS=1 to audit local thesis checkpoints",
    allow_module_level=True,
  )

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
  pytest.skip("retained-checkpoint audit requires CUDA", allow_module_level=True)


def _retained_models():
  from vbrl.evaluation.suite import load_config

  models = {}
  for config_path in (
    "configs/evaluation/peacock_24.yaml",
    "configs/evaluation/push_t_retained.yaml",
  ):
    for model in load_config(config_path).models:
      assert model.ref.checkpoint_file is not None
      models[model.ref.checkpoint_file] = model
  assert len(models) == 26
  return tuple(models.values())


@pytest.mark.parametrize(
  "model",
  _retained_models(),
  ids=lambda model: model.name,
)
def test_retained_checkpoint_strict_loads(model) -> None:
  from vbrl.runtime import build_env, load_trained_policy

  checkpoint = Path(model.ref.checkpoint_file).expanduser()
  if not checkpoint.is_file():
    pytest.fail(f"missing retained checkpoint: {checkpoint}")

  env = build_env(model.task_id, device="cuda:0", num_envs=1, seed=0)
  try:
    _, _, _, loaded = load_trained_policy(
      env,
      task_id=model.task_id,
      device="cuda:0",
      ref=model.ref,
    )
    assert loaded == checkpoint
  finally:
    env.close()
    gc.collect()
    torch.cuda.empty_cache()
