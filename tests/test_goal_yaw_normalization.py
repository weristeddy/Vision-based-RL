from types import SimpleNamespace

import numpy as np
import onnx
import pytest
import torch
from onnx.reference import ReferenceEvaluator
from rsl_rl.modules import EmpiricalNormalization

from vbrl.training.runner import GoalYawNormalization, VbrlOnPolicyRunner


def test_goal_yaw_stays_unscaled_after_pinned_training_and_export(tmp_path):
  normalizer = GoalYawNormalization(6, 3)
  reference = EmpiricalNormalization(6)
  pinned = torch.randn(256, 6)
  pinned[:, 3:5] = torch.tensor([0.0, 1.0])
  normalizer.update(pinned)
  reference.update(pinned)
  inputs = torch.randn(8, 6)
  inputs[:, 3:5] = torch.tensor([0.38268343, 0.92387953])
  normalizer.eval()
  actual = normalizer(inputs)
  torch.testing.assert_close(actual[:, 3:5], inputs[:, 3:5])
  torch.testing.assert_close(
    actual[:, [0, 1, 2, 5]], reference(inputs)[:, [0, 1, 2, 5]]
  )
  torch.testing.assert_close(torch.jit.script(normalizer)(inputs), actual)
  restored = GoalYawNormalization(6, 3)
  restored.load_state_dict(normalizer.state_dict())
  torch.testing.assert_close(restored(inputs), actual)
  path = tmp_path / "normalizer.onnx"
  torch.onnx.export(normalizer, inputs, path, opset_version=18, dynamo=False)
  graph = onnx.load(path)
  onnx.checker.check_model(graph)
  exported = ReferenceEvaluator(graph).run(
    None, {graph.graph.input[0].name: inputs.numpy()}
  )[0]
  np.testing.assert_allclose(exported, actual.numpy(), atol=1e-6)


@pytest.mark.parametrize("pixel", [False, True])
def test_runner_selects_goal_yaw_in_actor_and_critic(monkeypatch, pixel):
  from mjlab.rl.runner import MjlabOnPolicyRunner

  from vbrl.tasks.push_t.config.trossen_realistic.env_cfgs import (
    trossen_realistic_push_t_rgb_env_cfg,
  )

  cfg = trossen_realistic_push_t_rgb_env_cfg(
    action_scale=0.03, goal_in_observation=not pixel
  )
  sizes = {
    "qpos": 8,
    "qvel": 8,
    "target_qpos": 6,
    "tcp_pose": 7,
    "target_pose": 5,
    "obj_pose": 7,
    "relative_yaw": 2,
    "object_to_goal": 3,
    "ee_to_object": 3,
  }
  names = {group: list(cfg.observations[group].terms) for group in ("actor", "critic")}
  dims = {group: [(sizes[name],) for name in terms] for group, terms in names.items()}
  models = {
    group: SimpleNamespace(
      obs_dim=sum(sizes[name] for name in terms),
      obs_normalization=True,
      obs_normalizer=EmpiricalNormalization(sum(sizes[name] for name in terms)),
    )
    for group, terms in names.items()
  }
  env = SimpleNamespace(
    unwrapped=SimpleNamespace(
      observation_manager=SimpleNamespace(active_terms=names, group_obs_term_dim=dims)
    )
  )

  def init(runner, *args):
    runner.alg = SimpleNamespace(
      _raw_actor=models["actor"], _raw_critic=models["critic"]
    )

  monkeypatch.setattr(MjlabOnPolicyRunner, "__init__", init)
  VbrlOnPolicyRunner(env, {"obs_groups": {"actor": ("actor", "camera")}})
  assert isinstance(models["actor"].obs_normalizer, GoalYawNormalization) is not pixel
  assert isinstance(models["critic"].obs_normalizer, GoalYawNormalization)
  assert models["critic"].obs_normalizer.yaw_start == 32
  if not pixel:
    assert models["actor"].obs_normalizer.yaw_start == 32
