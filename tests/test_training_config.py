from __future__ import annotations

import pytest

pytest.importorskip("mjlab")


def _task_ids() -> set[str]:
  from vbrl.tasks import vbrl_task_ids

  return set(vbrl_task_ids())


def test_every_task_logs_to_wandb_under_its_own_id_tag() -> None:
  from mjlab.tasks.registry import load_rl_cfg

  from vbrl.tasks.utils import wandb_task_tag

  for task_id in _task_ids():
    agent = load_rl_cfg(task_id)
    tag = wandb_task_tag(task_id)
    assert agent.wandb_tags[0] == tag
    assert agent.wandb_tags.count(tag) == 1
    assert len(tag) <= 64
    assert agent.logger == "wandb"
    assert agent.wandb_project == "mjlab"
    assert agent.upload_model is True


def test_every_run_is_named_after_its_task() -> None:
  from mjlab.tasks.registry import load_rl_cfg

  from vbrl.tasks.utils import wandb_task_tag

  for task_id in _task_ids():
    assert load_rl_cfg(task_id).run_name == wandb_task_tag(task_id), task_id


def test_wandb_tags_never_exceed_the_limit_wandb_enforces() -> None:
  from mjlab.tasks.registry import load_rl_cfg

  from vbrl.tasks.utils.tags import WANDB_TAG_MAX_LENGTH, wandb_task_tag

  for task_id in _task_ids():
    for tag in load_rl_cfg(task_id).wandb_tags:
      assert 1 <= len(tag) <= WANDB_TAG_MAX_LENGTH, (task_id, tag)

  with pytest.raises(ValueError, match="the limit is 64"):
    wandb_task_tag("Mjlab-" + "X" * 65)


def test_push_t_students_distill_with_the_teachers_network() -> None:
  from mjlab.tasks.registry import load_rl_cfg

  from vbrl.training.distillation import VisualDistillationCfg

  state = load_rl_cfg("Mjlab-PushT-State-Trossen")
  student = load_rl_cfg("Mjlab-PushT-RealTable-DinoV2ViTS14-AfaFull6-Trossen")

  assert state.algorithm.learning_rate == 0.0001
  assert state.max_iterations == 8000
  assert isinstance(student.algorithm, VisualDistillationCfg)
  assert student.max_iterations == 8000
  assert student.obs_groups == {"student": ("actor", "camera"), "teacher": ("teacher",)}
  assert student.teacher.hidden_dims == state.actor.hidden_dims == (256, 256, 128)
  assert student.teacher.activation == state.actor.activation == "relu"
  assert student.teacher.distribution_cfg == state.actor.distribution_cfg
  assert student.student.distribution_cfg["init_std"] == 0.1
  assert student.clip_actions == state.clip_actions == 1.0


def test_push_t_visual_ppo_trains_exactly_like_the_teacher() -> None:
  from dataclasses import asdict

  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  from vbrl.tasks.push_t.config.trossen_realistic.rl_cfg import STATE_TASK_ID

  task_id = "Mjlab-PushT-RealTablePPO-DinoV2ViTS14-AfaFull6-Trossen"
  state, ppo = load_rl_cfg(STATE_TASK_ID), load_rl_cfg(task_id)
  batching = ("cache_frozen_features", "gradient_accumulation_steps")
  a, b = asdict(state.algorithm), asdict(ppo.algorithm)
  assert {k: v for k, v in a.items() if k not in batching} == {
    k: v for k, v in b.items() if k not in batching
  }
  assert ppo.algorithm.gradient_accumulation_steps == 8
  assert ppo.max_iterations == state.max_iterations
  assert ppo.obs_groups == {"actor": ("actor", "camera"), "critic": ("critic",)}
  assert ppo.actor.distribution_cfg == state.actor.distribution_cfg
  assert ppo.critic == state.critic

  state_env, ppo_env = load_env_cfg(STATE_TASK_ID), load_env_cfg(task_id)
  assert ppo_env.rewards == state_env.rewards
  assert ppo_env.terminations == state_env.terminations
  assert ppo_env.actions == state_env.actions
  assert ppo_env.commands["push_t_goal"].near_object_start is True
  assert "teacher" not in ppo_env.observations


def test_lift_cube_teacher_bounds_its_actions_like_push_t() -> None:
  from mjlab.tasks.registry import load_rl_cfg

  lift = load_rl_cfg("Mjlab-LiftCube-State-Trossen")
  push_t = load_rl_cfg("Mjlab-PushT-State-Trossen")

  assert lift.actor.distribution_cfg == push_t.actor.distribution_cfg
  assert lift.algorithm.bound_loss_coef == push_t.algorithm.bound_loss_coef == 1.0
  assert lift.algorithm.action_head_init_scale == 0.01
  assert lift.clip_actions == 1.0
