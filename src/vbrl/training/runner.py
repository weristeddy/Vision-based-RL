from __future__ import annotations

from typing import Any

from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.modules import EmpiricalNormalization


class GoalYawNormalization(EmpiricalNormalization):
  def __init__(self, shape: int, yaw_start: int) -> None:
    super().__init__(shape)
    self.yaw_start = yaw_start

  def forward(self, x):
    normalized = (x - self._mean) / (self._std + self.eps)
    start = self.yaw_start
    normalized[:, start : start + 2] = x[:, start : start + 2]
    return normalized


def policy_metadata(
  env: Any, run_name: str, clip_actions: float | None = None
) -> dict[str, Any]:
  import mjlab.rl.exporter_utils as exporter_utils
  from mjlab.envs.mdp.actions import (
    JointPositionAction,
    RelativeJointPositionActionCfg,
  )
  from mjlab.envs.mdp.actions.actions import BaseAction

  from vbrl.tasks.push_t.mdp.actions import TargetRelativeJointPositionActionCfg

  base = env.unwrapped if hasattr(env, "unwrapped") else env
  exporter_utils.JointPositionAction = BaseAction
  try:
    metadata = exporter_utils.get_base_metadata(base, run_name)
  finally:
    exporter_utils.JointPositionAction = JointPositionAction

  metadata["control_hz"] = 1.0 / base.step_dt
  action_cfg = next(iter(base.cfg.actions.values()))
  metadata["action_type"] = {
    RelativeJointPositionActionCfg: "relative",
    TargetRelativeJointPositionActionCfg: "target",
  }.get(type(action_cfg), "absolute")
  term = base.action_manager.get_term("joint_pos")
  if isinstance(action_cfg, TargetRelativeJointPositionActionCfg):
    if action_cfg.reference == "measured":
      metadata["action_type"] = "relative"
    metadata["ramp_s"] = float(action_cfg.ramp_s)
    metadata["target_low"] = term.low[0].cpu().tolist()
    metadata["target_high"] = term.high[0].cpu().tolist()
  # The clip is as much of the contract as the scale: without it Push-T sent
  # 0.31 rad against the 0.1 the simulator would ever apply.
  clip = getattr(term, "_clip", None)
  if clip is not None:
    metadata["action_clip_low"] = clip[0, :, 0].cpu().tolist()
    metadata["action_clip_high"] = clip[0, :, 1].cpu().tolist()
  # RSL-RL's wrapper clamps to +/-clip_actions before the environment sees it,
  # so the policy never trained on an unclipped `actions` observation.
  if clip_actions is not None:
    metadata["clip_actions"] = float(clip_actions)
  return metadata


class VbrlOnPolicyRunner(MjlabOnPolicyRunner):
  def __init__(self, env, train_cfg, log_dir=None, device="cpu") -> None:
    super().__init__(env, train_cfg, log_dir, device)
    if "camera" not in train_cfg["obs_groups"]["actor"]:
      return
    observations = env.unwrapped.observation_manager
    for group, model in (
      ("actor", self.alg._raw_actor),
      ("critic", self.alg._raw_critic),
    ):
      names = observations.active_terms[group]
      if "target_pose" in names and model.obs_normalization:
        dims = observations.group_obs_term_dim[group]
        yaw_start = sum(dim[0] for dim in dims[: names.index("target_pose")]) + 3
        model.obs_normalizer = GoalYawNormalization(model.obs_dim, yaw_start).to(device)

  def load(self, path: str, *args, **kwargs) -> dict:
    infos = super().load(path, *args, **kwargs)
    if hasattr(self.alg, "iteration"):
      self.alg.iteration = self.current_learning_iteration
    return infos

  def save(self, path: str, infos: Any = None) -> None:
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      import wandb
      from mjlab.rl.exporter_utils import attach_metadata_to_onnx

      self.export_policy_to_onnx(str(policy_dir), filename)
      to_wandb = self.logger.logger_type in ("wandb", "WandbLogWriter")
      run_name = wandb.run.name if to_wandb and wandb.run else "local"
      metadata = policy_metadata(
        self.env, str(run_name), self.cfg.get("clip_actions")
      )
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if to_wandb and self.cfg["upload_model"]:
        wandb.save(str(onnx_path), base_path=str(policy_dir))
    except Exception as error:
      print(f"[WARN] ONNX export failed (training continues): {error}")


__all__ = ["VbrlOnPolicyRunner", "policy_metadata"]
