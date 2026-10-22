from __future__ import annotations

from dataclasses import replace

import pytest

pytest.importorskip("numpy")


class _Session:
  def __init__(self, extra: dict[str, str]) -> None:
    from vbrl.deployment.policy import ARM_JOINTS

    self._meta = type(
      "Meta",
      (),
      {
        "custom_metadata_map": {
          "joint_names": ",".join(ARM_JOINTS),
          "default_joint_pos": ",".join("0" for _ in ARM_JOINTS),
          "observation_names": "qpos",
          "action_scale": ",".join("0.05" for _ in ARM_JOINTS),
          **extra,
        }
      },
    )()

  def get_modelmeta(self):
    return self._meta

  def get_inputs(self):
    return []

  def get_outputs(self):
    return [type("Output", (), {"shape": [1, 6]})()]


def _config(tmp_path, **fields):
  from vbrl.deployment.config import DeploymentConfig

  graph = tmp_path / "policy.onnx"
  graph.touch()
  defaults = {"control_hz": 20.0, "command_goal_time_s": 0.05}
  return replace(
    DeploymentConfig(onnx_file=str(graph), arm_ip="192.168.1.2"),
    **{**defaults, **fields},
  )


def _metadata(extra: dict[str, str]):
  from vbrl.deployment.policy import PolicyMetadata

  return PolicyMetadata.from_onnx(_Session(extra))


def test_matching_control_settings_pass(tmp_path) -> None:
  from vbrl.deployment.policy import check_control

  check_control(_metadata({"control_hz": "20.0", "ramp_s": "0.05"}), _config(tmp_path))


@pytest.mark.parametrize(
  ("extra", "name"),
  [
    ({"control_hz": "50.0", "ramp_s": "0.05"}, "control_hz"),
    ({"control_hz": "20.0", "ramp_s": "0.0"}, "ramp_s"),
  ],
)
def test_a_manifest_that_disagrees_with_training_is_refused(
  tmp_path, extra: dict[str, str], name: str
) -> None:
  from vbrl.deployment.policy import check_control

  with pytest.raises(ValueError, match=name):
    check_control(_metadata(extra), _config(tmp_path))


def test_an_export_without_control_metadata_is_not_checked(tmp_path) -> None:
  from vbrl.deployment.policy import check_control

  check_control(_metadata({}), _config(tmp_path, control_hz=50.0))
