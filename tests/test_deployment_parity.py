from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.sim, pytest.mark.gpu]

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")
if not torch.cuda.is_available():
  pytest.skip("deployment parity needs CUDA", allow_module_level=True)

TASK = "Mjlab-LiftCube-Sim2Real-DinoV2ViTS14-SpatialSoftmax-TrossenRealistic"


class _Meta:
  def __init__(self, values: dict[str, str]) -> None:
    self.custom_metadata_map = values


class _Session:
  def __init__(self, env) -> None:
    from mjlab.tasks.registry import load_rl_cfg

    from vbrl.training.runner import policy_metadata

    metadata = policy_metadata(env, "parity-test", load_rl_cfg(TASK).clip_actions)
    self._action_dim = env.unwrapped.action_manager.total_action_dim
    self._meta = _Meta(
      {
        key: (
          ",".join(f"{v}" for v in value) if isinstance(value, list) else str(value)
        )
        for key, value in metadata.items()
      }
    )

  def get_modelmeta(self):
    return self._meta

  def get_inputs(self):
    return [type("Input", (), {"name": "camera"})()]

  def get_outputs(self):
    return [type("Output", (), {"shape": [1, self._action_dim]})()]


@pytest.fixture(scope="module")
def simulation():
  from vbrl.runtime import build_env

  env = build_env(TASK, device="cuda:0", num_envs=1, seed=0)
  yield env
  env.close()


@pytest.fixture(scope="module")
def metadata(simulation):
  from vbrl.deployment.policy import PolicyMetadata

  return PolicyMetadata.from_onnx(_Session(simulation))


def test_the_contract_read_from_metadata_matches_the_simulator(
  simulation, metadata
) -> None:
  unwrapped = simulation.unwrapped
  robot = unwrapped.scene["robot"]
  action = unwrapped.action_manager.get_term("joint_pos")

  assert metadata.joint_names == tuple(robot.joint_names)
  np.testing.assert_allclose(
    metadata.default_joint_pos, robot.data.default_joint_pos[0].cpu().numpy(), atol=1e-6
  )
  np.testing.assert_allclose(
    metadata.action_scale, action._scale[0].cpu().numpy(), atol=1e-6
  )
  np.testing.assert_allclose(
    metadata.action_offset, action._offset[0].cpu().numpy(), atol=1e-6
  )
  assert metadata.observation_terms == tuple(unwrapped.observation_manager.active_terms["actor"])


def test_deployment_observation_matches_the_simulator(simulation) -> None:
  from vbrl.deployment.policy import Policy

  unwrapped = simulation.unwrapped
  observations = unwrapped.observation_manager.compute()
  expected = observations["actor"][0].cpu().numpy()

  robot = unwrapped.scene["robot"]
  command = unwrapped.command_manager.get_term("lift_height")
  # `Policy.goal` is in the arm's BASE frame, not the env origin.
  base = robot.data.root_link_pos_w[0].cpu().numpy()
  goal = command.target_pos[0].cpu().numpy() - base

  policy = Policy(_Session(simulation), goal=tuple(goal))
  policy._last_action = unwrapped.action_manager.action[0].cpu().numpy()

  observation = policy.observe(
    joint_pos=robot.data.joint_pos[0].cpu().numpy(),
    joint_vel=robot.data.joint_vel[0].cpu().numpy(),
    image=observations["camera"][0].cpu().numpy().transpose(1, 2, 0),
  )

  assert observation["obs"].shape == (1, expected.shape[0])
  # float32 built through different arithmetic; 1e-4 catches a swapped term and
  # tolerates a different order of floating-point operations.
  np.testing.assert_allclose(observation["obs"][0], expected, atol=1e-4, rtol=0)


def test_the_camera_feed_is_float_in_zero_to_one(simulation) -> None:
  from vbrl.deployment.policy import Policy

  unwrapped = simulation.unwrapped
  observations = unwrapped.observation_manager.compute()
  rendered = observations["camera"][0].cpu().numpy()  # uint8, C H W
  robot = unwrapped.scene["robot"]

  policy = Policy(_Session(simulation), goal=(0.35, 0.0, 0.35))
  observation = policy.observe(
    joint_pos=robot.data.joint_pos[0].cpu().numpy(),
    joint_vel=robot.data.joint_vel[0].cpu().numpy(),
    image=rendered.transpose(1, 2, 0),
  )

  camera = observation["camera"]
  assert camera.dtype == np.float32
  assert camera.shape == (1, 3, 224, 224)
  assert 0.0 <= camera.min() and camera.max() <= 1.0
  np.testing.assert_allclose(camera[0], rendered.astype(np.float32) / 255.0, atol=0)


def test_a_goal_outside_the_training_range_is_refused(tmp_path) -> None:
  from vbrl.deployment.config import GOAL_RANGE, DeploymentConfig

  graph = tmp_path / "policy.onnx"
  graph.touch()
  config = DeploymentConfig(
    onnx_file=str(graph),
    arm_ip="192.168.1.2",
    goal=(0.4, 0.0, GOAL_RANGE["z"][1] + 0.1),
  )
  with pytest.raises(ValueError, match="outside the range the policy trained on"):
    config.validate()


def test_joint_targets_match_the_action_term_the_policy_trained_under() -> None:
  from dataclasses import replace

  import numpy as np

  from vbrl.deployment.policy import ARM_JOINTS, Policy, PolicyMetadata

  default = np.linspace(0.1, 0.7, len(ARM_JOINTS))
  metadata = PolicyMetadata(
    joint_names=ARM_JOINTS,
    default_joint_pos=default,
    action_offset=default.copy(),
    action_scale=np.full(len(ARM_JOINTS), 0.1),
    observation_terms=("joint_pos",),
    action_dim=6,
    action_clip=None,
    action_type="relative",
    target_limits=None,
    clip_actions=None,
    needs_camera=False,
    source_run="test",
  )
  policy = Policy.__new__(Policy)
  policy.metadata = metadata
  policy._target = default.copy()
  action = np.array([1.0, -1.0, 0.5, 0.0, 0.25, -0.5])

  policy._position = default.copy()
  assert np.allclose(policy.joint_targets(action)[:6], default[:6] + 0.1 * action)

  moved = default + 0.4
  policy._position = moved
  targets = policy.joint_targets(action)
  assert np.allclose(targets[:6], moved[:6] + 0.1 * action)
  assert not np.allclose(targets[:6], default[:6] + 0.1 * action)

  assert targets.shape == (len(ARM_JOINTS),)
  assert targets[6] == pytest.approx(default[6])
  assert policy.has_gripper is False

  # The clip is part of the contract, not a safety extra: without it Push-T
  # commanded 0.31 rad against the 0.1 the simulator ever applies.
  policy.metadata = replace(
    metadata, action_clip=(np.full(6, -0.05), np.full(6, 0.05))
  )
  policy._position = default.copy()
  clipped = policy.joint_targets(action)
  assert np.allclose(
    clipped[:6], default[:6] + np.clip(0.1 * action, -0.05, 0.05)
  )
  assert clipped[0] == pytest.approx(default[0] + 0.05)
  assert clipped[1] == pytest.approx(default[1] - 0.05)

  high = default[:6] + 0.15
  policy.metadata = replace(
    metadata, action_type="target", target_limits=(default[:6] - 1.0, high)
  )
  policy._target = default.copy()
  policy._position = moved
  first = policy.joint_targets(action)[:6]
  assert np.allclose(first, default[:6] + 0.1 * action)
  policy._position = default.copy()
  second = policy.joint_targets(action)[:6]
  assert np.allclose(second, np.minimum(default[:6] + 0.2 * action, high))
  assert second[0] == pytest.approx(high[0])


def test_the_action_fed_back_as_an_observation_stays_inside_its_training_band() -> None:
  from dataclasses import replace

  import numpy as np

  from vbrl.deployment.policy import ARM_JOINTS, Policy, PolicyMetadata

  default = np.zeros(len(ARM_JOINTS) + 1)
  metadata = PolicyMetadata(
    joint_names=tuple(ARM_JOINTS) + ("left_carriage_joint",),
    default_joint_pos=default,
    action_offset=np.zeros(len(ARM_JOINTS)),
    action_scale=np.full(len(ARM_JOINTS), 0.05),
    observation_terms=("actions",),
    action_dim=6,
    action_clip=(np.full(6, -0.05), np.full(6, 0.05)),
    action_type="relative",
    target_limits=None,
    clip_actions=1.0,
    needs_camera=False,
    source_run="test",
  )

  def build(bound):
    policy = Policy.__new__(Policy)
    policy.metadata = replace(metadata, clip_actions=bound)
    policy._last_action = np.zeros(6)
    policy._network_action = np.zeros(6)
    policy._smoothing = 1.0
    # Divergent the way the real loop is: gain 1.5 > 1, so only a bound on what
    # gets fed back can stop it.
    policy._infer = lambda observation: 1.5 * observation["obs"][0] + 2.0
    return policy

  joint_pos = np.zeros(len(ARM_JOINTS))
  unbounded, bounded = build(None), build(1.0)
  for _ in range(40):
    for policy in (unbounded, bounded):
      policy.act(joint_pos=joint_pos, joint_vel=joint_pos, image=None)

  assert float(np.abs(unbounded.network_action).max()) > 100.0

  assert float(np.abs(bounded.network_action).max()) == pytest.approx(3.5)
  assert np.all(np.abs(bounded.act(
    joint_pos=joint_pos, joint_vel=joint_pos, image=None
  )) <= 1.0)
  assert float(np.abs(bounded.network_action).max()) > 1.0
