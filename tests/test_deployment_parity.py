"""The deployment observation must equal the one the simulator computes.

Deployment rebuilds the actor observation from scratch: sensor reads, a
configured target, and forward kinematics off the MJCF. Every step can be
subtly wrong -- a joint in the wrong slot, a velocity that should have been
relative, a quaternion applied forward instead of inverse -- and none of those
look like errors on hardware. They look like a policy that almost works.

So this feeds the simulator's own state through the deployment assembler and
demands the same vector back. It is the only check that can fail on a desk
instead of on a moving arm.
"""

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
  """The metadata an exported ONNX carries, without needing the file."""

  def __init__(self, values: dict[str, str]) -> None:
    self.custom_metadata_map = values


class _Session:
  """Enough of an ONNX Runtime session for PolicySpec to read the contract."""

  def __init__(self, env) -> None:
    from mjlab.rl.exporter_utils import get_base_metadata

    metadata = get_base_metadata(env.unwrapped, "parity-test")
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
  """The exported metadata must describe what the environment actually does."""
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
  origin = unwrapped.scene.env_origins[0].cpu().numpy()
  goal = command.target_pos[0].cpu().numpy() - origin

  policy = Policy(_Session(simulation), goal=tuple(goal))
  # ``actions`` is mdp.last_action, the raw policy output, which is exactly
  # what Policy.act remembers -- so no rescaling happens on either side.
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
  """Measured contract: raw 0..255 shifts the actions by 1.65 and looks fine."""
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

  # An empty file is enough: `validate` checks the graph exists before it checks
  # the goal, so naming a real export here would tie this assertion to whichever
  # 90 MB file happens to be in the checkout.
  graph = tmp_path / "policy.onnx"
  graph.touch()
  config = DeploymentConfig(
    onnx_file=str(graph),
    arm_ip="192.168.1.2",
    goal=(0.4, 0.0, GOAL_RANGE["z"][1] + 0.1),
  )
  with pytest.raises(ValueError, match="outside the range the policy trained on"):
    config.validate()
