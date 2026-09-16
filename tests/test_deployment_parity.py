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
    from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg
    from mjlab.rl.exporter_utils import get_base_metadata

    metadata = get_base_metadata(env.unwrapped, "parity-test")
    term = next(iter(env.unwrapped.cfg.actions.values()))
    metadata["action_type"] = (
      "relative"
      if isinstance(term, RelativeJointPositionActionCfg)
      else "absolute"
    )
    from mjlab.tasks.registry import load_rl_cfg

    clip_actions = load_rl_cfg(TASK).clip_actions
    if clip_actions is not None:
      metadata["clip_actions"] = float(clip_actions)
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
  # `Policy.goal` is in the arm's BASE frame -- the frame its forward kinematics
  # work in -- so the goal has to be referred to the robot's own root, not to the
  # env origin. The two coincided only while the base sat at z = 0; it now sits
  # on the 5 mm mounting plate, and using the origin here left the deployment
  # side reading a goal 5 mm high.
  base = robot.data.root_link_pos_w[0].cpu().numpy()
  goal = command.target_pos[0].cpu().numpy() - base

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


def test_joint_targets_match_the_action_term_the_policy_trained_under() -> None:
  """Relative and absolute mappings agree only at the home pose.

  Push-T uses `RelativeJointPositionAction` -- `target = current + scale *
  action` -- while Lift-Cube's is absolute against the default pose. Applying
  the absolute one to a relatively-trained policy is silent: the arm stays
  within one action scale of home for the whole episode and reads as a policy
  that does nothing, rather than as an error.
  """
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
    relative=True,
    clip_actions=None,
    needs_camera=False,
    source_run="test",
  )
  policy = Policy.__new__(Policy)
  policy.metadata = metadata
  policy._response_gain = 1.0
  action = np.array([1.0, -1.0, 0.5, 0.0, 0.25, -0.5])

  # At the home pose the two mappings coincide, which is exactly why the bug
  # survives a bench test and only shows up once the arm has moved.
  policy._position = default.copy()
  assert np.allclose(policy.joint_targets(action)[:6], default[:6] + 0.1 * action)

  # Away from home they do not, and the relative one must track the arm.
  moved = default + 0.4
  policy._position = moved
  targets = policy.joint_targets(action)
  assert np.allclose(targets[:6], moved[:6] + 0.1 * action)
  assert not np.allclose(targets[:6], default[:6] + 0.1 * action)

  # Push-T drives six joints; the seventh holds its default, which is how the
  # gripper stays closed without a channel of its own.
  assert targets.shape == (len(ARM_JOINTS),)
  assert targets[6] == pytest.approx(default[6])
  assert policy.has_gripper is False

  # `response_gain` scales the delta at every magnitude, which a rate clamp
  # cannot: sim realizes 0.267 of what it commands, so on hardware a clamp
  # throttles transport while leaving fine corrections 3.7x too responsive.
  policy._response_gain = 0.267
  scaled = policy.joint_targets(action)
  assert np.allclose(scaled[:6], moved[:6] + 0.267 * 0.1 * action)
  assert scaled[6] == pytest.approx(default[6])

  # The action term's clip is part of the contract, not a safety extra. Without
  # it a deployment sends whatever the policy asks for: on Push-T the commanded
  # gap reached 0.31 rad against the 0.1 the simulator ever applies, which is
  # the whole of "the robot moves way too fast".
  policy._response_gain = 1.0
  policy.metadata = replace(
    metadata, action_clip=(np.full(6, -0.05), np.full(6, 0.05))
  )
  policy._position = default.copy()
  clipped = policy.joint_targets(action)
  assert np.allclose(
    clipped[:6], default[:6] + np.clip(0.1 * action, -0.05, 0.05)
  )
  # 1.0 and -1.0 scale to +-0.1 and must come back at the bound.
  assert clipped[0] == pytest.approx(default[0] + 0.05)
  assert clipped[1] == pytest.approx(default[1] - 0.05)


def test_the_action_fed_back_as_an_observation_stays_inside_its_training_band() -> None:
  """`clip_actions` bounds the `actions` observation, or the policy winds up.

  RSL-RL's vector wrapper clamps the action to +/-`clip_actions` before the
  environment sees it, and the `actions` observation term reads that clamped
  value -- so a policy never saw one outside the band while training.
  Deployment has no wrapper. Feeding back the unclipped output closes a
  positive loop: a more extreme `actions` observation produces a more extreme
  action, which is fed back more extreme still.

  Measured on hardware, replaying the frame the aborted run logged: 2.1, 3.2,
  4.2, 5.6, 7.8, 10.7, 14.2, 17.4, diverging to 22.3, against 2.1, 2.8, 2.9,
  2.95 settling once the bound is applied.
  """
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
    relative=True,
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
    policy._response_gain = 1.0
    # A divergent map, in the sense the real loop is: the output grows with the
    # action fed back to it. Gain 1.5 > 1, so the only thing that can stop it
    # is a bound on what gets fed back.
    policy._infer = lambda observation: 1.5 * observation["obs"][0] + 2.0
    return policy

  joint_pos = np.zeros(len(ARM_JOINTS))
  unbounded, bounded = build(None), build(1.0)
  for _ in range(40):
    for policy in (unbounded, bounded):
      policy.act(joint_pos=joint_pos, joint_vel=joint_pos, image=None)

  # Unbounded, it runs away -- past the 6.0 the first hardware run aborted at.
  assert float(np.abs(unbounded.network_action).max()) > 100.0

  # Bounded, it settles at the fixed point of the map evaluated at the bound.
  assert float(np.abs(bounded.network_action).max()) == pytest.approx(3.5)
  # And what reaches the joints is the clamped action, never the raw output.
  assert np.all(np.abs(bounded.act(
    joint_pos=joint_pos, joint_vel=joint_pos, image=None
  )) <= 1.0)
  # while `network_action` keeps the unclamped value, so the loop's
  # out-of-distribution check still has something to measure.
  assert float(np.abs(bounded.network_action).max()) > 1.0
