"""The deployment observation must equal the one the simulator computes.

Deployment rebuilds the actor observation from scratch: sensor reads, a
configured target, and forward kinematics off the MJCF. Every one of those steps
can be subtly wrong -- a joint in the wrong slot, a velocity that should have
been relative, a quaternion applied forward instead of inverse -- and none of
those mistakes look like errors on hardware. They look like a policy that almost
works.

So this feeds the simulator's own state through the deployment assembler and
demands the very same vector back. It is the only check that can fail on a desk
instead of on a moving arm.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.sim, pytest.mark.gpu]

torch = pytest.importorskip("torch")
pytest.importorskip("mjlab")
if not torch.cuda.is_available():
  pytest.skip("deployment parity needs CUDA", allow_module_level=True)

TASK = "Mjlab-LiftCube-CollisionCam-DinoV2ViTS14-LocalGrid7-Trossen"


@pytest.fixture(scope="module")
def simulation():
  from vbrl.runtime import build_env

  env = build_env(TASK, device="cuda:0", num_envs=1, seed=0)
  yield env
  env.close()


def test_deployment_observation_matches_the_simulator(simulation) -> None:
  import numpy as np

  from vbrl.deployment.observations import ObservationAssembler
  from vbrl.deployment.spec import RobotSpec

  env = simulation
  unwrapped = env.unwrapped
  spec = RobotSpec.from_env(env, ee_site_name="ee_site")

  observations = unwrapped.observation_manager.compute()
  expected = observations["actor"][0].detach().cpu().numpy()

  robot = unwrapped.scene["robot"]
  joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()
  joint_vel = robot.data.joint_vel[0].detach().cpu().numpy()

  # The deployment target is a configured constant; here it has to be whatever
  # the command sampled, expressed in the same base frame deployment uses.
  command = unwrapped.command_manager.get_term("lift_height")
  origin = unwrapped.scene.env_origins[0].detach().cpu().numpy()
  goal = command.target_pos[0].detach().cpu().numpy() - origin

  assembler = ObservationAssembler(spec, goal=tuple(goal), device="cuda:0")
  # ``actions`` reports the previous command, which the env tracks itself.
  assembler.record_action(unwrapped.action_manager.action[0].detach().cpu().numpy())

  frame = observations["camera"][0].detach().cpu().numpy().transpose(1, 2, 0)
  actual = assembler.build(joint_pos=joint_pos, joint_vel=joint_vel, rgb=frame)
  produced = actual["actor"][0].detach().cpu().numpy()

  assert produced.shape == expected.shape, (
    f"deployment assembled {produced.shape}, simulator computed {expected.shape}"
  )
  # float32 observations built through different arithmetic; 1e-4 is tight
  # enough to catch a swapped term and loose enough to allow reordering of
  # floating-point operations.
  np.testing.assert_allclose(produced, expected, atol=1e-4, rtol=0)


def test_the_camera_observation_round_trips_channel_order(simulation) -> None:
  import numpy as np

  from vbrl.deployment.observations import ObservationAssembler
  from vbrl.deployment.spec import RobotSpec

  env = simulation
  unwrapped = env.unwrapped
  spec = RobotSpec.from_env(env, ee_site_name="ee_site")
  observations = unwrapped.observation_manager.compute()

  expected = observations["camera"][0].detach().cpu().numpy()
  frame = expected.transpose(1, 2, 0)  # what a camera hands us

  assembler = ObservationAssembler(spec, goal=(0.4, 0.0, 0.3), device="cuda:0")
  robot = unwrapped.scene["robot"]
  built = assembler.build(
    joint_pos=robot.data.joint_pos[0].detach().cpu().numpy(),
    joint_vel=robot.data.joint_vel[0].detach().cpu().numpy(),
    rgb=frame,
  )
  produced = built[spec.camera_group][0].detach().cpu().numpy()
  np.testing.assert_array_equal(produced, expected)


def test_joint_targets_reproduce_the_action_term(simulation) -> None:
  import numpy as np

  from vbrl.deployment.observations import ObservationAssembler
  from vbrl.deployment.spec import RobotSpec

  env = simulation
  spec = RobotSpec.from_env(env, ee_site_name="ee_site")
  assembler = ObservationAssembler(spec, goal=(0.4, 0.0, 0.3), device="cuda:0")

  action = np.linspace(-1.0, 1.0, spec.action_scale.shape[-1])
  targets = assembler.joint_targets(action)

  term = env.unwrapped.action_manager.get_term("joint_pos")
  term.process_actions(
    torch.as_tensor(action, dtype=torch.float32, device="cuda:0").unsqueeze(0)
  )
  expected = term._processed_actions[0].detach().cpu().numpy()
  np.testing.assert_allclose(targets, expected, atol=1e-5, rtol=0)


def test_a_goal_outside_the_training_range_is_refused() -> None:
  from vbrl.deployment.config import GOAL_RANGE, DeploymentConfig

  beyond = GOAL_RANGE["z"][1] + 0.1
  config = DeploymentConfig(
    task_id=TASK,
    checkpoint_file="ckpts/lift_cube/dinov2_vits14_local_grid7_real.pt",
    arm_ip="192.168.1.2",
    goal=(0.4, 0.0, beyond),
  )
  with pytest.raises(ValueError, match="outside the range the policy trained on"):
    config.validate()
