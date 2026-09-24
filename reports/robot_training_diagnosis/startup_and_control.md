# Follow-up: startup kick and control semantics

The earlier action-scale comparison establishes that the identified robot can learn, but does not establish that its control interface is appropriate. The follow-up uncovered a reproducible startup defect independent of policy actions.

## Confirmed startup defect

In installed MJLab 1.6.0, EntityData.clear_state sets joint_pos_target to zero. ManagerBasedRlEnv.reset resets the robot pose, then calls scene.write_data_to_sim before any action is applied. The delayed builtin actuator group appends the zero target into its history. On the first action, environments sampling a one-substep delay can receive that zero target instead of the reset joint angles. A zero policy action therefore does not prevent the kick.

GPU probe: 32 environments, seed 0, 100 zero-action steps (2 seconds), current Push-T scene, robot explicitly selected independently of local changes to the default factory:

| Case | Maximum arm drift | Maximum sampled joint speed |
| --- | ---: | ---: |
| Identified, current reset/delay | 0.104645 rad | 1.125271 rad/s |
| Identified, no delay | 0.00000313 rad | 0.000139 rad/s |
| Identified, reset targets initialized to reset pose | 0.00000313 rad | 0.000140 rad/s |
| Old realistic, current reset/delay | 0.032424 rad | 0.267342 rad/s |

11/32 environments exceeded 0.01 rad drift with the current reset in either model. None did with primed targets or delay disabled. The new-model elbow showed the largest drift. The native CPU scene with properly initialized controls also stayed stable. This isolates command initialization and delay history rather than spontaneous numerical collapse of the XML.

The in-memory correction appends a reset event after joint randomization:

```python
def prime_targets(env, env_ids):
    robot = env.scene['robot']
    robot.set_joint_position_target(
        robot.data.joint_pos[env_ids], env_ids=env_ids
    )
```

This event must run before reset writes controls into the delay buffer. It should use the randomized reset pose, not an unrandomized home constant. The probe includes an explicit partial-reset check; these particular partial-reset samples were stable even without the correction, so the measured kick is not universal at every reset. Production integration should also test automatic episode resets and unaffected environments.

Reproduce: `.venv/bin/python reports/robot_training_diagnosis/reset_probe.py`. Exact results: `reset_probe.json`. No production source changes were made.

## Why the same actions produce much more violent elbow movement

The current RelativeJointPositionAction recomputes target = measured_position + scale * action every 5 ms. Ignoring limits, coupling and the implicit integration details, the proportional actuator term becomes approximately kp * scale * action: the policy supplies a sustained position error rather than a fixed position destination. Zero action removes that error; it does not restore the preceding target after displacement.

The old elbow has actuator velocity feedback 10; the identified elbow has joint damping 0.725663 and zero armature. Other joints differ in the opposite direction. Consequently a single shared action scale cannot make the two robots respond similarly in every joint.

Native CPU probe at the same home pose, with gravity compensation, contacts disabled and no command delay:

| Elbow command | Old model final displacement | New model final displacement |
| --- | ---: | ---: |
| Recompute measured position + 0.03 for 0.1 s, then zero relative action | 0.05764 rad | 0.36813 rad |
| Set home + 0.03 once and hold it | 0.03002 rad | 0.02979 rad |

These are different command semantics with the same numeric initial error, not identical trajectories. They show why the moving-target interface amplifies small policy outputs on the identified elbow. It is a strong mechanism for the observed whipping; quantifying its share of learned-policy shaking would require replaying actions or retraining the alternative controller.

Reproduce: `.venv/bin/python reports/robot_training_diagnosis/control_probe.py`. Results and plot: `control_probe.json`, `control_response.png`.

## External examples and what they do establish

- [Exact Menagerie WXAI model](https://github.com/google-deepmind/mujoco_menagerie/blob/main/trossen_wxai/wxai_follower.xml): six arm joint and actuator parameter attributes match the local identified model. Local modifications include gravity compensation, camera/sites, rendering groups, and moving the gripper actuator to the left carriage. The arm gains have not been mistranscribed.
- [Menagerie WXAI README](https://github.com/google-deepmind/mujoco_menagerie/tree/main/trossen_wxai): says identification used four real trajectories. This is evidence of dynamics fitting, not proof that arbitrary RL action interfaces are equivalent or easy to learn.
- [Trossen official pick-and-place demo](https://github.com/TrossenRobotics/trossen_arm_mujoco/blob/main/trossen_arm_mujoco/scripts/wxai_pick_place.py) and [controller](https://github.com/TrossenRobotics/trossen_arm_mujoco/blob/main/trossen_arm_mujoco/src/controller.py): useful examples for the robot. The joint-position API writes desired joint angles into data.ctrl. Its Cartesian controller computes current_q + an IK correction based on Cartesian error, so it does not prove that all current-position-relative control is invalid. These are vendor-model examples, not a verified PPO baseline for the exact identified XML.

No well-matched public PPO example using the exact identified WXAI configuration was verified in the search.

## Recommended order

1. Initialize position targets to the reset pose before delay history is filled. This addresses a reproduced defect without changing identified physical parameters.
2. Compare action interfaces: either hold an absolute desired joint position, or update a persistent desired target once per policy step and hold it across physics substeps. Initialize/reset that reference correctly, clamp it to joint limits, bound its rate/tracking error, and make target tracking error observable if the controller introduces state.
3. Revisit action scale and exploration only after this control comparison. A blanket increase to 0.13 can restore task performance while making sensitive joints more violent.
4. Keep deployment action semantics aligned with whichever interface is adopted. Do not change physical damping merely to conceal an interface defect without checking hardware response.

The first W&B training clips for 1uqeno3i, riroa30p and pefvspmj were downloaded and contact sheets extracted. They show the initial motion; the controlled probes above separate the startup defect from policy-driven motion. User edits switched local factories back to realistic during the investigation; the reproducible probes select each model explicitly and preserve those edits.
