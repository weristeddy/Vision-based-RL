# Deployment commands

Copy-paste reference for running a trained policy on the real arm from the
Jetson AGX Thor. Every run lifts the arm to its upright home, lowers it to the
policy's start pose, and afterwards lifts it again before parking.

`--no-sync` is not optional here. ONNX Runtime is installed outside the
lockfile by `jetson/setup.sh`, so a plain `uv run` syncs the environment and
removes it. The alternative is `source .venv/bin/activate` once, after which
`vbrl-deploy ...` works on its own.

`--max-steps 400` is 20 s at 20 Hz.

## Push-T

Trained with the Trossen SDK's 50 ms linear ramp and its velocity feedforward
modelled in sim (`ramp_s 0.05`), so the manifests command with
`command_goal_time_s: 0.05` -- the two must match. Scale 0.05 at 20 Hz,
near-goal speed penalty -80 ramped, action rate -0.01. The goal marker is the
hollow outline: a 15 mm green frame tracing the T footprint. A filled marker is
out of distribution.

```bash
uv run --no-sync vbrl-deploy configs/deployment/push_t_realtable_wrist002_sdk50.yaml --dry-run
uv run --no-sync vbrl-deploy configs/deployment/push_t_realtable_wrist002_sdk50.yaml --max-steps 400
uv run --no-sync vbrl-deploy configs/deployment/push_t_realtable_target_sdk50.yaml --max-steps 400
uv run --no-sync vbrl-deploy configs/deployment/push_t_realtable_measured_sdk50.yaml --max-steps 400
```

`push_t_realtable_wrist002_sdk50` (W&B `o2uvn93k`, sim success 0.87) is the target
policy retrained with `--env.actions.joint-pos.wrist-roll-scale 0.02` and exported
with `--wrist-roll-scale 0.02`. It removed the gripper's wrist-roll jitter (0.78 to
0.27 rad/s) and is the best on the rig so far, 1 of 5 runs at the goal.

| manifest | W&B | reference | sim success | fingertip approaching T | T 1-3 cm from goal |
| --- | --- | --- | --- | --- | --- |
| `push_t_realtable_target_sdk50` | `ok5ae9ph` | target | 0.90 | 0.15 m/s | 0.10 m/s |
| `push_t_realtable_measured_sdk50` | `gh9nf7wc` | measured | 0.89 | 0.25 m/s | 0.13 m/s |

Mean action over 128 episodes at the manifest goal. ONNX exported with `--action-scale 0.05`,
`--ramp-s 0.05` and `--action-reference` set to the run's reference; a measured-reference
policy exported without it would integrate on the target instead of the measured position.
`vbrl-deploy` refuses a manifest whose `control_hz` or `command_goal_time_s` differs from
the `control_hz` and `ramp_s` the export carries, so export with the run's own flags.

`--action-gain` scales each action before smoothing and before it is added to
the target; `--action-smoothing` is the weight of the new action in the moving
average, so 1.0 is no smoothing. Both override the manifest's `motion:` block.

## Push-T: distilled students and the PPO reference

All five share the rig settings above (20 Hz, 50 ms SDK ramp, 11,000 us / 4,800 K)
and name their camera explicitly: `camera_serial` 260322270027 is the external
D405, 412622271761 the wrist D405. Students were distilled from the state policy
`yregddj3`; the PPO policy is `twub0nml` (4 epochs).

| manifest | W&B | encoder | camera | sim success |
| --- | --- | --- | --- | --- |
| `push_t_dinov2_afafull6_distill` | `7gazz0tn` | DINOv2-S + AFA (6 heads) | external | 0.89 |
| `push_t_dinov2_afa6_distill` | `j5vixm7j` | DINOv2-S + AFA6 | external | 0.87 |
| `push_t_r3m_afafull32_distill` | `v8qzh46c` | R3M + AFA (32 heads) | external | 0.43 |
| `push_t_dinov2_afa6_wrist_distill` | `v0x4uv8f` | DINOv2-S + AFA6 | wrist | 0.92 |
| `push_t_dinov2_afa6_rl_ep4` | `twub0nml` | DINOv2-S + AFA6, PPO | external | 0.80 |

```bash
uv run --no-sync vbrl-deploy configs/deployment/push_t_dinov2_afafull6_distill.yaml --max-steps 400 --no-keyboard-goal --log artifacts/deployment/afafull6_1.npz
uv run --no-sync vbrl-deploy configs/deployment/push_t_dinov2_afa6_distill.yaml --max-steps 400 --no-keyboard-goal --log artifacts/deployment/afa6_distill_1.npz
uv run --no-sync vbrl-deploy configs/deployment/push_t_r3m_afafull32_distill.yaml --max-steps 400 --no-keyboard-goal --log artifacts/deployment/r3m_afafull32_1.npz
uv run --no-sync vbrl-deploy configs/deployment/push_t_dinov2_afa6_wrist_distill.yaml --max-steps 400 --no-keyboard-goal --log artifacts/deployment/wrist_distill_1.npz
uv run --no-sync vbrl-deploy configs/deployment/push_t_dinov2_afa6_rl_ep4.yaml --max-steps 400 --no-keyboard-goal --log artifacts/deployment/rl_ep4_1.npz
```

Brightness is matched with gain, not exposure: both cameras keep 11,000 us (under the
16.7 ms frame time at 60 fps, so no blur or extra latency) and raise `camera_gain` until
the 224 x 224 crop's median matches the simulated median (external 128 at gain 27; wrist
110 at home at gain 26, measured 2026-10-20). Re-probe after any lighting change:

```bash
uv run --no-sync python -m vbrl.deployment.camera --serial 260322270027 --gain 27 --exposures 11000
uv run --no-sync python -m vbrl.deployment.camera --serial 412622271761 --gain 26 --exposures 11000
```

## Lift cube

```bash
uv run --no-sync vbrl-deploy configs/deployment/lift_cube.yaml --max-steps 1000
```

W&B `ntl27zt9`, a spatial-softmax policy, no goal marker. `goal:` is where to
lift the cube to in the base frame; keep z clear of the gripper's height at
home (0.205) or the policy reads "already there" and never lifts. The cube's
own position is never configured — the policy sees it only through the camera.
This is the one manifest that shapes the action in the manifest itself
(`action_smoothing: 0.25`, `max_joint_step: 0.035`).

## Viewing a policy in the browser

`vbrl-play` serves a Viser scene on port 8080 (`--host`, `--port` to change
it). The policies have no local `.pt`, only the exported ONNX, so the weights
come from W&B and the task ID supplies the architecture:

```bash
uv run --no-sync vbrl-play Mjlab-PushT-RealTable-DinoV2ViTS14-Afa6-TrossenIdentified \
  --wandb-run-path eduard-nicolae-robot-learning/mjlab/ok5ae9ph \
  --wandb-checkpoint-name model_14999.pt --action-scale 0.05 --num-envs 4
```

`vbrl-play` has no flag for the ramp or the action reference, so it replays
these policies with the registered defaults (no ramp, target reference).

Add `--goal 0.38442 0.01567 0.0124` to pin the goal to the rig's marker pose,
which is what the deployment manifests use; without it the goal randomizes over
the full circle. Other useful flags: `--agent zero` to see the scene with no
policy, `--scene {wood,plaster,peacock}` for an OOD tabletop, `--max-steps N`
to exit cleanly. Keep `--num-envs` small: per-env table textures are uploaded
as one mesh each.

The Jetson is headless, so the browser is on the laptop and port 8080 has to
reach it. Over VS Code Remote-SSH that happens by itself: the PORTS panel
forwards 8080 and <http://localhost:8080> opens. When the page does not load,
it is almost always that forward, not `vbrl-play` -- open the PORTS panel and
add 8080 by hand if auto-forwarding missed it. Without VS Code, tunnel it:

```bash
ssh -L 8080:localhost:8080 eddy@172.24.14.193
```

To check the server rather than the forward, from a shell on the Jetson:

```bash
uv run --no-sync python -c "import urllib.request as u; print(u.urlopen('http://127.0.0.1:8080/').status)"
```

200 means `vbrl-play` is fine and the problem is the forward. Note
`--max-steps` makes it exit when it is reached, closing the server with it --
leave it off to keep the viewer up.

## Before and after a run

```bash
uv run --no-sync vbrl-deploy configs/deployment/push_t_realtable_target_sdk50.yaml --dry-run
uv run --no-sync vbrl-deploy configs/deployment/push_t_realtable_target_sdk50.yaml --home
uv run --no-sync vbrl-deploy configs/deployment/push_t_realtable_target_sdk50.yaml --park
```

`--dry-run` homes the arm, reads the sensors and evaluates the policy without
commanding any motion. `--home` holds the home pose and needs no camera; Ctrl-C
parks. `--park` brings the arm down and releases torque.

Arrow keys move the goal while a run is going: up/down moves z, left/right
moves y, 0.02 m per press, clamped to the training range. `--no-keyboard-goal`
turns that off, and it turns itself off with no terminal on stdin.

## When the goal marker moves

The numbers in `goal:` and the marker on the table have to agree. Seat the two
tag25h9 markers in the notches, capture one external-camera frame, then take
the tags off — the policy must never see them.

```bash
uv run --no-sync python -m vbrl.deployment.goal_pose <frame.png> --margin-mm 7.0
```

## Exposure

`camera_exposure_us` and `camera_white_balance_k` belong to the scene the
policy was trained in, not to the room. The rig-table scene renders a narrow
brightness band, and a run below it fails completely while looking like a
policy failure: an earlier policy at 7000 us moved the T by 5.7 mm in 16 s, and
at 12000 us the same policy reached the goal. Auto white balance renders the
wood grey and the policies fail on it. Re-probe before a session rather than
trusting the numbers in the manifests:

```bash
uv run --no-sync python -m vbrl.deployment.camera
```
