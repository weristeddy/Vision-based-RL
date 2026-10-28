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

Two DAgger students of the state teacher `w924eo8a`, both DINOv2-S with AFA at
full width (6 heads), external D405 (`camera_serial` 260322270027):

| manifest | task ID | W&B |
| --- | --- | --- |
| `push_t_real_table_dinov2_afafull6` | `Mjlab-PushT-RealTable-DinoV2ViTS14-AfaFull6-Trossen` | `ed70t2t7` |
| `push_t_wood_table_dinov2_afafull6` | `Mjlab-PushT-WoodTable-DinoV2ViTS14-AfaFull6-Trossen` | `k1kqta8a` |

```bash
uv run --no-sync vbrl-deploy configs/deployment/push_t_real_table_dinov2_afafull6.yaml --dry-run
uv run --no-sync vbrl-deploy configs/deployment/push_t_real_table_dinov2_afafull6.yaml --max-steps 400 --no-keyboard-goal --log artifacts/deployment/real_table_1.npz
uv run --no-sync vbrl-deploy configs/deployment/push_t_wood_table_dinov2_afafull6.yaml --max-steps 400 --no-keyboard-goal --log artifacts/deployment/wood_table_1.npz
```

The policies trained at 20 Hz with the Trossen SDK's 50 ms linear goal ramp
modelled in sim, so the manifests command with `command_goal_time_s: 0.05`.
`vbrl-deploy` refuses a manifest whose `control_hz` or `command_goal_time_s`
differs from the `control_hz` and `ramp_s` in the ONNX metadata. The action
scale (0.025, wrist roll 0.02) travels in the metadata too. The goal marker is
the hollow outline: a 15 mm green frame tracing the T footprint. A filled marker
is out of distribution.

`motion:` in these manifests passes the policy through unshaped
(`action_smoothing: 1.0`, `max_joint_step: null`). `--action-gain` scales each
action before smoothing and before it is added to the target;
`--action-smoothing` is the weight of the new action in the moving average, so
1.0 is no smoothing. Both override the manifest's `motion:` block.

Brightness is matched with gain at a fixed exposure: 8,000 us, gain 27 and
4,400 K bring the 224 x 224 crop to median 128 and R/G 1.38 against the
simulated 127 and 1.36 (measured 2026-10-26). Re-probe after any lighting
change:

```bash
uv run --no-sync python -m vbrl.deployment.camera --serial 260322270027 --gain 27 --exposures 8000
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
uv run --no-sync vbrl-play Mjlab-PushT-RealTable-DinoV2ViTS14-AfaFull6-Trossen \
  --wandb-run-path eduard-nicolae-robot-learning/mjlab/ed70t2t7 \
  --wandb-checkpoint-name model_4999.pt --num-envs 4
```

The task ID fixes the action scale and the ramp, so the replay matches training.

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
uv run --no-sync vbrl-deploy configs/deployment/push_t_real_table_dinov2_afafull6.yaml --dry-run
uv run --no-sync vbrl-deploy configs/deployment/push_t_real_table_dinov2_afafull6.yaml --home
uv run --no-sync vbrl-deploy configs/deployment/push_t_real_table_dinov2_afafull6.yaml --park
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
