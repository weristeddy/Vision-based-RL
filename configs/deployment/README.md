# Deployment commands

Copy-paste reference for running a trained policy on the real arm from the
Jetson AGX Thor. Every run homes the arm first and parks it afterwards.

`--no-sync` is not optional here. ONNX Runtime is installed outside the
lockfile by `jetson/setup.sh`, so a plain `uv run` syncs the environment and
removes it. The alternative is `source .venv/bin/activate` once, after which
`vbrl-deploy ...` works on its own.

At 50 Hz, `--max-steps 1000` is 20 s.

## Which marker has to be on the table

The Push-T policies were not all trained against the same goal marker, and the
marker is drawn into the camera observation, so the wrong one is out of
distribution — fatally so for the pixel-only policies, which have no other goal
signal.

| marker | drawn as | policies |
| --- | --- | --- |
| **Hollow outline** | 15 mm green frame tracing the T footprint | `realtable`, `realtablepixel`, `goaloutline15k`, `goaloutline`, `goaloutlinepixel` |
| **Filled** | solid green T, the same two boxes as the object | `pixelgoal_fixed`, `visualslowstep` |
| none | — | `lift_cube` |

> **These policies are history, not the current configuration.** They were
> trained on `-TrossenRealistic` against the Trossen MJCF, a grey-rendering
> tabletop, a black background and no joint-velocity penalty. Push-T now
> registers `-TrossenIdentified` on the system-identified arm, so the task IDs
> below no longer exist and nothing here can be reproduced from source. The
> manifests still run, because the observation and action contract travels
> inside the ONNX graph.

## Hollow outline marker, 15,000 iterations

The current generation. Trained at the 0.03 per-step cap with 16 s episodes and
the linear orientation summand.

```bash
uv run --no-sync vbrl-deploy configs/deployment/push_t_realtable.yaml --max-steps 1000
uv run --no-sync vbrl-deploy configs/deployment/push_t_realtablepixel.yaml --max-steps 1000
uv run --no-sync vbrl-deploy configs/deployment/push_t_goaloutline15k.yaml --max-steps 1000
```

| manifest | W&B | goal signal | exposure | sim success |
| --- | --- | --- | --- | --- |
| `push_t_realtable` | `xbl8pqsz` | `target_pose` + marker | 12000 us | 0.727 |
| `push_t_realtablepixel` | `0vv02udk` | marker only | 12000 us | 0.500 |
| `push_t_goaloutline15k` | `ao5bzb64` | `target_pose` + marker | 9000 us | 0.409 |

`realtable` and `goaloutline15k` differ only by the tabletop texture — fixed
photograph against the randomized bank — so they are a controlled pair.

## Hollow outline marker, 6,000 iterations

```bash
uv run --no-sync vbrl-deploy configs/deployment/push_t_goaloutline.yaml --max-steps 1000
uv run --no-sync vbrl-deploy configs/deployment/push_t_goaloutlinepixel.yaml --max-steps 1000
```

| manifest | W&B | goal signal | exposure |
| --- | --- | --- | --- |
| `push_t_goaloutline` | `8vpxhjf2` | `target_pose` + marker | 7000 us |
| `push_t_goaloutlinepixel` | `9pqs3f02` | marker only | 7000 us |

Both were trained while the goal-yaw sampler was dropping its assignment, so
the goal yaw sat at 0 for all of training. They are only in distribution with
the marker near yaw 0, which is where this manifest's goal is. Rotating the
marker puts them out of distribution.

## Filled marker, 6,000 iterations

Superseded. Their task IDs are no longer registered, but the observation and
action contract travels inside the ONNX metadata, so the manifests still run.

```bash
uv run --no-sync vbrl-deploy configs/deployment/push_t_pixelgoal_fixed.yaml --max-steps 1000
uv run --no-sync vbrl-deploy configs/deployment/push_t_visualslowstep.yaml --max-steps 1000
```

| manifest | W&B | goal signal | per-step cap | exposure |
| --- | --- | --- | --- | --- |
| `push_t_pixelgoal_fixed` | `n92lvmra` | marker only | 0.05 | 7000 us |
| `push_t_visualslowstep` | `9w8z704o` | `target_pose` + marker | 0.03 | 7000 us |

`pixelgoal_fixed` was trained against **one** goal pose and no other — the one
in its manifest. The marker has to be at exactly that pose; move it back rather
than editing the file. It is the strongest of these in sim (0.967 success) only
because it faces one goal instead of the full distribution.

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
it). The entry point is `vbrl-play`, not `vbrl-visualize`. These policies have
no local `.pt`, only the exported ONNX, so the weights come from W&B and the
task ID supplies the architecture -- and the IDs below were renamed to
`-TrossenIdentified`, so these commands no longer resolve. They are kept as the
record of which run produced which policy.

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

200 means `vbrl-play` is fine and the problem is the forward. It binds
`0.0.0.0` and answers on every interface. Note `--max-steps` makes it exit when
it is reached, closing the server with it -- leave it off to keep the viewer up.

```bash
uv run --no-sync vbrl-play Mjlab-PushT-RealTable-DinoV2ViTS14-Afa6-TrossenRealistic \
  --wandb-run-path eduard-nicolae-robot-learning/mjlab/xbl8pqsz \
  --wandb-checkpoint-name model_14999.pt --num-envs 4

uv run --no-sync vbrl-play Mjlab-PushT-RealTablePixel-DinoV2ViTS14-Afa6-TrossenRealistic \
  --wandb-run-path eduard-nicolae-robot-learning/mjlab/0vv02udk \
  --wandb-checkpoint-name model_14999.pt --num-envs 4

uv run --no-sync vbrl-play Mjlab-PushT-GoalOutline-DinoV2ViTS14-Afa6-TrossenRealistic \
  --wandb-run-path eduard-nicolae-robot-learning/mjlab/ao5bzb64 \
  --wandb-checkpoint-name model_14999.pt --num-envs 4
```

Add `--goal 0.38442 0.01567 0.0124` to pin the goal to the rig's marker pose,
which is what the deployment manifests use; without it the goal randomizes over
the full circle. Other useful flags: `--agent zero` to see the scene with no
policy, `--scene {wood,plaster,peacock}` for an OOD tabletop, `--max-steps N`
to exit cleanly.

Keep `--num-envs` small. Per-env table textures are uploaded as one mesh each,
so this is for a handful of envs, not a training-sized batch.

## Before and after a run

```bash
uv run --no-sync vbrl-deploy configs/deployment/push_t_realtable.yaml --dry-run
uv run --no-sync vbrl-deploy configs/deployment/push_t_realtable.yaml --home
uv run --no-sync vbrl-deploy configs/deployment/push_t_realtable.yaml --park
```

`--dry-run` homes the arm, reads the sensors and evaluates the policy without
commanding any motion. `--home` holds the home pose and needs no camera; Ctrl-C
parks. `--park` brings the arm down and releases torque.

Arrow keys move the goal while a run is going: up/down moves z, left/right
moves y, 0.02 m per press, clamped to the training range. `--no-keyboard-goal`
turns that off, and it turns itself off with no terminal on stdin.

## When the goal marker moves

Where `target_pose` is an input, the numbers in `goal:` and the marker on the
table have to agree. Seat the two tag25h9 markers in the notches, capture one
external-camera frame, then take the tags off — the policy must never see them.

```bash
uv run --no-sync python -m vbrl.deployment.goal_pose <frame.png> --margin-mm 7.0
```

## Exposure

`camera_exposure_us` belongs to the scene the policy was trained in, not to the
room. The fixed-texture scenes render a narrow brightness band, and a run below
it fails completely while looking like a policy failure — `realtable` at
7000 us moved the T by 5.7 mm in 16 s, and at 12000 us the same policy reached
the goal. Re-probe before a session rather than trusting the number here:

```bash
uv run --no-sync python -m vbrl.deployment.camera
```
