# Real robot

Copy-paste reference for running a trained policy on the real arm from the
Jetson AGX Thor. Every run lifts the arm to its upright home, lowers it to the
policy's start pose, and afterwards lifts it again before parking.

`--no-sync` is not optional here. ONNX Runtime is installed outside the
lockfile by `jetson/setup.sh`, so a plain `uv run` syncs the environment and
removes it. The alternative is `source .venv/bin/activate` once, after which
`vbrl-deploy ...` works on its own.

`--max-steps 400` is 20 s at 20 Hz.

## Files

- `rig.yaml`: the arm and both cameras (serial, exposure, gain, white balance).
- `<task>.yaml`: one per task, read by every tool: the camera its policies see, its
  policies (the best seed of each), goals, episode length and runs, control and
  motion. `vbrl-deploy` uses the task's camera; collection records both.

Policies are listed by architecture; `fetch_policies` downloads the best seed of
every listed one to `ckpts/deployment/<task>/` (existing files are kept):

```bash
uv run --no-sync python -m vbrl.deployment.fetch_policies configs/deployment/push_t.yaml
```

## Single runs

`--policy` picks one of the task's policies (default the first). `--goal` is a goal
number, the Lift-Cube list's entry or the Push-T goal measured into
`artifacts/real_eval/push_t/goal<n>/goal.json`, a `goal.json` path, or `x,y,z` / `x,y,yaw`;
Push-T needs it.

```bash
uv run --no-sync vbrl-deploy configs/deployment/push_t.yaml --policy DinoV2ViTS14-AfaFull6 --goal 2 --max-steps 400 --no-keyboard-goal --log artifacts/deployment/dino_1.npz
uv run --no-sync vbrl-deploy configs/deployment/push_t.yaml --policy NatureCnn-FlattenRelu --goal 2 --dry-run
uv run --no-sync vbrl-deploy configs/deployment/lift_cube.yaml --policy R3MResNet50-Global --goal 1 --max-steps 400
uv run --no-sync vbrl-deploy configs/deployment/lift_cube.yaml --policy CompactVit-Svea --goal 0.40,0.10,0.30 --log artifacts/deployment/lift.npz --record-other
```

`goal` for Lift-Cube is where to lift the cube to in the base frame; keep z clear of
the gripper's height at home (0.205) or the policy reads "already there" and never
lifts.

## Evaluation (data collection)

One goal block at a time: every start, every policy, in the task file's order, both
cameras recorded, the object measured automatically. For Push-T the block starts by
measuring the taped goal with the T off the table; it is saved and reused.

```bash
uv run --no-sync python -m vbrl.deployment.collect configs/deployment/push_t.yaml --goal 2
```

Data goes to `artifacts/real_eval/<task>/goal<n>/<policy>/run<k>/` and one
`results.csv` per task; a folder that exists is skipped, Ctrl-C marks a failure.

The policies trained at 20 Hz with the Trossen SDK's 50 ms linear goal ramp
modelled in sim, so the task files command with `command_goal_time_s: 0.05`.
`vbrl-deploy` refuses a task file whose `control_hz` or `command_goal_time_s`
differs from the `control_hz` and `ramp_s` in the ONNX metadata. The action
scale (0.025, wrist roll 0.02) travels in the metadata too. The goal marker is
the hollow outline: a 15 mm green frame tracing the T footprint. A filled marker
is out of distribution.

`motion:` in the task files passes the policy through unshaped
(`action_smoothing: 1.0`, `max_joint_step: null`). `--action-gain` scales each
action before smoothing and before it is added to the target;
`--action-smoothing` is the weight of the new action in the moving average, so
1.0 is no smoothing. Both override the task file's `motion:` block.

Brightness and color are matched in the 224 x 224 crop the policy sees, with the room
light fixed: 6,500 us, gain 27 and 5,000 K give median 127 and R/G 1.40 against the
rig-table render's 125 and 1.39 (measured 2026-11-27; `rig.yaml`). Re-measure after any
lighting change; the whole-frame probe `python -m vbrl.deployment.camera` includes the
background and ignores the white balance, so measure the crop:

```bash
uv run --no-sync python - <<'PY'
import time, numpy as np
from vbrl.deployment.camera import RealSenseCamera
from vbrl.deployment.session import load_session
camera = RealSenseCamera(load_session("configs/deployment/push_t.yaml").config())
time.sleep(2.0); frames = [camera.frame() for _ in range(10)]; camera.close()
image = np.mean(frames, 0)
print("median", round(float(np.median(image)), 1), "R/G", round(float(np.median(image[..., 0]) / np.median(image[..., 1])), 3))
PY
```

## Viewing a policy in the browser

`vbrl-play` serves a Viser scene on port 8080 (`--host`, `--port` to change
it). The policies have no local `.pt`, only the exported ONNX, so the weights
come from W&B and the task ID supplies the architecture:

```bash
uv run --no-sync vbrl-play Mjlab-PushT-WoodTable-DinoV2ViTS14-AfaFull6-Trossen \
  --wandb-run-path eduard-nicolae-robot-learning/mjlab/rzf11em2 \
  --wandb-checkpoint-name model_7999.pt --num-envs 4
```

The task ID fixes the action scale and the ramp, so the replay matches training.

Add `--goal 0.38442 0.01567 0.0124` to pin the goal to the rig's marker pose,
which is what the task files use; without it the goal randomizes over
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
uv run --no-sync vbrl-deploy configs/deployment/push_t.yaml --goal 2 --dry-run
uv run --no-sync vbrl-deploy configs/deployment/push_t.yaml --home
uv run --no-sync vbrl-deploy configs/deployment/push_t.yaml --park
```

`--dry-run` homes the arm, reads the sensors and evaluates the policy without
commanding any motion. `--home` holds the home pose and needs no camera; Ctrl-C
parks. `--park` brings the arm down and releases torque.

Arrow keys move the goal while a run is going: up/down moves z, left/right
moves y, 0.02 m per press, clamped to the training range. `--no-keyboard-goal`
turns that off, and it turns itself off with no terminal on stdin.

## Measuring the goal and the overlap

Collection does this by itself; by hand: one 1280x720 frame from the external camera
(a task file in place of the image captures it), arm at home, nothing else over the
table. With the T off the table, fit the goal: the footprint placed best in the hole
the green tape encloses, the tape allowed to sit a few millimetres too wide or narrow
all round. Saving it as a goal block's goal.json makes `--goal <n>` use it:

```bash
uv run --no-sync vbrl-measure push_t configs/deployment/push_t.yaml --save artifacts/deployment/goal.json --annotate artifacts/deployment/goal_fit.png
```

After an episode, fit the T and print its overlap with that goal, the footprint
overlap the simulator scores:

```bash
uv run --no-sync vbrl-measure push_t configs/deployment/push_t.yaml --goal artifacts/deployment/goal.json --annotate artifacts/deployment/episode_fit.png
```

A fit below IoU 0.8 is printed as UNRELIABLE.

## Exposure

`camera_exposure_us` and `camera_white_balance_k` belong to the scene the
policy was trained in, not to the room. The rig-table scene renders a narrow
brightness band, and a run below it fails completely while looking like a
policy failure: an earlier policy at 7000 us moved the T by 5.7 mm in 16 s, and
at 12000 us the same policy reached the goal. Auto white balance renders the
wood grey and the policies fail on it. Re-probe before a session rather than
trusting the numbers in rig.yaml:

```bash
uv run --no-sync python -m vbrl.deployment.camera
```
