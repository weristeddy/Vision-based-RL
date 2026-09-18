# Vision-Based RL

Visual RL for robot manipulation, built on [MJLab](https://github.com/mujocolab/mjlab).
Two tasks (Lift-Cube, Push-T), a Trossen WidowX AI arm, and a visual-encoder stack
that trains either a frozen pretrained backbone or a CNN from scratch.

**The task ID is the contract.** An ID such as
`Mjlab-PushT-VisualSlowStep-DinoV2ViTS14-Afa6-TrossenRealistic` fixes the task,
robot, scene, camera, encoder, adapter and network. Learning rate, seed and
iteration count are flags, so many runs share one ID.

## Install

```bash
uv sync                       # creates .venv from uv.lock
uv run vbrl-fetch-backbones   # once: DINOv2 + R3M into .models/
uv run pytest -m "not sim" -q # ~1 min, no simulator needed
```

Prefix every command with `uv run`, or `source .venv/bin/activate` once.

## Try it locally

```bash
vbrl-list                     # every task ID, architecture, scene, robot
vbrl-list architectures       # one section
```

Watch an untrained policy in the browser — no checkpoint needed, opens Viser:

```bash
vbrl-play Mjlab-PushT-State-TrossenRealistic --agent random
```

Train the state-based Push-T task (fast, no camera):

```bash
vbrl-train Mjlab-PushT-State-TrossenRealistic --agent.max-iterations 50
```

Train a visual policy, and record a video while it runs:

```bash
vbrl-train Mjlab-PushT-VisualSlowStep-DinoV2ViTS14-Afa6-TrossenRealistic \
  --env.scene.num-envs 512 --agent.max-iterations 100 --video True
```

Watch a trained policy, or record it to a file:

```bash
vbrl-play <TASK_ID> --agent trained --checkpoint-file ckpts/<run>/model.pt
vbrl-play <TASK_ID> --agent trained --checkpoint-file ... --record artifacts/run.mp4
```

`vbrl-play`, `vbrl-evaluate`, `vbrl-analyze` and `vbrl-export-onnx` load weights
from a local file **or** a W&B run, never both:

```bash
--checkpoint-file ckpts/lift_cube/dinov2.pt
--wandb-run-path entity/project/run_id [--wandb-checkpoint-name model_5998.pt]
```

Every run writes a deployable ONNX beside its checkpoint automatically.

## The commands

| | |
|---|---|
| `vbrl-list` | every registry, read live from its table |
| `vbrl-train` | train a task ID; `vbrl-train <ID> --help` lists all flags |
| `vbrl-play` | Viser viewer, or `--record out.mp4` / `.gif` |
| `vbrl-evaluate` | success rates over a model × scene × seed grid, from YAML |
| `vbrl-analyze` | probe/PCA/occlusion on what an encoder sees, from YAML |
| `vbrl-deploy` | run a policy on the real arm, from YAML |
| `vbrl-export-onnx` | re-export a checkpoint that predates automatic export |
| `vbrl-fetch-backbones` | download the pretrained backbones |

`vbrl-evaluate`, `vbrl-analyze` and `vbrl-deploy` are driven by YAML under
[`configs/`](configs/) — copy one and edit it.

```bash
vbrl-evaluate configs/evaluation/ood_4096.yaml
vbrl-analyze  configs/analysis/lift_cube_rgb/trossen/default.yaml
vbrl-deploy   configs/deployment/push_t_visualslowstep.yaml --dry-run
```

## Real robot

The hardware drivers are an extra, so a cluster or development sync never pulls
them:

```bash
uv sync --extra deploy
```

On the Jetson AGX Thor there is one more step `uv` cannot express — its sm_110
wheels bundle no CUDA, so the libraries need preloading and ONNX Runtime comes
from a separate index:

```bash
bash jetson/setup.sh          # preload + ONNX Runtime, then verify GPU kernels run
```

Run a policy from a manifest under [`configs/deployment/`](configs/deployment/).
Every run homes the arm first and parks it afterwards:

```bash
vbrl-deploy configs/deployment/push_t_visualslowstep.yaml
vbrl-deploy configs/deployment/lift_cube.yaml --dry-run       # no motion, sanity check
vbrl-deploy configs/deployment/lift_cube.yaml --max-steps 250
vbrl-deploy configs/deployment/lift_cube.yaml --log run.npz   # per-step trace
vbrl-deploy configs/deployment/lift_cube.yaml --home          # hold home and stop
vbrl-deploy configs/deployment/lift_cube.yaml --park          # rest and release torque
```

Arrow keys move the goal while it runs (`--no-keyboard-goal` disables). A manifest
names the ONNX graph, the arm IP, the goal and the motion limits; the policy's
observation and action contract travel inside the ONNX metadata, so deployment
never reads the task registry.

### Calibration

Only needed when a camera moves or the printed goal marker is repositioned:

```bash
python -m vbrl.deployment.intrinsics --save artifacts/deployment/intrinsics
python -m vbrl.deployment.camera                      # pick camera_exposure_us
python -m vbrl.deployment.capture_board --execute --arm-ip 192.168.1.2
python -m vbrl.deployment.recalibrate --capture artifacts/deployment/charuco
python -m vbrl.deployment.goal_pose frame.png --margin-mm 7.0
```

`recalibrate` prints an MJCF `<camera>` block to paste into the robot XML.
`goal_pose` turns a photo of the two AprilTags into the `goal:` numbers a manifest
needs — then take the tags off, because the policy must never see them.

## Cluster (Slurm)

Training runs inside a pinned Apptainer image, because the cluster has no Python
on the host. Build it once:

```bash
apptainer build --force rl.sif rl.def
```

Each launcher takes `--profile {a100,b200,testing,a100-single,b200-single}` and
passes everything after `--` through unchanged:

```bash
# Four GPUs; --env.scene.num-envs and --gpu-ids are set by the profile.
bash cluster/submit.sh --profile a100 -- \
  Mjlab-PushT-State-TrossenRealistic --agent.max-iterations 500

# One GPU; you own --env.scene.num-envs. ~1.6x the wall clock, a quarter of the GPUs.
bash cluster/submit.sh --profile a100-single -- \
  Mjlab-PushT-State-TrossenRealistic --env.scene.num-envs 4096

# One job per architecture in a variant.
bash cluster/submit_sweep.sh --variant Sim2Real --profile b200-single --dry-run

# Viewing and evaluation.
bash cluster/view.sh     --profile testing -- Mjlab-PushT-State-TrossenRealistic --agent random
bash cluster/evaluate.sh --profile testing -- configs/evaluation/ood_4096.yaml
```

Run any command inside the image directly with:

```bash
apptainer exec --nv --env "PYTHONPATH=$PWD/src" rl.sif <command>
```

Profiles live in [`cluster/profiles/`](cluster/profiles/) — review one before
launching on another cluster.

## Layout

```
src/vbrl/
├── scripts/      the CLIs
├── tasks/        lift_cube/, push_t/ — env cfg, mdp/, config/<robot>/
├── vision/       one registry table: 4 encoders x 5 adapters
├── scenes/       presets.py (data) → materials.py → builder.py
├── training/     VisualPPO + the runner that exports ONNX on save
├── evaluation/   rollout, report, video recording
├── analysis/     capture → features → probe/pca/occlusion/comparison
├── deployment/   the real-robot loop, camera, calibration
└── asset_zoo/    robot XML, objects, textures
```

Adding a task ID is one `register_mjlab_task()` line in
`tasks/<task>/config/<robot>/__init__.py`. Adding an encoder or adapter is one
row in [`vision/registry.py`](src/vbrl/vision/registry.py) plus one in
[`vision/architectures.py`](src/vbrl/vision/architectures.py).

## Tests

```bash
uv run pytest -m "not sim"   # contract tests, ~1 min
uv run pytest                # adds the tests that build and step a simulator
```
