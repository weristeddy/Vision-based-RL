# Vision-Based RL

Robot-manipulation RL built on [MJLab](https://github.com/mujocolab/mjlab).
VBRL adds three manipulation tasks, two robots, a visual-encoder stack, and a
small PPO extension. Simulation, distributed launch, W&B logging, and
checkpoint handling are MJLab and RSL-RL doing their normal jobs.

**The task ID is the contract.** A registered ID such as
`Mjlab-PushT-RealTexture-DinoV2ViTS14-LocalGrid7-TrossenRealistic` fully
determines the task, robot, scene, camera, encoder, adapter, and network
architecture. It does *not* encode learning rate, seed, epochs, or iteration
count — those are flags, and many runs share one ID.

## Setup

Training runs inside the pinned Apptainer image (the cluster has no Python on
the host). It supplies MJLab 1.6.0, RSL-RL 5.4.2, MuJoCo 3.11, CUDA, and the
pretrained vision backbones, all installed from the committed `uv.lock`.

```bash
apptainer build --force rl.sif rl.def

# Prefix for every command below.
apptainer exec --nv \
  --env "PYTHONPATH=$PWD/src" --env "VBRL_REPO_ROOT=$PWD" \
  rl.sif <command>
```

On a **development workstation**, install the same stack natively with
[uv](https://docs.astral.sh/uv/) and drop the prefix:

```bash
uv sync                    # the environment, exactly as uv.lock pins it
uv run vbrl-fetch-backbones   # the pinned weights, into .models/
uv run vbrl-list tasks
```

`uv sync` reads `.python-version` and installs what `uv.lock` pins; `uv run`
executes inside `.venv` without activating it, and needs no environment variables
(`source .venv/bin/activate` works too, for an IDE or a plain shell). The image
builds from the same lockfile with `uv sync --frozen --no-dev`, so the two hosts
cannot drift. Change a dependency by editing `pyproject.toml`, running `uv lock`,
committing both, and rebuilding the image.

One GPU on a workstation against four on a cluster node, so smoke test here and
measure there. `uv run pytest -m "not sim"` is the fast contract pass;
`uv run pytest` adds [the tests that build and step a
simulator](tests/test_zero_shot_transfer.py) and is what to run before pushing.

### Jetson AGX Thor (robot deployment)

A Thor is the third target, for running trained policies on real hardware, and
it is the one host where `uv sync` alone is not enough:

```bash
bash jetson/setup.sh       # uv sync, install the loader preload, verify the GPU
```

Thor's GPU is **sm_110**, which the cu128 wheels the clusters use do not carry.
Installing them there succeeds and `torch.cuda.is_available()` returns `True`,
but every kernel launch fails with *no kernel image is available for execution
on the device* — so `pyproject.toml` routes `platform_machine == 'aarch64'` to
the CUDA 13 wheels on the [jetson-ai-lab index](https://pypi.jetson-ai-lab.io),
which do. Only the patch version differs (2.9.1 against the cluster's 2.9.0), so
checkpoints and module APIs are unchanged.

Those wheels link JetPack's system CUDA rather than bundling it, and expect NVPL
(ARM BLAS/LAPACK), cuDSS, and cuDNN from elsewhere. `uv sync` installs all
three from the lockfile, but into directories the dynamic loader does not
search, so [`jetson/_vbrl_jetson_preload.py`](jetson/_vbrl_jetson_preload.py)
loads them at interpreter startup. Installing that file is what `jetson/setup.sh`
adds over a plain sync; re-run it if you ever recreate `.venv`.
[`jetson/verify.py`](jetson/verify.py) is the gate — it launches real kernels
rather than trusting `is_available()`, and `bash jetson/setup.sh --no-sync`
re-runs it alone.

Two things the setup deliberately does not do. It does not touch the power mode,
which ships at 120 W of a possible MAXN (`sudo nvpmodel -m 0 && sudo
jetson_clocks`). And it installs no inference accelerator: TensorRT is present
system-wide and importable from the venv with
`PYTHONPATH=/usr/lib/python3.12/dist-packages`, and `triton` and
`torch-tensorrt` have aarch64 builds on the same index — but measure an eager
rollout against the control-rate budget before reaching for any of them, because
TensorRT's fp16 kernels change numerics and a policy driving real hardware is a
bad place to debug two things at once.

Then see what exists:

```bash
vbrl-list                  # every registry, and where to extend it
vbrl-list tasks            # just the task IDs
```

## The six commands

### `vbrl-list` — see every registry

Prints task IDs, architectures, encoders, adapters, scenes, robots, and analysis
steps, each read live from the table you would edit to add one. Start here.

```bash
vbrl-list                  # everything
vbrl-list architectures    # one section
```

### `vbrl-train` — train a policy

Every run is `vbrl-train <TASK_ID> [flags]`. `vbrl-train --help` prints usage
plus all 50 registered task IDs.

**Reading a task ID.** The scheme is `Mjlab-<Task>-<Variant>-<Arch>-<Robot>`:

```text
Mjlab-LiftCube-RealTexture-DinoV2ViTS14-LocalGrid7-Trossen
      └ task    └ variant   └ encoder    └ adapter  └ robot
```

- **Task** — `LiftCube`, `PushCube`, or `PushT`.
- **Variant** — what makes this environment distinct, which is normally the
  **training scene** (`RealTexture`, and any other preset you register).
  `SlowGoal` is the current Push-T generation: tilted camera, fixed-colour object
  on the photographic tabletop, and a goal yaw held fixed for 3,000 iterations
  before widening in 22.5° rungs.
  `State` marks a task with no camera at all. Defaults are silent: the camera
  renders the visual meshes unless the ID says otherwise, so the only other
  variant today is `CollisionCam` — see below.
- **Encoder + adapter** — present only on visual tasks. One of 4 backbones
  (`NatureCnn`, `CompactVit`, `DinoV2ViTS14`, `R3MResNet50`) paired with a head
  (`Linear`, `LocalGrid<N>`, `SpatialSoftmax`, `Afa<N>`). The registry in
  [`vision/registry.py`](src/vbrl/vision/registry.py) also has a `global` head
  that no registered ID currently uses.
- **Robot** — `Trossen` or `TrossenRealistic`.

**The one exception: `CollisionCam`.** Twelve Lift-Cube IDs render geom groups
`(0, 3)` — the collision proxies — instead of `(0, 2)`, the real meshes. That
was a mistake, caught after those checkpoints were trained; mjlab's own default
has always been `(0, 1, 2)`. They stay registered so the retained checkpoints
have a reproducible environment, and they are named so you can tell at a glance
that **their numbers are not comparable to a default-camera run**. Do not use
them for new work.

**What the ID fixes, and what you pass as flags.** The ID determines task,
robot, controller, scene, observations, camera, encoder, adapter, and
actor/critic architecture. It deliberately does *not* encode anything
run-specific, so many runs share one ID:

```bash
# Same architecture, three different runs.
vbrl-train Mjlab-PushCube-State-Trossen
vbrl-train Mjlab-PushCube-State-Trossen --agent.seed 1 --agent.max-iterations 3000
vbrl-train Mjlab-PushT-RealTexture-DinoV2ViTS14-LocalGrid7-TrossenRealistic \
  --agent.algorithm.learning-rate 2e-4 --env.scene.num-envs 1024 \
  --video True --gpu-ids all
```

Flags are nested Tyro paths over the two config trees — `--env.*` is the
MJLab environment, `--agent.*` is the RSL-RL runner. `vbrl-train <TASK_ID>
--help` prints all of them with defaults.

**Want a combination that is not registered?** Encoder, adapter, robot, and
scene are *not* flags — you add a task ID, which is a row in a table. See
[Extending the repo](#extending-the-repo). This is deliberate: the ID is what
makes a checkpoint reloadable years later without inspecting its W&B config.

**How this relates to MJLab.** [`src/vbrl/scripts/train.py`](src/vbrl/scripts/train.py)
is a copy of `mjlab.scripts.train` with the motion-tracking branch removed,
VBRL's task package imported, and the extra environment variables TorchrunX
workers need. So every MJLab/Tyro flag works unchanged, MJLab's own tasks are
selectable here too, and RSL-RL's W&B writer records `env_cfg`, `train_cfg`,
metrics, and uploads `model_N.pt`.

### `vbrl-visualize` — watch a policy in the browser

Wraps MJLab's `ViserPlayViewer` and serves it on a port you choose.

```bash
# The retained Lift-Cube checkpoints were trained on collision geometry, so
# they must be replayed against the CollisionCam ID.
vbrl-visualize Mjlab-LiftCube-CollisionCam-DinoV2ViTS14-LocalGrid7-Trossen \
  --checkpoint-file ckpts/lift_cube/dinov2_vits14_local_grid7_real.pt \
  --port 8080

# No checkpoint needed to inspect a task:
vbrl-visualize Mjlab-PushCube-State-Trossen --agent random
```

Add `--scene wood|plaster|peacock` to swap in an out-of-distribution tabletop.

### `vbrl-deploy` — run a policy on the real robot

VBRL-only, and only on the Jetson (`uv sync --extra deploy`). Reads a manifest
and drives a Trossen arm at the task's own control rate.

```bash
uv run vbrl-deploy configs/deployment/lift_cube.yaml --dry-run   # no arm commands
uv run vbrl-deploy configs/deployment/lift_cube.yaml
```

The task ID fixes the architecture, observations, camera, and action scaling, so
the manifest adds only what a real robot needs: the arm's address, the camera
serial, the lift target, and the per-step safety clamps the simulator never
needed (the action term declares `clip: None`, which is safe only in sim).

**The lift target is a parameter, and the cube's position is not.** The training
command sampled the target independently of the cube, uniformly over
`x(0.3, 0.5), y(-0.2, 0.2), z(0.2, 0.4)` in the base frame; a target outside
that box is refused as out of distribution. The cube is never configured at all
— the policy locates it through the camera alone, which is the whole point of a
visual task.

Deployment builds one simulated environment and never steps it, because the
constants hardware must reproduce — joint order, the default pose observations
are relative to, the action offset and scale, and the kinematics behind
`goal_position` — are derived in the sim from the same MJCF the policy trained
against. `tests/test_deployment_parity.py` feeds the simulator's own state
through the deployment assembler and demands the identical observation back;
that test is the reason to trust the loop before an arm moves.

### `vbrl-evaluate` — success rates over a model × scene × seed grid

VBRL-only. Reads a YAML and writes `episodes.csv`, `summary.csv`, and a plot.

```bash
vbrl-evaluate configs/evaluation/ood_4096.yaml
```

```yaml
version: 1
name: ood
models:
  - name: dinov2_local_grid
    task_id: Mjlab-LiftCube-CollisionCam-DinoV2ViTS14-LocalGrid7-Trossen
    checkpoint_file: ckpts/lift_cube/dinov2_vits14_local_grid7_real.pt
scenes: [wood_fixed, plaster_matched, peacock]
episodes: 64
seeds: [10, 20, 30]
output: artifacts/evaluation/ood
```

### `vbrl-analyze` — inspect what the encoder sees

VBRL-only. Runs an ordered list of steps against one checkpoint, building the
environment at most once. Available steps are the keys of `STEPS` in
[`scripts/analyze.py`](src/vbrl/scripts/analyze.py): `capture`, `features`,
`probe`, `pca`, `occlusion`, `comparison`.

```bash
vbrl-analyze lift_cube_rgb/trossen/default.yaml --device cuda:0
```

```yaml
version: 1
task_id: Mjlab-LiftCube-CollisionCam-DinoV2ViTS14-LocalGrid7-Trossen
checkpoint_file: ckpts/lift_cube/dinov2_vits14_local_grid7_real.pt
output: artifacts/analysis/default
steps:
  - script: capture
    args: {output: capture.npz, num_frames: 256, targets: [cube_position]}
  - script: features
    args: {capture: capture.npz, output: features.npz, stages: [adapter]}
  - script: probe
    args:
      capture: capture.npz
      features: features.npz
      output: probes/{stage}__{target}.npz
      stages: [adapter]
      targets: [cube_position]
```

### Loading weights

Every consumer takes weights from **exactly one** of a local file or a W&B run:

```yaml
checkpoint_file: ckpts/lift_cube/dinov2_vits14_local_grid7_real.pt
# ...or...
wandb_run_path: entity/project/run_id
wandb_checkpoint_name: model_2999.pt   # optional
```

The task ID rebuilds the architecture; the reference supplies only weights.

## Sweeps

[`configs/sweeps/`](configs/sweeps/) holds plain W&B sweeps — no VBRL
translation layer. Each fixes one task ID in `command` and varies only native
Tyro flag names, **with hyphens**:

```yaml
program: vbrl.scripts.train
method: grid
command:
  - ${env}
  - python
  - -m
  - ${program}
  - Mjlab-PushCube-State-Trossen
  - --env.scene.num-envs
  - "1024"
  - ${args_no_boolean_flags}
parameters:
  agent.seed: {values: [0, 1000, 2000]}
  agent.algorithm.learning-rate: {value: 0.0008938003178818718}
```

```bash
wandb sweep configs/sweeps/dinov2_vits14_local_grid.yaml
```

Two rules: verify a new flag with `vbrl-train <TASK_ID> --help` first, and
never sweep architecture — encoder, adapter, camera, and scene changes select a
*different registered task ID*.

## Cluster (Slurm)

Skip this section if you are not on a cluster. Each launcher takes
`--profile {a100,b200,testing,a100-single,b200-single}` and passes everything
after `--` through unchanged.

```bash
# Train on four GPUs. --env.scene.num-envs and --gpu-ids are appended.
bash cluster/submit.sh --profile a100 -- \
  Mjlab-PushCube-State-Trossen --agent.max-iterations 500

# Train on one GPU. Same job, roughly 1.6x the wall clock, a quarter of the GPUs.
bash cluster/submit.sh --profile a100-single -- \
  Mjlab-PushCube-State-Trossen --agent.max-iterations 500 --env.scene.num-envs 4096

# Run a sweep agent instead.
bash cluster/submit.sh --profile a100 --sweep-agent entity/mjlab/sweep-id

# One GPU for viewing or evaluating.
bash cluster/view.sh --profile testing -- Mjlab-PushCube-State-Trossen --agent random
bash cluster/evaluate.sh --profile testing -- configs/evaluation/ood_4096.yaml
```

`cluster/submit_sweep.sh` submits one job per architecture in a variant and
takes `--arch` / `--exclude` (comma-separated substrings) to narrow that set.
Narrowing belongs there rather than in the task registry: a task ID has to stay
registered for its existing checkpoints to remain loadable, so unregistering an
architecture to shrink a sweep would orphan every run already trained under it.

```bash
# Every architecture in the variant.
bash cluster/submit_sweep.sh --variant SlowGoal --profile b200-single

# Drop R3M layer 4 and keep layer 3 -- the trailing hyphen is what
# distinguishes "R3MResNet50-" from "R3MResNet50L3-".
bash cluster/submit_sweep.sh --variant SlowGoal --profile b200-single \
  --exclude "R3MResNet50-"
```

Profiles in [`cluster/profiles/`](cluster/profiles/) hold the partition,
account, and resources — review one before launching on another cluster. A
profile's `GPU_COUNT` selects the topology and only `1` and `4` are accepted;
both run through [`cluster/train.slurm`](cluster/train.slurm), since mjlab's
`train.py` skips torchrunx entirely at `num_gpus <= 1`.

**The two topologies are equivalent, and differ over one flag.** RSL-RL
all-reduces gradients and divides by the world size, so one rank at 4,096
environments performs the same 2,048-sample updates as four ranks at 1,024. A
four-GPU profile therefore *owns* `--env.scene.num-envs` and pins 1,024 per
rank: the per-rank count is topology, not tuning, because changing it silently
changes the effective batch every rank contributes to. A `*-single` profile
defaults to 1,024 and lets an override through, because with one rank the
environment count *is* the batch. Both refuse a user-supplied `--gpu-ids`, since a job must not reach outside
its allocation. The four-GPU path passes `--gpu-ids all`; the single-GPU path
passes nothing, matching mjlab's own single-GPU command — the default `[0]`
indexes into `CUDA_VISIBLE_DEVICES`, which Slurm has already narrowed to the
one granted GPU.

## Extending the repo

**Contracts are Python tables. Runs are YAML.**

Anything a checkpoint must be reloadable against — task, robot, scene, encoder,
adapter — is a row in a Python table: importable, type-checked, and pinned by a
test. Anything describing one experiment — which checkpoints to evaluate, which
analysis steps to run, which hyperparameters to sweep — is YAML. That is why a
new encoder is a Python row while a new evaluation is a new `.yaml`.

Every table is one place, and `vbrl-list` prints all of them:

| To add a… | Edit | Picked up by |
|---|---|---|
| Encoder or adapter | `ENCODERS` / `ADAPTERS` in [vision/registry.py](src/vbrl/vision/registry.py) | `build_encoder`, everything downstream |
| Named architecture | `ARCHITECTURES` in [vision/architectures.py](src/vbrl/vision/architectures.py) | every task that crosses the table |
| Scene / material bank / DR | `_PRESETS` in [scenes/presets.py](src/vbrl/scenes/presets.py) | `apply_scene`, `replace_scene`, `--scene` |
| Robot | `ROBOTS` in [asset_zoo/robots/__init__.py](src/vbrl/asset_zoo/robots/__init__.py) + a package beside it | `get_robot`, task `env_cfgs.py` |
| Analysis step | `STEPS` in [scripts/analyze.py](src/vbrl/scripts/analyze.py) | `vbrl-analyze` manifests |
| Task ID | one line in `tasks/<task>/config/<robot>/__init__.py` | the MJLab registry, so all four scripts |
| Task | a new `tasks/<task>/` package | `import_packages()` at import time |
| Evaluation / analysis / sweep run | a file in [configs/](configs/) | `vbrl-evaluate` / `vbrl-analyze` / `wandb sweep` |

### Worked example: a new encoder, end to end

1. Add the backbone module under [vision/backbones/](src/vbrl/vision/backbones/)
   exposing `load()`.
2. Add one row to `ENCODERS` in `vision/registry.py` — channels, weights,
   whether it trains, and its builder.
3. Add one row to `ARCHITECTURES` naming the pairing you want, e.g.
   `"MyEncoder-LocalGrid7": vision_cfg("my_encoder", "local_grid", target_grid_size=7)`.
4. It now appears in `vbrl-list`. Give it a task ID by adding one registration
   line to the task you want it on.
5. `pytest` will tell you to extend `tests/data/vision_checkpoint_layout.json`
   with the new architecture — that snapshot is what stops a later refactor
   from silently breaking checkpoint keys.

### Worked example: registering a task ID

Every ID is written out literally, so you can grep any task ID and land on the
single line that creates it. The per-scene helper binds the environment; the
architecture token keys `ARCHITECTURES`:

```python
# tasks/push_t/config/trossen_realistic/__init__.py
_real_texture(
  "Mjlab-PushT-RealTexture-DinoV2ViTS14-LocalGrid7-TrossenRealistic",
  "DinoV2ViTS14-LocalGrid7",
)
_default(
  "Mjlab-PushT-Default-DinoV2ViTS14-LocalGrid7-TrossenRealistic",
  "DinoV2ViTS14-LocalGrid7",
)
```

Adding a row to `ARCHITECTURES` therefore does **not** register anything by
itself — you choose which tasks get it. `test_the_architecture_table_and_the_registry_cannot_drift`
fails if a table row never reaches a task ID, so a forgotten registration is
caught rather than silently ignored.

The task's `rl_cfg.py` owns the hyperparameters (they differ per task); the
shared table owns the architectures. Register only what you will actually
train — Lift-Cube's `COLLISION_CAM_ARCHITECTURES` is deliberately frozen at the
12 its checkpoints were trained with, rather than following the table.

### Where the scene lives

Scenes work differently on purpose, because you pick one at training time and
compare against others afterwards:

- **Training scene is part of the task ID**, and it is the variant token.
  `default` is the fixed-colour baseline; `RealTexture` swaps in 1203
  photographic tabletops. Both randomize lighting and camera pose identically,
  so a pair differs only in appearance. Every task's `env_cfgs.py` takes a
  `scene` argument, so training on another preset is a one-line registration —
  `Mjlab-LiftCube-Procedural-…` — not a change to the task.
- **A task with no camera gets no visual randomization at all**, whatever scene
  it names: `apply_scene` passes `camera_model=None` and `_events` returns
  empty. That is what keeps the `State` tasks free of dead lighting terms.
- **Evaluation and analysis swap scenes at runtime** through `replace_scene`:
  `--scene wood` on `vbrl-visualize`, `scenes:` in an evaluation YAML, `scene:`
  plus `eval_dr:` in an analysis manifest. No retraining, no new ID.
- Only OOD presets can be swapped in (`replace_scene` passes
  `require_ood=True`), which today means `wood`, `plaster`, and `peacock`. Mark
  a preset `ood=True` in `_PRESETS` to make it a valid evaluation target.

### Layout

```text
src/vbrl/
├── tasks/<task>/          <task>_env_cfg.py, mdp/, config/<robot>/
├── asset_zoo/             robots, objects, textures
├── scenes/                presets.py (data) → materials.py → builder.py
├── vision/                registry.py (capabilities) + architectures.py (named pairings)
├── training/ppo.py        VisualPPO
├── runtime.py             shared env + checkpoint loading
├── scripts/               list, train, evaluate, analyze, play
├── evaluation/            rollouts and reports
└── analysis/              capture → features → probe/pca/occlusion
```

A whole new *task* is the same shape as the existing ones: a robot-agnostic
`<task>_env_cfg.py`, term functions under `mdp/`, and a `config/<robot>/`
package that registers its IDs.

## Tests

```bash
apptainer exec --env "PYTHONPATH=$PWD/src" --env "VBRL_REPO_ROOT=$PWD" \
  rl.sif python -m pytest -q
```
