# Vision-Based RL

Visual encoders for sim-to-real manipulation on a Trossen WidowX AI arm, built on
[MJLab](https://github.com/mujocolab/mjlab). Three tasks of increasing visual demand
(Lift-Cube, Push-T, Push-Chain). Students trained from scratch (Nature CNN, compact
ViT) or on frozen DINOv2, DINOv3 and R3M, with a global readout or AFA, are
distilled from one state teacher per task.

**The task ID is the contract.** `Mjlab-PushT-WoodTable-DinoV2ViTS14-AfaFull6-Trossen`
fixes task, robot, scene, camera, encoder, adapter, network and training recipe.
Seed, env count, iterations, video and logging are flags.

## Install

```bash
uv sync                       # .venv from uv.lock
uv run vbrl-fetch-backbones   # once: DINOv2, DINOv3, R3M into .models/
uv run pytest -m "not sim" -q # ~1 min, no simulator
```

Run every command with `uv run`, or `source .venv/bin/activate` once.

## Tasks

| task | teacher | students | camera |
| --- | --- | --- | --- |
| Lift-Cube | `Mjlab-LiftCube-State-Trossen`, 1,500 it. | `Mjlab-LiftCube-WoodTable-<Arch>-Trossen`, 3,000 it. | wrist |
| Push-T | `Mjlab-PushT-State-Trossen`, 8,000 it. | `Mjlab-PushT-WoodTable-<Arch>-Trossen`, 8,000 it. | external |
| Push-Chain | `Mjlab-PushChain3-State-Trossen`, 20,000 it. | `Mjlab-PushChain3-WoodTable-<Arch>-Trossen`, 15,000 it. | external |

`<Arch>` is one of `NatureCnn-FlattenRelu`, `CompactVit-Svea`,
`DinoV2ViTS14-{Global,AfaFull6}`, `R3MResNet50-{Global,AfaFull32}`; Push-T adds
`DinoV3ViTS16-{Global,AfaFull6}` and `R3MResNet50L3-AfaFull16`. `vbrl-list` prints
every registered ID.

## Local

```bash
vbrl-list                                            # task IDs, architectures, scenes
vbrl-play Mjlab-PushT-State-Trossen --agent random   # Viser viewer, no checkpoint

# A short teacher run, then a student distilled from it.
vbrl-train Mjlab-PushT-State-Trossen --env.scene.num-envs 512 --agent.max-iterations 50
vbrl-train Mjlab-PushT-WoodTable-DinoV2ViTS14-AfaFull6-Trossen --env.scene.num-envs 64 \
  --agent.resume True --wandb-run-path <entity>/mjlab/<teacher run>

# Watch or record a trained policy.
vbrl-play <TASK_ID> --agent trained --wandb-run-path <entity>/mjlab/<run>
vbrl-play <TASK_ID> --agent trained --checkpoint-file <model.pt> --record artifacts/run.mp4
```

After training, `vbrl-train` evaluates every saved checkpoint (training curve,
`eval/*`), evaluates the final one in all evaluation scenes (`final/*`, color sweeps
as tables), deletes the intermediate checkpoints and uploads the final checkpoint
and its ONNX to W&B. `--evaluate False` skips this. The same evaluation by hand:

```bash
vbrl-evaluate <TASK_ID> --wandb-run-path <entity>/mjlab/<run> --log   # final scenes into the run
vbrl-evaluate <TASK_ID> --run-dir logs/rsl_rl/<experiment>/<run>      # curve and final scenes
vbrl-evaluate <TASK_ID> --checkpoint <model.pt> --scenes wood049 marble019
```

## Cluster (Slurm)

Everything runs inside the pinned Apptainer image; the cluster has no Python on the
host. Build it once, and again after a dependency change:

```bash
apptainer build --force rl.sif rl.def
```

`cluster/submit.sh --profile {a100,b200,testing,a100-single,b200-single}` passes
everything after `--` to `vbrl-train`. Four-GPU profiles run 1,024 envs per rank
unless `--env.scene.num-envs` says otherwise.

```bash
# Teacher, three seeds one after another in one job.
bash cluster/submit.sh --profile b200 --seeds 3 -- \
  Mjlab-PushT-State-Trossen --env.scene.num-envs 8192 --label pusht-teacher

# Students of several architectures, three seeds each, from the best teacher seed.
bash cluster/submit.sh --profile b200 --seeds 3 -- \
  Mjlab-PushT-WoodTable-DinoV2ViTS14-AfaFull6-Trossen,Mjlab-PushT-WoodTable-DinoV2ViTS14-Global-Trossen \
  --agent.resume True --wandb-run-path <entity>/mjlab/<teacher run> --label distill-<teacher run>

# Push-Chain with its own iteration count.
bash cluster/submit.sh --profile b200 --seeds 3 -- \
  Mjlab-PushChain3-State-Trossen --env.scene.num-envs 8192 \
  --agent.max-iterations 20000 --label pushchain3-teacher

# One job per architecture of a variant.
bash cluster/submit_sweep.sh --variant WoodTable --arch PushT --dry-run \
  -- --agent.resume True --wandb-run-path <entity>/mjlab/<teacher run>

# A viewer, or any command inside the image.
bash cluster/view.sh --profile testing -- Mjlab-PushT-State-Trossen --agent random
apptainer exec --nv --env "PYTHONPATH=$PWD/src" rl.sif <command>
```

Pull new code on the cluster only while none of your jobs runs: a job reads the
checkout, and its evaluation imports modules after training.

## Real robot

```bash
uv sync --extra deploy        # hardware drivers
bash jetson/setup.sh          # Jetson AGX Thor only: CUDA preload, ONNX Runtime, GPU check
```

One file per task in [`configs/deployment/`](configs/deployment/) (README there has the details)
lists its camera, its policies (the best seed of each) and its goals; `rig.yaml`
holds the arm and both cameras. Every tool reads the task file:

```bash
python -m vbrl.deployment.fetch_policies configs/deployment/push_t.yaml       # ckpts/deployment/push_t/
vbrl-deploy configs/deployment/push_t.yaml --policy R3MResNet50L3-AfaFull16 --goal 2
vbrl-deploy configs/deployment/lift_cube.yaml --policy NatureCnn-FlattenRelu --goal 1 --dry-run
vbrl-deploy configs/deployment/lift_cube.yaml --home                           # hold home and stop
python -m vbrl.deployment.collect configs/deployment/push_t.yaml --goal 2     # evaluation block
```

Object and goal poses from an external-camera frame, the same fit for every task.
The goal is measured once from a frame without the object:

```bash
vbrl-measure push_t goal_frame.png --save goal.json
vbrl-measure push_t end_frame.png --goal goal.json        # pose and overlap
vbrl-measure push_chain frame.png --links 3 --goal goal.json
vbrl-measure lift_cube frame.png
```

## Figures

```bash
uv run python -m vbrl.analysis.final_evaluation --output reports/students
uv run python -m vbrl.analysis.real_robot           # reports/real
uv run python reports/eval_scenes/render_eval_scenes.py reports/eval_scenes/figures/scenes
```

Learning curves: see [`reports/students/README.md`](reports/students/README.md).

## Feature analysis

```bash
# A real frame's state (joints at the frame's exposure, measured object and goal) in sim scenes.
uv run python -m vbrl.analysis.pairs artifacts/real_eval/push_t/goal1/DinoV2ViTS14-AfaFull6/run4 \
  --step 16 --task-id Mjlab-PushT-WoodTable-DinoV2ViTS14-AfaFull6-Trossen
# Images once per task, then probes, Grad-CAM and feature distance per deployed student.
uv run python -m vbrl.analysis.dataset Mjlab-PushT-WoodTable-DinoV2ViTS14-AfaFull6-Trossen --run <its W&B run>
uv run python -m vbrl.analysis.feature_analysis push_t    # artifacts/features/push_t/measures.csv
```

## Layout

```
src/vbrl/
├── scripts/      the CLIs
├── tasks/        lift_cube/, push_t/, push_chain/: env cfg, mdp/, config/<robot>/
├── vision/       encoders x adapters in one table; architectures.py names them
├── scenes/       presets.py (data) → materials.py → builder.py
├── training/     VisualPPO, distillation, the runner that exports ONNX on save
├── evaluation/   rollout, checkpoint evaluation, video recording
├── analysis/     thesis figures
├── deployment/   real-robot loop, policy, camera, data collection
├── rig/          camera calibration, external view, object and goal measurement
└── asset_zoo/    robot XML, objects, textures
```

A task ID is one `register_mjlab_task()` line in
`tasks/<task>/config/<robot>/__init__.py`; an encoder or adapter is one row in
[`vision/registry.py`](src/vbrl/vision/registry.py) plus one in
[`vision/architectures.py`](src/vbrl/vision/architectures.py).

## Tests

```bash
uv run pytest -m "not sim"   # contract tests, ~1 min
uv run pytest                # adds the tests that build and step a simulator
```
