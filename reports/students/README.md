# Students: final evaluation and learning curves (first view)

Push-T and Lift-Cube, six encoders, three seeds each; Push-T also DINOv3 (two more).
Push-Chain follows.

| encoder | Push-T runs (seeds 1000, 2000, 3000) | Lift-Cube runs |
| --- | --- | --- |
| DINOv2 + AFA | `ohkztpwp`, `q34wcoo7`, `omkgrm7i` | `oaq4s80l`, `qm999vka`, `buljwnnl` |
| DINOv2 global | `1fmoiz87`, `2e07q98c`, `686bg8dm` | `lxecudf0`, `v3zwmalb`, `q274r4ra` |
| R3M + AFA | `d4upzxez`, `a3tkw1b4`, `8hrb42a7` | `94dlia0s`, `zyiyjmnb`, `vqm70h53` |
| R3M global | `r7zhk0cc`, `1sn561i7`, `uvygjxty` | `csfdx04e`, `4md02xdm`, `himcw4vq` |
| Nature CNN | `lvvsamd6`, `pg149st9`, `jxmgv168` | `vwr1mvaf`, `5o1prtxa`, `rumte7b7` |
| CompactViT | `9ndoybcd`, `ka06fn9o`, `859y417j` | `xzshwd6v`, `qhqnoijf`, `swff3q1s` |
| DINOv3 + AFA | `5t6nx3qk`, `gbq6yqkg`, `fs8ruz1w` | not trained |
| DINOv3 global | `gbl6ao1n`, `ilt0hk04`, `ueg86l0f` | not trained |

Teachers: Push-T `auxcyfuo`, Lift-Cube `u6bm8899`. Students: DAgger distillation,
8,000 (Push-T) and 3,000 (Lift-Cube) iterations, 1,024 envs × 4 ranks, all trained on
`real_texture_wood`.

## Evaluation protocol

Final checkpoint, 512 episodes per scene, mean action, seed 0. Every scene is `wood049`
with one factor changed; everything else (light, camera pose, object and goal color
within the rig's range, friction) is randomized as in training and drawn from the same
seed, so the 512 episodes are identical across scenes.

| scene | change against `wood049` |
| --- | --- |
| In distribution (`wood049`) | a training wood texture near the bank's median |
| Rig table (`real_table`) | the rig's own tabletop photo (unseen, wood) |
| Marble table (`marble019`) | unseen material, brightness matched to training |
| Light color (`ood_light`) | table light fixed to RGB (0.08, 0.40, 0.45), Colosseum-style |
| Color swap (`colour_swap`) | Push-T: object green, goal bordeaux (each the other's training median). Lift-Cube: the goal is never in view, so only the cube turns green; labeled "Object green" there |
| Object color sweep (`object_color_sweep`) | object over a 256-color chart (15 hues x 16 shades + 16 grays), 2 episodes per color |
| Goal color sweep (`goal_color_sweep`, Push-T) | goal over the same chart |

## Files

- `final_evaluation_runs.csv`: one row per run, scene and metric (`task, encoder, run,
  seed, scene, metric, value`).
- `final_evaluation_summary.csv`: per task, encoder, scene and metric over the three
  seeds (`seeds, mean, std, min, max`). Filter `metric == success` for the LaTeX table.
- `<task>_overview.{pdf,png}`: success per encoder and scene, mean ± std (heatmap).
- `<task>_scenes.{pdf,png}`: the same per scene, with every seed shown.
- `<task>_learning_curves.{pdf,png}`: per-checkpoint evaluation over training, mean
  over seeds with the min-max range shaded, teacher's final value dashed.
  `*_curves.csv` and `*_final.csv` hold the numbers.

## Captions

The figures carry no titles; these go into `\caption{}`.

- `pusht_overview`: *Push-T, success rate (%) of the distilled students per evaluation
  scene, mean ± standard deviation over three seeds (512 episodes each). Every scene
  is the in-distribution table with one factor changed.*
- `pusht_overview_overlap`: *Push-T, final overlap with the goal (%) at the last step of
  the episode, mean ± standard deviation over three seeds, same scenes as the success
  rate.*
- `liftcube_overview`: *Lift-Cube, success rate (%) per evaluation scene, mean ± standard
  deviation over three seeds: the cube held within 5 cm of the goal at the end of the
  episode. The goal is outside the wrist camera's view, so the color swap only turns
  the cube green.*
- `<task>_scenes`: *Success rate per evaluation scene and encoder: mean (large marker),
  individual seeds (small markers) and their range (line).*
- `pusht_learning_curves`: *Push-T, success rate and final overlap over distillation,
  evaluated every 100 iterations in the training scene; line, mean over three seeds;
  band, range across seeds; dashed, the teacher's final value.*
- `liftcube_learning_curves`: *Lift-Cube, success rate and final position error of the
  cube over distillation; line, mean over three seeds; band, range across seeds; dashed,
  the teacher's final value.*

## Caption notes

- Success: Push-T, overlap with the goal ≥ 0.90 reached during the episode. Lift-Cube,
  as on the real robot (`held_at_goal`): at the last step the gripper is closing and
  blocked by the cube, and its TCP is within 5 cm of the goal. MJLab's own success
  (the cube within 5 cm at any step) stays in the CSVs as `success`; it counts a cube
  carried to the goal and dropped, which separates the encoders in the green-cube
  scene (R3M + AFA 70 % against 12 % held). Final overlap and position error at the
  last step.
- The learning curves evaluate every 100th checkpoint in the training scene
  (`real_texture_wood`, randomized table), the overview in the fixed scenes, so the
  final points differ slightly.
- Push-T DINOv2 + AFA (all seeds) and DINOv2 global (seeds 1000, 2000) were trained and
  their curves evaluated with the goal tape 2.4 mm proud; all other runs and every
  final scene evaluation use the flat tape (0.2 mm). Re-evaluated under the flat tape,
  DINOv2 + AFA gained 3-8 points.
- Lift-Cube success saturates for every encoder in distribution; the position error
  separates them there.

## Regenerate

```bash
uv run python -m vbrl.analysis.final_evaluation --output reports/students

uv run python -m vbrl.analysis.learning_curves --project eduard-nicolae-robot-learning/mjlab \
  --curve "DINOv2 + AFA=ohkztpwp,q34wcoo7,omkgrm7i" --curve "DINOv2 global=1fmoiz87,2e07q98c,686bg8dm" \
  --curve "R3M + AFA=d4upzxez,a3tkw1b4,8hrb42a7" --curve "R3M global=r7zhk0cc,1sn561i7,uvygjxty" \
  --curve "Nature CNN=lvvsamd6,pg149st9,jxmgv168" --curve "CompactViT=9ndoybcd,ka06fn9o,859y417j" \
  --curve "DINOv3 + AFA=5t6nx3qk,gbq6yqkg,fs8ruz1w" --curve "DINOv3 global=gbl6ao1n,ilt0hk04,ueg86l0f" \
  --reference "Teacher=auxcyfuo" --metric success --metric overlap \
  --name pusht_learning_curves --output reports/students

uv run python -m vbrl.analysis.learning_curves --project eduard-nicolae-robot-learning/mjlab \
  --curve "DINOv2 + AFA=oaq4s80l,qm999vka,buljwnnl" --curve "DINOv2 global=lxecudf0,v3zwmalb,q274r4ra" \
  --curve "R3M + AFA=94dlia0s,zyiyjmnb,vqm70h53" --curve "R3M global=csfdx04e,4md02xdm,himcw4vq" \
  --curve "Nature CNN=vwr1mvaf,5o1prtxa,rumte7b7" --curve "CompactViT=xzshwd6v,qhqnoijf,swff3q1s" \
  --reference "Teacher=u6bm8899" --metric success --metric position_error --ylim position_error=0,40 \
  --name liftcube_learning_curves --output reports/students
```

The encoder colors are fixed in `vbrl.analysis.final_evaluation.ENCODERS`; the learning
curves look them up by curve name, so the `--curve` order only sets the legend order.
