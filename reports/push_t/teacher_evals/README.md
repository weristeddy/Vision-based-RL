# Push-T state teacher, three seeds

`Mjlab-PushT-State-Trossen`, 8k iterations, 8,192 envs × 4 ranks.

| seed | W&B run | commit |
| --- | --- | --- |
| 0 | `auxcyfuo` | `ae8a542` |
| 1 | `y19k4ry4` | `1cff6f2` |
| 2 | `6lsan1db` | `1cff6f2` |

The commits train the same recipe; between them Push-T only moved files into
`tasks/utils`.

Every 100 iterations each checkpoint is evaluated in the play config: home start,
mean action, 512 episodes, env seed 0, so all checkpoints and seeds see the same
start and goal draws. Overlap, position and yaw error are taken at the last step
of the episode; success means overlap ≥ 0.90 was reached at any point.

Caption: bold line, mean over three training seeds; thin lines, the individual
seeds. The mean's plateau at about 63 % between iterations 3,500 and 5,700 is not
a learning stage of any run: two seeds have learned and seed 1 is still at zero.

| final checkpoint | mean ± std over seeds |
| --- | --- |
| overlap | 94.4 ± 0.5 % |
| success | 95.8 ± 1.2 % |
| position error | 6.3 ± 0.2 mm |
| yaw error | 2.6 ± 0.1° |

Seeds 0 and 2 first exceed 50 % success at iteration 3,100 and 3,200, seed 1 at
6,100; the plateau of the mean between those is seed 1 still at zero.

- `learning_curves.{pdf,png}`: final overlap and success rate
- `errors.{pdf,png}`: final position and yaw error
- `*_curves.csv`: every evaluated checkpoint per run; `*_final.csv`: the table above

Regenerate:

```bash
uv run python -m vbrl.analysis.learning_curves --project eduard-nicolae-robot-learning/mjlab \
  --curve "State teacher=auxcyfuo,y19k4ry4,6lsan1db" --metric overlap --metric success \
  --name learning_curves --seeds --output reports/push_t/teacher_evals
uv run python -m vbrl.analysis.learning_curves --project eduard-nicolae-robot-learning/mjlab \
  --curve "State teacher=auxcyfuo,y19k4ry4,6lsan1db" --metric position_error --metric yaw_error \
  --name errors --seeds --output reports/push_t/teacher_evals
```

For the students, one `--curve` per encoder (three runs each) and the teacher as
`--reference`, drawn dashed at its final mean; leave out `--seeds` there, so each
encoder is its mean with the range between its lowest and highest seed shaded.
