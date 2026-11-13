# Push-T on the real robot

Five episodes of the DINOv2 + AFA wood student (`rzf11em2`, iteration 7,999) on the
rig after the camera recalibration of 2026-11-11, 400 steps (20 s) each, goal
`[0.38587, 0.01701, 0.0273]` measured from the taped outline. Each final state is
measured from one 1280 × 720 external frame with the arm parked, as in
`../measurement/`.

```bash
uv run --no-sync python -m vbrl.analysis.push_t_rig_runs artifacts/rig_runs/calibrated \
  --title "Push-T on the real robot: DINOv2 + AFA student, 5 runs"
```

| run | Δx | Δy | Δyaw | final overlap |
| --- | --- | --- | --- | --- |
| 01 | −7.0 mm | +0.4 mm | −0.35° | 88.6 % |
| 02 | −8.7 mm | +5.3 mm | +2.52° | 75.7 % |
| 03 | −8.7 mm | +0.5 mm | +0.53° | 83.2 % |
| 04 | −5.6 mm | +0.5 mm | +0.65° | 88.9 % |
| 05 | −5.3 mm | +6.5 mm | −7.24° | 84.2 % |

Mean overlap 84.1 %, none at the 90 % success threshold. Every run ends 5–9 mm short
in x, toward the robot (mean −7.1 mm); in simulation, at the same goal over 256
episodes, the same student ends −1.7 mm short and succeeds in 73 %. With the overlap
counted on 2.8 mm raster cells, a pure shift of 3–5 mm gives 94.3 % and one of
6–7 mm 88.6 %.

**`rig_runs`**: per run the start (policy camera, 224 × 224) and the end (crop of the
1280 × 720 frame) without and with the goal footprint and the T fit; (a) final
overlap, dashed the success threshold; (b) final offset of the T from the goal, the
shaded square ±5 mm; (c) yaw error. `rig_runs.csv` holds the numbers.
