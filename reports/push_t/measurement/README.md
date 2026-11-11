# Push-T rig measurement against simulator ground truth

The real-world evaluation measures the goal from the green outline and the T after
each episode, from one external-camera frame with the arm at home
(`vbrl.deployment.push_t.measure_goal`, `measure_overlap`), and scores the same
footprint overlap as the simulator. This benchmark runs exactly that code on
rendered frames whose poses are known.

```bash
uv run python -m vbrl.analysis.push_t_measurement            # 512 samples, ~10 min
uv run python -m vbrl.analysis.push_t_measurement --replot   # figures from samples.csv
```

## Setup

- Scene: `Mjlab-PushT-WoodTable-DinoV2ViTS14-AfaFull6-Trossen` play config, AmbientCG
  wood tabletops and light pose and colour randomised as in training, arm at home.
- Camera: the calibrated external camera pose, rendered at 1280 × 720 with the
  D405's focal length at that mode (pinhole, no distortion), camera randomisation off.
- T and outline at the centre of their measured colour ranges: the rig has one T
  and one marker, not the training spread. The outline is drawn 0.2 mm above the
  table like the rig's print, not as the simulator's 2.4 mm slab.
- Per sample: goal uniform over x 0.2–0.5 m, y ±0.2 m, the full circle; the T at the
  goal plus a Gaussian offset (30 mm, 30 mm, 30°) scaled by U(0, 1), so the true
  overlap covers 0–100 %. One frame without the T gives the goal, one with it the T.
- A fit counts only when its silhouette IoU reaches the threshold
  (`MIN_T_IOU`, `MIN_OUTLINE_IOU`); a rejected frame is a failed measurement, not
  an error.

## Results (512 samples, seed 0)

| | accepted | x bias ± σ | y bias ± σ | yaw bias ± σ | footprint displacement median / 95 % |
| --- | --- | --- | --- | --- | --- |
| goal outline | 508 | −0.26 ± 0.22 mm | −0.02 ± 0.16 mm | +0.01 ± 0.14° | 0.31 / 0.67 mm |
| T | 497 | −0.16 ± 0.32 mm | +0.00 ± 0.48 mm | +0.03 ± 0.30° | 0.35 / 0.97 mm |

Overlap, 493 samples with both fits accepted: mean absolute error 0.50 pp, median
0.27 pp, 95 % 1.68 pp, maximum 8.9 pp. Success at 90 % changed in 6 of 493
samples, all with a true overlap between 89.3 % and 90.7 %.

Of the 15 rejected T frames, 9 had no T pixels at all: one AmbientCG texture is a
red-stained wood whose chromaticity equals the T's. The rig's table is light wood.
The outline's −0.26 mm x bias is the 0.2 mm height of the drawn outline, which the
fit takes as flat.

## Figures

**`measurement_summary`** (thesis figure). Mean error ± standard deviation of the
outline and the T fits against the simulator's ground truth: (a) the xy distance
between fitted and true pose, (b) the absolute yaw difference; n counts the fits
accepted out of the 512 scenes. The overlap the
two poses give is off by 0.50 pp on average, and success at 90 % came out
differently in 6 of 493 samples, all within 0.7 pp of the threshold. (c) A rig
frame (the policy's 224 × 224 crop of the 424 × 240 stream, so about 2.8 mm per
pixel) with the T fit, the outline fit and the goal the manifest holds from the
AprilTags. Repeatability is the standard deviation over the 59 frames of
`wood_table_cnn_1`, in which the T never moves.

**`measurement_benchmark`** (detail). (a–f) Signed error, fitted minus true, of x, y
and yaw for the outline and the T, against the true x (distance from the robot
base; the camera sits at x = 0.81 m). Shaded band: bias ± 2σ. (g) Signed overlap
error against the true overlap; orange, the samples whose success at 90 % changed.
(h) As (c) above.

The rig frame shows the manifest's AprilTag goal 7.9 mm from the printed outline the
policy sees. The outline fit measures the marker itself. The old AprilTag solver
defaulted to a camera file that no longer exists, so the stored goal may come from
an earlier calibration rather than from the tags themselves. A 1280 × 720 frame
with the tags seated, solved with both methods and the current calibration,
separates the two.
