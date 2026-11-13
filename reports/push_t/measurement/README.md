# Push-T rig measurement against simulator ground truth

The real-world evaluation measures the goal from the taped outline and the T after
each episode, from one external-camera frame with the arm at home
(`vbrl.deployment.push_t.measure_goal`, `measure_overlap`), and scores the same
footprint overlap as the simulator. This benchmark runs exactly that code on
rendered frames whose poses are known.

```bash
uv run python -m vbrl.analysis.push_t_measurement --scene real_table   # ~10 min
uv run python -m vbrl.analysis.push_t_measurement --scene wood
uv run python -m vbrl.analysis.push_t_measurement --scene real_table --replot
```

## Method

The T and the goal are each found by comparing two masks. The observed mask comes
from colour: pixels whose green chromaticity lies 0.04 below the table's median are
the T, 0.04 above it the green tape, whose enclosed hole is the goal. The predicted
mask comes from the model: the T as a 24 mm prism, or the goal as the flat
footprint with the tape's offset from the T's edge as a fourth parameter, projected
through the calibrated camera onto the table plane at a candidate pose (x, y, yaw).
A sweep over yaw in 10° steps and a Nelder–Mead search find the pose with the
highest IoU between the two masks; below 0.8 the measurement counts as failed. The
overlap then comes from the two poses, on the simulator's 64 × 64 raster.

## Setup

- Scenes: the play config of `Mjlab-PushT-RealTable-DinoV2ViTS14-AfaFull6-Trossen`
  (the rig's tabletop photo) and of the `WoodTable` task (153 AmbientCG woods), light
  pose and colour randomised as in training, no shadows (the rig's room light casts
  almost none), arm at home.
- Camera: the external camera at its calibrated pose, rendered at 1280 × 720 with the
  D405's focal length at that mode (pinhole, no distortion), camera randomisation
  off. The simulated table is flat at −5 mm.
- T and outline at the centre of their measured colour ranges: the rig has one T and
  one marker, not the training spread. The outline is drawn 0.2 mm above the table
  like the rig's tape, not as the simulator's 2.4 mm slab.
- Per sample: goal uniform over x 0.2–0.5 m, y ±0.2 m, the full circle; the T at the
  goal plus a Gaussian offset (30 mm, 30 mm, 30°) scaled by U(0, 1), so the true
  overlap covers 0–100 %. One frame without the T gives the goal, one with it the T.

## Results (512 samples each, seed 0)

| scene | | accepted | x bias ± σ | y bias ± σ | yaw bias ± σ |
| --- | --- | --- | --- | --- | --- |
| real_table | goal | 512 | −0.10 ± 0.12 mm | −0.04 ± 0.10 mm | +0.00 ± 0.10° |
| real_table | T | 512 | −0.12 ± 0.27 mm | −0.01 ± 0.20 mm | +0.03 ± 0.25° |
| wood | goal | 512 | −0.11 ± 0.12 mm | −0.04 ± 0.10 mm | +0.00 ± 0.10° |
| wood | T | 503 | −0.14 ± 0.26 mm | −0.01 ± 0.20 mm | +0.03 ± 0.26° |

As distances (`measurement_summary`): goal 0.17 ± 0.10 mm and 0.07 ± 0.06°, T
0.31 ± 0.18 mm and 0.20 ± 0.16°, on both scenes.

Overlap from the two fitted poses: mean absolute error 0.33 pp on both scenes
(median 0.27 pp, maximum 2.0 pp). Success at 90 % changed in 6 of 512 (real_table)
and 7 of 503 (wood) samples, all with a true overlap between 89.2 % and 90.7 %.

The 9 rejected T frames on wood are one AmbientCG texture, a red-stained wood whose
chromaticity equals the T's; the rig's table is light wood. The goal's −0.1 mm x
bias is the 0.2 mm height of the drawn outline, which the fit takes as flat.

## Figures

Each scene folder holds the same two figures.

**`measurement_summary`** (thesis figure). (a, b) Mean error ± standard deviation of
the goal and the T fits against the simulator's ground truth, as the xy distance and
the absolute yaw difference; n counts the accepted fits. (c) The rig: a crop of a
1280 × 720 frame (`rig_frame.png`) measured with the current calibration and table
plane, with the T fit, the goal fit and the goal the AprilTags gave before
2026-11-11. That goal is 2.3 mm and 0.77° from the goal fit; it was solved with the
earlier camera calibration, so part of the difference is that calibration's error.

**`measurement_benchmark`** (detail). (a–f) Signed error, fitted minus true, of x, y
and yaw for the goal and the T, against the true x (distance from the robot base;
the camera sits at x ≈ 0.80 m). Shaded band: bias ± 2σ. (g) Signed overlap error
against the true overlap; orange, the samples whose success at 90 % changed.
(h) As (c) above.
