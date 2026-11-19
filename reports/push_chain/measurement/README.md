# Push-Chain rig measurement against simulator ground truth

The real-world evaluation measures the goal from the taped outline and the chain after
each episode, from one external-camera frame with the arm at home
(`vbrl.deployment.push_chain.measure_goal`, `measure_overlap`), and scores the same
whole-chain coverage as the simulator. This benchmark runs exactly that code on
rendered frames whose poses are known.

```bash
uv run python -m vbrl.analysis.push_chain_measurement --scene real_table   # ~25 min
uv run python -m vbrl.analysis.push_chain_measurement --scene wood
uv run python -m vbrl.analysis.push_chain_measurement --scene real_table --links 4
uv run python -m vbrl.analysis.push_chain_measurement --scene real_table --summarize
```

## Method

As for Push-T (`reports/push_t/measurement`): the chain and the goal are each found
by comparing an observed mask from colour (red chain, the hole inside the green tape)
with a mask rendered from the model through the calibrated camera, maximising their
IoU. The chain model is read from `push_chain_{N}.xml`'s collision geoms, each box
or cylinder projected as a prism at its height above the table, so any chain length
is measured without new code (`--links`); results go to `push_chain_{N}/<scene>`. The goal model is the flat footprint with
the mitred corners the straight tape makes outside each bend, and the tape's offset
from the chain's edge as an extra parameter. A pose is the halfway point of the chain
(the centre link for three links), its yaw and the hinge angles. The search starts
from a grid of 36 yaws and hinge steps (at most 1,764 candidates), each centred on
the mask, refines the three best with Nelder–Mead and keeps the best; below IoU 0.8
the measurement counts as failed. The overlap then comes from the two poses.

## Setup

- Scenes: the play config of the chain's RGB task on the rig's tabletop photo
  (`real_table`) and on AmbientCG woods, light randomised as in training, no shadows,
  arm at home.
- Camera: the external camera at its calibrated pose, 1280 × 720, pinhole, camera
  randomisation off. On the table one pixel covers 0.85–1.8 mm across the
  workspace. Chain and tape at the centre of their measured colour ranges; the
  tape flat, 0.2 mm above the table, as in every chain task.
- The chain is rendered at rest on the table, not at the height the simulation drops
  it from: 1 mm of float showed as a 1 mm x bias at the camera's 45° tilt.
- Per sample: the goal from the task's sampler (x 0.2–0.5 m, y ±0.2 m, the full
  circle, hinges ±95°); the chain at the goal plus a Gaussian offset (30 mm, 30 mm,
  30° on yaw and each hinge) scaled by U(0, 1).
- Segmentation window x 0.05–0.65 m, y ±0.35 m, wider than Push-T's, which cut 11 %
  of the sampled chain outlines. Check once with `--annotate` that it lies on the
  rig's table.

## Results (`push_chain_3`, 512 samples each, seed 0)

Position, yaw and hinge errors are fit minus truth (bias ± sd), with the two
orderings of the chain aligned. The worst landmark is the largest error over all
2N + 1 landmarks (tips, centres, pins).

| scene | accepted goal / chain | goal position | chain position | chain yaw, hinges | chain worst landmark | overlap error | success flips at 90 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| real_table | 512 / 512 | 0.20 ± 0.13 mm | 0.50 ± 0.38 mm | 0.81°, 1.7° | 1.10 ± 0.75 mm (max 7.2) | 0.50 pp (max 3.2) | 3 of 512 |
| wood | 511 / 506 | 0.21 ± 0.18 mm | 0.52 ± 0.41 mm | 0.77°, 1.7° | 1.10 ± 0.76 mm (max 5.6) | 0.47 pp (max 7.9) | 5 of 505 |

No fit is biased: x and y biases are at most 0.10 mm for the chain and 0.08 mm for
the goal, yaw and hinge biases at most 0.25°. The goal is measured about twice as
precisely as the chain. The tape is flat, so its image is an exact projection of
the model; its sharp mitred corners fix each hinge; and the free tape offset absorbs
a constant edge error (fitted −0.58 ± 0.12 mm against a true 0, from rasterising
the tape's inner edge), which in the chain fit lands in the pose. The chain is 30 mm
tall with shaded side faces, its joints are round, and its best IoU is 0.975 against
the goal's 0.987. Wood rejects about 1 % of the chain fits and gives the
few large overlap errors.
