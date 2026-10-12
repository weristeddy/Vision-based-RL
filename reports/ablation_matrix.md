# Ablation matrix: what moves sim success and sim2real

Draft 1, 2026-10-12. Built from W&B (2,554 runs over four projects, Apr 13 – Oct 12),
git history, `reports/`, and the deployment notes. Push-T sim success in the E and G rows
is the README's mean-action evaluation where one exists, otherwise the final training value.

## How to read it

**Grade** says how much a number can carry in the thesis:

| grade | meaning |
|---|---|
| **M** | measured, controlled: the two arms differ in this factor only, ≥2 runs per arm or a multi-seed eval |
| **M1** | measured, controlled, but one run per arm. Seed spread is 0.03 when training is stable (5 DinoV2-Afa6 VisualSlow runs: 0.64–0.70 overlap), but 0.41–0.63 on state Push-T and 0.00–0.76 when a run collapses, so differences under ~0.15 are noise |
| **C** | measured, confounded: something else changed too (code generation, scale, curriculum) |
| **E** | estimated from analysis, geometry, a probe, or a replay, not from a trained-policy A/B |
| **R** | real-robot observation, qualitative (no trial count) |
| **?** | not measured; marked where it matters |

**Metrics.** Push-T overlap and yaw error are comparable across generations; *success* is not,
because the threshold moved from 0.98 to 0.90 and training success is latched and stochastic
(45% "ever reached" vs 28% "still there"). Yaw chance level ≈ 1.57 rad. Lift-Cube is
`episode_success` during training, or the 3-seed × 32-episode eval in
`reports/lift_cube_rgb/trossen/ood_evaluation_4096`.

**Encoder families** (adapter set reduced to AFA, as decided 2026-10-12):

| family | architectures in the data |
|---|---|
| scratch | `NatureCnn-Flatten`, `CompactVit-Flatten` (their native heads; AFA on a scratch trunk fails, row A8) |
| global pretrained | `DinoV2ViTS14-Linear` (CLS), `R3MResNet50-Linear` (avg-pool) |
| local pretrained | `DinoV2ViTS14-Afa6`, `R3MResNet50L3-Afa16`, `R3MResNet50-Afa32` |

---

## A. Representation — the thesis axis

| # | factor | levels | sim effect | real effect | evidence | grade | cost |
|---|---|---|---|---|---|---|---|
| A1 | family, Push-T, uniform goals (no curriculum), same PPO | scratch / global / AFA | overlap: scratch 0.34–0.51, global 0.03–0.45, **DinoV2-AFA 0.73–0.90**, R3M-L3-AFA 0.38–0.54. Yaw: scratch and global at chance (1.0–1.8), DinoV2-AFA **0.06–0.18** | ? only DinoV2-AFA was ever deployed | Uniform `yw5mcdjb` `k9suf1ek` vs `ph3fxq00` `d5ban84x`…; UniformQuad `na8v3f12` `2zydonnk`; VisualGoal `62e4elj6`; NearGoal `ee5sks4p` | **M** (4 sweeps, 1–4 runs each) | — |
| A2 | family, Push-T, *with* goal-yaw curriculum (SlowGoal), same PPO | scratch / global / AFA | **NatureCnn-Flatten 0.87/0.87** (yaw 0.10), DinoV2-AFA 0.76/0.73, R3M-L3-AFA 0.76/0.61, CompactVit 0.53/0.46, R3M-Linear 0.42/0.39, DinoV2-Linear 0.26/0.17 | ? | `nfyrxver` `d5oj40ua` vs `uwu1ctnk` `flyfi2em` | **M** (2 runs each) | curriculum: 3,000-iteration pin + 22.5° rungs |
| A3 | encoder × curriculum interaction | A1 vs A2 | scratch CNN needs the curriculum to learn yaw at all (0.50 → 0.87); DinoV2-AFA learns yaw from uniform goals (0.78 either way). **The pretrained local features buy learnability, not final sim performance** | ? | A1, A2 | **M** | — |
| A4 | encoder × PPO settings | kl 0.2 / ent 0 / lr 3e-4 / no std floor vs kl 0.05 / ent 0.001 / lr 2e-4 / floor 0.15 | NatureCnn 0.55 → 0.87; DinoV2-AFA 0.54 → 0.76; R3M-L3-AFA 0.43 → 0.76; DinoV2-Linear 0.03 → 0.26. Scratch gains most | — | SlowGoal `323da6q8` `s8tx7u0u` vs `nfyrxver`; `z1uccona` `c17s6fqp` vs `uwu1ctnk` | **M** | the old "Afa6 wins SlowGoal" verdict (`c17s6fqp`) came from the weaker config |
| A5 | collapse rate | — | at the good PPO config 3 of 7 DinoV2-AFA SlowGoal runs and 1 of 3 NatureCnn ended at overlap 0.00 | — | `c96ok0s6` `k5vwopwm` `pq155sf7` | M | budget ≥3 seeds per cell |
| A6 | family, Lift-Cube, in sim (train success, 3k iterations, CollisionCam) | scratch (SS/LG heads) / global / AFA | scratch 0.59–0.85, DinoV2-Linear 0.55, R3M-Linear 0.63, **DinoV2-AFA 0.70, R3M-AFA 0.74** | — | `hmace229` `njz9q97r` `d8s3z42w` `m5dnsjcc` … | M1 | scratch Lift-Cube has **no Flatten-head run**: dropping SS/LG leaves the scratch row empty |
| A7 | family, Lift-Cube OOD tabletops (trained with texture DR, mean of 3 seeds × 32 episodes) | wood / plaster | scratch 0.94–0.99; **global DinoV2 0.63, R3M 0.49 (wood 0.04)**; **AFA DinoV2 1.00**, R3M 0.77 (wood 0.53) | ? | `reports/lift_cube_rgb/trossen/ood_evaluation_4096` | **M** | — |
| A8 | AFA on a scratch trunk | NatureCnn-Afa1, CompactVit-Afa2 | Curriculum 0.00 / 0.07 overlap; probe 117 mm (predicts the mean) | — | `i8nuow54` `x9dpf27k`; probe 7ed6400 | M1 + E | — |
| A9 | R3M tap layer for AFA | layer4 (Afa32) vs layer3 (Afa16) | yaw: layer4 object spans 1.5 feature cells vs 3.1; Uniform L3-Afa16 0.53 vs Afa32 diverged; SlowGoal best 0.76 vs 0.60 | — | `v46njetp`; probe | M1 + E | layer3 is cheaper too |
| A10 | cube-localization probe (Lift-Cube, 1,196 frames) | linear probe on features | centroid error: DinoV2-AFA 14.6 px, R3M-AFA **36.5**, DinoV2-Linear 27.6, R3M-Linear 22.5, scratch 18–27 | — | `cube_localization_4096/summary.csv` | E | — |
| A11 | throughput | per family | Push-T, 4 GPUs × 1,024 envs, frozen features cached: hours per 1k iterations NatureCnn-Flatten 0.55, R3M 0.81–0.91, DinoV2-Linear 0.95, CompactVit-Flatten 1.00, R3M-L3-Afa16 1.03, DinoV2-Afa6 1.07. Pretrained cost is the rollout forward (2.7 s vs 1.3 s per iteration); updates are as cheap as NatureCnn (0.6–0.9 s) and cheaper than CompactVit (2.3 s). Lift-Cube CollisionCam fps (NatureCnn 36–42k vs DinoV2 8.6–12.5k) is not comparable | — | SlowGoal/Uniform/VisualGoal/NearGoal/FreeStart runs | **M** | pretrained ≈ 2× NatureCnn, ≈ CompactVit |
| A12 | sample efficiency (iterations to overlap 0.5) | per family | ? computable from W&B history, not yet pulled | — | — | ? | — |
| A13 | real transfer per family | — | — | Lift-Cube: only `ntl27zt9` (DinoV2-**SpatialSoftmax**) has run on hardware, and it works. Push-T: only DinoV2-AFA, none reliable | deployment README | R | **the dropped adapter is the only one that has transferred** |

## B. Encoder × appearance DR — the hypothesis "DINO transfers without appearance DR"

Nothing in the history measures this interaction. Every encoder comparison above trained
with the AmbientCG texture bank; the only scene comparisons (B1, B2) used DinoV2-AFA alone.

| # | factor | levels | sim effect | real effect | evidence | grade | cost |
|---|---|---|---|---|---|---|---|
| B1 | tabletop texture DR, Push-T DinoV2-AFA, 15k | `real_table` (rig photo) vs `real_texture` (1,203-texture bank) | 0.727 vs 0.409 success, overlap 0.805 vs 0.632, yaw 0.137 vs 0.339 — **but each is evaluated in its own training scene**, and the DR run was still climbing (+0.03 / 1k iterations) | real_table reached the goal by step ~300 at 12000 µs, then pushed it off; GoalOutline (DR) failed (6k version) | `xbl8pqsz` vs `ao5bzb64` | C (eval scene ≠) | DR costs sample efficiency; real_table costs calibration (B4) |
| B2 | same, 20 Hz measured-arm generation | real_table vs real_texture | success 0.84 vs 0.34, overlap 0.87 vs 0.68, yaw 0.12 vs 0.36 | **both fail on hardware the same way: "different push directions on the real image"** | `kreek7ud` vs `g177vptm` | C + R | — |
| B3 | colour DR on, Push-T DinoV2-SpatialSoftmax 3k | `default` (no colour DR) vs `real_texture` | overlap 0.650 vs 0.466, yaw 0.30 vs 1.46 | — | `j39edqx1` vs `oku8kenn` | M1 | appearance DR slows learning badly at 3k |
| B4 | brittleness of the overfitted scene | luminance band | real_table renders p1 126 – p95 151 (later widened to 103–173); the bank spans 23–255 | at 7000 µs (median 95, below every training frame) the policy moved the T 5.7 mm in 16 s | memory 09-23 | R + E | **every real session needs exposure and white balance checked** |
| B5 | **the interaction itself** | {scratch, global, DinoV2-AFA} × {real_table, real_texture} | ? | ? | — | **?** | 6 cells × 3 seeds; see plan P1 |

## C. Appearance and camera

| # | factor | levels | effect | evidence | grade |
|---|---|---|---|---|---|
| C1 | camera pose | 45° tilt vs near-overhead | 79% vs 37% of the T visible while the gripper touches it | 7ed6400 | E |
| C2 | FOV | D405 horizontal FOV written as vertical vs measured | objects rendered 1.72× too small; pre-fix checkpoints not comparable | fb67b28 | E (bug) |
| C3 | object colour DR | uniform RGB vs fixed / narrow | uniform put 24% of resets within a 1.2:1 luminance ratio of the table | 7ed6400 | E |
| C4 | T colour range | old vs pixel-matched (120 renders) | old range missed 63% of rig matches | e74509d | E |
| C5 | table tint | rgba (0.44, 0.50, 0.55) vs warm | sim table grey 136/136/133 vs rig 186/134/88; no light setting could fix it | 00cb59c | E |
| C6 | light colour DR | independent R, G, B vs shared draw + red-blue tilt | per-channel lit tables green/purple; rig red/blue 1.11 | 00cb59c | E |
| C7 | light intensity range | (0.70, 1.00) vs (0.34, 1.30) | old range rendered 122–160; the rig drifts 94–161 within a day | 00cb59c | E |
| C8 | camera pose jitter per step + FOV ±2° | on vs off | ? | — | ? |
| C9 | exposure on hardware | 7000 vs 12000 µs (real_table) | total failure vs reaching the goal | `xbl8pqsz` | **R (largest real effect on record)** |
| C10 | white balance | auto vs 5700 K | auto renders the wood grey; post-09-23 policies fail on it | g4 manifest | R |
| C11 | render background | dark blue (MJLab < 1.6) vs black | shifts every background pixel; old checkpoints OOD | CLAUDE.md | E |

## D. Goal specification

| # | factor | levels | sim effect | real effect | evidence | grade |
|---|---|---|---|---|---|---|
| D1 | goal as numbers + marker vs marker only | `target_pose` in obs or not | success +0.23, +0.22, and ≈0 on the stiff arm (final 0.82 vs 0.90, overlap 0.90 vs 0.88; the README's 0.86 vs 0.81 is a last-200-iteration mean) | pixel-only fails "harder" (6k generation) | `xbl8pqsz`/`0vv02udk`; `kreek7ud`/`8mqxbt06`; `q0yd31jj`/`rtu81ivl` | M1 ×3, sign not stable |
| D2 | fixed goal vs distribution | one pose vs full | 0.967 vs 0.4–0.7 | placed the T but inconsistently | `n92lvmra` | M1 + R |
| D3 | goal-yaw curriculum vs uniform (50 Hz era) | — | curriculum worked for 9/15 architectures, uniform for 2/15; 45° rungs made yaw worse in 8/15 | — | sweeps | M |
| D4 | goal calibration noise in obs | none vs 3 mm / 1.5° | ? | ? | — | ? |
| D5 | marker style | filled vs hollow outline | mismatch is OOD for pixel-only policies | design | E |

## E. Control and action interface

| # | factor | levels | sim effect | real effect | evidence | grade |
|---|---|---|---|---|---|---|
| E1 | control rate | 50 vs 20 Hz (measured arm) | 0.64 vs 0.96 mean-action success | 20 Hz + instant commands: wrist moves in bursts | c3b512f | M1 + R |
| E2 | controller target in the observation | without vs with `target_qpos` | 0.06 at it 500 vs 0.93 at it 300 (state) | — | a00f4f3 | M1 |
| E3 | action reference | measured-relative vs target-relative | on Menagerie gains measured-relative made per-joint speed 13× uneven; with the SDK ramp modelled: 0.89 vs 0.90 | ? (both candidates untested) | `gh9nf7wc` vs `ok5ae9ph` | M1 + E |
| E4 | SDK goal-time ramp | unmodelled ramp / instant / ramp modelled in sim | instant vs modelled: ? | unmodelled ramp: 35–56% action flips, drove into the table; instant: smooth | 9144d88, memory | R |
| E5 | per-step action scale | 0.1 vs 0.03 (VisualSlow vs VisualSlowStep) | 0.501 vs 0.131, but the curriculum collapse confounds it | — | `yhhr77ar` vs `9w8z704o` | C |
| E6 | action clip = scale | clip beyond scale vs equal | outputs drifted to \|a\| 4.25 in sim, 5.16 on hardware | — | c833dab | E + R |
| E7 | action smoothing on hardware | none vs 0.25 (Lift-Cube) | — | needed for Lift-Cube transfer | `ntl27zt9` | R |
| E8 | deployment rate clamp | 0.010 vs none | — | fired on 100% of steps; transport worked, placement failed | c833dab | R |
| E9 | actor distribution | Beta vs Gaussian (+ std cap 1.5, bound loss) | Beta froze without entropy; uncapped Gaussian std hit 1e6 | "days lost blaming Beta" | 8924e15, memory | E |
| E10 | wrist-roll scale | new option | ? | ? | 492f40e | ? |

## F. Dynamics and sim fidelity

| # | factor | levels | effect | evidence | grade |
|---|---|---|---|---|---|
| F1 | arm model | Trossen MJCF / Menagerie identified / measured gains | fraction of the gap closed per 50 Hz step: 0.35 / 0.125 / rig 0.15 | CLAUDE.md, 2c3eb6d | E |
| F2 | gravity compensation | off vs on | rig holds 0.005/0.018 rad; uncompensated model sags 0.055/0.052 | 3602fc4 | E |
| F3 | object mass | 172.8 vs 50.7 g | 11.6× less slide per push | c833dab | E |
| F4 | T–table friction | 0.30 → 0.40 → 0.55 → 0.35 | at 0.55 the real T kept sliding after release | de424b0 | R |
| F5 | penalty tuned on the wrong arm | velocity hinge sized on the realistic model | inert there; on the identified arm overlap 0.395 → 0.043 | CLAUDE.md | M1 |
| F6 | **end-to-end latency vs `action_delay`** | 1-step delay vs measured | **never measured** | memory | **?** |
| F7 | start pose | low vs home (3 seeds) | 0.06/0.02/0.00 vs 0.00/0.94/0.96 | dbc8217 | M |

## G. Rewards (mostly state Push-T)

| # | factor | levels | effect | grade |
|---|---|---|---|---|
| G1 | any penalty on gripper–T contact (vertical force, barrier, top-contact termination, late top-contact penalty) | on vs off | −73 to −100% success; the policy learns "don't touch" | M (6 mechanisms) |
| G2 | height ceiling weight | −0.02 / −0.05 / −0.1 | −0.1: success 0.008; −0.05: −8 to −19% overlap for 1.4 mm | M1 |
| G3 | action-rate / acceleration penalties from step 0 | on vs off | overlap 0.077 vs 0.804 (σ collapse) | M1 |
| G4 | action-rate weight | −0.002 vs −0.01/−0.02 | −0.002 let the mean action reverse every step; it chattered on hardware | R |
| G5 | near-goal fingertip speed penalty, visual 15k | none / −20 / −40 × ramped / from start | success 0.94/0.95 ramped, 0.77/0.92 from start; tip speed 0.21 → 0.10–0.17 m/s | M1 (4 arms) |
| G6 | orientation term | ((cos e + 1)/2)² vs linear | gradient 0 at 180°: a quarter of uniform starts get none | E |
| G7 | table-touch weight | −2/3 vs −0.1 | no difference; the failing run was the seed | M1 |

## H. Training

| # | factor | levels | effect | grade |
|---|---|---|---|---|
| H1 | `desired_kl` | 0.05 vs 0.1 | at 0.05, 71% of the update budget thrown away | E |
| H2 | PPO config bundle | see A4 | +0.2 to +0.3 overlap, larger for scratch | **M** |
| H3 | iterations, Lift-Cube | 3k vs 6k | 0.361 vs 0.800 at_goal; ground collisions 29% → 0% | M1 |
| H4 | penalty ramp timing | from start vs ramped (G5) | −0.17 at −20, −0.03 at −40 | M1 |
| H5 | ONNX fp32 vs bf16 training | — | actions differ by up to 0.24; success 0.78 vs 0.75 | M1 |

---

## What it already supports

1. **Local pretrained features (DinoV2-AFA) learn what scratch and global features cannot,
   unless the task is made easier.** On uniform goals DinoV2-AFA alone gets the yaw right
   (A1); a scratch CNN matches it only with a yaw curriculum and tuned PPO (A2–A4). That is a
   cleaner thesis claim than "simple encoders cannot encode it": they can, if you hand them a
   curriculum.
2. **Global pretrained features are the worst family on both tasks**, in sim (A1, A6) and on
   OOD tables (A7). R3M's global vector is the most fragile (wood 0.04).
3. **On Lift-Cube, texture DR made scratch encoders OOD-robust (0.94–0.99).** So the
   advantage of pretraining there, with DR on, is small; the hypothesis in B5 is about whether
   it grows once the DR is removed.
4. **The largest real-robot effects on record are not representation choices:** exposure and
   white balance (C9, C10), the SDK command mode (E4), and deployment clamps (E8). They are
   preconditions; any encoder comparison on hardware is void until they are pinned.
5. **Sim success is not evidence of transfer.** The best sim policies (real_table 0.84,
   PixelGoalFixed 0.97) failed or were inconsistent on hardware.

## Gaps that decide the thesis

| gap | why it matters | cheapest way to close it |
|---|---|---|
| **B5** encoder × appearance DR | your central hypothesis | P1 |
| A13 real transfer per family | the thesis's sim2real column is empty for scratch and global | P1 on Lift-Cube, which already transfers |
| A6 scratch with a Flatten head on Lift-Cube | dropping SS/LG leaves the Lift-Cube scratch row empty | part of P1 |
| A12 sample efficiency | one of your four comparison axes | pull W&B history for the runs above |
| visual vs dynamics share of the Push-T gap | B2 says both scenes fail identically, which points away from appearance | P2 |
| F6 latency | the one control assumption never checked | P2c |

**P1 — the factorial, on Lift-Cube first.** {`NatureCnn-Flatten`, `DinoV2ViTS14-Linear`,
`DinoV2ViTS14-Afa6`} × {`real_table`, `real_texture`} × 3 seeds = 18 runs. Evaluate every run
on the same set: both training scenes, wood/plaster/peacock, and ≥20 real trials. The thesis
number is the interaction: (DR gain for scratch) − (DR gain for DinoV2). Lift-Cube first
because it transfers and is cheap; repeat on Push-T once P2 has said whether its gap is visual.

**P2 — split the Push-T gap.**
- (a) Run the state policy on the real arm with the T pose from a tag on the T. The state
  policy never sees an image, so a tag is fine. If it fails, the gap is dynamics or control,
  and no encoder will fix it.
- (b) Replay one real episode's images through the visual policy next to sim renders of the
  same states, and compare actions and DinoV2 features.
- (c) Measure camera-to-command latency against the env's one-step `action_delay`.
