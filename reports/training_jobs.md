# Training jobs

One row per submitted Slurm job; fill in the job ID right after `sbatch` prints it.
Every student job runs 3 seeds (1000, 2000, 3000) one after another on 4 B200.

| job ID | submitted | task IDs | label | iterations | status |
| --- | --- | --- | --- | --- | --- |
| 3498729 | 2026-11-26 | `Mjlab-PushT-WoodTable-DinoV3ViTS16-AfaFull6-Trossen` | `distill-auxcyfuo-dinov3-256` | 8,000 | running (reservation) |
| 3498733 | 2026-11-26 | `Mjlab-PushChain3-WoodTableImageGoal-DinoV3ViTS16-AfaFull6-Trossen` | `distill-yxkvn60y-imagegoal-dinov3-256` | 15,000 | queued |
| R1 | | `Mjlab-PushChain3-WoodTableImageGoal-NatureCnn-FlattenRelu-Trossen`, `…-CompactVit-Svea-Trossen` | `distill-yxkvn60y-imagegoal` | 15,000 | planned (reservation) |
| R2 | | `Mjlab-PushChain3-WoodTableImageGoal-DinoV2ViTS14-Global-Trossen` | `distill-yxkvn60y-imagegoal` | 15,000 | planned (reservation, after R1) |
| Q1 | | `Mjlab-PushChain3-WoodTableImageGoal-R3MResNet50-Global-Trossen` | `distill-yxkvn60y-imagegoal` | 15,000 | planned |
| Q2 | | `Mjlab-PushChain3-WoodTableImageGoal-R3MResNet50-AfaFull32-Trossen` | `distill-yxkvn60y-imagegoal` | 15,000 | planned |
| Q3 | | `Mjlab-PushChain3-WoodTableImageGoal-R3MResNet50L3-AfaFull16-Trossen` | `distill-yxkvn60y-imagegoal` | 15,000 | planned |

Done before: `Mjlab-PushChain3-WoodTableImageGoal-DinoV2ViTS14-AfaFull6-Trossen`,
label `distill-yxkvn60y-imagegoal`, runs `nnlg9thd`, `k0bi70sg`, `s4exluhh`.
