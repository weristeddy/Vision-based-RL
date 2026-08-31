# AmbientCG texture dataset

This directory contains the prepared realistic texture pool used by the
canonical `real_texture` scene for every robot.

The setup follows Garcia et al., *Robust visual sim-to-real transfer for
robotic manipulation* (IROS 2023), which uses 1,203 AmbientCG textures:
https://arxiv.org/abs/2307.15320

AmbientCG assets are CC0: https://ambientcg.com/

`basecolor_256/` contains the deterministic paper-sized set of 1,203 RGB
base-color crops. PBR normal, roughness, displacement, and AO maps are excluded
because this experiment only consumes the RGB appearance map. `raw/` is ignored
and retained only as an optional staging location for future dataset updates.

`vbrl.scenes.presets.ambientcg_texture_paths()` inventories every prepared
asset. The runtime MuJoCo model uses a fixed, seeded sample of 768 files via
`ambientcg_runtime_texture_paths()`. This leaves headroom below MuJoCo's public
renderer limit of 1,000 materials while retaining the full source pool here;
training, visualization, and analysis must not maintain separate texture lists.
