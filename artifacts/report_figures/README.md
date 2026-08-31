# Report figures — SlowGoal + gamma996 sweep

15 runs: `Mjlab-PushT-SlowGoal-<Arch>-TrossenRealistic`, gamma 0.996, 6000
iterations, four GPUs x 1024 envs. This is the best-performing sweep in the
project.

Two encodings, each chosen for how many series share an axes:

* **figs 1-2 (small multiples)** -- one panel per encoder, **colour = adapter**.
  Fifteen curves on one axes cannot be read, and five dash patterns over noisy
  overlapping lines is past what line style can carry, so the panels split by
  encoder and colour is freed for the adapter. Every panel also draws all 15 runs
  in grey, so each encoder is read against the whole field.
* **figs 3-6 (summary)** -- **colour = encoder**, since the adapter is an axis or
  a row label there.

Encoder titles carry the native spatial map handed to the adapter, measured by
hooking each built encoder rather than assumed: Nature CNN 24x24x64, Compact ViT
14x14x128, DINOv2 16x16x384, R3M layer 3 14x14x1024, R3M layer 4 7x7x2048.

| file | what it shows |
|---|---|
| `fig1_yaw_error` | orientation error over training, faceted by encoder |
| `fig2_success` | success rate over training, faceted by encoder |
| `fig3_heatmap_yaw` | final yaw error, encoder x adapter |
| `fig4_heatmap_success` | final success rate, encoder x adapter |
| `fig5_adapter_ranking` | adapter ranking, one point per encoder |
| `fig6_widening_cost` | yaw at iteration 3000 (pinned) vs 6000 (full circle) |

Each is written as both `.pdf` (vector, for LaTeX `\includegraphics`) and `.png`
(300 dpi). Curves in figs 1-2 are a 41-point centred moving average; final values
in figs 3-6 are the mean of the last 50 logged iterations.

Palette: validated five-slot categorical set, all-pairs CVD deltaE 13.0 and
normal-vision deltaE 16.3 (OKLab x100). Two slots fall below 3:1 against the
surface, so every line figure carries direct labels and the heatmaps print their
values.

Regenerate figs 3-6 with `scratchpad/figs.py` and figs 1-2 with
`scratchpad/facet.py`, both against `scratchpad/hist.json`.
