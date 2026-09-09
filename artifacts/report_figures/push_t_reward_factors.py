"""Push-T dense reward factors, transcribed from
src/vbrl/tasks/push_t/mdp/rewards.py so the figure matches what was trained.

  maniskill_dense_reward     w*((cos e + 1)/2)^2 + (1-w)*(1-tanh 5d)^2 + tcp
  linear_orientation_reward  w*(1 - |e|/pi)      + (1-w)*(1-tanh 5d)^2 + tcp
  tcp term                   sqrt(max(1 - tanh 5*d_ee, 0)) / 20
  at_goal                    the whole reward is replaced by _MAX_REWARD
  return                     reward / _MAX_REWARD
Registered orientation_weight w = 0.5; _DISTANCE_SCALE = 5.0; _MAX_REWARD = 3.

Both distances reference the T's centre of mass. The reward reads
root_link_pos_w = data.xpos, the body frame origin, and push_t.xml is centred so
that origin *is* the centre of mass -- the same construction ManiSkill3 uses.
"""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Categorical slots 1-2 of the validated reference palette. Each panel carries
# its own legend, so a hue identifies a series only within its own panel.
S1, S2 = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e2e1dd"
SCALE = 5.0

mpl.rcParams.update({
  "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
  "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
  "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
  "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
  "axes.labelcolor": INK2, "figure.facecolor": "white", "axes.facecolor": "white",
  "xtick.major.width": 0.6, "ytick.major.width": 0.6,
  "xtick.major.size": 3, "ytick.major.size": 3, "legend.labelcolor": INK2,
  "mathtext.fontset": "dejavusans",
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 4.25))

# --- the two complete rewards, above both panels ----------------------------
POS = r"(1-w)\left(1-\tanh 5d\right)^{2}"
EE = r"\frac{1}{20}\sqrt{1-\tanh 5d_{ee}}"
fig.text(0.5, 0.995,
         r"ManiSkill:   $r=\frac{1}{3}\left[\,w\left(\frac{\cos e+1}{2}"
         r"\right)^{2}+" + POS + "+" + EE + r"\,\right]$",
         ha="center", va="top", fontsize=9.5, color=INK)
fig.text(0.5, 0.915,
         r"Linear variant:   $r=\frac{1}{3}\left[\,w\left(1-\frac{|e|}{\pi}"
         r"\right)+" + POS + "+" + EE + r"\,\right]$",
         ha="center", va="top", fontsize=9.5, color=INK)
fig.text(0.5, 0.838,
         r"$w=0.5$;  $d$ measures the T's centre of mass to the goal, $d_{ee}$ the "
         r"end-effector to that same point.  At goal" "\n"
         r"(overlap $\geq 0.90$) the bracket is replaced by $3$, so $r=1$: a value "
         r"the shaped terms cannot reach.",
         ha="center", va="top", fontsize=8.5, color=INK2)

# --- (a) orientation factors ------------------------------------------------
e = np.linspace(0.0, math.pi, 2000)
deg = np.degrees(e)
ax1.plot(deg, ((np.cos(e) + 1.0) / 2.0) ** 2, color=S1, lw=2.0,
         solid_capstyle="round",
         label=r"ManiSkill  $\left(\frac{\cos e+1}{2}\right)^{2}$")
ax1.plot(deg, 1.0 - np.abs(e) / math.pi, color=S2, lw=2.0,
         solid_capstyle="round", label=r"Linear  $1-|e|/\pi$")
ax1.set_xlim(0, 180); ax1.set_ylim(-0.02, 1.04)
ax1.set_xticks(range(0, 181, 45))
ax1.set_xlabel("goal yaw error $e$  (degrees)")
ax1.set_ylabel("orientation factor")
ax1.set_title("(a)  Orientation factors", loc="left", color=INK, pad=8)
ax1.legend(frameon=False, loc="lower left", handlelength=1.5, borderpad=0.2)

# --- (b) distance factors ---------------------------------------------------
d = np.linspace(0.0, 0.40, 2000)
ax2.plot(d * 100, (1.0 - np.tanh(SCALE * d)) ** 2, color=S1, lw=2.0,
         solid_capstyle="round", label=r"Position  $(1-\tanh 5d)^{2}$")
ax2.plot(d * 100, np.sqrt(np.clip(1.0 - np.tanh(SCALE * d), 0.0, None)) / 20.0,
         color=S2, lw=2.0, solid_capstyle="round",
         label=r"End-effector  $\frac{1}{20}\sqrt{1-\tanh 5d_{ee}}$")
ax2.set_xlim(0, 40); ax2.set_ylim(-0.02, 1.04)
ax2.set_xlabel("distance $d$  (cm)")
ax2.set_ylabel("distance factor")
ax2.set_title("(b)  Distance factors", loc="left", color=INK, pad=8)
ax2.legend(frameon=False, loc="upper right", handlelength=1.5, borderpad=0.2)

for ax in (ax1, ax2):
  ax.grid(True, color=GRID, lw=0.6, zorder=0)
  ax.set_axisbelow(True)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)

fig.tight_layout(pad=0.6, rect=(0, 0, 1, 0.745))
for ext in ("png", "pdf"):
  fig.savefig(f"artifacts/report_figures/push_t_reward_factors.{ext}",
              dpi=300, bbox_inches="tight")
print("saved png + pdf")
