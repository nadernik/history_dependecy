"""
distribution_check.py
======================
Visual sanity check of the raw trial distributions that feed into the
choice-effect and reward-effect psychometric analyses.

Two figures are produced:

Figure 1 — Choice distribution (feeds serial_dependence_psychometric.py)
  Filter : angle_n-1 == 45°  AND  hitmiss_n-1 == 1  (correct trials only)
  Shows  : for each rat, side-by-side bars of H vs V choices at each
           n-1 angle in [25, 65°].  Lets you see whether the two
           conditions feeding the choice-effect PF are balanced.

Figure 2 — Reward distribution (feeds reward_effect_psychometric.py)
  Filter : angle_n-1 == 45° (both rewarded and unrewarded)
  Shows  : for each rat, counts of rewarded vs unrewarded trials at each
           n-1 angle in [25, 65°], with prev-V and prev-H side by side
           in the same figure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from serial_dependence_analysis import SerialDependenceAnalyzer

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

RAT_COL        = "rat"
ANGLE_N1_COL   = "angle_n-1"
ACTION_N1_COL  = "action_n-1"
HITMISS_N1_COL = "hitmiss_n-1"

ANGLE_LO       = 25    # lower bound of the n-1 angle window to display
ANGLE_HI       = 65    # upper bound
BOUNDARY_ANGLE = 45

# How many rats to show per row.  Increase to 4 or 5 if you have many rats
# and a wide screen; decrease to 2 for a narrow/portrait layout.
RATS_PER_ROW   = 3

OUTPUT_CHOICE = "distribution_choice.pdf"
OUTPUT_REWARD = "distribution_reward.pdf"

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA  (one call, reused for both figures)
# ──────────────────────────────────────────────────────────────────────────────

an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data(fold=False)
df = an.create_lagged_features()

required = [RAT_COL, ANGLE_N1_COL, ACTION_N1_COL, HITMISS_N1_COL]
df = df.dropna(subset=required).copy()

# Keep only n-1 angles in the display window
# This is the only filter shared by both figures.
#print(unique_angles := sorted(df[ANGLE_N1_COL].unique()))
angles = [25, 30, 35, 40 , 45, 50 , 55, 60, 65]

df_window = df[df[ANGLE_N1_COL].between(ANGLE_LO, ANGLE_HI)].copy()
df_window = df[df[ANGLE_N1_COL].isin(angles)].copy()
rats      = sorted(df_window[RAT_COL].unique())
angles    = sorted(df_window[ANGLE_N1_COL].unique())
n_rats    = len(rats)
n_angles  = len(angles)

# x positions and bar width for grouped bars
x      = np.arange(n_angles)
bw     = 0.35   # bar width

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — CHOICE DISTRIBUTION
# Filter: correct (hitmiss_n-1==1) boundary trials only, same as the
# choice-effect analysis.
# Each subplot shows, for one rat, how many n-1 trials at each angle were
# followed by an H choice vs a V choice on trial n-1.
# This directly shows the data that feeds into the blue/orange PF curves.
# ──────────────────────────────────────────────────────────────────────────────

#df_choice = df_window[df_window[HITMISS_N1_COL] == 1].copy()
df_choice = df_window

# Compute grid dimensions: rats wrap onto a new row every RATS_PER_ROW panels
ncols1 = min(n_rats, RATS_PER_ROW)
nrows1 = int(np.ceil(n_rats / RATS_PER_ROW))

fig1, axes1 = plt.subplots(
    nrows1, ncols1,
    figsize=(4 * ncols1, 4 * nrows1),
    sharey=True, constrained_layout=True,
)
axes1 = np.atleast_1d(axes1).reshape(-1)   # flatten to 1-D for easy iteration

for ax, rat in zip(axes1, rats):
    sub = df_choice[df_choice[RAT_COL] == rat]

    counts_H = [len(sub[(sub[ANGLE_N1_COL] == a) & (sub[ACTION_N1_COL] == 0)])
                for a in angles]
    counts_V = [len(sub[(sub[ANGLE_N1_COL] == a) & (sub[ACTION_N1_COL] == 1)])
                for a in angles]

    ax.bar(x - bw/2, counts_H, bw, label="prev H", color="tab:blue",   alpha=0.8)
    ax.bar(x + bw/2, counts_V, bw, label="prev V", color="tab:orange", alpha=0.8)

    if BOUNDARY_ANGLE in angles:
        bi = angles.index(BOUNDARY_ANGLE)
        ax.axvspan(bi - 0.5, bi + 0.5, color="grey", alpha=0.12, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(a)) for a in angles], fontsize=8)
    ax.set_xlabel("n-1 angle (°)")
    ax.set_ylabel("Trial count")
    ax.set_title(f"Rat {rat}", fontsize=9)

# Hide any empty slots in the last row
for ax in axes1[n_rats:]:
    ax.set_visible(False)

axes1[0].legend(fontsize=8)
fig1.suptitle(
    "Choice distribution | n-1 angles 25–65°\n"
    "(data feeding the choice-effect psychometric analysis)",
    fontsize=10,
)
fig1.savefig(OUTPUT_CHOICE, bbox_inches="tight", dpi=150)

# ──────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — REWARD DISTRIBUTION
# Filter: ALL boundary trials (rewarded + unrewarded), same as the
# reward-effect analysis.
# Two rows: top = previous choice was V, bottom = previous choice was H.
# Each subplot shows rewarded vs unrewarded counts at each n-1 angle.
# This tells you how many trials you have in each cell of the
# reward × prev-choice design before balancing.
# ──────────────────────────────────────────────────────────────────────────────

# Figure 2 has two condition rows (prev V, prev H) for each rat-grid row.
# So the total number of matplotlib rows = n_rat_rows * 2.
ncols2     = min(n_rats, RATS_PER_ROW)
n_rat_rows = int(np.ceil(n_rats / RATS_PER_ROW))
nrows2     = n_rat_rows * 2   # one pair of condition rows per rat-grid row

fig2, axes2 = plt.subplots(
    nrows2, ncols2,
    figsize=(4 * ncols2, 4 * nrows2),
    constrained_layout=True,
)
axes2 = np.atleast_2d(axes2)

for rat_idx, rat in enumerate(rats):
    sub = df_window[df_window[RAT_COL] == rat]

    # Which column and which pair-of-rows does this rat belong to?
    col       = rat_idx % RATS_PER_ROW          # column within the current rat-grid row
    rat_row   = rat_idx // RATS_PER_ROW         # which rat-grid row (0, 1, 2 …)
    base_row  = rat_row * 2                     # first matplotlib row for this rat-grid row

    for offset, (prev_choice, choice_label) in enumerate([(1, "prev V"), (0, "prev H")]):
        ax  = axes2[base_row + offset, col]
        grp = sub[sub[ACTION_N1_COL] == prev_choice]

        counts_rew   = [len(grp[(grp[ANGLE_N1_COL] == a) & (grp[HITMISS_N1_COL] == 1)])
                        for a in angles]
        counts_unrew = [len(grp[(grp[ANGLE_N1_COL] == a) & (grp[HITMISS_N1_COL] == 0)])
                        for a in angles]

        ax.bar(x - bw/2, counts_rew,   bw, label="rewarded",   color="tab:green", alpha=0.8)
        ax.bar(x + bw/2, counts_unrew, bw, label="unrewarded", color="tab:red",   alpha=0.8)

        if BOUNDARY_ANGLE in angles:
            bi = angles.index(BOUNDARY_ANGLE)
            ax.axvspan(bi - 0.5, bi + 0.5, color="grey", alpha=0.12, zorder=0)

        ax.set_xticks(x)
        ax.set_xticklabels([str(int(a)) for a in angles], fontsize=8)
        ax.set_xlabel("n-1 angle (°)")
        ax.set_ylabel("Trial count")
        ax.set_title(f"Rat {rat} | {choice_label}", fontsize=9)

# Hide empty slots (last rat-grid row may not be full)
for rat_idx in range(n_rats, n_rat_rows * RATS_PER_ROW):
    col      = rat_idx % RATS_PER_ROW
    rat_row  = rat_idx // RATS_PER_ROW
    base_row = rat_row * 2
    for offset in range(2):
        axes2[base_row + offset, col].set_visible(False)

axes2[0, 0].legend(fontsize=8)
fig2.suptitle(
    "Reward distribution | n-1 angles 25–65° | rewarded vs unrewarded\n"
    "(data feeding the reward-effect psychometric analysis)",
    fontsize=10,
)
fig2.savefig(OUTPUT_REWARD, bbox_inches="tight", dpi=150)

plt.show()