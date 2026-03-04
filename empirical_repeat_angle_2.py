'''
SAFELY CODING: 

START session: 

- check branch --> on the bottom left or run 'git branch' in terminal
-UPDATE branch --> 
git checkout main (it goes to main branch)
git pull (it updates my version of main branch to the current one if anybody changed it)
- UPDATE your working branch -->
git checkout feature/exploratory_psych_curves --> goes to your working branch
git merge main --> merges main into your working branch, so that on my branch I have the latest updates from main

START CODING


END session:
(optional) check changes made: git status
git add exploratory_psych_curves.py --> stages the changes made to this file
git commit -m "progress" --> makes a local checkpoint in your branch.
(only the first time) git push -u origin feature/exploratory_psych_curves --> pushes the changes to my remote working branch
git push --> from the second time onwards
'''





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from serial_dependence_analysis import SerialDependenceAnalyzer

# -----------------------------
# CONFIG
# -----------------------------
ANGLE_COL     = 'angle'
ANGLE_N1_COL  = 'angle_n-1'
RESP_COL      = 'action'
RESP_N1_COL   = 'action_n-1'
RAT_COL       = 'rat'
SESSION_COL   = 'date'
HITMISS_N1_COL = 'hitmiss_n-1'  # previous trial outcome (0/1)
BOUNDARY      = 45

REP_COL       = 'repeated_action'
CAT_COL       = 'stim_cat'
CAT_N1_COL    = 'stim_cat_n-1'
TRANS_COL     = 'cat_transition'   # "same" / "different" / (optional) boundary-involved

# -----------------------------
# CATEGORY RULE: include boundary as its own label
# -----------------------------
def stim_category(angle, boundary=45):
    if pd.isna(angle):
        return np.nan
    if angle < boundary:
        return "H"
    if angle > boundary:
        return "V"
    return "B"  # <- 45° is its own category

# -----------------------------
# UNIQUE-ANGLE EMPIRICAL MEAN
# -----------------------------
def prob_repeat_by_unique_angle(df, angle_col=ANGLE_COL, rep_col=REP_COL, min_count=5):
    g = df.groupby(angle_col)[rep_col].agg(p_repeat="mean", n="size").reset_index()
    g = g[g["n"] >= min_count].sort_values(angle_col)
    g = g.rename(columns={angle_col: "angle"}) # I think this is useless
    return g

# For the "current angle = 45" bias points (prev H vs prev V)
def prob_repeat_at_current_45_by_prev_cat(df, boundary=45, min_count=10):
    sub = df[df[ANGLE_COL] == boundary]
    out = (
        sub.groupby(CAT_N1_COL)[REP_COL]
        .agg(p_repeat="mean", n="size")
        .reset_index()
    )
    out = out[out["n"] >= min_count]
    return out  # rows for CAT_N1_COL in {"H","V","B"} depending on your data

# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data()
df = an.create_lagged_features()

df[REP_COL] = (df[RESP_COL] == df[RESP_N1_COL]).astype(int)

df[CAT_COL]    = df[ANGLE_COL].apply(lambda x: stim_category(x, BOUNDARY))
df[CAT_N1_COL] = df[ANGLE_N1_COL].apply(lambda x: stim_category(x, BOUNDARY))

# same vs different category transition (B will behave consistently)
df[TRANS_COL] = np.where(df[CAT_COL] == df[CAT_N1_COL], "same", "different")


# IMPORTANT: keep 45° trials, but do NOT let them enter blue/orange
boundary_mask = (df[ANGLE_COL] == BOUNDARY) | (df[ANGLE_N1_COL] == BOUNDARY)
df.loc[boundary_mask, TRANS_COL] = "boundary_involved"

# Keep boundary trials; just drop missing essentials
df = df.dropna(subset=[
    RAT_COL, SESSION_COL, ANGLE_COL, ANGLE_N1_COL, RESP_COL, RESP_N1_COL, CAT_COL, CAT_N1_COL, TRANS_COL
])

# -----------------------------
# PLOT: one figure per rat
# -----------------------------
rats = sorted(df[RAT_COL].unique())
n = len(rats)
ncols = int(np.ceil(np.sqrt(n)))
nrows = int(np.ceil(n / ncols))
colors = {
    "same": "tab:blue",
    "different": "tab:orange",
    "prev45": "tab:green",
    "prevH_at45": "tab:purple",
    "prevV_at45": "tab:red",
}

min_count_per_angle = 5
for rw in [0,1]:  # optionally split by reward state
    df_rw = df[df[HITMISS_N1_COL] == rw]
    print(f"Reward state {rw}: {len(df_rw)} trials")
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, rat in zip(axes, rats):
        rat_df = df_rw[df_rw[RAT_COL] == rat].copy()

        # (A) Main curves: same vs different categories (includes 45 if it passes min_count)
        for transition in ["same", "different"]:
            sub = rat_df[rat_df[TRANS_COL] == transition]
            if len(sub) < 30:
                continue
            g = prob_repeat_by_unique_angle(sub, min_count=min_count_per_angle)

            ax.plot(
                g["angle"], g["p_repeat"],
                marker="o", linestyle="-",
                color=colors[transition],
                label=f"{transition} cat (n vs n-1)"
            )

        # (B) EXTRA curve: previous angle == 45, plot vs current angle
        ''' 
        sub_prev45 = rat_df[rat_df[ANGLE_N1_COL] == BOUNDARY]
        if len(sub_prev45) >= 30:
            g_prev45 = prob_repeat_by_unique_angle(sub_prev45, min_count=min_count_per_angle)
            ax.plot(
                g_prev45["angle"], g_prev45["p_repeat"],
                marker="o", linestyle="--",
                color=colors["prev45"],
                label="prev angle = 45° (n-1 boundary)"
            )
        ''' 
        # (C) SPECIAL points at current angle == 45: split by prev category (H vs V)
        # This gives you exactly: “bias at ambiguous current stimulus”
        bias45 = prob_repeat_at_current_45_by_prev_cat(rat_df, boundary=BOUNDARY, min_count=10)

        # Plot only H and V (optionally also B if you want)
        for prev_cat, col, lab in [
            ("H", colors["prevH_at45"], "current 45°, prev stim = H"),
            ("V", colors["prevV_at45"], "current 45°, prev stim = V"),
            # ("B", "tab:gray", "current 45°, prev stim = 45°"),  # optional
        ]:
            row = bias45[bias45[CAT_N1_COL] == prev_cat]
            if len(row) == 1:
                y = float(row["p_repeat"].iloc[0])
                ax.scatter([BOUNDARY], [y], s=140, color=col, edgecolor="k", zorder=5, label=lab)

        # Cosmetics
        ax.axhline(0.5, color="k", ls="--", alpha=0.4)
        ax.axvline(BOUNDARY, color="k", ls=":", alpha=0.4)

        ax.set_ylim(0, 1)
        ax.set_xlabel("Current angle (trial n)")
        ax.set_ylabel("P(repeat previous action)")
        ax.set_title(f"Rat {rat}: repetition vs current angle (45° isolated)")
    # hide unused axes
    for ax in axes[len(rats):]:
        ax.set_visible(False)

    # ---- CREATE LEGEND ONCE ----
    handles, labels = [], []
    for ax in axes[:len(rats)]:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l

    # remove duplicates
    unique = dict(zip(labels, handles))

    fig.legend(unique.values(), unique.keys(),
               loc="upper center",
               bbox_to_anchor=(0.5, 1.02),
               ncol=3)

    plt.show()
        
