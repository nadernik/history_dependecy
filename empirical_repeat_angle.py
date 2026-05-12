# this scripts plots empirical p(repeat) curves conditioned on n-1 angle category transition (same vs different) for all rats, 
# split by n-1 reward (hit vs miss). It also plots special points for the bias at current angle = 45° (ambiguous stimulus), split by previous category (H vs V).
# Futher conditioning on n-1 difficulty is implemented dividing each curve in n-1 difficulty (easy - hard)

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
import functions_Ale as fa

# -----------------------------
# CONFIG / COLUMN NAMES
# -----------------------------
ANGLE_COL     = 'angle'        # current stimulus angle (trial n)
ANGLE_N1_COL  = 'angle_n-1'    # previous stimulus angle (trial n-1)
RESP_COL      = 'action'       # current action
RESP_N1_COL   = 'action_n-1'   # previous action
RAT_COL       = 'rat'
HITMISS_N1_COL = 'hitmiss_n-1'  # previous trial outcome (0/1)
SESSION_COL   = 'date'
BOUNDARY      = 45

REP_COL       = 'repeated_action'
CAT_COL       = 'stim_cat'      # current stimulus category (H/V)
CAT_N1_COL    = 'stim_cat_n-1'  # previous stimulus category (H/V)
TRANS_COL     = 'cat_transition'  # "same" / "different"
BIN_COL_N_1 = 'angle_n-1_bin'  # optional: binned version of angle_n-1 for easier plotting or stratification

# NEW: difficulty split based on previous angle (n-1)
DIFF_N1_COL = 'diff_label_n-1'          # "easy" / "hard"
DIFF_THRESHOLD_DEG = 20                 # <-- EDIT: boundary-distance threshold in degrees

# -----------------------------
# CATEGORY RULE (edit if needed)
# -----------------------------
def stim_category(angle, boundary=45):
    if pd.isna(angle):
        return np.nan
    if angle < boundary:
        return "H"
    if angle > boundary:
        return "V"
    return np.nan  # exclude exactly at boundary

# NEW: difficulty rule for angle_n-1
def difficulty_from_boundary(angle, boundary=45, thr=DIFF_THRESHOLD_DEG):
    """
    Labels previous angle difficulty based on distance from the category boundary.
    hard = closer to boundary (small |angle-boundary|)
    easy = farther from boundary (large |angle-boundary|)
    """
    if pd.isna(angle) or angle == boundary:
        return np.nan
    return "hard" if abs(angle - boundary) < thr else "easy"

# -----------------------------
# UNIQUE-ANGLE EMPIRICAL MEAN
# -----------------------------
def prob_repeat_by_unique_angle(df, angle_col=ANGLE_COL, rep_col=REP_COL, min_count=5):
    """
    Returns a dataframe with columns:
      angle, p_repeat, n
    computed at each unique angle (no binning).
    Angles with fewer than min_count trials are dropped.
    """
    g = df.groupby(angle_col)[rep_col].agg(p_repeat="mean", n="size").reset_index()
    g = g[g["n"] >= min_count].sort_values(angle_col)
    g = g.rename(columns={angle_col: "angle"})
    return g

# Optional: session bootstrap CI at each unique angle
def bootstrap_session_unique_angle(df, angle_col=ANGLE_COL, rep_col=REP_COL,
                                   session_col=SESSION_COL, min_count=5,
                                   n_boot=500, seed=0):
    """
    Session bootstrap over sessions; within each bootstrap sample compute p_repeat per unique angle.
    Returns:
      point_df: angle, p_repeat, n
      ci_df: angle, low, high (aligned to the angles that survive min_count in point_df)
    """
    # point estimate
    point_df = prob_repeat_by_unique_angle(df, angle_col, rep_col, min_count=min_count)
    angles = point_df["angle"].to_numpy()

    sessions = df[session_col].dropna().unique()
    if len(sessions) < 2 or len(angles) == 0:
        ci_df = point_df[["angle"]].copy()
        ci_df["low"] = np.nan
        ci_df["high"] = np.nan
        return point_df, ci_df

    rng = np.random.default_rng(seed)

    # collect bootstrap p_repeat for each angle in `angles`
    Y = np.full((n_boot, len(angles)), np.nan, dtype=float)

    for b in range(n_boot):
        sampled = rng.choice(sessions, size=len(sessions), replace=True)
        boot = pd.concat([df[df[session_col] == s] for s in sampled], ignore_index=True)

        boot_df = prob_repeat_by_unique_angle(boot, angle_col, rep_col, min_count=min_count)
        # align by angle
        m = boot_df.set_index("angle")["p_repeat"]
        Y[b, :] = [m.get(a, np.nan) for a in angles]

    low = np.nanquantile(Y, 0.025, axis=0)
    high = np.nanquantile(Y, 0.975, axis=0)

    ci_df = pd.DataFrame({"angle": angles, "low": low, "high": high})
    return point_df, ci_df

# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data()
df = an.create_lagged_features()

# repeat indicator
df[REP_COL] = (df[RESP_COL] == df[RESP_N1_COL]).astype(int)

# stimulus categories for n and n-1
df[CAT_COL]    = df[ANGLE_COL].apply(lambda x: stim_category(x, BOUNDARY))
df[CAT_N1_COL] = df[ANGLE_N1_COL].apply(lambda x: stim_category(x, BOUNDARY))

# same vs different transition
df[TRANS_COL] = np.where(df[CAT_COL] == df[CAT_N1_COL], "same", "different")

# NEW: difficulty of previous angle (n-1)
df[DIFF_N1_COL] = df[ANGLE_N1_COL].apply(lambda x: difficulty_from_boundary(x, BOUNDARY, DIFF_THRESHOLD_DEG))

# drop unusable rows
df = df.dropna(subset=[
    RAT_COL, SESSION_COL, ANGLE_COL, ANGLE_N1_COL, RESP_COL, RESP_N1_COL,
    CAT_COL, CAT_N1_COL, TRANS_COL, DIFF_N1_COL
])

# -----------------------------
# PLOT: one figure per rat, now FOUR curves (same/different) x (easy/hard) based on n-1 difficulty
# -----------------------------
rats = sorted(df[RAT_COL].unique())

colors = {
    ("same", "easy"): "tab:blue",
    ("same", "hard"): "tab:cyan",
    ("different", "easy"): "tab:orange",
    ("different", "hard"): "tab:red",
}
min_count_per_angle = 5

for rw in [1, 1]:  # keep reward split as-is
    df_rw = df[df[HITMISS_N1_COL] == rw]
    print(f"Reward state {rw}: {len(df_rw)} trials")

    for rat in rats:
        rat_df = df_rw[df_rw[RAT_COL] == rat].copy()

        fig, ax = plt.subplots(figsize=(7, 5))

        for transition in ["same", "different"]:
            for diff in ["easy", "hard"]:
                sub = rat_df[(rat_df[TRANS_COL] == transition) & (rat_df[DIFF_N1_COL] == diff)]
                if len(sub) < 30:
                    continue

                # point estimate per unique angle
                g = prob_repeat_by_unique_angle(sub, min_count=min_count_per_angle)

                # OPTIONAL CI
                g, ci = bootstrap_session_unique_angle(
                    sub, min_count=min_count_per_angle, n_boot=6, seed=1000 + int(rat)
                )
                ax.fill_between(
                    ci["angle"], ci["low"], ci["high"],
                    alpha=0.2, linewidth=0, color=colors[(transition, diff)]
                )

                ax.plot(
                    g["angle"], g["p_repeat"],
                    marker='o', linestyle='-',
                    color=colors[(transition, diff)],
                    label=f"{transition} cat; n-1 {diff}"
                )

        ax.axhline(0.5, color='k', ls='--', alpha=0.4)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Current angle (trial n) — unique values")
        ax.set_ylabel("P(repeat previous action)")
        ax.set_title(f"Rat {rat}: repetition vs current angle (unique angles) | hitmiss n-1={rw}")
        ax.legend()
        plt.show()


# -----------------------------
'''




'''
# -----------------------------