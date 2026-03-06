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

ANGLE_COL       = 'angle'
ANGLE_COL_N1        = 'angle_n-1'
RAT_COL = 'rat'
SESSION_COL = 'date'
EVIDENCE_COL = 'evidence_4_repetition'
REP_COL = 'repeated_action'
STATE_COL = 'hitmiss_n-1'      # 0/1
state_values = [0, 1]
BOUNDARY        = 45
RESP_COL        = 'action'

# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data()

df_processed = an.create_lagged_features()
modalities = df_processed["mod"].dropna().unique()


# angle is in [0, 90]
bins = np.linspace(0, 90, 13)   # 12 bins; tweak as you like
bin_centers = (bins[:-1] + bins[1:]) / 2
bins_curr = np.linspace(0, 90, 5)  # 4 bins
df_processed["curr_bin"] = pd.cut(df_processed[ANGLE_COL], bins=bins_curr, include_lowest=True)

def binned_prob(df, bins):
    """Return P(vertical) per angle bin (NaN if bin empty)."""
    b = pd.cut(df[ANGLE_COL_N1], bins=bins, include_lowest=True)
    out = df.groupby(b)[RESP_COL].mean()
    # align to all bins
    out = out.reindex(pd.IntervalIndex.from_breaks(bins), fill_value=np.nan)
    return out.to_numpy()

def bootstrap_session_binned(df, bins, n_boot=500, seed=0):
    """
    Session bootstrap: resample SESSION_COL with replacement; keep all trials in each sampled session.
    Returns: point estimate (empirical) + CI low/high per bin.
    """
    # point estimate
    y_hat = binned_prob(df, bins)

    sessions = df[SESSION_COL].dropna().unique()
    if len(sessions) < 2:
        return y_hat, np.full_like(y_hat, np.nan, dtype=float), np.full_like(y_hat, np.nan, dtype=float)

    rng = np.random.default_rng(seed)
    Y = []

    for b in range(n_boot):
        sampled = rng.choice(sessions, size=len(sessions), replace=True)
        boot = pd.concat([df[df[SESSION_COL] == s] for s in sampled], ignore_index=True)
        Y.append(binned_prob(boot, bins))

    Y = np.asarray(Y)  # shape (n_boot, n_bins)
    y_low  = np.nanquantile(Y, 0.025, axis=0)
    y_high = np.nanquantile(Y, 0.975, axis=0)
    return y_hat, y_low, y_high

# -----------------------------
# PLOT one panel per rat
# -----------------------------
rats = sorted(df_processed[RAT_COL].dropna().unique())
n = len(rats)
ncols = int(np.ceil(np.sqrt(n)))
nrows = int(np.ceil(n / ncols))

color_map = {0: "violet", 1: "yellow"}  # match your earlier description if you want
modalities= sorted(modalities)
#fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows), sharex=True, sharey=True, constrained_layout=True)
#axes = np.atleast_1d(axes).reshape(-1)
angles = df_processed["angle"].dropna().unique()


cmap = plt.cm.viridis
bins_unique = df_processed["curr_bin"].dropna().unique()
for rw in [0,1]:  # optionally split by reward state
    df_rw = df_processed[df_processed[STATE_COL] == rw]
    print(f"Reward state {rw}: {len(df_rw)} trials")
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(-1)
    for ax, rat in zip(axes, rats):

        rat_df = df_rw[df_rw[RAT_COL] == rat]

        for i, cb in enumerate(sorted(bins_unique)):

            sub = rat_df[rat_df["curr_bin"] == cb]

            if len(sub) < 20:
                continue

            y_hat, y_low, y_high = bootstrap_session_binned(sub, bins, n_boot =1)

            color = cmap(i / len(bins_unique))

            ax.plot(bin_centers, y_hat, marker='o', color=color,
                    label=f"{cb.left:.0f}-{cb.right:.0f}")

        ax.axhline(0.5, color='k', ls='--', alpha=0.4)
        ax.axvline(0, color='k', ls='--', alpha=0.4)      # evidence boundary is 0
        ax.set_xlim(0, 90)
        ax.set_ylim(0, 1)
        ax.set_title(f"Rat {rat}")
        ax.set_xlabel("angle n-1")
        ax.set_ylabel(f"P(C = vertical)")

    # hide unused axes
    for ax in axes[len(rats):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2)

    plt.show()