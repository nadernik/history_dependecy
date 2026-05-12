# this script plots empirical P(repeat) curves conditioned on evidence for repetition, split by reward state of the previous trial (hit vs miss).
# evidence for repetition is defined as the signed distance of the current angle from the boundary, 
# signed according to the previous choice (positive = evidence for repeating previous action, negative = evidence for switching to the other action).
# The output is a series of triplet plots (one per rat) where each triplet contains the curves for the three sensory modalities, with different colors for previous hit vs miss.


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
RAT_COL = 'rat'
SESSION_COL = 'date'
EVIDENCE_COL = 'evidence_4_repetition'
REP_COL = 'repeated_action'
STATE_COL = 'hitmiss_n-1'      # 0/1
state_values = [0, 1]
BOUNDARY        = 45
RESP_COL        = 'action'
MODALITY_COL = 'mod'            

# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data()

df_processed = an.create_lagged_features()

df_processed[REP_COL] = (df_processed["action_n-1"] == df_processed[RESP_COL]).astype(int)
df_processed[EVIDENCE_COL] = (df_processed[ANGLE_COL] - BOUNDARY) * (
    2 * df_processed["action_n-1"] - 1
)
modalities = df_processed["mod"].dropna().unique()

# to use angle-based levels
x_col = EVIDENCE_COL
#x_levels = np.sort(df_processed[EVIDENCE_COL].dropna().unique())

# evidence is in [-45,+45]
bins = np.linspace(-45, 45, 13)   # 12 bins; tweak as you like
bin_centers = (bins[:-1] + bins[1:]) / 2


def binned_prob(df, bins):
    """Return P(repeat) per evidence bin (NaN if bin empty)."""
    b = pd.cut(df[EVIDENCE_COL], bins=bins, include_lowest=True)
    out = df.groupby(b)[REP_COL].mean()
    # align to all bins
    out = out.reindex(pd.IntervalIndex.from_breaks(bins), fill_value=np.nan)
    return out.to_numpy()

def level_prob(df, x_col, x_levels):
    """
    Return mean REP_COL per discrete x level (NaN if missing).
    x_levels controls ordering and which levels are shown.
    """
    out = df.groupby(x_col)[REP_COL].mean()
    out = out.reindex(x_levels, fill_value=np.nan)
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

def bootstrap_session_levels(df, x_col, x_levels, n_boot=500, seed=0):
    """
    Session bootstrap: resample SESSION_COL with replacement; keep all trials in each sampled session.
    Returns: point estimate (empirical) + CI low/high per x level.
    """
    # point estimate
    y_hat = level_prob(df, x_col, x_levels)

    sessions = df[SESSION_COL].dropna().unique()
    if len(sessions) < 2:
        return (
            y_hat,
            np.full_like(y_hat, np.nan, dtype=float),
            np.full_like(y_hat, np.nan, dtype=float),
        )

    rng = np.random.default_rng(seed)
    Y = []

    for _ in range(n_boot):
        sampled = rng.choice(sessions, size=len(sessions), replace=True)
        boot = pd.concat([df[df[SESSION_COL] == s] for s in sampled], ignore_index=True)
        Y.append(level_prob(boot, x_col, x_levels))

    Y = np.asarray(Y)  # shape (n_boot, n_levels)
    y_low  = np.nanquantile(Y, 0.025, axis=0)
    y_high = np.nanquantile(Y, 0.975, axis=0)
    return y_hat, y_low, y_high
# -----------------------------
# PLOT one panel per rat
# -----------------------------
rats = sorted(df_processed[RAT_COL].dropna().unique())
n = len(rats)
ncols = int(np.ceil(np.sqrt(3)))
nrows = int(np.ceil(3 / ncols))
#ncols = 3
#nrows = 1
mod_map = {1: "T", 2: "V", 3: "VT"}
color_map = {0: "violet", 1: "yellow"}  # match your earlier description if you want
modalities= sorted(modalities)
print(f"unique evidence levels: {np.sort(df_processed[EVIDENCE_COL].dropna().unique())}")
for rat in rats:
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows), sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(-1)
    rat_df = df_processed[df_processed[RAT_COL] == rat].dropna(subset=[EVIDENCE_COL, REP_COL, SESSION_COL, STATE_COL])
    print(f"Rat {rat}: {len(rat_df)} trials")
    for ax, md in zip(axes, modalities):
        rat_md = rat_df[rat_df[MODALITY_COL] == md]
        x_levels = np.sort(rat_md[EVIDENCE_COL].dropna().unique())
        md_label = mod_map.get(md, f"aaah")
        for rs in state_values:
            sub = rat_md[rat_md[STATE_COL] == rs]
            if len(sub) < 20:
                continue

            y_hat, y_low, y_high = bootstrap_session_binned(sub, bins, n_boot=5, seed=1000 + 10*rs + int(rat))
            c = color_map.get(rs, "gray")
            # CI band
            #ax.fill_between(bin_centers, y_low, y_high, alpha=0.2, linewidth=0, color=c)
            # empirical points
            ax.plot(bin_centers, y_hat, marker='o', linestyle='-', color=c, label=f"prev {'hit' if rs==1 else 'miss'}")

        ax.axhline(0.5, color='k', ls='--', alpha=0.4)
        ax.axvline(0, color='k', ls='--', alpha=0.4)      # evidence boundary is 0
        ax.set_xlim(-45, 45)
        ax.set_ylim(0, 1)
        ax.set_title(f"Rat {rat}, , md: {md_label}")
        ax.set_xlabel("Evidence for repetition")
        ax.set_ylabel(f"P(repeat)")

    # hide unused axes
    for ax in axes[len(rats):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2)

    plt.show()