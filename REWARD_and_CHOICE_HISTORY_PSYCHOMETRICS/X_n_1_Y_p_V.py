# this script plots empirical P(repeat) curves conditioned on n-1 angle, split by n-1 reward (hit vs miss). 
# to observe the dependence of trial n choice on trial n-1 angle split by reward state
# filter here by n angle so that we can look at the effect of n-1 angle separately for each n angle (e.g. to see if the history effects are more pronounced when n is difficult)


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
# FLAG: switch between binning modes
# -----------------------------
USE_BINS = False       # True = binned (current behaviour) | False = unique angles with enough trials
MIN_TRIALS = 20       # minimum trials per angle to be included when USE_BINS = False

# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data()

df_processed = an.create_lagged_features()
modalities = df_processed["mod"].dropna().unique()
# filter here by n angle 
df_processed = df_processed[df_processed['angle'].isin([40, 45, 50])]

bins = np.linspace(0, 90, 13)
bin_centers = (bins[:-1] + bins[1:]) / 2


def binned_prob(df, bins):
    """Return P(vertical) per angle bin (NaN if bin empty)."""
    b = pd.cut(df[ANGLE_COL_N1], bins=bins, include_lowest=True)
    out = df.groupby(b)[RESP_COL].mean()
    out = out.reindex(pd.IntervalIndex.from_breaks(bins), fill_value=np.nan)
    return out.to_numpy()


def angle_prob(df, angles):
    """Return P(vertical) per unique angle value."""
    out = df.groupby(ANGLE_COL_N1)[RESP_COL].mean()
    out = out.reindex(angles, fill_value=np.nan)
    return out.to_numpy()


def get_valid_angles(df, min_trials=MIN_TRIALS):
    """Return unique angle_n-1 values that have at least min_trials across all conditions."""
    counts = df.groupby(ANGLE_COL_N1)[RESP_COL].count()
    return np.sort(counts[counts >= min_trials].index.to_numpy())


def bootstrap_session_binned(df, bins, n_boot=500, seed=0):
    """Session bootstrap for BINNED mode."""
    y_hat = binned_prob(df, bins)
    sessions = df[SESSION_COL].dropna().unique()
    if len(sessions) < 2:
        return y_hat, np.full_like(y_hat, np.nan), np.full_like(y_hat, np.nan)

    rng = np.random.default_rng(seed)
    Y = [binned_prob(
            pd.concat([df[df[SESSION_COL] == s] for s in rng.choice(sessions, size=len(sessions), replace=True)]),
            bins
        ) for _ in range(n_boot)]

    Y = np.asarray(Y)
    return y_hat, np.nanquantile(Y, 0.025, axis=0), np.nanquantile(Y, 0.975, axis=0)


def bootstrap_session_angles(df, angles, n_boot=500, seed=0):
    """Session bootstrap for UNIQUE ANGLES mode."""
    y_hat = angle_prob(df, angles)
    sessions = df[SESSION_COL].dropna().unique()
    if len(sessions) < 2:
        return y_hat, np.full_like(y_hat, np.nan), np.full_like(y_hat, np.nan)

    rng = np.random.default_rng(seed)
    Y = [angle_prob(
            pd.concat([df[df[SESSION_COL] == s] for s in rng.choice(sessions, size=len(sessions), replace=True)]),
            angles
        ) for _ in range(n_boot)]

    Y = np.asarray(Y)
    return y_hat, np.nanquantile(Y, 0.025, axis=0), np.nanquantile(Y, 0.975, axis=0)

# -----------------------------
# PLOT one panel per rat
# -----------------------------
rats = sorted(df_processed[RAT_COL].dropna().unique())
n = len(rats)
ncols = int(np.ceil(np.sqrt(n)))
nrows = int(np.ceil(n / ncols))

color_map = {0: "violet", 1: "green"}
modalities = sorted(modalities)

for md in modalities:
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(-1)
    print(f"Plotting modality: {md}")
    df_md = df_processed[df_processed['mod'] == md]

    for ax, rat in zip(axes, rats):
        rat_df = df_md[df_md[RAT_COL] == rat].dropna(
            subset=[RESP_COL, SESSION_COL, STATE_COL, ANGLE_COL_N1])

        # compute x-axis once per rat (so both conditions share the same x)
        if not USE_BINS:
            valid_angles = get_valid_angles(rat_df)

        for rs in state_values:
            sub = rat_df[rat_df[STATE_COL] == rs]
            if len(sub) < MIN_TRIALS:
                continue

            if USE_BINS:
                x = bin_centers
                y_hat, y_low, y_high = bootstrap_session_binned(
                    sub, bins, n_boot=5, seed=1000 + 10*rs + int(rat))
            else:
                x = valid_angles
                y_hat, y_low, y_high = bootstrap_session_angles(
                    sub, valid_angles, n_boot=1, seed=1000 + 10*rs + int(rat))

            c = color_map.get(rs, "gray")
            ax.fill_between(x, y_low, y_high, alpha=0.2, linewidth=0, color=c)
            ax.plot(x, y_hat, marker='o', linestyle='-', color=c,
                    label=f"prev {'hit' if rs==1 else 'miss'}")

        ax.axhline(0.5, color='k', ls='--', alpha=0.4)
        ax.axvline(45, color='k', ls='--', alpha=0.4)   # boundary
        ax.set_xlim(0, 90)
        ax.set_ylim(0, 1)
        ax.set_title(f"Rat {rat}")
        ax.set_xlabel("angle n-1")
        ax.set_ylabel("P(C = vertical)")

    for ax in axes[len(rats):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(0.5, 1.02), ncol=2)
    plt.suptitle(f"Modality: {md}", y=1.04)
    plt.show()