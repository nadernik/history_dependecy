# evidence for repetition = signed distance of current angle from boundary, where sign depends on previous action
#  (positive if n-1 action was "repeat" side, negative if n-1 action was "alternate" side)
# this script plots empirical P(repeat) curves conditioned on evidence for repetition.
# output = one plot with all rats together


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from serial_dependence_analysis import SerialDependenceAnalyzer
from serial_dependence_analysis import fit_psychometric_curve, cumulative_gaussian_lapse
import functions_Ale as fa

# -----------------------------
# CONFIG / COLUMN NAMES
# -----------------------------
ANGLE_COL       = 'angle'
ANGLE_N_1_COL   = 'angle_n-1'
BIN_COL_N_1     = 'bin_angle_n-1'     # previous-trial angle bin label
BIN_COL_N       = 'bin_angle_n'
RESP_COL        = 'action'        
SESSION_COL     = 'date'                # <-- CHANGE if needed

RAT_COL = 'rat'
EVIDENCE_COL = 'evidence_4_repetition'
STATE_COL = 'hitmiss_n-1'      # 0/1
state_values = [0, 1]

BOUNDARY        = 45
REP_COL         = 'repeated_action'

# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data()
df_processed = an.create_lagged_features()

df_processed[REP_COL] = (df_processed['action_n-1'] == df_processed[RESP_COL]).astype(int)
df_processed[EVIDENCE_COL] = (df_processed[ANGLE_COL] - BOUNDARY) * (2*df_processed['action_n-1'] - 1)

# evidence should be in roughly [-45,+45]
bins = np.linspace(-45, 45, 13)   # 12 bins; tweak as you like
bin_centers = (bins[:-1] + bins[1:]) / 2

def binned_prob(df, bins):
    """Return P(repeat) per evidence bin (NaN if bin empty)."""
    b = pd.cut(df[EVIDENCE_COL], bins=bins, include_lowest=True)
    out = df.groupby(b)[REP_COL].mean()
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

fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows), sharex=True, sharey=True, constrained_layout=True)
axes = np.atleast_1d(axes).reshape(-1)

color_map = {0: "violet", 1: "yellow"}  # match your earlier description if you want

for ax, rat in zip(axes, rats):

    rat_df = df_processed[df_processed[RAT_COL] == rat].dropna(subset=[EVIDENCE_COL, REP_COL, SESSION_COL, STATE_COL])

    for rs in state_values:
        sub = rat_df[rat_df[STATE_COL] == rs]
        if len(sub) < 20:
            continue

        y_hat, y_low, y_high = bootstrap_session_binned(sub, bins, n_boot=5, seed=1000 + 10*rs + int(rat))

        c = color_map.get(rs, "gray")
        # CI band
        ax.fill_between(bin_centers, y_low, y_high, alpha=0.2, linewidth=0, color=c)
        # empirical points
        ax.plot(bin_centers, y_hat, marker='o', linestyle='-', color=c, label=f"prev {'hit' if rs==1 else 'miss'}")

    ax.axhline(0.5, color='k', ls='--', alpha=0.4)
    ax.axvline(0, color='k', ls='--', alpha=0.4)      # evidence boundary is 0
    ax.set_xlim(-45, 45)
    ax.set_ylim(0, 1)
    ax.set_title(f"Rat {rat}")
    ax.set_xlabel("Evidence for repetition")
    ax.set_ylabel("P(repeat)")

# hide unused axes
for ax in axes[len(rats):]:
    ax.set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
if handles:
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2)

plt.show()
