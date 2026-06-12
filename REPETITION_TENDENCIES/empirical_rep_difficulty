# this script plots empirical P(repeat) curves conditioned on evidence for repetition, split by difficulty of the previous trial (difficulty_n-1).
# evidence for repetition is defined as the signed distance of the current angle from the boundaryfor one target rat (changeable at the top of the script).
# It uses a session bootstrap to compute confidence intervals, resampling sessions with replacement.
# n-1 is always correct to avoid confounding effects of n-1 reward,so this results are conditioned on n-1 correctness.
# n-1 difficulty is binned into 3 levels (easy/medium/hard) based on the distance of n-1 angle from the boundary (45°), 
# and the curves are plotted separately for each level of n-1 difficulty, with different colors.
# difficulty_n-1 = 0 (i.e. n-1 angle = 45°) is excluded from this analysis, as it is corrupted by a bug in the data collection script
#  that caused the hitmiss of those trials to be recorded as randomly 0/1, so we cannot reliably determine if they were hits or misses 
# there is also the psychometric curve version of this analysis: psych_rep_difficulty.py

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
from functions_Ale import filter_training_trials
from serial_dependence_analysis import SerialDependenceAnalyzer

ANGLE_COL       = 'angle'
RAT_COL         = 'rat'
SESSION_COL     = 'date'
EVIDENCE_COL    = 'evidence_4_repetition'
REP_COL         = 'repeated_action'
STATE_COL       = 'hitmiss_n-1'      # 0/1
DIFFICULTY_NM1_COL = 'difficulty_n-1'
BOUNDARY        = 45
RESP_COL        = 'action'

# -----------------------------
# TUNABLE PARAMETERS
# -----------------------------
TARGET_RAT = 5   # <-- change this to run on a different rat

# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data()

df_processed = an.create_lagged_features()
df_processed  = filter_training_trials(df_processed, method='criterion', criterion=0.80, min_consec=2)
modalities = df_processed["mod"].dropna().unique()
df_processed[REP_COL] = (df_processed["action_n-1"] == df_processed[RESP_COL]).astype(int)
df_processed[EVIDENCE_COL] = (df_processed[ANGLE_COL] - BOUNDARY) * (
    2 * df_processed["action_n-1"] - 1
)

# -----------------------------
# KEY FILTERS:
#   1. keep only trials where n-1 was correct
#   2. exclude corrupted n-1 trials at the boundary (difficulty_n-1 == 0, i.e. angle_n-1 == 45°)
# -----------------------------
df_processed = df_processed[df_processed[STATE_COL] == 1].copy()
df_processed = df_processed[df_processed[DIFFICULTY_NM1_COL] != 0].copy()

# evidence bins
bins = np.linspace(-45, 45, 13)   # 12 bins
bin_centers = (bins[:-1] + bins[1:]) / 2

def binned_prob(df, bins):
    """Return P(repeat) per evidence bin (NaN if bin empty)."""
    b = pd.cut(df[EVIDENCE_COL], bins=bins, include_lowest=True)
    out = df.groupby(b)[REP_COL].mean()
    out = out.reindex(pd.IntervalIndex.from_breaks(bins), fill_value=np.nan)
    return out.to_numpy()


def bootstrap_session_binned(df, bins, n_boot=500, seed=0):
    """
    Session bootstrap: resample SESSION_COL with replacement.
    Returns: point estimate (empirical) + CI low/high per bin.
    """
    y_hat = binned_prob(df, bins)

    sessions = df[SESSION_COL].dropna().unique()
    if len(sessions) < 2:
        return y_hat, np.full_like(y_hat, np.nan, dtype=float), np.full_like(y_hat, np.nan, dtype=float)

    rng = np.random.default_rng(seed)
    Y = []

    for _ in range(n_boot):
        sampled = rng.choice(sessions, size=len(sessions), replace=True)
        boot = pd.concat([df[df[SESSION_COL] == s] for s in sampled], ignore_index=True)
        Y.append(binned_prob(boot, bins))

    Y = np.asarray(Y)
    y_low  = np.nanquantile(Y, 0.025, axis=0)
    y_high = np.nanquantile(Y, 0.975, axis=0)
    return y_hat, y_low, y_high


# -----------------------------
# PLOT: one figure for the target rat,
# one panel per difficulty_n-1 level,
# curves conditioned on difficulty_n-1
# -----------------------------
rat_df = df_processed[df_processed[RAT_COL] == TARGET_RAT].dropna(
    subset=[EVIDENCE_COL, REP_COL, SESSION_COL, DIFFICULTY_NM1_COL]
)
print(f"Rat {TARGET_RAT}: {len(rat_df)} trials (after filtering n-1 correct, excluding diff_n-1=45)")

# -----------------------------
# DIAGNOSTIC: trial counts per difficulty_n-1 level
# -----------------------------
MIN_TRIALS = 20   # <-- threshold below which a level is flagged/dropped

counts = rat_df[DIFFICULTY_NM1_COL].value_counts().sort_index()
print(f"\nTrial counts per difficulty_n-1 level (rat {TARGET_RAT}):")
for level, n in counts.items():
    flag = "  <-- WARNING: too few trials, will be skipped" if n < MIN_TRIALS else ""
    print(f"  difficulty_n-1 = {level:>5}:  {n:>4} trials{flag}")

# keep only levels with enough trials
valid_levels = counts[counts >= MIN_TRIALS].index.tolist()
dropped = counts[counts < MIN_TRIALS].index.tolist()
if dropped:
    print(f"\nDropping levels with < {MIN_TRIALS} trials: {dropped}")
rat_df = rat_df[rat_df[DIFFICULTY_NM1_COL].isin(valid_levels)]

# creation of bins of n-1 difficulty
difficulty_levels = sorted(valid_levels)

bin_size = len(difficulty_levels) // 3

low_levels = difficulty_levels[:bin_size] # harder trials
mid_levels = difficulty_levels[bin_size:2*bin_size] # medium difficulty trials
high_levels = difficulty_levels[2*bin_size:] # easier trials

difficulty_bin_map = {}
# assign bin labels (1, 2, 3) to difficulty levels, where 3 = hardest and 1 = easiest
# bc difficulty here is distance from the boundary, the lower the difficulty_n-1 level, the harder the trial, so we assign bin 3 to low levels and bin 1 to high levels
for level in low_levels:
    difficulty_bin_map[level] = 3

for level in mid_levels:
    difficulty_bin_map[level] = 2

for level in high_levels:
    difficulty_bin_map[level] = 1

rat_df["difficulty_n-1_bin"] = rat_df[DIFFICULTY_NM1_COL].map(difficulty_bin_map)
difficulty_levels = sorted(rat_df["difficulty_n-1_bin"].dropna().unique())
#difficulty_levels = sorted(valid_levels)
print(f"\nDifficulty n-1 levels kept for plotting: {difficulty_levels}")

# colormap: one color per difficulty level, sorted easy→hard
cmap = plt.cm.viridis
colors = cmap(np.linspace(0.1, 0.9, len(difficulty_levels)))
diff_colors = {lv: colors[i] for i, lv in enumerate(difficulty_levels)}

#fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
for mod in modalities:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    print(f"\nProcessing modality {mod} for Rat {TARGET_RAT}")
    sub_md = rat_df[rat_df['mod'] == mod]
    for diff_level in difficulty_levels:
        sub = sub_md[sub_md["difficulty_n-1_bin"] == diff_level]

        y_hat, y_low, y_high = bootstrap_session_binned(
            sub, bins, n_boot=1, seed=42 + int(diff_level)
        )

        c = diff_colors[diff_level]
        ax.fill_between(bin_centers, y_low, y_high, alpha=0.15, linewidth=0, color=c)
        ax.plot(bin_centers, y_hat, marker='o', linestyle='-', color=c,
                label=f"diff_n-1 = {diff_level}")

    ax.axhline(0.5, color='k', ls='--', alpha=0.4)
    ax.axvline(0,   color='k', ls='--', alpha=0.4)
    ax.set_xlim(-45, 45)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Evidence for repetition")
    ax.set_ylabel("P(repeat)")
    ax.legend(title="difficulty n-1", fontsize=9, title_fontsize=9, loc='upper left')
    ax.set_title(
        f"Rat {TARGET_RAT}  —  P(repeat) conditioned on difficulty n-1\n"
        f"(only trials where n-1 was correct)"
    )
    plt.show()