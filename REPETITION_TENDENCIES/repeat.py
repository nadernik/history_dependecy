
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
import matplotlib.cm as cm
from serial_dependence_analysis import SerialDependenceAnalyzer
from serial_dependence_analysis import fit_psychometric_curve, cumulative_gaussian_lapse
import functions_Ale as fa

# -----------------------------
# CONFIG / COLUMN NAMES
# -----------------------------
BIN_COL_N_1     = 'bin_angle_n-1'     # previous-trial angle bin label
BIN_COL_N       = 'bin_angle_n'       # current-trial angle bin label                        
BIN_COL_N_MID   = 'bin_n_midpoint'    # numeric midpoint for current-trial bin (x-axis)
RAT_COL         = 'rat'               # subject/animal ID
ACTION_N_1      = 'action_n-1'  # previous-trial response (0/1)
ACTION          = 'action'    # current-trial response (0/1)
TRANS_COL  = 'mod_transition_n-1'     # e.g. 'T->V', 'V->VT', ...
ANGLE_COL  = 'angle'
ANGLE_N_1_COL  = 'angle_n-1'
MOD_COL = 'mod'  # e.g. 1 -->'T', 2-->'V', 3-->'VT'
REPETITION_COL = 'repeat_n'  # e.g. 'rep', 'alt'
REWARD_COL = "hitmiss_n-1"   # 1 rewarded, 0 not rewarded
BINARY_ANGLE_COL_N_1 = 'bin_angle_n-1_binary'  # previous-trial angle bin label binary (2 groups)
SWITCH_COL = 'switch_n'  # 1 if switch, 0 if repetition
DIFFICULTY_NM1_COL = 'difficulty_n-1'
TARGET_RAT = 6   # <-- change this to run on a different rat
# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data()
df_processed = an.create_lagged_features()
df_processed = df_processed[df_processed['rat'].isin([TARGET_RAT])].copy()
df_processed, _ = fa.bin_by_unique_angles(df_processed, name_new_col=BIN_COL_N_1, n_groups=2, angle_col='angle_n-1', exclude_angle=45)
df_processed, angle_to_label   = fa.bin_by_unique_angles(df_processed, name_new_col=BIN_COL_N,   n_groups=6, angle_col='angle', exclude_angle=45)
dfc = df_processed.dropna(subset=[BIN_COL_N_1, BIN_COL_N]).copy()
dfc[BIN_COL_N_MID] = dfc[BIN_COL_N].map(angle_to_label)
#dfc[REPETITION_COL] = (dfc[ACTION_N_1] == dfc[ACTION]).astype(int) # 1: repetition, 0: alternation
bin_to_binary = {
    "0–40°": 0,
    "50–90°": 1
}
dfc[BINARY_ANGLE_COL_N_1] = dfc[BIN_COL_N_1].map(bin_to_binary)
#dfc = dfc[dfc[ANGLE_N_1_COL] == 45]
dfc[REPETITION_COL] = (dfc[ACTION_N_1] == dfc[ACTION]).astype(int) # 1: repetition, 0: alternation
dfc[SWITCH_COL] = (dfc[ACTION_N_1] != dfc[ACTION]).astype(int) # 1: switch, 0: repetition
mask = dfc[[ACTION_N_1, ACTION]].notna().all(axis=1)
dfc.loc[mask, REPETITION_COL] = (dfc.loc[mask, ACTION_N_1] == dfc.loc[mask, ACTION]).astype(int)
dfc = dfc.dropna(subset=[REPETITION_COL])


rats = sorted(dfc[RAT_COL].dropna().unique())
cmap = cm.get_cmap("tab20", len(rats))  # tab20 = 20 distinct colors
# -----------------------------
# COMPUTE AND PRINT REPETITION RATES
rep_rates = (
    dfc
    .dropna(subset=[REPETITION_COL, "hitmiss_n-1"])
    .groupby([RAT_COL, "hitmiss_n-1", BIN_COL_N ])[REPETITION_COL]
    .mean()
    .reset_index()
)
print(rep_rates.sort_values([RAT_COL, "hitmiss_n-1", BIN_COL_N]))


# difficulty on trial n
dfc["difficulty_x"] = dfc[ANGLE_COL].abs()

rats = sorted(dfc[RAT_COL].dropna().unique())
min_trials_per_x = 5

reward_states = [
    (1, "Rewarded at n-1"),
    (0, "NOT rewarded at n-1")
]

#for reward_state, title_state in reward_states:
transitions = sorted(dfc[TRANS_COL].dropna().unique(), key=str)
n_mod = len(transitions)

# -----------------------------
# DIAGNOSTIC: trial counts per difficulty_n-1 level
# -----------------------------
MIN_TRIALS = 20   # <-- threshold below which a level is flagged/dropped
dfc_rat = dfc[dfc[RAT_COL] == TARGET_RAT]
counts = dfc_rat[DIFFICULTY_NM1_COL].value_counts().sort_index()
print(f"\nTrial counts per difficulty_n-1 level (rat {TARGET_RAT}):")
for level, n in counts.items():
    flag = "  <-- WARNING: too few trials, will be skipped" if n < MIN_TRIALS else ""
    print(f"  difficulty_n-1 = {level:>5}:  {n:>4} trials{flag}")

# keep only levels with enough trials
valid_levels = counts[counts >= MIN_TRIALS].index.tolist()

difficulty_levels = sorted(valid_levels)

bin_size = len(difficulty_levels) // 3

low_levels = difficulty_levels[:bin_size]
mid_levels = difficulty_levels[bin_size:2*bin_size]
high_levels = difficulty_levels[2*bin_size:]

difficulty_bin_map = {}

for level in low_levels:
    difficulty_bin_map[level] = 3

for level in mid_levels:
    difficulty_bin_map[level] = 2

for level in high_levels:
    difficulty_bin_map[level] = 1

dfc["difficulty_n-1_bin"] = dfc[DIFFICULTY_NM1_COL].map(difficulty_bin_map)
difficulty_levels = sorted(dfc["difficulty_n-1_bin"].dropna().unique())


# COLOR MAP FOR DIFFICULTY BINS
difficulty_levels = sorted(dfc["difficulty_n-1_bin"].dropna().unique())
cmap = cm.get_cmap("tab10", len(difficulty_levels))
color_map = {level: cmap(i) for i, level in enumerate(difficulty_levels)}


#for side in [0,1]:
#    dfc_side = dfc[dfc[ACTION_N_1] == side]
for rw, title_state in reward_states:
    print(f"\n\n=== {title_state}")
    plt.figure(figsize=(8, 6))
    for diff_level in difficulty_levels:
        dfc_D = dfc[dfc["difficulty_n-1_bin"] == diff_level]
        rat = TARGET_RAT
        #for rat in rats:
        dfr = dfc_D[
            (dfc_D[RAT_COL] == rat) &
            (dfc_D[REWARD_COL] == rw)
            ].dropna(subset=["difficulty_x", REPETITION_COL])
        print(f"Rat {rat} has {len(dfr)} trials for reward state {rw}.")
        if dfr.empty:
            continue

        agg = (
            dfr
            .groupby("difficulty_x")[REPETITION_COL]
            .agg(mean="mean", n="size")
            .reset_index()
            .sort_values("difficulty_x")
        )

        agg = agg[agg["n"] >= min_trials_per_x]
        if agg.empty:
            print(f"Skipping rat {rat} due to no bins with >= {min_trials_per_x} trials.")
            continue

        plt.plot(
            agg["difficulty_x"],
            agg["mean"],
            marker="o",
            linewidth=1.5,
            color= color_map[diff_level],
            label=f"diff {diff_level}"
        )

    plt.axhline(0.5, linestyle="--", linewidth=1)
    plt.ylim(0, 1)
    plt.xlabel("|angleₙ| (difficulty)")
    plt.ylabel("P(repeat)")
    plt.title(f"Repeat vs difficulty — {title_state}")

    plt.legend(
        title="Rat ID",
        fontsize=8,
        title_fontsize=9,
        frameon=False
    )

    plt.tight_layout()
    plt.show()

        # ---------- Plot 2: Switch rate ----------
    '''
    
    plt.figure(figsize=(8, 6))
    for rat in rats:
        dfr = dfc[
            (dfc[RAT_COL] == rat) &
            (dfc[REWARD_COL] == rw)
        ].dropna(subset=["difficulty_x", REPETITION_COL])

        if dfr.empty:
            continue

        agg = (
            dfr.groupby("difficulty_x")[REPETITION_COL]
                .agg(mean="mean", n="size", switch_rate=lambda x: 1 - x.mean())
                .reset_index()
                .sort_values("difficulty_x")
        )
        agg = agg[agg["n"] >= min_trials_per_x]
        if agg.empty:
            continue

        plt.plot(
            agg["difficulty_x"], agg["switch_rate"],
            marker="o", linewidth=1.5,
            color=cmap(rats.index(rat)),
            label=f"rat {rat}"
        )

plt.axhline(0.5, linestyle="--", linewidth=1)
plt.ylim(0, 1)
plt.xlabel("|angleₙ| (difficulty)")
plt.ylabel("P(switch)")
plt.title(f"Switch rate vs difficulty — {title_state}")
plt.legend(title="Rat ID", fontsize=8, title_fontsize=9, frameon=False)
plt.tight_layout()
plt.show()

'''