
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
MOD_COL = 'mod'  # e.g. 1 -->'T', 2-->'V', 3-->'VT'
REPETITION_COL = 'repetition_n-1'  # e.g. 'rep', 'alt'


# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data()
df_processed = an.create_lagged_features()

df_processed, _ = fa.bin_by_unique_angles(df_processed, name_new_col=BIN_COL_N_1, n_groups=2, angle_col='angle_n-1', exclude_angle=45)
df_processed, angle_to_label   = fa.bin_by_unique_angles(df_processed, name_new_col=BIN_COL_N,   n_groups=2, angle_col='angle', exclude_angle=45)
dfc = df_processed.dropna(subset=[BIN_COL_N_1, BIN_COL_N]).copy()
dfc[REPETITION_COL] = (dfc[ACTION_N_1] == dfc[ACTION]).astype(int) # 1: repetition, 0: alternation
# --- Get unique transitions ---
#dfc here is inherited from before so we have already added the bins, rat and bin mid points columns
#transitions = sorted(dfc[TRANS_COL].dropna().unique(), key=str)
#n_trans = len(transitions)
# -----------------------------
# --- Get unique modalities ---
modalities = sorted(dfc[MOD_COL].dropna().unique(), key=str)
n_mod = len(modalities)

# Current-bin midpoints (for x-axis)
bin_means_cur  = {b: fa.midpoint(b) for b in pd.Series(dfc[BIN_COL_N]).dropna().unique()}

# Attach current-bin numeric midpoint to dfc so curve fits have proper x-values
dfc[BIN_COL_N_MID] = dfc[BIN_COL_N].map(bin_means_cur)
# -----------------------------
# BOOTSTRAP PER TRANSITION TYPE
# -----------------------------
rat = 10  # which rat to analyze
boot_results = []
for md in modalities:
        dfc_tr = dfc[dfc[MOD_COL] == md].copy()
        dfc_tr = dfc_tr[dfc_tr[RAT_COL] == rat].copy()
        grup_col = [BIN_COL_N_1, RAT_COL, MOD_COL]   

        agg = (
            dfc_tr
            .groupby(grup_col)
            .agg(
                right_trials   = (ACTION_N_1, 'sum'),                        # number of 1s
                left_trials = (ACTION_N_1, lambda x: x.size - x.sum()),   # number of 0s
            )
            .reset_index()
        )

        # compute bottleneck side per rat/bin and store it per rat
        agg['min_side_count'] = agg[['right_trials', 'left_trials']].min(axis=1)
        agg['min_side_action'] = (agg['right_trials'] < agg['left_trials']).astype(int)
        worst = (
            agg.sort_values('min_side_count')
            .groupby(RAT_COL)
            .head(1)[[RAT_COL, BIN_COL_N_1, MOD_COL, 'min_side_count', 'min_side_action']]
        )
        worst = worst.reset_index(drop=True)

        print(f'modality: {md},\ntable:\n{worst}')


        n_bootstraps = 1000
        
        n_samples_per_condition_list = {}
        
        # bottleneck number: worst side, worst bin, per rat
        n_samples_per_condition = int(worst.loc[worst['rat'] == rat, 'min_side_count'].item())     
        n_samples_per_condition_list[rat] = n_samples_per_condition

        
        for b in range(n_bootstraps):
            for bin_label in agg[BIN_COL_N_1].unique():
                # subset per rat & prev_bin
                df_subset = dfc_tr[(dfc_tr[RAT_COL] == rat) & (dfc_tr[BIN_COL_N_1] == bin_label)]
                
                # split in "right" and "left" according to ACTION_N_1
                right_trials = df_subset[df_subset[ACTION_N_1] == 1]
                left_trials  = df_subset[df_subset[ACTION_N_1] == 0]
                # resample with replacement
                samp_r = right_trials.sample(n_samples_per_condition, replace=True)
                samp_l = left_trials.sample(n_samples_per_condition, replace=True)
                
                samp_dataset = pd.concat([samp_r, samp_l], ignore_index=True)

                popt, ok, x_fit_dummy, y_fit_dummy = fit_psychometric_curve(
                    samp_dataset[ANGLE_COL],
                    samp_dataset[ACTION]  # action on current trial
                )

                if not ok:
                    continue

                boot_results.append({
                    'md_mode': md,
                    'rat': rat,
                    'prev_bin': bin_label,
                    'bootstrap': b,
                    'mu':    popt[0],
                    'sigma': popt[1],
                    'gamma': popt[2],
                    'lapse': popt[3],
                    'n_per_side': n_samples_per_condition, # for reference
                })
                

# -----------------------------
# Analyses and bootstrap + plotting
# -----------------------------
boot_df = pd.DataFrame(boot_results)

group_cols = ['rat', 'prev_bin', 'md_mode',]
g = boot_df.groupby(group_cols)

summary = (
    g.agg(
        mu_med     = ('mu', 'median'),
        mu_low     = ('mu', lambda x: x.quantile(0.025)),
        mu_high    = ('mu', lambda x: x.quantile(0.975)),
        sigma_med  = ('sigma', 'median'),
        sigma_low  = ('sigma', lambda x: x.quantile(0.025)),
        sigma_high = ('sigma', lambda x: x.quantile(0.975)),
        gamma_med  = ('gamma', 'median'),
        lapse_med  = ('lapse', 'median'),
    )
    .reset_index()
)
# -----------------------------
# SAVE SUMMARY
# -----------------------------

from pathlib import Path

output_path_1 = Path("rat10_allmodalities_bootstrap_psychometric_params.csv")


if not output_path_1.exists():
    boot_df.to_csv(output_path_1, index=False)
    print(f"Saved {output_path_1}")
else:
    print(f"{output_path_1} already exists — not overwriting.")
