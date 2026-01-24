
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
ANGLE_COL       = 'angle'             # current-trial raw angle 
RAT_COL         = 'rat'               # subject/animal ID
ACTION_N_1      = 'action_n-1'  # previous-trial response (0/1)
ACTION          = 'action'    # current-trial response (0/1)

# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data()
df_processed = an.create_lagged_features()

df_processed, _ = fa.bin_by_unique_angles(df_processed, name_new_col=BIN_COL_N_1, n_groups=2, angle_col='angle_n-1', exclude_angle=45)
df_processed, angle_to_label   = fa.bin_by_unique_angles(df_processed, name_new_col=BIN_COL_N,   n_groups=2, angle_col='angle', exclude_angle=45)
dfc = df_processed.dropna(subset=[BIN_COL_N_1, BIN_COL_N]).copy()


# Current-bin midpoints (for x-axis)
bin_means_cur  = {b: fa.midpoint(b) for b in pd.Series(dfc[BIN_COL_N]).dropna().unique()}

# Attach current-bin numeric midpoint to dfc so curve fits have proper x-values
dfc[BIN_COL_N_MID] = dfc[BIN_COL_N].map(bin_means_cur)
# -----------------------------
# DETERMINE BOTTLENECK SAMPLE SIZE PER RAT
# -----------------------------
grup_col = [BIN_COL_N_1, RAT_COL]   

agg = (
    dfc
    .groupby(grup_col)
    .agg(
        right_trials   = (ACTION_N_1, 'sum'),                        # number of 1s
        left_trials = (ACTION_N_1, lambda x: x.size - x.sum()),   # number of 0s
    )
    .reset_index()
)

# determine bottleneck: min number of trials on the less-sampled side (left/right) per rat and prev_bin
agg['min_side_count'] = agg[['right_trials', 'left_trials']].min(axis=1)
agg['min_side_action'] = (agg['right_trials'] < agg['left_trials']).astype(int)
worst = (
    agg.sort_values('min_side_count')
       .groupby(RAT_COL)
       .head(1)[[RAT_COL, BIN_COL_N_1, 'min_side_count', 'min_side_action']]
)
worst = worst.reset_index(drop=True)
print(f"Bottleneck (min side) per rat:{worst}")

# -----------------------------
# BOOTSTRAPPING PSYCHOMETRIC CURVE FITS
# -----------------------------
n_bootstraps = 1000
boot_results = []
n_samples_per_condition_80_list = {}
n_samples_per_condition_list= {}
for rat in agg[RAT_COL].unique():
    # bottleneck number: worst side, worst bin, per rat
    n_samples_per_condition = int(worst.loc[worst['rat'] == rat, 'min_side_count'].item())
    
    # use 80% rounded, minus 1 so that we don't round above to 81%
    #n_samples_per_condition_80 = max(1, int(np.ceil(n_samples_per_condition * 0.8)) - 1) #this is the number of samples per side(R\L\action) that we will use for bootstrapping
    #print(f'Rat {rat} will use {n_samples_per_condition_80}, so 80% of {n_samples_per_condition} samples per condition (side) for bootstrapping.')
    #n_samples_per_condition_80_list[rat] = n_samples_per_condition_80
    n_samples_per_condition_list[rat] = n_samples_per_condition
    for bin_label in agg[BIN_COL_N_1].unique():
        # subset per rat & prev_bin
        df_subset = dfc[(dfc[RAT_COL] == rat) & (dfc[BIN_COL_N_1] == bin_label)]
        
        # create a right and left dataset to sample from by splitting in "right" and "left" according to ACTION_N_1
        right_trials = df_subset[df_subset[ACTION_N_1] == 1]
        left_trials  = df_subset[df_subset[ACTION_N_1] == 0]
        
        for b in range(n_bootstraps):
            # resample with replacement
            samp_r = right_trials.sample(n_samples_per_condition, replace=True)
            samp_l = left_trials.sample(n_samples_per_condition, replace=True)

            samp_dataset = pd.concat([samp_r, samp_l], ignore_index=True)

            popt, ok, x_fit_dummy, y_fit_dummy = fit_psychometric_curve(
                samp_dataset[ANGLE_COL],
                samp_dataset[ACTION]
            )
            #print(f"Rat {rat}, prev_bin {bin_label}, bootstrap {b}: popt = {popt}, ok = {ok}")

            if not ok:
                print(f"Fit failed for rat {rat}, prev_bin {bin_label}, bootstrap {b}. Skipping.")
                continue
            # store results so we can compute median and CIs later
            boot_results.append({
                'rat': rat,
                'prev_bin': bin_label,
                'bootstrap': b,
                'mu':    popt[0],
                'sigma': popt[1],
                'gamma': popt[2],
                'lapse': popt[3],
                'n_per_side': n_samples_per_condition, # for reference
            })
# END OF  FIRST PART
# -----------------------------
# Analyses and bootstrap + plotting
# -----------------------------
boot_df = pd.DataFrame(boot_results)

group_cols = ['rat', 'prev_bin']
g = boot_df.groupby(group_cols)

summary = (
    g.agg(
        mu_med     = ('mu', 'median'),
        mu_q025    = ('mu', lambda x: x.quantile(0.025)),
        mu_q975    = ('mu', lambda x: x.quantile(0.975)),

        sigma_med  = ('sigma', 'median'),
        sigma_q025 = ('sigma', lambda x: x.quantile(0.025)),
        sigma_q975 = ('sigma', lambda x: x.quantile(0.975)),

        gamma_med  = ('gamma', 'median'),
        gamma_q025 = ('gamma', lambda x: x.quantile(0.025)),
        gamma_q975 = ('gamma', lambda x: x.quantile(0.975)),

        lapse_med  = ('lapse', 'median'),
        lapse_q025 = ('lapse', lambda x: x.quantile(0.025)),
        lapse_q975 = ('lapse', lambda x: x.quantile(0.975)),
    )
    .reset_index()
)
# -----------------------------
# SAVE RESULTS SAFELY

from pathlib import Path

output_path_1 = Path("ALL_bootstrap_psychometric_params.csv")
output_path_2 = Path("summary_bootstrap_psychometric_params.csv")

if not output_path_1.exists():
    boot_df.to_csv(output_path_1, index=False)
    print(f"Saved {output_path_1}")
else:
    print(f"{output_path_1} already exists — not overwriting.")

if not output_path_2.exists():
    summary.to_csv(output_path_2, index=False)
    print(f"Saved {output_path_2}")
else:
    print(f"{output_path_2} already exists — not overwriting.")