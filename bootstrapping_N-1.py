
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
ANGLE_COL       = 'angle'             # current-trial raw angle
RESP_COL        = 'action'            # current-trial response (0/1)
HIT_COL         = 'hitmiss_n-1'       # filter on previous-trial hits (True/1) if desired
BIN_COL_N_MID   = 'bin_n_midpoint'    # numeric midpoint for current-trial bin (x-axis)
RAT_COL         = 'rat'               # subject/animal ID
trialID         = 'trialID'  
ACTION_N_1      = 'action_n-1'  # unique trial identifier
ACTION          = 'action'    # current-trial response (0/1)

# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data()
df_processed = an.create_lagged_features()

df_processed, _ = fa.bin_by_unique_angles(df_processed, name_new_col=BIN_COL_N_1, n_groups=6, angle_col='angle_n-1', exclude_angle=45)
df_processed, angle_to_label   = fa.bin_by_unique_angles(df_processed, name_new_col=BIN_COL_N,   n_groups=6, angle_col='angle', exclude_angle=45)
dfc = df_processed.dropna(subset=[BIN_COL_N_1, BIN_COL_N]).copy()
# Current-bin midpoints (for x-axis)
bin_means_cur  = {b: fa.midpoint(b) for b in pd.Series(dfc[BIN_COL_N]).dropna().unique()}

# Attach current-bin numeric midpoint to dfc so curve fits have proper x-values
dfc[BIN_COL_N_MID] = dfc[BIN_COL_N].map(bin_means_cur)
# -----------------------------
grup_col = [BIN_COL_N_1, RAT_COL]   # <-- remove HIT_COL from grouping

agg = (
    dfc
    .groupby(grup_col)
    .agg(
        right_trials   = (ACTION_N_1, 'sum'),                        # number of 1s
        left_trials = (ACTION_N_1, lambda x: x.size - x.sum()),   # number of 0s
    )
    .reset_index()
)


agg['min_side_count'] = agg[['right_trials', 'left_trials']].min(axis=1)
agg['min_side_action'] = (agg['right_trials'] < agg['left_trials']).astype(int)
worst = (
    agg.sort_values('min_side_count')
       .groupby(RAT_COL)
       .head(1)[[RAT_COL, BIN_COL_N_1, 'min_side_count', 'min_side_action']]
)
worst = worst.reset_index(drop=True)

#print(f'{worst}')
n_bootstraps = 1000
boot_results = []
# extract precise number of trials per rat to sample,  but it needs to be done inside the loop
#n_samples_per_condition = worst.loc[worst['Rat'] == rat, 'min_side_count'].item()
#n_samples_per_condition_80 = n_samples_per_condition*0.8
for rat in agg[RAT_COL].unique():
    # compute per rat the number of samples to draw 
    n_samples_per_condition = worst.loc[worst['rat'] == rat, 'min_side_count'].item()
    print(f'Rat {rat} will use {n_samples_per_condition} samples per condition (side).')
    n_samples_per_condition_80 = int(np.ceil(n_samples_per_condition*0.8)) - 1
    print(f'--> Using {n_samples_per_condition_80} samples per condition (side) for bootstrapping.')
    for bin_label in agg[BIN_COL_N_1].unique():
        # compute per rat and per bin the subset of data
        df_subset = dfc[(dfc[RAT_COL] == rat) & (dfc[BIN_COL_N_1] == bin_label)]
        right_trials = df_subset[df_subset[ACTION_N_1] == 1]
        left_trials  = df_subset[df_subset[ACTION_N_1] == 0]
        for b in range(n_bootstraps):
    # resample with replacement
            samp_r = right_trials.sample(n_samples_per_condition_80, replace=True)
            samp_l = left_trials.sample(n_samples_per_condition_80, replace=True)
            
            samp_dataset = pd.concat([samp_r, samp_l], ignore_index=True)

            popt, ok, x_fit, y_fit = fit_psychometric_curve(
            samp_dataset[BIN_COL_N_MID], 
            samp_dataset[ACTION]
            )

            if not ok:
                continue

            boot_results.append({
                'rat': rat,
                'prev_bin': bin_label,
                'bootstrap': b,
                'mu':    popt[0],
                'sigma': popt[1],
                'gamma': popt[2],
                'lapse': popt[3],
            })
            # PSE|mu, sensitivity|sigma, guess_rate|gamma, lapse_rate|lapse = popt
        # collect results per n-1 bin and per rat
boot_df = pd.DataFrame(boot_results)
        #print(boot_df.head(15))
        # compute mean and ci per rat and per n-1 bin
group_cols = [RAT_COL, 'prev_bin']   # or just [RAT_COL] if you ignore prev_bin
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
        # For plotting the bootstrap curves
        x_fit = np.linspace(0, 90, 200)

        for _, row in summary.iterrows():
            y_fit = cumulative_gaussian_lapse(
                x_fit,
                row['mu_med'],
                row['sigma_med'],
                row['gamma_med'],
                row['lapse_med'],
            )
            label = f"rat {row[RAT_COL]}, prev_bin {row['prev_bin']}"
            plt.plot(x_fit, y_fit, label=label)
            plt.show()


print("end")
        # FINAL REPORT




            




































'''
for rat_id in agg[RAT_COL].unique():
    for bin_label in agg[BIN_COL_N_1].unique():
        df_subset = agg[(agg[RAT_COL] == rat_id) & (agg[BIN_COL_N_1] == bin_label)]
        if df_subset.empty:
            print(f"Rat {rat_id}, Bin {bin_label}: 0 trials (no data).")
            continue
        
        total = int(df_subset['total'].values[0])
        correct = int(df_subset['correct_trials'].values[0])
        incorrect = int(df_subset['incorrect_trials'].values[0])
        print(f"Rat {rat_id}, Bin {bin_label} has: {df_subset['total'].values[0]} trials in TOTAL, of which {df_subset['correct_trials'].values[0]} are CORRECT and {df_subset['incorrect_trials'].values[0]} are INCORRECT.")
 
'''




        
        
        


