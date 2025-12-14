
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
from serial_dependence_analysis import fit_psychometric_curve, cumulative_gaussian_lapse,cumulative_gaussian_fixed_lapse
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


agg['min_side_count'] = agg[['right_trials', 'left_trials']].min(axis=1)
agg['min_side_action'] = (agg['right_trials'] < agg['left_trials']).astype(int)
worst = (
    agg.sort_values('min_side_count')
       .groupby(RAT_COL)
       .head(1)[[RAT_COL, BIN_COL_N_1, 'min_side_count', 'min_side_action']]
)
worst = worst.reset_index(drop=True)


n_bootstraps = 1000
boot_results = []
n_samples_per_condition_80_list = {}
for rat in agg[RAT_COL].unique():
    # bottleneck number: worst side, worst bin, per rat
    n_samples_per_condition = worst.loc[worst['rat'] == rat, 'min_side_count'].item()
    
    # use 80% rounded, minus 1 so that we don't round above to 81%
    n_samples_per_condition_80 = max(1, int(np.ceil(n_samples_per_condition * 0.8)) - 1)
    print(f'Rat {rat} will use {n_samples_per_condition_80}, so 80% of {n_samples_per_condition} samples per condition (side) for bootstrapping.')
    n_samples_per_condition_80_list[rat] = n_samples_per_condition_80
    for bin_label in agg[BIN_COL_N_1].unique():
        # subset per rat & prev_bin
        df_subset = dfc[(dfc[RAT_COL] == rat) & (dfc[BIN_COL_N_1] == bin_label)]
        
        # split in "right" and "left" according to ACTION_N_1
        right_trials = df_subset[df_subset[ACTION_N_1] == 1]
        left_trials  = df_subset[df_subset[ACTION_N_1] == 0]
        
        # if data are missing in one side, skip this rat/bin combo
        #if (len(right_trials) == 0) or (len(left_trials) == 0):
            #print(f"Skipping rat {rat}, prev_bin {bin_label}: not enough trials on at least one side.")
            #continue
        
        for b in range(n_bootstraps):
            # resample with replacement
            samp_r = right_trials.sample(n_samples_per_condition_80, replace=True)
            samp_l = left_trials.sample(n_samples_per_condition_80, replace=True)
            
            samp_dataset = pd.concat([samp_r, samp_l], ignore_index=True)

            popt, ok, x_fit_dummy, y_fit_dummy = fit_psychometric_curve(
                samp_dataset[ANGLE_COL],
                samp_dataset[ACTION], minimal_curvefit=True
            )
            print(f"Rat {rat}, prev_bin {bin_label}, bootstrap {b}: popt = {popt}, ok = {ok}")

            if not ok:
                print(f"Fit failed for rat {rat}, prev_bin {bin_label}, bootstrap {b}. Skipping.")
                continue

            boot_results.append({
                'rat': rat,
                'prev_bin': bin_label,
                'bootstrap': b,
                'mu':    popt[0],
                'sigma': popt[1],
                'gamma': 0.02,
                'lapse': 0.02,
            })

# -----------------------------
# Analyses and bootstrap + plotting
# -----------------------------
boot_df = pd.DataFrame(boot_results)

group_cols = ['rat', 'prev_bin']
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
# SETUP SUBPLOTS (one per rat)
# -----------------------------
rats  = sorted(pd.Series(dfc[RAT_COL]).dropna().unique())
n     = len(rats)
ncols = int(np.ceil(np.sqrt(n))) 
nrows = int(np.ceil(n / ncols))  

fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows),
                         sharex=True, sharey=True, constrained_layout=True)
axes = np.atleast_1d(axes).reshape(-1)
# plotting finale
x_fit = np.linspace(0, 90, 200)
color_by_bin, _   = fa.color_bin(bin_means_cur)

for ax, rat in zip(axes, boot_df[RAT_COL].unique()):
    #plt.figure()
    sub = summary[summary[RAT_COL] == rat]
    raw_r = dfc[dfc[RAT_COL] == rat]
    popt_r, ok_r, x_fit_r, y_fit_r = fit_psychometric_curve(raw_r[ANGLE_COL],raw_r[ACTION], min_trials=5, minimal_curvefit=True)
    if ok_r:
        ax.plot(x_fit_r, y_fit_r, color='black', alpha=0.8, ls='--')
    for _, row in sub.iterrows():
        #raw_r_b= dfc[(dfc[RAT_COL] == rat) & (dfc[BIN_COL_N_1] == row['prev_bin'])]

        y_fit = cumulative_gaussian_fixed_lapse(
            x_fit,
            row['mu_med'],
            row['sigma_med'],
            row['gamma_med'],
            row['lapse_med'],
        )

        prev_bin = row['prev_bin'] 
        color = color_by_bin[prev_bin]

        # -------------------------
        # 2) CURVE BOOTSTRAP → CI
        # -------------------------
        # all bootstraps of this rat/prev_bin combination
        boot_sub = boot_df[(boot_df[RAT_COL] == rat) &
                        (boot_df['prev_bin'] == prev_bin)]

        # if we have enough bootstrap samples, we compute CI bands
        if len(boot_sub) > 0:
            params = boot_sub[['mu', 'sigma', 'gamma', 'lapse']].to_numpy()

            y_boot = np.array([
                cumulative_gaussian_fixed_lapse(x_fit, p[0], p[1]) #(x_fit, p[0], p[1], p[2], p[3])
                for p in params
            ])

            
            y_low  = np.quantile(y_boot, 0.025, axis=0)
            y_high = np.quantile(y_boot, 0.975, axis=0)

            # CI band
            ax.fill_between(
                x_fit,
                y_low,
                y_high,
                color=color,
                alpha=0.6,   # transparency
                linewidth=0
            )

        label = f"prev_bin {row['prev_bin']}"
        ax.plot(x_fit, y_fit, color=color, alpha=0.9, label=label)
        
        #plt.plot(x_fit, y_fit, label=label, color = color)
        #popt_r_b, ok_r_b, x_fit_r_b, y_fit_r_b = fit_psychometric_curve(raw_r_b[BIN_COL_N_MID],raw_r_b[ACTION], min_trials=5)
        #if ok_r_b:
            #ax.plot(x_fit_r_b, y_fit_r_b, color='black', alpha=0.3, ls='--')
    
    '''
    plt.xlabel("Angle (bin midpoint)")
    plt.ylabel("P(right)")
    plt.legend()
    plt.title("Bootstrap psychometric curves by rat and previous bin")
    plt.show()
    '''
    # decorations
    ax.axhline(0.5, color='k', ls='--', alpha=0.4)
    ax.axvline(45,  color='k', ls='--', alpha=0.4)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1)
    ax.set_title(f"Rat {rat}, N = {n_samples_per_condition_80_list[rat]} samples")
    ax.set_xlabel('Current angle (deg)')
    ax.set_ylabel('P(action = 1)')

# hide any unused axes
for ax in axes[len(rats):]:
    ax.set_visible(False)

# shared legend (only if we actually plotted something)
handles, labels = axes[0].get_legend_handles_labels()
if handles:
    fig.legend(handles, labels, title='Prev angle bin',
               loc='upper center', ncol=min(6, len(handles)),
               bbox_to_anchor=(0.5, 1.02), fontsize=9)

fig.suptitle('Bootstrapped Psychometric curves conditioned on previous-trial angle bin (per rat)',
             y=1.06, fontsize=14)

plt.show()