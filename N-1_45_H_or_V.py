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
BIN_COL_N_1     = 'bin_angle_n-1'     # previous-trial angle bin label
BIN_COL_N       = 'bin_angle_n'       # current-trial angle bin label
ANGLE_COL       = 'angle'             # current-trial raw angle          
HIT_COL         = 'hitmiss_n-1'       # filter on previous-trial hits (True/1) if desired
BIN_COL_N_MID   = 'bin_n_midpoint'    # numeric midpoint for current-trial bin (x-axis)
RAT_COL         = 'rat'               # subject/animal ID  
ACTION_N_1      = 'action_n-1'  # unique trial identifier
ACTION          = 'action'    # current-trial response (0/1)

an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data()
df_processed = an.create_lagged_features()

df_processed, angle_to_label_N_1 = fa.bin_by_unique_angles(df_processed, name_new_col=BIN_COL_N_1, n_groups=6, angle_col='angle_n-1', exclude_angle= None)
df_processed, angle_to_label_N  = fa.bin_by_unique_angles(df_processed, name_new_col=BIN_COL_N,   n_groups=6, angle_col='angle', exclude_angle= None)

# MY DATATSETS

dfc_H = df_processed[
    (df_processed[BIN_COL_N_1] == '45°') &
    (df_processed['action_n-1'] == 0) &
    (df_processed[BIN_COL_N].notna())
].copy()
blue_color = np.array([0.2, 0.4, 0.9, 1.0])

dfc_V = df_processed[
    (df_processed[BIN_COL_N_1] == '45°') &
    (df_processed['action_n-1'] == 1) &
    (df_processed[BIN_COL_N].notna())
].copy()
red_color = np.array([0.9, 0.2, 0.2, 1.0])

dataset_list = [dfc_H, dfc_V]

dfc = df_processed[
    (df_processed[BIN_COL_N_1] == '45°') &
    (df_processed[BIN_COL_N].notna())
].copy()

#THINGS THAT I NEED TO PLOT
# Current-bin midpoints (for x-axis)
bin_means_cur  = {b: fa.midpoint(b) for b in pd.Series(df_processed[BIN_COL_N]).dropna().unique()}

# Attach current-bin numeric midpoint to dfc so curve fits have proper x-values
dfc_V[BIN_COL_N_MID] = dfc_V[BIN_COL_N].map(bin_means_cur)
dfc_H[BIN_COL_N_MID] = dfc_H[BIN_COL_N].map(bin_means_cur)
dfc[BIN_COL_N_MID]   = dfc[BIN_COL_N].map(bin_means_cur)

# -----------------------------
# SETUP SUBPLOTS (one per rat)
# -----------------------------
rats  = sorted(pd.Series(df_processed[RAT_COL]).dropna().unique())
n     = len(rats)
ncols = int(np.ceil(np.sqrt(n))) 
nrows = int(np.ceil(n / ncols))  

fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows),
                         sharex=True, sharey=True, constrained_layout=True)
axes = np.atleast_1d(axes).reshape(-1)
# -----------------------------
for i in range(len(dataset_list)):
    agg_rats = fa.aggregate_data(dataset_list[i], BIN_COL_N_1, ANGLE_COL, ACTION, RAT_COL=RAT_COL)
    cur_dataset = dataset_list[i]
    # agg_rats has: mean, n, se per (prev-bin, current angle, rat) — filtered to min trials = 5.
    # -----------------------------
    # PLOT PER RAT
    # -----------------------------
    for ax, rat in zip(axes, rats):
        agg_rat = agg_rats[agg_rats[RAT_COL] == rat]
        sub = agg_rat[agg_rat[BIN_COL_N_1] == '45°']
        
        # iterate prev-bins in order of their numeric midpoint (so legend/colors match)
        # successes per current angle, then collapse into current-angle bins
        sub = sub.assign(successes=sub['mean'] * sub['n']).copy()
        sub[BIN_COL_N] = sub[ANGLE_COL].map(angle_to_label_N)
        sub = sub.dropna(subset=[BIN_COL_N]).copy()

        subN = (sub.groupby(BIN_COL_N, as_index=False)
                    .agg(n=('n', 'sum'), successes=('successes', 'sum')))
        subN['mean'] = subN['successes'] / subN['n']
        subN['se']   = np.sqrt(subN['mean'] * (1 - subN['mean']) / subN['n'])
        subN[BIN_COL_N_MID] = subN[BIN_COL_N].map(bin_means_cur)
        subN = subN.dropna(subset=[BIN_COL_N_MID]).sort_values(BIN_COL_N_MID)

        if i == 0:
                color = blue_color 
                lab = "45° judged H" 
        else:
                color = red_color
                lab = "45° judged V"

            # points with error bars
        ax.errorbar(
             subN[BIN_COL_N_MID], subN['mean'], yerr=subN['se'],
            fmt='o', ms=4, alpha=0.9, color=color
        )
            # fit raw trials for this rat × previous-bin using current-bin midpoint as x
        raw_r = dfc[(dfc[RAT_COL] == rat)  & (dfc[BIN_COL_N_1].astype(str) == str('45°'))]
        raw = cur_dataset[(cur_dataset[RAT_COL] == rat) & (cur_dataset[BIN_COL_N_1].astype(str) == str('45°'))]
        if not raw.empty:
                x = pd.to_numeric(raw[BIN_COL_N_MID], errors='coerce')
                y = pd.to_numeric(raw[ACTION],      errors='coerce')
                m = x.notna() & y.notna()
                if m.sum() >= 5 and x[m].nunique() >= 4:  # avoid degenerate fits
                    popt, ok, x_fit, y_fit = fit_psychometric_curve(x[m], y[m], min_trials=5)
                    popt_r, ok_r, x_fit_r, y_fit_r = fit_psychometric_curve(raw_r[BIN_COL_N_MID],raw_r[ACTION], min_trials=5)
                    if ok:
                        ax.plot(x_fit, y_fit, color=color,label = lab, alpha=0.9)
                        #'''
                        bias, slope, lapse_low, lapse_high = popt
                        text = (
                            f"μ = {bias:.2f}\n"
                            f"slope = {slope:.2f}\n"
                            f"λ_low = {lapse_low:.2f}\n"
                            f"λ_high = {lapse_high:.2f}"
                        )

                        # Different position for each condition
                        # Place the parameter text in different corners depending on condition
                        if i == 0:
                            # Blue curve → top-left corner
                            tx, ty = 0.02, 0.98
                            ha, va = "left", "top"
                        else:
                            # Red curve → bottom-right corner
                            tx, ty = 0.98, 0.02
                            ha, va = "right", "bottom"

                        ax.text(
                            tx, ty, text,
                            transform=ax.transAxes,   # axes coords: (0,0)=bottom-left, (1,1)=top-right
                            ha=ha,
                            va=va,
                            fontsize=8,
                            color=color,               # match curve color
                            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2)
                        )


                        #'''
                    if ok_r:
                        ax.plot(x_fit_r, y_fit_r, color='grey', alpha=0.3, ls='--')

        # decorations
        ax.axhline(0.5, color='k', ls='--', alpha=0.4)
        ax.axvline(45,  color='k', ls='--', alpha=0.4)
        ax.set_xlim(0, 90)
        ax.set_ylim(0, 1)
        ax.set_title(f"Rat {rat}")
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

fig.suptitle('Psychometric curves conditioned on previous-trial angle (per rat)',
                y=1.06, fontsize=14)

plt.show()