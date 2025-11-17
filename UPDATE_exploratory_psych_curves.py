UPDATE_exploratory_psych_curves.py
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

# serial_psychometric_by_prev_angle.py
# ------------------------------------------------------------
# Plots psychometric curves conditioned on previous-trial angle bins,
# using your existing functions_Ale and serial_dependence_analysis modules.
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from serial_dependence_analysis import SerialDependenceAnalyzer
from serial_dependence_analysis import fit_psychometric_curve
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

# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data()
df_processed = an.create_lagged_features()

# -----------------------------
# BINNING (6 groups; exclude 45°)
# NOTE: we build both prev-bin and current-bin labels; the second call returns angle_to_label
# that is the mapping for angles (the same for prev and curr), which we use later to plot by current bins.
# -----------------------------
df_processed, _ = fa.bin_by_unique_angles(df_processed, name_new_col=BIN_COL_N_1, n_groups=6, angle_col='angle_n-1', exclude_angle=45)
df_processed, angle_to_label   = fa.bin_by_unique_angles(df_processed, name_new_col=BIN_COL_N,   n_groups=6, angle_col='angle', exclude_angle=45)

# -----------------------------
# CLEANING
# Option A: just drop rows missing either prev-bin or current-bin labels
# -----------------------------
dfc = df_processed.dropna(subset=[BIN_COL_N_1, BIN_COL_N]).copy()

# -----------------------------
# Option B (comment ''' to filter only trials whose previous trial was a HIT and add ~ before d[HIT_COL] to keep False / 0s):
# -----------------------------
'''
dfc = (df_processed
        .dropna(subset=[BIN_COL_N_1, BIN_COL_N, HIT_COL])
        .loc[lambda d: d[HIT_COL].astype(bool)]   # add ~ before d[HIT_COL] to keep misses
        .copy())
'''

# -----------------------------
# AGGREGATE (mean & counts per prev-bin × current angle)
# -----------------------------
agg_rats = fa.aggregate_data(dfc, BIN_COL_N_1, ANGLE_COL, RESP_COL, RAT_COL=RAT_COL)
# agg_rats has: mean, n, se per (prev-bin, current angle, rat) — filtered to min trials = 5.

# -----------------------------
# COLOR MAPPING / MIDPOINTS
# -----------------------------
# Prev-bin midpoints (for color & legend sorting)
bin_means_prev = {b: fa.midpoint(b) for b in pd.Series(dfc[BIN_COL_N_1]).dropna().unique()}
color_by_bin   = fa.color_bin(bin_means_prev)

# Current-bin midpoints (for x-axis)
bin_means_cur  = {b: fa.midpoint(b) for b in pd.Series(dfc[BIN_COL_N]).dropna().unique()}

# Attach current-bin numeric midpoint to dfc so curve fits have proper x-values
dfc[BIN_COL_N_MID] = dfc[BIN_COL_N].map(bin_means_cur)

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

# -----------------------------
# PLOT PER RAT
# -----------------------------
for ax, rat in zip(axes, rats):
    agg_rat = agg_rats[agg_rats[RAT_COL] == rat]

    # iterate prev-bins in order of their numeric midpoint (so legend/colors match)
    for b in sorted(bin_means_prev, key=bin_means_prev.get):
        sub = agg_rat[agg_rat[BIN_COL_N_1] == b]
        if sub.empty:
            continue

        # successes per current angle, then collapse into current-angle bins
        sub = sub.assign(successes=sub['mean'] * sub['n']).copy()
        sub[BIN_COL_N] = sub[ANGLE_COL].map(angle_to_label)
        sub = sub.dropna(subset=[BIN_COL_N]).copy()

        subN = (sub.groupby(BIN_COL_N, as_index=False)
                  .agg(n=('n', 'sum'), successes=('successes', 'sum')))
        subN['mean'] = subN['successes'] / subN['n']
        subN['se']   = np.sqrt(subN['mean'] * (1 - subN['mean']) / subN['n'])
        subN[BIN_COL_N_MID] = subN[BIN_COL_N].map(bin_means_cur)
        subN = subN.dropna(subset=[BIN_COL_N_MID]).sort_values(BIN_COL_N_MID)

        color = color_by_bin[b]

        # points with error bars
        ax.errorbar(subN[BIN_COL_N_MID], subN['mean'], yerr=subN['se'],
                    fmt='o', ms=4, alpha=0.9, label=str(b), color=color)

        # fit raw trials for this rat × previous-bin using current-bin midpoint as x
        raw_r = dfc[dfc[RAT_COL] == rat]
        raw = dfc[(dfc[RAT_COL] == rat) & (dfc[BIN_COL_N_1].astype(str) == str(b))]
        if not raw.empty:
            x = pd.to_numeric(raw[BIN_COL_N_MID], errors='coerce')
            y = pd.to_numeric(raw[RESP_COL],      errors='coerce')
            m = x.notna() & y.notna()
            if m.sum() >= 5 and x[m].nunique() >= 4:  # avoid degenerate fits
                popt, ok, x_fit, y_fit = fit_psychometric_curve(x[m], y[m], min_trials=5)
                popt_r, ok_r, x_fit_r, y_fit_r = fit_psychometric_curve(raw_r[BIN_COL_N_MID],raw_r[RESP_COL], min_trials=5)
                if ok:
                    ax.plot(x_fit, y_fit, color=color, alpha=0.9)
                    '''
                    bias, slope, lapse_low, lapse_high = popt
                    text = (
                    f"μ = {bias:.2f}\n"
                    f"slope = {slope:.2f}\n"
                    f"λ_low = {lapse_low:.2f}\n"
                    f"λ_high = {lapse_high:.2f}"
                    )

                    ax.text(
                    0.98, 0.05, text,
                    transform=ax.transAxes,
                    ha="right", va="bottom",
                    fontsize=7,
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=2)
                    )
                    '''
                if ok_r:
                    ax.plot(x_fit_r, y_fit_r, color='gray', alpha=0.3, ls='--')

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

BIN_COL_N_1   = 'bin_angle_n-1'  
BIN_COL_N     = 'bin_angle_n'         # your previous-angle bins
TRANS_COL  = 'mod_transition_n-1'     # e.g. 'T->V', 'V->VT', ...
ANGLE_COL  = 'angle'
RESP_COL   = 'action'
HIT_COL = 'hitmiss_n-1'
BIN_COL_N_MID = 'bin_n_midpoint'
RAT_COL='rat'

# --- Get unique transitions ---
#dfc here is inherited from before so we have already added the bins, rat and bin mid points columns
transitions = sorted(dfc[TRANS_COL].dropna().unique(), key=str)
n_trans = len(transitions)

# --- recreate proper agg_rats with RAT_COL included ---
agg_rats = fa.aggregate_data(dfc, BIN_COL_N_1, ANGLE_COL, RESP_COL, RAT_COL= RAT_COL, MOD_TRANS_COL= TRANS_COL)

# --- Plot each transition in its own subplot per rat ---
for rat in dfc['rat'].unique():

    # --- Prepare subplot grid ---
    ncols = int(np.ceil(np.sqrt(n_trans)))
    nrows = int(np.ceil(n_trans / ncols))
    #fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows), sharex=True, sharey=True, constrained_layout=True)
    fig, axes = plt.subplots(nrows, ncols,figsize=(4*ncols, 4*nrows),sharex=True, sharey=True,constrained_layout=True)

    axes = np.array(axes).reshape(-1)  # flatten in case grid isn't full

    

    mask = agg_rats['rat'] == rat
    agg_rat = agg_rats[mask]
    print(f"Processing rat {rat} with {len(agg_rat)} aggregated rows.")
    for ax, tr in zip(axes, transitions):

        agg_tr = agg_rat[agg_rat[TRANS_COL] == tr]
        print(f"  Transition {tr} with {len(agg_tr)} aggregated rows.")
        # Loop over previous-angle bins
        for i, b in enumerate(sorted(bin_means_cur, key=lambda x: bin_means_cur[x])):
            print(f"    Previous bin {b}")
            sub = agg_tr[agg_tr[BIN_COL_N_1] == b].sort_values(ANGLE_COL)
            if sub.empty:
                continue

            sub = sub.assign(successes=sub['mean'] * sub['n'])
            sub[BIN_COL_N] = sub[ANGLE_COL].map(angle_to_label)
            sub = sub.dropna(subset=[BIN_COL_N])

            subN = (
                sub.groupby(BIN_COL_N, as_index=False)
                   .agg(n=('n', 'sum'),
                        successes=('successes', 'sum'))
            )
            subN['mean'] = subN['successes'] / subN['n']
            subN['se'] = np.sqrt(subN['mean'] * (1 - subN['mean']) / subN['n'])
            subN[BIN_COL_N_MID] = subN[BIN_COL_N].map(bin_means_cur)

            color = color_by_bin[b]

            ax.errorbar(
                subN[BIN_COL_N_MID], subN['mean'],
                yerr=subN['se'], fmt='o', ms=3.5,
                alpha=0.9, label=str(b), color=color
            )


            # fit reference curve
            raw_r = dfc[(dfc[RAT_COL] == rat) & (dfc[TRANS_COL] == tr)]
            # Fit curve
            mask_raw = (
                (dfc[RAT_COL] == rat) &
                (dfc[BIN_COL_N_1].astype(str) == str(b)) &
                (dfc[TRANS_COL] == tr)
            )
            raw = dfc[mask_raw]

           
            popt, ok, x_fit, y_fit = fit_psychometric_curve( raw[BIN_COL_N_MID], raw[RESP_COL], min_trials=5)
            popt_r, ok_r, x_fit_r, y_fit_r = fit_psychometric_curve(raw_r[BIN_COL_N_MID],raw_r[RESP_COL], min_trials=5)
            if ok:
                ax.plot(x_fit, y_fit, color=color, alpha=0.9)
                #bias, slope, lapse_low, lapse_high = popt # to be plotted but commented out for now
            if ok_r:
                ax.plot(x_fit_r, y_fit_r, color='gray', alpha=0.3, ls='--')

        # subplot decorations
        ax.axhline(0.5, color='k', ls='--', alpha=0.4)
        ax.axvline(45, color='k', ls='--', alpha=0.4)
        ax.set_xlim(0, 90)
        ax.set_ylim(0, 1)
        ax.set_aspect('auto')
        ax.set_title(f'{tr}', fontsize=12)
        ax.set_xlabel('Current angle (deg)')
        ax.set_ylabel('P(action = 1)')

    # --- Global figure tweaks (OUTSIDE the 'for ax, tr' loop) ---
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, title='Prev angle bin',
        loc='upper center', ncol=8,
        bbox_to_anchor=(0.5, 1.08), fontsize=9
    )
    fig.suptitle(
        f'Rat {rat} Psychometric curves by previous-trial angle '
        'for each modality transition',
        fontsize=14, y=1.04
    )
    plt.show()


print("DONE")
