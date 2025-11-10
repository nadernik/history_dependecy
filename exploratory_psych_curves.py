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

# coping from serial_dependence_analysis.py and functions_Ale.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from serial_dependence_analysis import SerialDependenceAnalyzer
from serial_dependence_analysis import fit_psychometric_curve
import functions_Ale as fa

# LOAD DATA AND PREPROCESS
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)  # k=1 is enough for n-1 effects
an.load_and_preprocess_data()
df_processed = an.create_lagged_features()
#Converts already angles to 0–90°.
#Creates lagged features (so you get angle_n-1, action_n-1, etc.).

# BINNING n-1 BY UNIQUE ANGLES (and excluding 45°)
df_processed = fa.bin_by_unique_angles(df_processed, name_new_col='bin_angle_n-1', n_groups=11, angle_col='angle_n-1', exclude_angle=45) 
# we have 33 unique angles excluding 45, so 11 groups of 3 angles each
# these are my 11 bins: 0–5°, 8–10°, 15–20°, 21–26°, 27–30°, 34–40°, 50–53°, 55–60°, 65–70°, 73–80°, 82–90

# PLOTTING PSYCHOMETRIC CURVES CONDITIONED ON PREVIOUS-TRIAL ANGLE BINS

BIN_COL   = 'bin_angle_n-1'  
ANGLE_COL = 'angle'
RESP_COL  = 'action'
HIT_COL = 'hitmiss_n-1'   # <-- not used if you don't want to filter by hits/misses 


# Cleaning --> DROP NaNs(45°) in angle n-1 bins
dfc = df_processed.dropna(subset=[BIN_COL]).copy()

'''
#UNLOCK TO SEE THE EFFECTS CONTROLLED FOR PREVIOUS TRIAL HITS/MISSES
dfc = (df_processed
       .dropna(subset=[BIN_COL, HIT_COL])
       .loc[lambda d:  d[HIT_COL].astype(bool)]      # keeps True / 1 # add ~ before d[HIT_COL] to keep False / 0
       .copy())
''' 

# Create a summary table paired per (n-1_bin, current angle)
agg = fa.aggregate_data(dfc, BIN_COL, ANGLE_COL, RESP_COL)
# agg is my mini dataframe with mean, n and standard error for each combination of previous angle bin and current angle (only for combinations that have at least 5 trials)

# --- Color mapping ---
bin_means = {b: fa.midpoint(b) for b in dfc[BIN_COL].unique()}
plt.figure(figsize=(10, 7))
# --- Prepare unique color per bin, grouped by angle family (<45° = blues, >45° = reds)
color_by_bin = fa.color_bin(bin_means)

# --- Plot each previous-angle bin ---
for i, b in enumerate(sorted(bin_means, key=lambda x: bin_means[x])):
    sub = agg[agg[BIN_COL] == b] #sub --> all rows where for ex. BIN_COL == (0,15]:
    sub = sub.sort_values(ANGLE_COL) # apparently not necessary, I can't notice any difference in the plot but just to be sure
    if sub.empty: #sub --> there are no matches for that bin (but this should have been filtered out before)
        continue

    color = color_by_bin[b]

    # Dots with error bars
    plt.errorbar(sub[ANGLE_COL], sub['mean'], yerr=sub['se'],
                 fmt='o', ms=4, alpha=0.9, label=str(b), color=color)
    
    # Fit psychometric curve using function from serial_dependence_analysis.py
    raw = dfc[dfc[BIN_COL] == b]
    popt, ok, x_fit, y_fit = fit_psychometric_curve(
        raw[ANGLE_COL], raw[RESP_COL], min_trials=5
    )
    if ok:
        plt.plot(x_fit, y_fit, color=color, alpha=0.9)
# the gaussian curves are fitted with raw data not with means

# Decorations
plt.axhline(0.5, color='k', ls='--', alpha=0.4)
plt.axvline(45, color='k', ls='--', alpha=0.4)
plt.xlim(0, 90)
plt.ylim(0, 1)
plt.xlabel('Current angle (deg)')
plt.ylabel('P(action = 1)')
plt.title('Psychometric curves conditioned on previous-trial angle, only n-1 miss', fontsize=14)
plt.legend(title='Prev angle bin', fontsize=9)
plt.tight_layout()
plt.show()

#outliers = agg[(agg['mean'] == 0) | (agg['mean'] == 1)]
#print(outliers[['angle', BIN_COL, 'mean', 'n']])


# NOW PLOTS PER MODALITY TRANSITION TYPE
# let's redifine some variables for clarity

BIN_COL    = 'bin_angle_n-1'          # your previous-angle bins
TRANS_COL  = 'mod_transition_n-1'     # e.g. 'T->V', 'V->VT', ...
ANGLE_COL  = 'angle'
RESP_COL   = 'action'
HIT_COL = 'hitmiss_n-1'

#dfc = df_processed.dropna(subset=[BIN_COL, TRANS_COL]).copy()

# --- Helper: get numeric midpoint of bin label ---

#bin_means = {b: fa.midpoint(b) for b in dfc[BIN_COL].unique()}


# --- Get unique transitions ---
transitions = sorted(dfc[TRANS_COL].dropna().unique(), key=str)
n_trans = len(transitions)

# --- Prepare subplot grid ---
ncols = int(np.ceil(np.sqrt(n_trans)))
nrows = int(np.ceil(n_trans / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4.5*nrows), sharex=True, sharey=True, constrained_layout=True)
axes = np.array(axes).reshape(-1)  # flatten in case grid isn't full

# --- Plot each transition in its own subplot ---
for ax, tr in zip(axes, transitions):
    sub_tr = dfc[dfc[TRANS_COL] == tr]
    agg = fa.aggregate_data(sub_tr, BIN_COL, ANGLE_COL, RESP_COL)

    # Loop over previous-angle bins
    for i, b in enumerate(sorted(bin_means, key=lambda x: bin_means[x])):
        sub = agg[agg[BIN_COL] == b].sort_values(ANGLE_COL)
        if sub.empty:
            continue
        # choose color family
        color = color_by_bin[b]

        # dots with error bars
        ax.errorbar(sub[ANGLE_COL], sub['mean'], yerr=sub['se'],
                    fmt='o', ms=3.5, alpha=0.9, label=str(b), color=color)

        # fit psychometric curve
        raw = sub_tr[sub_tr[BIN_COL] == b]
        popt, ok, x_fit, y_fit = fit_psychometric_curve(
            raw[ANGLE_COL], raw[RESP_COL], min_trials=5
        )
        if ok:
            ax.plot(x_fit, y_fit, color=color, alpha=0.9)

    # subplot decorations
    ax.axhline(0.5, color='k', ls='--', alpha=0.4)
    ax.axvline(45, color='k', ls='--', alpha=0.4)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1)
    ax.set_title(f'{tr}', fontsize=12)
    ax.set_xlabel('Current angle (deg)')
    ax.set_ylabel('P(action = 1)')

# --- Global figure tweaks ---
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, title='Prev angle bin', loc='upper center',
           ncol=8, bbox_to_anchor=(0.5, 1.08), fontsize=9)
fig.suptitle('Psychometric curves by previous-trial angle for each modality transition', fontsize=14, y=1.04)
plt.show()
































'''
# Save original  angles between 90° and 134° and their indices --> check that the transformation is correct
in the future maybe we want to plot original vs mirrored angles to be see if they behave as we expected ro not
mask_check = (df['angle'] > 90) & (df['angle'] < 134)
original_angles = df.loc[mask_check, 'angle'].copy()

# Apply the transformation
df.loc[mask_check, 'angle'] = (90 - (df.loc[mask_check, 'angle'] - 90))

# Compare old vs new
comparison = pd.DataFrame({
    'original_angle': original_angles,
    'mirrored_angle': df.loc[mask_check, 'angle']
}).reset_index(drop=True)

print(comparison)
'''

'''
# 5. Filter to only angles 0-90
df = df[(df['angle'] >= 0) & (df['angle'] <= 90)]

# 6. Define reversed rule rats
rev_rule_rats = [6, 7, 14, 15, 16, 17]

# 7. Define colors for modalities
modality_colors = [[0, 2/3, 0], [0, 0.4470, 0.7410], [1, 0, 0], [0, 0, 0]]

print(f"Data loaded and simplified successfully!")
print(f"Total trials after filtering: {len(df)}")
print(f"Rats included: {sorted(rats)}")
print(f"Angle range: {df['angle'].min():.1f} to {df['angle'].max():.1f}")
print(f"Action values: {sorted(df['action'].unique())}")
print(f"Hit/Miss values: {sorted(df['hitmiss'].unique())}")
print(f"Modalities: {sorted(df['mod'].unique())}")
print(f"Reversed rule rats: {rev_rule_rats}")
print(f"\nFirst few rows:")
print(df[['rat', 'action', 'hitmiss', 'mod', 'angle']].head())



# Plot simplified distributions
fig, axes = plt.subplots(2,2 figsize=(15, 10))

# Angle distribution (0-90)
axes[0,0].hist(df['angle'], bins=20, alpha=0.7, edgecolor='black', color='black')
axes[0,0].set_title('Distribution of Angles (0-90°)')
axes[0,0].set_xlabel('Angle (degrees)')
axes[0,0].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
'''