# Unified GLM analysis (replaces glm_families.py + glm_familiy1.py stil retrivable from git commit 10/11/2026 at 16:23).
# Per-rat GLMs with different sets of features (= different models), to disentangle the contribution of
# current stimulus, modality and history effects on rats' choices.
#
# All the models live in the MODEL_LIBRARY dict below, grouped in families:
#   A: current angle + modality effects
#   B: current angle + modality + history effects (angle n-1, action n-1, success/failure n-1, angle split by hit/miss)
#   C: categorical angle + history + interactions of history with ambiguity (WORK IN PROGRESS)
#   D: current angle + history effects, WITHOUT modality terms
# Choose which models to run by listing their keys in MODELS_TO_RUN.
#
# Trials where angle_n-1 == 45 are excluded by default (EXCLUDE_CORRUPTED_N1), because a bug in the data
# collection script recorded their hit/miss randomly as 0/1, so we cannot know if they were hits or misses.

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


from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))   # project root: functions_Ale.py, serial_dependence_analysis.py
from functions_Ale import flatten_model_results

from serial_dependence_analysis import SerialDependenceAnalyzer


# =============================
# CONFIG
# =============================
HISTORY_DEPTH = 1
BOUNDARY_ANGLE = 45
EXCLUDE_CORRUPTED_N1 = True   # drop trials where angle_n-1 == 45 (hit/miss of those trials is corrupted)
N_BOOT = 1000                 # bootstrap iterations for the CIs (use a small number, e.g. 1, only for quick tests)
CI_LEVEL = 95
SCALING = 'cont_only'
RANDOM_STATE = 42
N_JOBS = 4


# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=HISTORY_DEPTH, save_figures=False)
an.load_and_preprocess_data()
df = an.create_lagged_features()

if EXCLUDE_CORRUPTED_N1:
    df = df[df['angle_n-1'] != BOUNDARY_ANGLE]   # remove trials where n-1 = 45 as reward data there are corrupted


# FEATURE CREATION:
# modality dummies
df['tactile'] = df['mod'] == 1
# df['visual'] = df['mod'] == 2
df['visuotactile'] = df['mod'] == 3

# categorical angles, recoded as -1 left and +1 right
df['angle_cat'] = (df['angle'] > BOUNDARY_ANGLE).astype(int)  # 0 for angles <= 45, 1 for angles > 45
df['angle_cat'] = df['angle_cat']*2 - 1
df['angle_n-1_cat'] = (df['angle_n-1'] > BOUNDARY_ANGLE).astype(int)
df['angle_n-1_cat'] = df['angle_n-1_cat']*2 - 1
df['angle_cat_tactile'] = df['angle_cat']*df['tactile']
df['angle_cat_visuotactile'] = df['angle_cat']*df['visuotactile']

# re-code action_n-1 to be -1 for left and +1 for right, instead of 0/1, so that it can be used in the model as a signed variable (e.g. for interaction with hit_n-1)
df['action_n-1'] = df['action_n-1']*2 - 1
df['success_n-1'] = df['hit_n-1']*df['action_n-1']        # 1 if previous trial was a hit and the action was right (1*1), -1 if hit and left (1*-1), 0 if miss (0*anything)
df['failure_n-1'] = (1 - df['hit_n-1'])*df['action_n-1']  # 1 if previous trial was a miss and the action was right (1*1), -1 if miss and left (1*-1), 0 if hit (0*anything)

# continuous angles centred on the boundary
df['angle'] = df['angle'] - BOUNDARY_ANGLE
df['angle_n-1'] = df['angle_n-1'] - BOUNDARY_ANGLE
df['angle_hit_n-1'] = df['angle_n-1']*df['hit_n-1']         # angle of the previous trial if it was a hit, 0 if it was a miss
df['angle_miss_n-1'] = df['angle_n-1']*(1 - df['hit_n-1'])  # angle of the previous trial if it was a miss, 0 if it was a hit
df['angle_tactile'] = df['angle']*df['tactile']
df['angle_visuotactile'] = df['angle']*df['visuotactile']

# ambiguity and its interactions with history
df['ambiguity'] = -df['difficulty']  # higher values means more ambiguous trials, and we expect that ambiguity should have a negative effect on performance (i.e. higher ambiguity should lead to more errors)
df['H_action_n-1_ambiguity'] = df['action_n-1']*df['ambiguity']
df['H_success_n-1_ambiguity'] = df['success_n-1']*df['ambiguity']
df['H_failure_n-1_ambiguity'] = df['failure_n-1']*df['ambiguity']
df['H_angle_n-1_ambiguity'] = df['angle_n-1']*df['ambiguity']


# -----------------------------
# MODEL LIBRARY
# -----------------------------
# the key is the model name (used as label in plots and tables); the original name from the old scripts is in the comment
MODEL_LIBRARY = {

    # ---- FAMILY A: current angle + modality ----
    'A0_angle':                 ['angle'],                                                                      # features_0
    'A1_angle_mod':             ['angle', 'tactile', 'visuotactile'],                                           # features_1
    'A2_angle_mod_inter':       ['angle', 'tactile', 'visuotactile', 'angle_tactile', 'angle_visuotactile'],    # features_2
    # best model according to the previous analysis, but not very interpretable: we don't know if the effect of angle
    # is driven by tactile trials, visuotactile trials, or both (likely tactile, as they are more numerous)
    'A3_angle_inter_only':      ['angle', 'angle_visuotactile', 'angle_tactile'],                               # features_3

    # ---- FAMILY B: current angle + modality + history ----
    'B4_prev_angle':            ['angle', 'tactile', 'visuotactile', 'angle_tactile', 'angle_visuotactile', 'angle_n-1'],                                           # features_4
    'B5_prev_action':           ['angle', 'tactile', 'visuotactile', 'angle_tactile', 'angle_visuotactile', 'action_n-1'],                                          # features_5
    'B6_prev_angle_action':     ['angle', 'tactile', 'visuotactile', 'angle_tactile', 'angle_visuotactile', 'angle_n-1', 'action_n-1'],                             # features_6
    'B7_succ_fail':             ['angle', 'tactile', 'visuotactile', 'angle_tactile', 'angle_visuotactile', 'success_n-1', 'failure_n-1'],                          # features_7
    'B8_prev_angle_succ_fail':  ['angle', 'tactile', 'visuotactile', 'angle_tactile', 'angle_visuotactile', 'angle_n-1', 'success_n-1', 'failure_n-1'],             # features_8
    'B9_angle_hit_miss':        ['angle', 'tactile', 'visuotactile', 'angle_tactile', 'angle_visuotactile', 'angle_hit_n-1', 'angle_miss_n-1'],                     # features_9a
    'B10_angle_hit_miss_action':['angle', 'tactile', 'visuotactile', 'angle_tactile', 'angle_visuotactile', 'angle_hit_n-1', 'angle_miss_n-1', 'action_n-1'],        # features_10a
    'B11_angle_hit_miss_sf':    ['angle', 'tactile', 'visuotactile', 'angle_tactile', 'angle_visuotactile', 'angle_hit_n-1', 'angle_miss_n-1', 'success_n-1', 'failure_n-1'],  # features_11a (this might be a bit BS)

    # ---- FAMILY C: categorical angle + history x ambiguity interactions (WORK IN PROGRESS) ----
    # I still need to understand how I should code the interactions with ambiguity and how to interpret them
    'C1_action_x_ambiguity':    ['angle_cat', 'tactile', 'visuotactile', 'angle_cat_tactile', 'angle_cat_visuotactile', 'action_n-1', 'H_action_n-1_ambiguity'],                                                                         # features_9 (families)
    'C2_prev_angle_x_ambiguity':['angle_cat', 'tactile', 'visuotactile', 'angle_cat_tactile', 'angle_cat_visuotactile', 'angle_n-1', 'H_angle_n-1_ambiguity'],                                                                          # features_10 (families)
    'C3_succ_fail_x_ambiguity': ['angle_cat', 'tactile', 'visuotactile', 'angle_cat_tactile', 'angle_cat_visuotactile', 'angle_n-1', 'success_n-1', 'failure_n-1', 'H_success_n-1_ambiguity', 'H_failure_n-1_ambiguity'],                 # features_11 (families)

    # ---- FAMILY D: current angle + history, no modality terms ----
    'D0_angle':                         ['angle'],                                                              # features_10 (familiy1) = features_01 (families)
    'D1_action':                        ['angle', 'action_n-1'],                                                # features_11 (familiy1)
    'D2_prev_angle':                    ['angle', 'angle_n-1'],                                                 # features_12 (familiy1) = features_02 (families)
    'D3_prev_angle_action':             ['angle', 'angle_n-1', 'action_n-1'],                                   # features_13 (familiy1)
    'D4_action_success':                ['angle', 'action_n-1', 'success_n-1'],                                 # features_14 (familiy1)
    'D5_action_failure':                ['angle', 'action_n-1', 'failure_n-1'],                                 # features_15 (familiy1)
    'D6_action_prev_angle_success':     ['angle', 'action_n-1', 'angle_n-1', 'success_n-1'],                    # features_16 (familiy1)
    'D7_action_prev_angle_failure_hit': ['angle', 'action_n-1', 'angle_n-1', 'failure_n-1', 'hit_n-1'],         # features_17 (familiy1)
    'D8_hit_prev_angle_succ_fail':      ['angle', 'hit_n-1', 'angle_n-1', 'success_n-1', 'failure_n-1'],        # features_18 (familiy1)
    'D9_prev_angle_succ_fail':          ['angle', 'angle_n-1', 'success_n-1', 'failure_n-1'],                   # features_19 (familiy1)
    'D10_succ_fail':                    ['angle', 'success_n-1', 'failure_n-1'],                                # features_20 (familiy1)
    'D11_angle_hit_miss':               ['angle', 'angle_hit_n-1', 'angle_miss_n-1'],                           # features_03 (families)
    'D12_angle_hit_miss_succ_fail':     ['angle', 'angle_hit_n-1', 'angle_miss_n-1', 'success_n-1', 'failure_n-1'],  # features_04 (families)
}

# choose here which models to run (keys of MODEL_LIBRARY). Examples:
# family A: ['A0_angle', 'A1_angle_mod', 'A2_angle_mod_inter', 'A3_angle_inter_only']
# family B: ['B4_prev_angle', 'B5_prev_action', 'B6_prev_angle_action', 'B7_succ_fail', 'B8_prev_angle_succ_fail']
MODELS_TO_RUN = ['D11_angle_hit_miss', 'D12_angle_hit_miss_succ_fail']


# -----------------------------
# FIT MODELS (one GLM per rat, per model)
# -----------------------------
results = []
for model_name in MODELS_TO_RUN:
    result = an.analyze_individual_rats_ale_N(features_keys=MODEL_LIBRARY[model_name], df_processed=df, k=None, scaling=SCALING)
    # result is a dict with one key per rat, and each key is another dict with keys:
    # 'data', 'model', 'scaler', 'feature_names', 'train_accuracy', 'n_trials', 'coefficients', 'cross_validation_scores', 'aic', 'bic'
    results.append(result)


# -----------------------------
# PLOT RESULTS
# -----------------------------
# ---- one flat dataframe per model + one shared beta-column order (union across models) ----
flat_dfs = [flatten_model_results(m) for m in results]  # list of n dataframes, one per model, with columns: rat_id, feature_name, beta_mean, beta_ci_lower, beta_ci_upper, score_mean, score_std
beta_cols = []
for flat_df in flat_dfs:
    beta_cols.extend([col for col in flat_df.columns if col.startswith("beta_")])
beta_cols = sorted(set(beta_cols))  # unique and sorted (by alphabetical order, this is also something to change, as it is a bit arbitrary)
score_cols = ["score_mean", "score_std"]


for i, result in enumerate(results):
    ci_df = an.bootstrap_ci(result, n_iterations=N_BOOT, ci=CI_LEVEL,
                            random_state=RANDOM_STATE, n_jobs=N_JOBS)
    an.plot_bootstrap_ci(flat_dfs[i], ci_df, model_label=MODELS_TO_RUN[i], beta_cols=beta_cols)
    an.plot_forest_ci(ci_df, model_label="")


# -----------------------------
# MODEL COMPARISON (AIC / BIC)
# -----------------------------
plot_data = []
for i, model_results in enumerate(results):
    for rat_id, d in model_results.items():
        plot_data.append({
            'rat_id': rat_id,
            'model_name': MODELS_TO_RUN[i],  # Label for color coding
            'aic': d['aic'],
            'bic': d['bic']
        })
df_plot = pd.DataFrame(plot_data)
df_plot['delta_aic'] = df_plot.groupby('rat_id')['aic'].transform(lambda x: x - x.min())
df_plot['delta_bic'] = df_plot.groupby('rat_id')['bic'].transform(lambda x: x - x.min())

df_plot = df_plot.sort_values(['rat_id', 'model_name'])

# Create a pivot table for easy comparison
comparison_df = df_plot.pivot(index='rat_id', columns='model_name', values='delta_aic')

# Add a 'Winner' column to see which model hit 0.0
comparison_df['Best Model aic'] = comparison_df.idxmin(axis=1)

print("\n=== MODEL COMPARISON SUMMARY (Delta AIC) ===")
print(comparison_df.round(2).to_string())
print("============================================\n")

# Create a pivot table for easy comparison
comparison_df = df_plot.pivot(index='rat_id', columns='model_name', values='delta_bic')

# Add a 'Winner' column to see which model hit 0.0
comparison_df['Best Model bic'] = comparison_df.idxmin(axis=1)

print("\n=== MODEL COMPARISON SUMMARY (Delta BIC) ===")
print(comparison_df.round(2).to_string())
print("============================================\n")
