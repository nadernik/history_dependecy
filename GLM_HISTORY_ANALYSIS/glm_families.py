# glm analysis on the hole dataset, but excluding trials where n-1 angle = 45° (as they are corrupted by a bug in the data collection script 
# that caused the hitmiss of those trials to be recorded as randomly 0/1, so we cannot reliably determine if they were hits or misses), 
# and with a specific focus on the effect of current angle and its interaction with modality, history effects, and ambiguity.
# three families of models are implemented:
# FAMILY 1: current angle + modaliy effects
# FAMILY 2: current angle and history effects (e.g. hit/miss, choice, angle of the previous trial) + modality effects
# FAMILY 3: current angle, history effects, and interactions between current angle and history effects + modality effects (WORKIN PROGRESS)

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
from functions_Ale import flatten_model_results
   
from serial_dependence_analysis import SerialDependenceAnalyzer
HISTORY_DEPTH = 1


# =============================
# CONFIG
# =============================
RAT_COL        = "rat"
ANGLE_COL      = "angle"          # trial n angle
ACTION_COL     = "action"         # trial n choice (0/1)
ANGLE_N1_COL   = "angle_n-1"      # trial n-1 angle
ACTION_N1_COL  = "action_n-1"     # trial n-1 choice (0/1)
BOUNDARY_ANGLE = 45
HITMISS_N1_COL = "hitmiss_n-1"    # trial n-1 outcome (0/1)

# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=HISTORY_DEPTH, save_figures=False)
an.load_and_preprocess_data()
df = an.create_lagged_features()

df = df[df['angle_n-1'] != 45] #remove trials where n-1 = 45 as reward data there are corrupted 


# FEATURE CREATION:
# categorical angles
df['tactile'] = df['mod'] == 1
# df['visual'] = df['mod'] == 2
df['visuotactile'] = df['mod'] == 3
df['angle_cat'] = (df['angle'] > BOUNDARY_ANGLE).astype(int) # 0 for angles <= 45, 1 for angles > 45
df['angle_cat'] = df['angle_cat']*2 -1 # recoded as -1 left and +1 right
df['angle_n-1_cat'] = (df['angle_n-1'] > BOUNDARY_ANGLE).astype(int) # 0 for angles <= 45, 1 for angles > 45
df['angle_n-1_cat'] = df['angle_n-1_cat']*2 -1 # recoded as -1 left and +1 right
df['angle_cat_tactile'] = df['angle_cat']*df['tactile']
df['angle_cat_visuotactile'] = df['angle_cat']*df['visuotactile']
# re-code action_n-1 to be -1 for left and +1 for right, instead of 0/1, so that it can be used in the model as a signed variable (e.g. for interaction with hit_n-1)
df['action_n-1'] = df['action_n-1']*2 -1
#df['hitmiss_n-1'] = df['hitmiss_n-1']*2 -1
df['success_n-1'] = df['hit_n-1']*df['action_n-1'] # 1 if previous trial was a hit and the action was right (1*1), -1 if hit and left (1*-1), 0 if miss (0*anything)
df['failure_n-1'] = (1 - df['hit_n-1'])*df['action_n-1'] # 1 if previous trial was a miss and the action was right (1*1), -1 if miss and left (1*-1), 0 if hit (0*anything)
#df['angle_hit_n-1'] = df['angle_n-1']*df['hit_n-1'] # angle of the previous trial if it was a hit, 0 if it was a miss
#df['angle_miss_n-1'] = df['angle_n-1']*(1 - df['hit_n-1']) # angle of the previous trial if it was a miss, 0 if it was a hit
df['angle'] = df['angle'] - 45
df['angle_n-1'] = df['angle_n-1'] - 45
df['angle_hit_n-1'] = df['angle_n-1']*df['hit_n-1'] # angle of the previous trial if it was a hit, 0 if it was a miss
df['angle_miss_n-1'] = df['angle_n-1']*(1- df['hit_n-1']) # angle of the previous trial if it was a miss, 0 if it was a hit
df['angle_tactile'] = df['angle']*df['tactile']
df['angle_visuotactile'] = df['angle']*df['visuotactile']
df['ambiguity'] = -df['difficulty'] # higher values means more ambiguous trials, and we expect that ambiguity should have a negative effect on performance (i.e. higher ambiguity should lead to more errors)
df['H_action_n-1_ambiguity'] = df['action_n-1']*df['ambiguity']
df['H_success_n-1_ambiguity'] = df['success_n-1']*df['ambiguity']
df['H_failure_n-1_ambiguity'] = df['failure_n-1']*df['ambiguity']
df['H_angle_n-1_ambiguity'] = df['angle_n-1']*df['ambiguity']


# 4 models with increasing complexity: now it is a bit hardcoded but we can easily make it more flexible later
# FAMILY 1: currentangle
features_0 = ['angle']
features_1 = ['angle', 'tactile', 'visuotactile'] 
features_2 = ['angle', 'tactile', 'visuotactile', 'angle_tactile', 'angle_visuotactile']
features_3 = ['angle', 'angle_visuotactile', 'angle_tactile'] # this is the best model according to the previous analysis, but it is not very interpretable, as we don't know if the effect of angle is driven by tactile trials, visuotactile trials, or both (it is likely that it is driven by tactile trials, as they are more numerous and have a stronger effect, but we cannot be sure without looking at the coefficients)
# FAMILY 2: current angle and history effects (e.g. hit/miss, choice, angle of the previous trial)
features_4 = ['angle', 'tactile', 'visuotactile', 'angle_tactile', 'angle_visuotactile', 'angle_n-1']
features_5 = ['angle', 'tactile', 'visuotactile', 'angle_tactile', 'angle_visuotactile', 'action_n-1']
features_6 = ['angle', 'tactile', 'visuotactile', 'angle_tactile', 'angle_visuotactile', 'angle_n-1', 'action_n-1']
features_7 = ['angle', 'tactile', 'visuotactile', 'angle_tactile', 'angle_visuotactile', 'success_n-1', 'failure_n-1'] 
features_8 = ['angle', 'tactile', 'visuotactile', 'angle_tactile', 'angle_visuotactile', 'angle_n-1', 'success_n-1', 'failure_n-1']
# other members of the family 2 could be:
features_9a = ['angle', 'tactile', 'visuotactile', 'angle_tactile', 'angle_visuotactile', 'angle_hit_n-1', 'angle_miss_n-1'] 
features_10a = ['angle', 'tactile', 'visuotactile', 'angle_tactile', 'angle_visuotactile', 'angle_hit_n-1', 'angle_miss_n-1', 'action_n-1']
features_11a = ['angle', 'tactile', 'visuotactile', 'angle_tactile', 'angle_visuotactile', 'angle_hit_n-1', 'angle_miss_n-1', 'success_n-1', 'failure_n-1'] # this might be a bit BS

# FAMILY 3: current angle, history effects, and interactions between current angle and history effects 
# I still need to understand how I should code the interctions with ambiguity and how to interpret them
features_9 = ['angle_cat', 'tactile', 'visuotactile', 'angle_cat_tactile', 'angle_cat_visuotactile', 'action_n-1', 'H_action_n-1_ambiguity']
features_10 = ['angle_cat', 'tactile', 'visuotactile', 'angle_cat_tactile', 'angle_cat_visuotactile', 'angle_n-1', 'H_angle_n-1_ambiguity']
features_11 = ['angle_cat', 'tactile', 'visuotactile', 'angle_cat_tactile', 'angle_cat_visuotactile', 'angle_n-1', 'success_n-1', 'failure_n-1', 'H_success_n-1_ambiguity', 'H_failure_n-1_ambiguity']


features_01 = ['angle']
features_02 = ['angle', 'angle_n-1']
features_03 = ['angle', 'angle_hit_n-1', 'angle_miss_n-1']
features_04 = ['angle', 'angle_hit_n-1', 'angle_miss_n-1', 'success_n-1', 'failure_n-1']
#FAMILY 1
#models = [features_0, features_1, features_2, features_3]
# FAMILY 2
#models = [features_4, features_5, features_6, features_7, features_8]
# FAMILY 3
#models = [features_0, features_1, features_2, features_3]
models = [features_03, features_04]
results = []
for model in models:
    result = an.analyze_individual_rats_ale_N(features_keys=model, df_processed=df, k=None, scaling='cont_only')
    # result is a dict with one key per rat, and each key is another dict with keys: 
    # 'data', 'model', 'scaler', 'feature_names', 'train_accuracy', 'n_trials', 'coefficients', 'cross_validation_scores'
    results.append(result)

# -----------------------------
# PLOT RESULTS
# -----------------------------
# ---- build 4 dfs in a list + one shared beta-column order (union across models) ----
dfs=[flatten_model_results(m) for m in results] # dfs is a list of n dataframes, one per model, with columns: rat_id, feature_name, beta_mean, beta_ci_lower, beta_ci_upper, score_mean, score_std
beta_cols = []
for df in dfs:
    beta_cols.extend([col for col in df.columns if col.startswith("beta_")])
beta_cols = sorted(set(beta_cols))  # unique and sorted (by alphabetical order, this is also something to change, as it is a bit arbitrary)
score_cols=["score_mean","score_std"]


for i, result in enumerate(results):
    
    ci_df = an.bootstrap_ci(result, n_iterations=1000, ci=95,
                        random_state=42, n_jobs=4)
    an.plot_bootstrap_ci(dfs[i], ci_df, model_label=f"Model {i}", beta_cols=beta_cols)
    an.plot_forest_ci(ci_df, model_label="")


plot_data = []
for i, model_results in enumerate(results):
    for rat_id, d in model_results.items():
        plot_data.append({
            'rat_id': rat_id,
            'model_name': f"Model {i}", # Label for color coding
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
