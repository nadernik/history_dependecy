

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
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools import add_constant

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
df = df.dropna(subset=['angle_n-1', 'action_n-1', 'hit_n-1'])
df = df[df['angle_n-1'] != 45] #remove trials where n-1 = 45 as reward data there are corrupted 


# FEATURE CREATION:
# categorical angles
df['tactile'] = (df['mod'] == 1).astype(int)
df['visuotactile'] = (df['mod'] == 3).astype(int)
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

#predictor_cols = ['angle', 'tactile', 'visuotactile', 'angle_n-1', 'action_n-1', 'success_n-1', 'failure_n-1', 'angle_hit_n-1', 'angle_miss_n-1', 'angle_tactile', 'angle_visuotactile']





predictor_cols = [
    'angle',
    'success_n-1', 'angle_n-1',
    
]

vif_results = []

rat = 11
group = df[df[RAT_COL] == rat]

df_vif = group[predictor_cols].copy()

# force numeric
for col in predictor_cols:
    df_vif[col] = pd.to_numeric(df_vif[col], errors='coerce')

# check problematic rows BEFORE dropping
mask_bad = ~np.isfinite(df_vif.to_numpy()).all(axis=1)
if mask_bad.any():
    print("Bad rows (NaN or inf):")
    print(df_vif.loc[mask_bad])

# clean
df_vif = df_vif.replace([np.inf, -np.inf], np.nan).dropna()

print(f"Remaining rows: {len(df_vif)}")

print("\n=== CORRELATION MATRIX (Pearson) ===")
corr_matrix = df_vif.corr(method='pearson')
print(corr_matrix.round(3))
X = add_constant(df_vif).astype(float)

# compute VIF
vif_results = []
for i, col in enumerate(X.columns):
    vif_results.append({
        "rat": rat,
        "feature": col,
        "VIF": variance_inflation_factor(X.values, i)
    })

vif_df = pd.DataFrame(vif_results)
print(vif_df)






