
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


"WHAT THIS FILE DOES: "

" - This file load and pre-process the data"
" - Then it runs the glm analysis with lapses"
" - Finally, it extracts coefficients (kernels)"
" - Plot them as a heatmap"


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from serial_dependence_analysis import SerialDependenceAnalyzer
import functions_to_run_toso_glm as ft


# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=5, save_figures=False)
an.load_and_preprocess_data()
df_processed = an.create_lagged_features()

# -----------------------------
# ANALYZE: fit lapse GLM with angle lags 1 to 5,
# -----------------------------

results = an.analyze_individual_rats_toso_lapse(
    min_trials=200,
    lapse_penalty=200.0,
    max_iter=4000
)

# --- extract current-angle beta ---
rats = sorted(results.keys())
curr_angle_beta = []
# We will use this to add a "lag 0" column to the stimulus kernel, so that we can plot it together with the lag 1-5 coefficients.
for rat in rats:
    names = results[rat]["feature_names"]
    beta  = results[rat]["beta"]
    idx = names.index("angle")
    curr_angle_beta.append(beta[idx])

# --- extract history kernels ---
k = an.k
rats, stim_kernel   = ft.extract_kernel(results, prefix="angle",   k=k)
_,    choice_kernel = ft.extract_kernel(results, prefix="action",  k=k)
_,    out_kernel    = ft.extract_kernel(results, prefix="hitmiss", k=k)
curr_col = np.array(curr_angle_beta)[:, None] 
K_with_curr = np.hstack([curr_col, stim_kernel]) 

# Combine all kernels into one array for heatmap plotting
K_all = np.hstack([K_with_curr, choice_kernel, out_kernel])
# Create xtick labels
xticks = (
    ["angle (curr)"] +
    [f"ang {i}" for i in range(1, k+1)] +
    [f"cho {i}" for i in range(1, k+1)] +
    [f"out {i}" for i in range(1, k+1)]
)

plt.figure(figsize=(1.0*K_all.shape[1], 0.4*len(rats)))
sns.heatmap(
    K_all,
    cmap="coolwarm",
    center=0,
    annot=True,
    fmt=".2f",
    xticklabels=xticks,
    yticklabels=rats
)
plt.xlabel("Predictor (type × lag)")
plt.ylabel("Rat")
plt.title("History kernels: angle, choice, outcome")
plt.tight_layout()
plt.show()




