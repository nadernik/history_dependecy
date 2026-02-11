
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


"""
GLM ANALYSIS WITH LAPSING + ANGLE LAGS 1 to 5 and choice and outcome lags fixed to 5
======================================================
PROCEDURE TO OBTAIN RESULTS:

“Coefficients were estimated using the full dataset for each subject. Each model include the current stimulus and the 5 most recent lags of choice and outcome as predictors, 
and models that included angle history lags from 0 (no angle history) to 5 are compared on two measures: Negative Log Likelihood (NLL) per trial, and Δ NLL per trial (compared to model with L-1 angle lags).

Run this file as a standalone script.

What it does:
1) Loads & preprocesses data with SerialDependenceAnalyzer
2) Fits a lapse-GLM per rat with angle lags 1 to 5, and choice and outcome fixed to 5 lags
3) Plots  Negative Log Likelihood per trial vs Lag (one line per rat)
4) Plots Δ Log Likelihood per trial --> this shows the incremental value of adding each angle lag (vs model with Lag -1)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from serial_dependence_analysis import SerialDependenceAnalyzer


# -----------------------------
# CONFIG / COLUMN NAMES
# -----------------------------
RAT_COL   = "rat"
ANGLE_COL = "angle"
TRIAL_COL = "trialID" # trial index within session (optional but recommended)

# Analysis params
HISTORY_DEPTH = 5
MIN_TRIALS    = 50
LAPSE_PENALTY = 200.0
MAX_ITER      = 4000

# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=5, save_figures=False)
an.load_and_preprocess_data()
df_processed = an.create_lagged_features()

rats = df_processed[RAT_COL].dropna().unique()

# -----------------------------
# ANALYZE: fit lapse GLM with angle lags 1 to 5, and choice and outcome fixed to 5 lags
# -----------------------------
rows = []
for rat in rats:
    rat_df = df_processed[df_processed[RAT_COL] == rat].copy()
    #print(f"Rat {rat}: {len(rat_df)} trials after preprocessing")

    res_rows = an.nested_angle_lags( 
        rat_df,
        K=HISTORY_DEPTH,
        lapse_penalty=LAPSE_PENALTY,
        max_iter=MAX_ITER,
        scaler=None,          
        min_trials=MIN_TRIALS
    )

    # res_rows is a list of dicts (one per L). Add them all.
    rows.extend(res_rows)

nll_df = pd.DataFrame(rows)

# add nll per trial (positive)
nll_df["nllpt"] = -nll_df["llpt"]

# optional: drop failed fits, but there are none in this case
nll_df_ok = nll_df[nll_df["success"] == True].copy()


# -----------------------------
# PLOT: NLL per trial vs L (one line per rat)
# -----------------------------
plt.figure(figsize=(10, 6))
sns.lineplot(data=nll_df_ok, x="L", y="nllpt", hue="rat", marker="o", palette="tab20")
plt.title("NLL per trial vs included angle history depth (L)\n(choice/outcome lags fixed to 5)")
plt.xlabel("Max angle lag included (L)")
plt.ylabel("NLL per trial")
plt.xticks(range(0, HISTORY_DEPTH + 1))
plt.grid(True)
plt.legend(title="Rat", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

# -----------------------------
# PLOT: Δ LL per trial (vs L-1) (one line per rat)
# -----------------------------

plt.figure(figsize=(10, 6))
sns.lineplot(data=nll_df_ok, x="L", y="dllpt", hue="rat", marker="o", palette="tab20")
plt.axhline(0, linestyle="--")
plt.title("Incremental value of adding angle lag L (Δ LL per trial)")
plt.xlabel("Added angle lag (L)")
plt.ylabel("Δ LL per trial (vs L-1)")
plt.xticks(range(0, HISTORY_DEPTH + 1))
plt.grid(True)
plt.legend(title="Rat", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()





