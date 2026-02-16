
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
import seaborn as sns

from serial_dependence_analysis import SerialDependenceAnalyzer
HISTORY_DEPTH = 5

# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=HISTORY_DEPTH, save_figures=False)
an.load_and_preprocess_data()
df_p = an.create_lagged_features()
# re-code action_n-1 to be -1 for left and +1 for right, instead of 0/1, so that it can be used in the model as a signed variable (e.g. for interaction with hit_n-1)
df_p['action_n-1'] = df_p['action_n-1']*2 -1
df_p['success_n-1'] = df_p['hit_n-1']*df_p['action_n-1'] # 1 if previous trial was a hit and the action was right (1*1), -1 if hit and left (1*-1), 0 if miss (0*anything)

print('success_n-1', df_p['success_n-1'].value_counts()) # just to check that we have all 3 values (1, -1, 0) and that they are not too imbalanced
df_p['failure_n-1'] = (1 - df_p['hit_n-1'])*df_p['action_n-1'] # 1 if previous trial was a miss and the action was right (1*1), -1 if miss and left (1*-1), 0 if hit (0*anything)
print('failure_n-1', df_p['failure_n-1'].value_counts()) # again just to check that we have all 3 values (1, -1, 0) and that they are not too imbalanced

# 4 models with increasing complexity: now it is a bit hardcoded but we can easily make it more flexible later
features_1 = ['angle']
features_2 = ['angle', 'action_n-1', ]
features_3 = ['angle', 'success_n-1', 'failure_n-1']
features_4 = ['angle', 'success_n-1', 'failure_n-1', 'angle_n-1']
#features_5 = ['angle', 'angle_n-1', 'angle_n-2', 'angle_n-3', 'angle_n-4', 'angle_n-5']
#feature_5 = ['angle', 'success_n-1', 'failure_n-1', 'angle_n-1', 'success_n-2', 'failure_n-2', 'angle_n-2', 'success_n-3', 'failure_n-3', 'angle_n-3']

models = [features_1, features_2, features_3, features_4] #, features_5]
results = []
for model in models:
    result = an.analyze_individual_rats_ale(features_keys=model, df_processed=df_p, k=None)
    # result is a dict with one entry per rat, and each entry is another dict with keys: 
    # 'data', 'model', 'scaler', 'feature_names', 'train_accuracy', 'n_trials', 'coefficients', 'cross_validation_scores'
    results.append(result)
# -----------------------------
# PLOT RESULTS
# -----------------------------

def flatten_model_results(model_results_dict):
    # this is a helper function to flatten the nested dict structure of model results into a single dataframe, 
    # with one row per rat and columns for each feature coefficient and Cross Validation scores
    rows=[]
    for rat_id,d in model_results_dict.items():
        fn=d["feature_names"]
        b=np.asarray(d["coefficients"]).ravel()
        print(f"Rat {rat_id} – features: {fn}, check function")
        cv=d.get("cross_validation_scores", {})
        row={"rat_id": rat_id, "score_mean": cv.get("score_mean", np.nan), "score_std": cv.get("score_std", np.nan)}
        row.update({f"beta_{n}": v for n,v in zip(fn,b)})
        rows.append(row)
    return pd.DataFrame(rows).set_index("rat_id").sort_index()

# ---- build 4 dfs in a list + one shared beta-column order (union across models) ----
dfs=[flatten_model_results(m) for m in results]
beta_cols = []
for df in dfs:
    beta_cols.extend([col for col in df.columns if col.startswith("beta_")])
beta_cols = sorted(set(beta_cols))  # unique and sorted (by alphabetical order, this is also something to change, as it is a bit arbitrary)
score_cols=["score_mean","score_std"]

# ---- PLOTS ----
# ---- plot: 1 heatmap per model (betas centered at 0)  ----
for i,df in enumerate(dfs):
    B=df.reindex(columns=beta_cols)                 # shared x-axis across models
    S=df.reindex(columns=score_cols)                # always same 2 columns
    plt.figure(figsize=(0.55*len(beta_cols)+3, 0.55*len(df)+2))
    sns.heatmap(
        B, 
        cmap="coolwarm",
        center=0, 
        annot=True,        
        fmt=".2f" ); 
    plt.title(f"Model {i} – betas") 
    plt.ylabel("rat_id"); plt.xlabel("") 
    plt.tight_layout()
    plt.show()

# creates a single dataframe with all the CV scores for all models and rats, to be used for the line plot below (one line per rat, x-axis: model, y-axis: CV score)
cv_long = (
    pd.concat(
        [df[["score_mean", "score_std"]].assign(model=i) for i, df in enumerate(dfs)],
        axis=0
    )
    .reset_index()  
    .rename(columns={"index": "rat_id"})
)
# labels that are more descriptive than just the model number, to be used in the line plot below (x-axis)
model_labels = {
    0: "M1: angle",
    1: "M2: angle + action_n-1",
    2: "M3: angle + success/failure",
    3: "M4: angle + success/failure + angle_n-1",
}
cv_long["model_label"] = cv_long["model"].map(model_labels).fillna(cv_long["model"].astype(str))

#print(cv_long.head())

# ---- plot: 1 lineplot  ----

plt.figure(figsize=(10, 5))

for rat_id, g in cv_long.groupby("rat_id"):
    g = g.sort_values("model")
    plt.plot(g["model"], g["score_mean"], marker="o", linewidth=1, alpha=0.8, label=str(rat_id))

plt.xticks(sorted(cv_long["model"].unique()),
           [model_labels.get(i, str(i)) for i in sorted(cv_long["model"].unique())],
           rotation=20, ha="right")

plt.ylabel("CV accuracy (mean)")
plt.xlabel("Model")
plt.title("Cross-validated accuracy across models (one line per rat)")
plt.tight_layout()


plt.show()

# to have a heatmap of CV scores per model and rat instead of a line plot:

'''
    plt.figure(figsize=(4, 0.55*len(df)+2))
    sns.heatmap(S, cmap="viridis", annot=True, fmt=".3f")
    plt.title(f"Model {i} – CV scores")
    plt.ylabel("rat_id")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()

'''




