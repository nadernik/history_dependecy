
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
SESSION-BOOTSTRAP for SIGNIFICANCE + KERNEL HEATMAP from best fit betas
======================================================
PROCEDURE TO OBTAIN RESULTS:

“Coefficients were estimated using the full dataset for each subject.
Statistical reliability was assessed using session-level bootstrap confidence intervals, obtained by resampling sessions with replacement and refitting the model.
An effect was considered reliable if the bootstrap CI excluded zero.”

What it does:
1) Loads & preprocesses data with SerialDependenceAnalyzer
2) Performs SESSION bootstrap (resample whole 'date' sessions with replacement)
3) Computes percentile CIs per predictor
4) Saves CI table to CSV
5) Fits a lapse-GLM per rat
6) Plots kernel heatmap (angle curr + history kernels) and significance stars

"""
import pandas as pd
from serial_dependence_analysis import SerialDependenceAnalyzer
from joblib import Parallel, delayed
import os
import functions_to_run_toso_glm as ft


# -----------------------------
# CONFIG / COLUMN NAMES
# -----------------------------
RAT_COL   = "rat"
ANGLE_COL = "angle"
DATE_COL  = "date"   # session identifier
TRIAL_COL = "trialID" # trial index within session (optional but recommended)

# Analysis params
HISTORY_DEPTH = 1
MIN_TRIALS    = 200
LAPSE_PENALTY = 200.0
MAX_ITER      = 800

# Bootstrap params: these need to be set in "fctions_to_run_toso_glm.py" too, which is a bit messy and I should change it in the future
# if we run these analysis again
N_BOOT = 2
ALPHA  = 0.05

# Output
CI_CSV_PATH = "session_bootstrap_ci_table.csv"


# ------------------------------------------------------------
# MAIN: run everything end-to-end
# ------------------------------------------------------------
def main():
    #1) --- load + preprocess ---
    an = SerialDependenceAnalyzer(history_depth=HISTORY_DEPTH, save_figures=False)
    an.load_and_preprocess_data()
    df_processed =an.create_lagged_features()  # df is usually stored inside analyzer too
    an.df_processed = df_processed  # ensure it's set inside analyzer

    #2-3) --- run in parallel without over load session bootstrap CIs---
    rats  = sorted(pd.Series(df_processed[RAT_COL]).dropna().unique())

    n_jobs = min(4, max(1, os.cpu_count() // 2))  # keep more then 1 core free

    all_ci_list = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
        delayed(ft._ci_one_rat)(rat, an, df_processed) for rat in rats
    )

    all_ci = pd.concat(all_ci_list, ignore_index=True)


    # 4) Save table so you can inspect/filter later
    all_ci.to_csv(CI_CSV_PATH, index=False)
    print(f"[saved] {CI_CSV_PATH}  (rows={len(all_ci)}, rats={len(rats)}, failed fits total={all_ci['n_boot_failed'].sum()})")
    # --- 5) fit per rat (fit to the original dataset) ---
    results = an.analyze_individual_rats_toso_lapse(
        min_trials=MIN_TRIALS,
        lapse_penalty=LAPSE_PENALTY,
        max_iter=MAX_ITER
    )

    # --- 6) kernel heatmap from fitted betas ---
    ft.plot_kernel_heatmap(an, results, all_ci = all_ci)

    # Optional: print significant effects (compact)
    sig = all_ci.loc[all_ci["significant"]].sort_values(["rat", "predictor"])
    print("\nSignificant predictors (CI excludes 0):")
    if len(sig) == 0:
        print("  (none)")
    else:
        print(sig[["rat", "predictor", "ci_low", "ci_high", "n_boot_success"]].to_string(index=False))


if __name__ == "__main__":
    main()