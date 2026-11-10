"""
GLM null / pattern testing for history effects in rat psychophysics.

Two modes:
    mode = 'pooled'        -> one fit per rat, all trials, no modality regressors
    mode = 'per_modality'  -> three fits per rat (tactile / visual / visuotactile),
                              no modality regressors inside each fit

Per-rat (per-modality) outputs:
    - Bootstrap CIs on each coefficient (session-level resampling)
    - CIs on (beta_success - beta_failure), (beta_success + beta_failure)
    - P(opposite signs) between success_n-1 and failure_n-1
    - Rat classification: reward x choice / pure choice / ambiguous
    - Delta AIC/BIC for two nested comparisons:
        (i)  features_11 (action_n-1) vs features_19 (success + failure)
             -> pure choice vs reward x choice
        (ii) features_18/19 vs same without angle_n-1
             -> perceptual bias null

Population-level:
    - Count of rats per classification, binomial test vs 50%

Plots per (rat, modality):
    - Forest plot of coefficients with CIs
    - Histogram of bootstrap distribution for (beta_s - beta_f)

Notes on statistical choices:
    - L2 regularized logistic regression, ridge strength C fixed at 1.0 (matches
      the existing pipeline). Set FIT_LAMBDA_PER_FIT = True to re-CV C on each
      fit; kept off by default for speed and comparability with prior analyses.
    - Bootstrap resamples SESSIONS (not trials) with replacement -> respects
      within-session correlations that raw trial-level bootstrap ignores.
    - Ridge lambda held fixed across bootstrap resamples (standard, understates
      uncertainty only slightly compared to re-CV on each resample).
    - Scaling: StandardScaler(with_mean=False) on continuous regressors only,
      matching the existing pipeline (angle is already centered meaningfully).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ============================================================
# PATHS (computed from this file, so the script runs from any folder / computer)
# ============================================================

HERE = Path(__file__).resolve().parent     # results are saved in this folder
ROOT = HERE.parents[2]                      # project root (history_dependecy)

# ============================================================
# CONFIG
# ============================================================

# Preprocessing choices. Both False = the August 2026 setup (glm_summary.csv was made like that).
APPLY_TRAINING_FILTER = True    # functions_Ale.filter_training_trials, criterion 0.80, 2 consecutive sessions
DROP_PREV_ANGLE_45 = True       # drop trials whose previous angle is 45 (their hit/miss was recorded at random)

MODE = "per_modality"           # 'pooled' or 'per_modality'
MIN_TRIALS = 200                # minimum trials per fit
N_BOOT = 1000                   # 5000 for publication
CI_LEVEL = 95
RANDOM_STATE = 42
N_JOBS = 4
RIDGE_C = 1.0                   # matches existing pipeline
OPPOSITE_SIGN_THRESHOLD = 0.95  # P(opposite signs) needed to call reward x choice

# feature sets (using your naming). Modality regressors intentionally excluded:
# in 'pooled' mode we ignore modality entirely; in 'per_modality' mode we
# subset the data by modality so modality regressors are redundant.
FEATURES_11 = ["angle", "action_n-1"]
FEATURES_18 = ["angle", "hit_n-1", "angle_n-1", "success_n-1", "failure_n-1"]
FEATURES_19 = ["angle", "angle_n-1", "success_n-1", "failure_n-1"]
FEATURES_18_NO_ANGLE_N1 = ["angle", "hit_n-1", "success_n-1", "failure_n-1"]
FEATURES_19_NO_ANGLE_N1 = ["angle", "success_n-1", "failure_n-1"]

# which of features_18 / features_19 is the "main" model for the pattern test.
# both are computed; MAIN_MODEL determines which is used for classification and plots.
MAIN_MODEL_NAME = "features_19"   # 'features_18' or 'features_19'

# modality codes (from your existing code: 1=tactile, 2=visual, 3=visuotactile)
MODALITY_CODES = {"tactile": 1, "visual": 2, "visuotactile": 3}

# regressors treated as continuous for scaling purposes
CONTINUOUS_REGRESSORS = {"angle", "angle_n-1"}


# ============================================================
# CORE FITTING
# ============================================================

def build_design_matrix(rat_data: pd.DataFrame, feature_names: list[str]):
    """Return X, y, scaler, ordered feature names.

    Scaling: StandardScaler(with_mean=False) on continuous regressors only.
    """
    cont_cols = [f for f in feature_names if f in CONTINUOUS_REGRESSORS]
    bin_cols = [f for f in feature_names if f not in CONTINUOUS_REGRESSORS]

    X_parts = []
    ordered_names = []
    scaler = None

    if cont_cols:
        X_cont = rat_data[cont_cols].values.astype(float)
        scaler = StandardScaler(with_mean=False)
        X_cont = scaler.fit_transform(X_cont)
        X_parts.append(X_cont)
        ordered_names.extend(cont_cols)

    if bin_cols:
        X_bin = rat_data[bin_cols].values.astype(float)
        X_parts.append(X_bin)
        ordered_names.extend(bin_cols)

    X = np.column_stack(X_parts)
    y = rat_data["action"].values.astype(int)
    return X, y, scaler, ordered_names


def fit_ridge_logistic(X, y, C=RIDGE_C):
    m = LogisticRegression(penalty="l2", C=C, max_iter=1000, solver="lbfgs")
    m.fit(X, y)
    return m


def compute_aic_bic(model, X, y):
    """AIC/BIC from the fitted logistic model.

    For ridge, the effective degrees of freedom differ from raw parameter count;
    we use k = n_features + 1 (intercept) as a standard approximation, matching
    the existing pipeline. This is fine for *relative* comparison between nested
    models fit with the same C.
    """
    from sklearn.metrics import log_loss

    n = len(y)
    proba = model.predict_proba(X)
    ll = -log_loss(y, proba, normalize=False)
    k = X.shape[1] + 1
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll
    return aic, bic, ll


# ============================================================
# BOOTSTRAP (session-level)
# ============================================================

def _one_bootstrap_fit(X, y, row_indices, C):
    m = fit_ridge_logistic(X[row_indices], y[row_indices], C=C)
    return m.coef_[0]


def bootstrap_coefficients(rat_data, X, y, feature_names,
                           n_boot=N_BOOT, C=RIDGE_C,
                           random_state=RANDOM_STATE, n_jobs=N_JOBS):
    """Session-level bootstrap. Returns array (n_boot, n_features)."""
    rng = np.random.default_rng(random_state)
    sessions = rat_data["date"].unique()
    n_sessions = len(sessions)
    session_to_rows = {s: np.where(rat_data["date"].values == s)[0] for s in sessions}

    boot_session_idx = rng.integers(0, n_sessions, size=(n_boot, n_sessions))

    def build_rows(session_choice):
        return np.concatenate([session_to_rows[sessions[s]] for s in session_choice])

    coefs = Parallel(n_jobs=n_jobs)(
        delayed(_one_bootstrap_fit)(X, y, build_rows(boot_session_idx[it]), C)
        for it in range(n_boot)
    )
    return np.asarray(coefs)


# ============================================================
# PER-FIT ANALYSIS
# ============================================================

def analyze_single_fit(rat_data, feature_names, n_boot=N_BOOT,
                       C=RIDGE_C, random_state=RANDOM_STATE, n_jobs=N_JOBS):
    """Fit one GLM + bootstrap + summary stats for a single (rat, modality) subset.

    Returns dict with: coefficients (real fit), bootstrap draws, per-coef CIs,
    AIC/BIC, and (if success_n-1 and failure_n-1 are present) the pattern stats.
    """
    X, y, scaler, ordered_names = build_design_matrix(rat_data, feature_names)
    model = fit_ridge_logistic(X, y, C=C)
    aic, bic, ll = compute_aic_bic(model, X, y)

    boot = bootstrap_coefficients(rat_data, X, y, ordered_names,
                                  n_boot=n_boot, C=C,
                                  random_state=random_state, n_jobs=n_jobs)

    alpha = (100 - CI_LEVEL) / 2
    ci_low, ci_high = np.percentile(boot, [alpha, 100 - alpha], axis=0)

    result = {
        "feature_names": ordered_names,
        "coefficients": model.coef_[0],
        "boot_draws": boot,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "aic": aic,
        "bic": bic,
        "log_lik": ll,
        "n_trials": len(y),
    }

    # pattern stats if success_n-1 and failure_n-1 both present
    if "success_n-1" in ordered_names and "failure_n-1" in ordered_names:
        i_s = ordered_names.index("success_n-1")
        i_f = ordered_names.index("failure_n-1")
        diff = boot[:, i_s] - boot[:, i_f]
        summ = boot[:, i_s] + boot[:, i_f]
        result["diff_boot"] = diff
        result["diff_ci"] = np.percentile(diff, [alpha, 100 - alpha])
        result["sum_boot"] = summ
        result["sum_ci"] = np.percentile(summ, [alpha, 100 - alpha])
        result["p_opposite_signs"] = float(
            np.mean(np.sign(boot[:, i_s]) != np.sign(boot[:, i_f]))
        )
        result["p_same_signs"] = 1 - result["p_opposite_signs"]

    return result


def classify_rat(fit_result, opp_thresh=OPPOSITE_SIGN_THRESHOLD):
    """Classify pattern based on success/failure coefficients.

    Categories:
        'reward_x_choice' : P(opposite signs) >= threshold
        'pure_choice'     : difference CI includes 0 AND same-sign majority
        'ambiguous'       : otherwise
    """
    if "p_opposite_signs" not in fit_result:
        return "n/a"

    p_opp = fit_result["p_opposite_signs"]
    diff_ci_low, diff_ci_high = fit_result["diff_ci"]
    diff_includes_zero = diff_ci_low <= 0 <= diff_ci_high

    if p_opp >= opp_thresh:
        return "reward_x_choice"
    if diff_includes_zero and p_opp < 0.5:
        return "pure_choice"
    return "ambiguous"


# ============================================================
# TOP-LEVEL ANALYSIS
# ============================================================

def run_analysis(df, mode="per_modality", n_boot=N_BOOT,
                 random_state=RANDOM_STATE, n_jobs=N_JOBS,
                 min_trials=200):
    """Main entry point.

    mode: 'pooled' or 'per_modality'
    df:   preprocessed dataframe with columns:
          rat, date, action, angle, angle_n-1, action_n-1, hit_n-1,
          success_n-1, failure_n-1, mod
    """
    assert mode in ("pooled", "per_modality")
    rats = sorted(df["rat"].unique())

    main_features = FEATURES_18 if MAIN_MODEL_NAME == "features_18" else FEATURES_19
    no_angle_features = (FEATURES_18_NO_ANGLE_N1 if MAIN_MODEL_NAME == "features_18"
                         else FEATURES_19_NO_ANGLE_N1)

    all_results = {}   # {rat_id: {modality_or_'pooled': {...fits...}}}

    for rat_id in rats:
        rat_df = df[df["rat"] == rat_id].copy()
        all_results[rat_id] = {}

        if mode == "pooled":
            subsets = {"pooled": rat_df}
        else:
            subsets = {name: rat_df[rat_df["mod"] == code].copy()
                       for name, code in MODALITY_CODES.items()}

        for subset_name, subset_df in subsets.items():
            subset_df = subset_df.dropna(subset=(["action", "date"]
                                                 + list(set(FEATURES_18) | {"action_n-1"}))).copy()
            if len(subset_df) < min_trials:
                print(f"  Rat {rat_id} [{subset_name}]: skip ({len(subset_df)} trials)")
                continue

            print(f"  Rat {rat_id} [{subset_name}]: {len(subset_df)} trials")

            fits = {}
            # main model (features_18 or _19) — used for pattern test
            fits["main"] = analyze_single_fit(
                subset_df, main_features, n_boot=n_boot,
                random_state=random_state, n_jobs=n_jobs,
            )
            # no-angle_n-1 variant — for perceptual bias null via delta AIC/BIC
            fits["main_no_angle_n1"] = analyze_single_fit(
                subset_df, no_angle_features, n_boot=n_boot,
                random_state=random_state, n_jobs=n_jobs,
            )
            # features_11 — for pure choice vs reward x choice comparison
            fits["features_11"] = analyze_single_fit(
                subset_df, FEATURES_11, n_boot=n_boot,
                random_state=random_state, n_jobs=n_jobs,
            )

            # model comparison deltas
            fits["delta_aic_11_vs_main"] = fits["features_11"]["aic"] - fits["main"]["aic"]
            fits["delta_bic_11_vs_main"] = fits["features_11"]["bic"] - fits["main"]["bic"]
            fits["delta_aic_no_angle_vs_main"] = (fits["main_no_angle_n1"]["aic"]
                                                  - fits["main"]["aic"])
            fits["delta_bic_no_angle_vs_main"] = (fits["main_no_angle_n1"]["bic"]
                                                  - fits["main"]["bic"])

            # classification from main model
            fits["classification"] = classify_rat(fits["main"])

            all_results[rat_id][subset_name] = fits

    return all_results


# ============================================================
# SUMMARY TABLE
# ============================================================

def build_summary_table(all_results):
    """Long-format summary table across (rat, subset)."""
    rows = []
    for rat_id, per_subset in all_results.items():
        for subset_name, fits in per_subset.items():
            main = fits["main"]
            names = main["feature_names"]

            row = {"rat_id": rat_id, "subset": subset_name,
                   "n_trials": main["n_trials"],
                   "classification": fits["classification"]}

            for feat in ["success_n-1", "failure_n-1", "angle_n-1", "hit_n-1", "angle"]:
                if feat in names:
                    j = names.index(feat)
                    row[f"beta_{feat}"] = main["coefficients"][j]
                    row[f"ci_low_{feat}"] = main["ci_low"][j]
                    row[f"ci_high_{feat}"] = main["ci_high"][j]

            if "diff_ci" in main:
                row["diff_s_minus_f"] = np.mean(main["diff_boot"])
                row["diff_ci_low"] = main["diff_ci"][0]
                row["diff_ci_high"] = main["diff_ci"][1]
                row["p_opposite_signs"] = main["p_opposite_signs"]

            row["delta_aic_11_vs_main"] = fits["delta_aic_11_vs_main"]
            row["delta_bic_11_vs_main"] = fits["delta_bic_11_vs_main"]
            row["delta_aic_no_angle_vs_main"] = fits["delta_aic_no_angle_vs_main"]
            row["delta_bic_no_angle_vs_main"] = fits["delta_bic_no_angle_vs_main"]

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# POPULATION-LEVEL TEST
# ============================================================

def population_binomial_test(summary_df, subset=None, target="reward_x_choice"):
    """Binomial test: fraction of rats classified as `target` vs 50%.

    If subset is None, uses all rows; else filters to summary_df['subset'] == subset.
    """
    df = summary_df if subset is None else summary_df[summary_df["subset"] == subset]
    n = len(df)
    k = int((df["classification"] == target).sum())
    if n == 0:
        return None
    test = binomtest(k, n, p=0.5, alternative="greater")
    return {"subset": subset or "all", "n_rats": n, "n_target": k,
            "target": target, "p_value": test.pvalue,
            "proportion": k / n}


# ============================================================
# PLOTTING
# ============================================================

def plot_forest_per_fit(fit_result, title=""):
    names = fit_result["feature_names"]
    betas = fit_result["coefficients"]
    lo = fit_result["ci_low"]
    hi = fit_result["ci_high"]

    y_pos = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(names) + 1.5))
    ax.errorbar(betas, y_pos,
                xerr=[betas - lo, hi - betas],
                fmt="o", capsize=4, color="black")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("coefficient (log-odds)")
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def plot_diff_histogram(fit_result, title=""):
    if "diff_boot" not in fit_result:
        return None
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(fit_result["diff_boot"], bins=50, color="steelblue", edgecolor="white")
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    lo, hi = fit_result["diff_ci"]
    ax.axvline(lo, color="black", linestyle=":", linewidth=1)
    ax.axvline(hi, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel(r"$\beta_{success} - \beta_{failure}$")
    ax.set_ylabel("bootstrap count")
    p_opp = fit_result["p_opposite_signs"]
    ax.set_title(f"{title}\nP(opposite signs) = {p_opp:.3f}")
    plt.tight_layout()
    return fig


def plot_all(all_results, out_dir=None):
    """Generate forest + diff-hist plots per (rat, subset). Optionally save."""
    import os
    for rat_id, per_subset in all_results.items():
        for subset_name, fits in per_subset.items():
            title = f"Rat {rat_id} [{subset_name}] — {fits['classification']}"
            f1 = plot_forest_per_fit(fits["main"], title=title)
            f2 = plot_diff_histogram(fits["main"], title=title)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                f1.savefig(os.path.join(out_dir, f"forest_rat{rat_id}_{subset_name}.png"),
                           dpi=120)
                if f2 is not None:
                    f2.savefig(os.path.join(out_dir,
                               f"diff_rat{rat_id}_{subset_name}.png"), dpi=120)
                plt.close(f1)
                if f2 is not None:
                    plt.close(f2)


# ============================================================
# RUN METADATA
# ============================================================

def write_run_metadata(path, df, config):
    """Save what is needed to repeat the run: date, git commit, settings,
    library versions and number of trials per rat."""
    import joblib, numpy, scipy, sklearn
    git = {"commit": None, "uncommitted_changes": None}
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                                        text=True, timeout=10).stdout.strip()
        git = {"commit": run("rev-parse", "HEAD") or None,
               "uncommitted_changes": bool(run("status", "--porcelain"))}
    except Exception:
        pass
    meta = {
        "script": Path(__file__).name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git": git,
        "config": config,
        "library_versions": {
            "python": sys.version.split()[0], "numpy": numpy.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "joblib": joblib.__version__,
        },
        "rows_after_preprocessing": int(len(df)),
        "rows_per_rat_after_preprocessing": {str(k): int(v) for k, v in df.groupby("rat").size().items()},
    }
    Path(path).write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # ---- data loading and preprocessing (explicit; same block in the three scripts) ----
    sys.path.append(str(ROOT))      # serial_dependence_analysis.py and functions_Ale.py are in the project root
    from serial_dependence_analysis import SerialDependenceAnalyzer
    from functions_Ale import filter_training_trials

    an = SerialDependenceAnalyzer(history_depth=1, save_figures=False,
                                  data_path=str(ROOT / "data" / "behavior_data.mat"),
                                  csv_path=str(ROOT / "processed_behavior_data.csv"))
    an.load_and_preprocess_data()
    df = an.create_lagged_features()

    if APPLY_TRAINING_FILTER:
        df = filter_training_trials(df, method='criterion', criterion=0.80, min_consec=2)
    if DROP_PREV_ANGLE_45:
        # the hit/miss of 45-degree trials was recorded at random (data collection bug), so the
        # trial after them has no reliable outcome history. Done before the angles are centered.
        df = df[df['angle_n-1'] != 45].copy()

    df['action_n-1'] = df['action_n-1'] * 2 - 1
    df['success_n-1'] = df['hit_n-1'] * df['action_n-1']
    df['failure_n-1'] = (1 - df['hit_n-1']) * df['action_n-1']
    df['angle'] = df['angle'] - 45
    df['angle_n-1'] = df['angle_n-1'] - 45
    # ---------------------------------------------

    results = run_analysis(df, mode=MODE, n_boot=N_BOOT, min_trials=MIN_TRIALS)
    summary = build_summary_table(results)
    print(summary.to_string())
    out_csv = HERE / "glm_summary.csv"      # name read by hit_n_1_check2.py (per_modality mode)
    summary.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(summary)} rows)")

    if MODE == "per_modality":
         for mod_name in MODALITY_CODES:
             print(population_binomial_test(summary, subset=mod_name))
    else:
         print(population_binomial_test(summary, subset="pooled"))

    # metadata is saved right after the table, before the plots, so it exists whenever the CSV does
    write_run_metadata(HERE / "glm_summary_metadata.json", df, {
        "apply_training_filter": APPLY_TRAINING_FILTER, "drop_prev_angle_45": DROP_PREV_ANGLE_45,
        "mode": MODE, "min_trials": MIN_TRIALS, "n_boot": N_BOOT, "ci_level": CI_LEVEL,
        "random_state": RANDOM_STATE, "ridge_c": RIDGE_C, "main_model": MAIN_MODEL_NAME,
    })

    plot_all(results, out_dir=HERE / "glm_plots")