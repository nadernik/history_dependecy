"""
Test whether hit_n-1 (pure reward main effect) is present beyond reward x choice.

Fits features_18 = ['angle', 'hit_n-1', 'angle_n-1', 'success_n-1', 'failure_n-1']
alongside features_19 = ['angle', 'angle_n-1', 'success_n-1', 'failure_n-1'] and:
  - reports beta_hit_n-1 with bootstrap CI per (rat, modality)
  - reports delta AIC / delta BIC (features_19 vs features_18)
    positive = features_18 wins (hit_n-1 earns its keep)
    negative = features_19 wins (hit_n-1 does not help; null claim on pure reward)
  - classifies each fit: 'pure_reward_present' / 'no_pure_reward' / 'ambiguous'

Usage: run it from anywhere. The preprocessing is written out in the MAIN block
(switches APPLY_TRAINING_FILTER and DROP_PREV_ANGLE_45 below). Mirrors the
structure of glm_test_choice_reward_choice_x_reward.py (same settings).
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from scipy.stats import binomtest

# ============================================================
# PATHS (computed from this file, so the script runs from any folder / computer)
# ============================================================

HERE = Path(__file__).resolve().parent     # results are saved in this folder
ROOT = HERE.parents[2]                      # project root (history_dependecy)

# ============================================================
# CONFIG (same settings as glm_test_choice_reward_choice_x_reward.py)
# ============================================================

# Preprocessing choices. Both False = the August 2026 setup (hit_n1_test_summary.csv was made like that).
APPLY_TRAINING_FILTER = True    # functions_Ale.filter_training_trials, criterion 0.80, 2 consecutive sessions
DROP_PREV_ANGLE_45 = True       # drop trials whose previous angle is 45 (their hit/miss was recorded at random)

MODE = "per_modality"           # 'pooled' or 'per_modality'
N_BOOT = 1000
CI_LEVEL = 95
RANDOM_STATE = 42
N_JOBS = 4
RIDGE_C = 1.0

FEATURES_18 = ["angle", "hit_n-1", "angle_n-1", "success_n-1", "failure_n-1"]
FEATURES_19 = ["angle", "angle_n-1", "success_n-1", "failure_n-1"]

MODALITY_CODES = {"tactile": 1, "visual": 2, "visuotactile": 3}
CONTINUOUS_REGRESSORS = {"angle", "angle_n-1"}
MIN_TRIALS = 200


# ============================================================
# FITTING (identical logic to main script)
# ============================================================

def build_design_matrix(rat_data, feature_names):
    cont_cols = [f for f in feature_names if f in CONTINUOUS_REGRESSORS]
    bin_cols = [f for f in feature_names if f not in CONTINUOUS_REGRESSORS]

    X_parts, ordered_names = [], []
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
    return X, y, ordered_names


def fit_ridge(X, y, C=RIDGE_C):
    m = LogisticRegression(penalty="l2", C=C, max_iter=1000, solver="lbfgs")
    m.fit(X, y)
    return m


def aic_bic(model, X, y):
    n = len(y)
    ll = -log_loss(y, model.predict_proba(X), normalize=False)
    k = X.shape[1] + 1
    return 2 * k - 2 * ll, k * np.log(n) - 2 * ll


def _bootstrap_one(X, y, rows, C):
    m = fit_ridge(X[rows], y[rows], C=C)
    return m.coef_[0]


def bootstrap_coefs(rat_data, X, y, n_boot, C, random_state, n_jobs):
    rng = np.random.default_rng(random_state)
    sessions = rat_data["date"].unique()
    n_sessions = len(sessions)
    session_to_rows = {s: np.where(rat_data["date"].values == s)[0] for s in sessions}
    boot_idx = rng.integers(0, n_sessions, size=(n_boot, n_sessions))

    def build_rows(choice):
        return np.concatenate([session_to_rows[sessions[s]] for s in choice])

    return np.asarray(Parallel(n_jobs=n_jobs)(
        delayed(_bootstrap_one)(X, y, build_rows(boot_idx[it]), C)
        for it in range(n_boot)
    ))


# ============================================================
# ANALYSIS
# ============================================================

def analyze_subset(rat_data, n_boot=N_BOOT, random_state=RANDOM_STATE, n_jobs=N_JOBS):
    # fit features_18 and features_19
    X18, y, names18 = build_design_matrix(rat_data, FEATURES_18)
    X19, _, _ = build_design_matrix(rat_data, FEATURES_19)

    m18 = fit_ridge(X18, y)
    m19 = fit_ridge(X19, y)

    aic18, bic18 = aic_bic(m18, X18, y)
    aic19, bic19 = aic_bic(m19, X19, y)

    # bootstrap only features_18 (we need CI on beta_hit_n-1)
    boot18 = bootstrap_coefs(rat_data, X18, y, n_boot, RIDGE_C, random_state, n_jobs)

    alpha = (100 - CI_LEVEL) / 2
    ci_low, ci_high = np.percentile(boot18, [alpha, 100 - alpha], axis=0)

    i_hit = names18.index("hit_n-1")
    beta_hit = m18.coef_[0][i_hit]
    hit_ci_low, hit_ci_high = ci_low[i_hit], ci_high[i_hit]
    hit_excludes_zero = (hit_ci_low > 0) or (hit_ci_high < 0)

    # delta = 19 - 18. positive -> features_18 (with hit_n-1) wins
    delta_aic = aic19 - aic18
    delta_bic = bic19 - bic18

    # classification
    if hit_excludes_zero and delta_bic > 0:
        classification = "pure_reward_present"
    elif (not hit_excludes_zero) and delta_bic < 0:
        classification = "no_pure_reward"
    else:
        classification = "ambiguous"

    return {
        "n_trials": len(y),
        "beta_hit_n-1": beta_hit,
        "ci_low_hit_n-1": hit_ci_low,
        "ci_high_hit_n-1": hit_ci_high,
        "hit_ci_excludes_zero": hit_excludes_zero,
        "delta_aic_19_vs_18": delta_aic,
        "delta_bic_19_vs_18": delta_bic,
        "classification": classification,
    }


def run(df, mode="per_modality", n_boot=N_BOOT):
    assert mode in ("pooled", "per_modality")
    rats = sorted(df["rat"].unique())
    rows = []

    for rat_id in rats:
        rat_df = df[df["rat"] == rat_id].copy()
        subsets = ({"pooled": rat_df} if mode == "pooled"
                   else {name: rat_df[rat_df["mod"] == code].copy()
                         for name, code in MODALITY_CODES.items()})

        for subset_name, subset_df in subsets.items():
            subset_df = subset_df.dropna(subset=(["action", "date"] + FEATURES_18)).copy()
            if len(subset_df) < MIN_TRIALS:
                print(f"  Rat {rat_id} [{subset_name}]: skip ({len(subset_df)} trials)")
                continue
            print(f"  Rat {rat_id} [{subset_name}]: {len(subset_df)} trials")
            res = analyze_subset(subset_df, n_boot=n_boot)
            res["rat_id"] = rat_id
            res["subset"] = subset_name
            rows.append(res)

    return pd.DataFrame(rows)


def summarize(summary_df):
    print("\n=== Per-fit hit_n-1 test ===")
    cols = ["rat_id", "subset", "n_trials", "beta_hit_n-1",
            "ci_low_hit_n-1", "ci_high_hit_n-1", "hit_ci_excludes_zero",
            "delta_aic_19_vs_18", "delta_bic_19_vs_18", "classification"]
    print(summary_df[cols].to_string(index=False))

    print("\n=== Class counts per subset ===")
    counts = (summary_df.groupby(["subset", "classification"])
              .size().unstack(fill_value=0))
    print(counts.to_string())

    print("\n=== Binomial test: fraction 'pure_reward_present' vs 50% ===")
    for sub in summary_df["subset"].unique():
        sub_df = summary_df[summary_df["subset"] == sub]
        n = len(sub_df)
        k = int((sub_df["classification"] == "pure_reward_present").sum())
        p = binomtest(k, n, p=0.5, alternative="greater").pvalue
        print(f"  {sub:>12}: {k}/{n} ({k/n:.2%}), p = {p:.4f}")

    print("\n=== Fraction of fits with beta_hit_n-1 CI excluding 0 ===")
    for sub in summary_df["subset"].unique():
        sub_df = summary_df[summary_df["subset"] == sub]
        k = int(sub_df["hit_ci_excludes_zero"].sum())
        n = len(sub_df)
        print(f"  {sub:>12}: {k}/{n} ({k/n:.2%})")


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
    # ---------------------------------------------------------------------

    results_df = run(df, mode=MODE, n_boot=N_BOOT)
    summarize(results_df)
    out_csv = HERE / "hit_n1_test_summary.csv"      # read by hit_n_1_check2.py
    results_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(results_df)} rows)")

    write_run_metadata(HERE / "hit_n1_test_metadata.json", df, {
        "apply_training_filter": APPLY_TRAINING_FILTER, "drop_prev_angle_45": DROP_PREV_ANGLE_45,
        "mode": MODE, "min_trials": MIN_TRIALS, "n_boot": N_BOOT, "ci_level": CI_LEVEL,
        "random_state": RANDOM_STATE, "ridge_c": RIDGE_C,
    })