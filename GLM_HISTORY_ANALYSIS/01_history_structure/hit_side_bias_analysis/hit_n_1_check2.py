"""
Side-bias analysis for the hit_n-1 finding.

Goals:
1. Quantify side bias per (rat, modality) two ways:
   - GLM intercept from features_18 (log-odds of vertical given all regressors = 0)
   - Raw P(vertical) on real 45 trials (choice under weak/no sensory evidence)
   Cross-check the two measures.

2. For the 8 fits with delta_BIC > 2 (hit_n-1 clearly relevant):
   - Report beta_hit_n-1 alongside beta_success_n-1 and beta_failure_n-1 from the
     existing summary CSV.
   - Categorize each as 'repeater after win' (beta_success > 0) or 'switcher
     after win' (beta_success < 0), and see which category the pure-reward rats
     fall in.

3. Across all 36 fits:
   - Correlation between beta_hit_n-1 and side bias (both measures).
   - Correlation between beta_hit_n-1 and beta_success_n-1.
   - Scatter plots.

Usage: run it from anywhere. The preprocessing is written out in the MAIN block
(switches APPLY_TRAINING_FILTER and DROP_PREV_ANGLE_45 below). It reads
glm_summary.csv (written by glm_test_choice_reward_choice_x_reward.py) and
hit_n1_test_summary.csv (written by hit_n_1_check.py) from this folder, and
saves side_bias_per_fit.csv and side_bias_analysis.png in this folder too.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

# ============================================================
# PATHS (computed from this file, so the script runs from any folder / computer)
# ============================================================

HERE = Path(__file__).resolve().parent     # inputs are read from, and results saved in, this folder
ROOT = HERE.parents[2]                      # project root (history_dependecy)

# ============================================================
# CONFIG (same settings as the two scripts that produce its inputs)
# ============================================================

# Preprocessing choices. Both False = the August 2026 setup (side_bias_per_fit.csv was made like that).
APPLY_TRAINING_FILTER = True    # functions_Ale.filter_training_trials, criterion 0.80, 2 consecutive sessions
DROP_PREV_ANGLE_45 = True       # drop trials whose previous angle is 45 (their hit/miss was recorded at random)

SCRIPT_DIR = str(HERE)
SUMMARY_CSV = os.path.join(SCRIPT_DIR, "glm_summary.csv")
HIT_TEST_CSV = os.path.join(SCRIPT_DIR, "hit_n1_test_summary.csv")
MODE = "per_modality"
RIDGE_C = 1.0
MIN_TRIALS = 200
FEATURES_18 = ["angle", "hit_n-1", "angle_n-1", "success_n-1", "failure_n-1"]
MODALITY_CODES = {"tactile": 1, "visual": 2, "visuotactile": 3}
CONTINUOUS_REGRESSORS = {"angle", "angle_n-1"}


DELTA_BIC_THRESHOLD = 2                # for identifying "pure_reward_present"


# ============================================================
# FITTING (identical logic)
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


# ============================================================
# SIDE BIAS
# ============================================================

def side_bias_from_45(subset_df, angle_col_original_scale=False):
    """P(vertical) on real 45 degree trials.

    Assumes the current-trial angle column in `subset_df` is `angle` and that
    it has been centered by subtracting 45 (as in the preprocessing). If your
    df still has raw angles (not centered), set angle_col_original_scale=True.
    """
    if angle_col_original_scale:
        mask = subset_df["angle"] == 45
    else:
        mask = subset_df["angle"] == 0    # centered: 45 -> 0
    trials = subset_df[mask]
    n = len(trials)
    if n == 0:
        return np.nan, 0
    p_vert = trials["action"].mean()
    return p_vert, n


# ============================================================
# MAIN
# ============================================================

def run(df, mode="per_modality"):
    rats = sorted(df["rat"].unique())
    rows = []
    for rat_id in rats:
        rat_df = df[df["rat"] == rat_id].copy()
        subsets = ({"pooled": rat_df} if mode == "pooled"
                   else {name: rat_df[rat_df["mod"] == code].copy()
                         for name, code in MODALITY_CODES.items()})

        for subset_name, subset_df in subsets.items():
            clean = subset_df.dropna(subset=(["action", "date"] + FEATURES_18)).copy()
            if len(clean) < MIN_TRIALS:
                continue

            X, y, names = build_design_matrix(clean, FEATURES_18)
            m = fit_ridge(X, y)
            intercept = float(m.intercept_[0])
            i_hit = names.index("hit_n-1")
            beta_hit = float(m.coef_[0][i_hit])

            p_vert_45, n_45 = side_bias_from_45(clean)

            rows.append({
                "rat_id": rat_id,
                "subset": subset_name,
                "n_trials": len(clean),
                "beta_hit_n-1": beta_hit,
                "intercept": intercept,
                "p_vert_at_45": p_vert_45,
                "n_45_trials": n_45,
            })
    return pd.DataFrame(rows)


def merge_with_summary(side_df, summary_path=SUMMARY_CSV):
    """Merge side-bias output with the existing summary CSV to get
    beta_success_n-1, beta_failure_n-1, delta_bic_19_vs_18 in one place.

    Note: delta_bic_19_vs_18 was computed by the new hit_n-1 test script and is
    not in glm_summary.csv. If you have the hit_n-1 test summary CSV, pass its
    path instead / additionally. For now this merges what's available.
    """
    summ = pd.read_csv(summary_path)
    keep = ["rat_id", "subset",
            "beta_success_n-1", "beta_failure_n-1",
            "diff_s_minus_f", "p_opposite_signs"]
    merged = side_df.merge(summ[keep], on=["rat_id", "subset"], how="left")
    return merged


def summarize_and_plot(merged_df, hit_test_df=None):
    """Print correlations and produce scatter plots.

    hit_test_df: optional DataFrame from the hit_n-1 test with columns
    rat_id, subset, delta_bic_19_vs_18. If provided, used to filter
    delta_BIC>2 rats.
    """
    if hit_test_df is not None:
        merged_df = merged_df.merge(
            hit_test_df[["rat_id", "subset", "delta_bic_19_vs_18",
                          "ci_low_hit_n-1", "ci_high_hit_n-1"]],
            on=["rat_id", "subset"], how="left"
        )

    print("\n=== 1. Side bias measures per (rat, modality) ===")
    print(merged_df[["rat_id", "subset", "intercept", "p_vert_at_45",
                     "n_45_trials", "beta_hit_n-1"]].to_string(index=False))

    print("\n=== 2. Cross-check: intercept vs P(vertical) at 45 ===")
    # convert intercept (log-odds) to prob for direct comparison
    merged_df["p_from_intercept"] = 1 / (1 + np.exp(-merged_df["intercept"]))
    r_p, p_p = pearsonr(merged_df["p_from_intercept"], merged_df["p_vert_at_45"])
    r_s, p_s = spearmanr(merged_df["p_from_intercept"], merged_df["p_vert_at_45"])
    print(f"  Pearson  r = {r_p:.3f} (p = {p_p:.4f})")
    print(f"  Spearman r = {r_s:.3f} (p = {p_s:.4f})")

    print("\n=== 3. Correlation of beta_hit_n-1 with side bias (all 36 fits) ===")
    for bias_col in ["intercept", "p_vert_at_45"]:
        d = merged_df.dropna(subset=[bias_col, "beta_hit_n-1"])
        r_p, p_p = pearsonr(d[bias_col], d["beta_hit_n-1"])
        r_s, p_s = spearmanr(d[bias_col], d["beta_hit_n-1"])
        print(f"  beta_hit_n-1 vs {bias_col}: "
              f"Pearson r = {r_p:.3f} (p = {p_p:.4f}), "
              f"Spearman r = {r_s:.3f} (p = {p_s:.4f})")

    print("\n=== 4. Correlation of beta_hit_n-1 with beta_success_n-1 (all 36 fits) ===")
    d = merged_df.dropna(subset=["beta_success_n-1", "beta_hit_n-1"])
    r_p, p_p = pearsonr(d["beta_success_n-1"], d["beta_hit_n-1"])
    r_s, p_s = spearmanr(d["beta_success_n-1"], d["beta_hit_n-1"])
    print(f"  Pearson  r = {r_p:.3f} (p = {p_p:.4f})")
    print(f"  Spearman r = {r_s:.3f} (p = {p_s:.4f})")

    if "delta_bic_19_vs_18" in merged_df.columns:
        strong = merged_df[merged_df["delta_bic_19_vs_18"] > DELTA_BIC_THRESHOLD]
        print(f"\n=== 5. delta_BIC > {DELTA_BIC_THRESHOLD} fits: "
              f"beta_hit_n-1 alongside beta_success_n-1 and beta_failure_n-1 ===")
        cols = ["rat_id", "subset", "n_trials",
                "beta_hit_n-1", "ci_low_hit_n-1", "ci_high_hit_n-1",
                "beta_success_n-1", "beta_failure_n-1",
                "p_vert_at_45", "intercept",
                "delta_bic_19_vs_18"]
        cols = [c for c in cols if c in strong.columns]
        print(strong[cols].sort_values("delta_bic_19_vs_18", ascending=False).to_string(index=False))

        print("\n=== 6. Repeater vs switcher among delta_BIC > 2 rats ===")
        strong = strong.copy()
        strong["after_win_strategy"] = np.where(
            strong["beta_success_n-1"] > 0, "repeater", "switcher"
        )
        print(strong.groupby("after_win_strategy").size().to_string())
        print()
        print(strong[["rat_id", "subset", "beta_hit_n-1",
                      "beta_success_n-1", "after_win_strategy"]].to_string(index=False))

    # ---- plots ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.scatter(merged_df["p_from_intercept"], merged_df["p_vert_at_45"], alpha=0.7)
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("P(vertical) from intercept")
    ax.set_ylabel("P(vertical) raw at 45")
    ax.set_title("Side-bias cross-check")

    ax = axes[1]
    ax.scatter(merged_df["intercept"], merged_df["beta_hit_n-1"], alpha=0.7)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("intercept (side bias, log-odds vertical)")
    ax.set_ylabel(r"$\beta_{hit_{n-1}}$")
    ax.set_title(r"$\beta_{hit_{n-1}}$ vs side bias")

    ax = axes[2]
    if "beta_success_n-1" in merged_df.columns:
        ax.scatter(merged_df["beta_success_n-1"], merged_df["beta_hit_n-1"], alpha=0.7)
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel(r"$\beta_{success_{n-1}}$")
        ax.set_ylabel(r"$\beta_{hit_{n-1}}$")
        ax.set_title(r"$\beta_{hit_{n-1}}$ vs $\beta_{success_{n-1}}$")

    plt.tight_layout()
    out_png = os.path.join(SCRIPT_DIR, "side_bias_analysis.png")
    plt.savefig(out_png, dpi=120)
    print(f"\nSaved plot to {out_png}")


# ============================================================
# GUARD: do the input tables match this run's preprocessing?
# ============================================================

def check_input_consistency(df, config):
    """Compare this run's preprocessing with the runs that made glm_summary.csv (step 1) and
    hit_n1_test_summary.csv (step 2), using the metadata file each of them saves next to its CSV.
    Returns a list of problems (empty list = consistent)."""
    problems = []
    inputs = [("glm_summary.csv", "glm_summary_metadata.json"),
              ("hit_n1_test_summary.csv", "hit_n1_test_metadata.json")]
    for csv_name, meta_name in inputs:
        meta_path = HERE / meta_name
        if not meta_path.exists():
            problems.append(f"{csv_name} has no metadata file ({meta_name}): it comes from the original "
                            "run, with no training filter and trials after a 45-degree trial kept.")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for key, label in [("apply_training_filter", "training filter"),
                           ("drop_prev_angle_45", "exclusion of previous angle 45")]:
            if meta["config"].get(key) != config[key]:
                problems.append(f"{csv_name} was made with {label} = {meta['config'].get(key)}, "
                                f"this run uses {config[key]}.")
        if meta.get("rows_after_preprocessing") != int(len(df)):
            problems.append(f"{csv_name} was made on {meta.get('rows_after_preprocessing')} rows after "
                            f"preprocessing, this run has {len(df)} (different data or preprocessing).")
    return problems


def print_consistency_warning(problems):
    if not problems:
        return
    print("\n" + "!" * 78)
    print("WARNING: the input tables do not match the preprocessing of this run")
    for p in problems:
        print("  - " + p)
    print("The merged outputs that use them mix two datasets: the correlation of beta_hit with")
    print("beta_success, the list of fits with delta BIC > 2, the repeater / switcher labels and the")
    print("third scatter plot. The side-bias comparisons (intercept, P(vertical) at 45) are not affected.")
    print("To fix it, rerun glm_test_choice_reward_choice_x_reward.py and hit_n_1_check.py (steps 1")
    print("and 2) with the same switches as this script, then rerun this one.")
    print("!" * 78 + "\n")


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
    # -----------------------------------------

    # Guard: the two input tables (steps 1 and 2) must have been made with the same preprocessing
    # as this run, otherwise the merged outputs mix two datasets. This only warns, it does not stop.
    # The warning is printed now and again at the end, so it is not lost in the long output.
    problems = check_input_consistency(df, {
        "apply_training_filter": APPLY_TRAINING_FILTER, "drop_prev_angle_45": DROP_PREV_ANGLE_45,
    })
    print_consistency_warning(problems)

    side_df = run(df, mode=MODE)
    out_csv = HERE / "side_bias_per_fit.csv"
    side_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(side_df)} rows)")
    merged = merge_with_summary(side_df)
    hit_df = pd.read_csv(HIT_TEST_CSV)
    summarize_and_plot(merged, hit_test_df=hit_df)

    write_run_metadata(HERE / "side_bias_metadata.json", df, {
        "apply_training_filter": APPLY_TRAINING_FILTER, "drop_prev_angle_45": DROP_PREV_ANGLE_45,
        "mode": MODE, "min_trials": MIN_TRIALS, "ridge_c": RIDGE_C,
        "delta_bic_threshold": DELTA_BIC_THRESHOLD,
        "input_tables_consistent": not problems, "input_table_warnings": problems,
    })
    print_consistency_warning(problems)