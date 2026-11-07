"""
Analysis 2 - sign-aligned analysis of history-effect modality modulation.

Pipeline:
    1. Per-rat per-modality ridge logistic fits (features_19) - point estimates only
       by default. Toggle BOOTSTRAP_PER_MODALITY = True to also compute session-level
       bootstrap CIs on per-modality coefficients (needed for the follow-up
       mixed-model analysis on the 36 aligned values, not for the Wilcoxon).
    2. Per-rat baseline sign fits with session-level bootstrap CIs:
        - pooled (all modalities)      - sanity check
        - leave-one-out (x3)           - primary sign source; the LOO fit that
                                         holds out modality m is used to align
                                         that rat's beta in modality m
    3. Sign alignment: multiply per-modality beta by the LOO sign (holding out
       that modality).
    4. Wilcoxon signed-rank tests on per-rat modality contrasts of sign-aligned
       values, per regressor and per modality pair (T vs V, T vs VT, V vs VT).

Outputs (all CSVs in OUT_DIR):
    per_modality_coefficients.csv        rat x modality x regressor
    baseline_signs.csv                   rat x sign_method x regressor
    sign_aligned_coefficients.csv        rat x modality x regressor (aligned values)
    wilcoxon_modality_contrasts.csv      regressor x modality_pair (test results)

Runtime note:
    With N_BOOT = 1000, N_JOBS = 4, 12 rats, and BOOTSTRAP_PER_MODALITY = False,
    expect roughly 1-2 hours dominated by pooled and LOO bootstraps. Set N_BOOT
    lower for exploration.
"""

from __future__ import annotations
import os
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIG
# ============================================================

FEATURES = ["angle", "angle_n-1", "success_n-1", "failure_n-1"]  # features_19
CONTINUOUS_REGRESSORS = {"angle", "angle_n-1"}
ALIGN_REGRESSORS = ["success_n-1", "failure_n-1"]                # sign-align these

MODALITY_CODES = {"tactile": 1, "visual": 2, "visuotactile": 3}
MODALITY_NAMES = list(MODALITY_CODES.keys())

RIDGE_C = 1.0                 # matches existing pipeline
N_BOOT = 1000                 # 5000 for publication
CI_LEVEL = 95
RANDOM_STATE = 42
N_JOBS = 4
MIN_TRIALS = 200

BOOTSTRAP_PER_MODALITY = False  # False: Wilcoxon only. True: also compute CIs
                                # on per-modality betas (for later mixed model).

OUT_DIR = "./analysis2_outputs"


# ============================================================
# ATOMIC FUNCTIONS (mirroring the existing pipeline)
# ============================================================

def build_design_matrix(data: pd.DataFrame, feature_names: list[str]):
    """Return (X, y, ordered_feature_names).

    Continuous regressors are scaled with StandardScaler(with_mean=False);
    binary regressors are left as-is. Matches the existing pipeline.
    """
    cont = [f for f in feature_names if f in CONTINUOUS_REGRESSORS]
    binn = [f for f in feature_names if f not in CONTINUOUS_REGRESSORS]

    parts, names = [], []
    if cont:
        Xc = data[cont].values.astype(float)
        Xc = StandardScaler(with_mean=False).fit_transform(Xc)
        parts.append(Xc); names.extend(cont)
    if binn:
        Xb = data[binn].values.astype(float)
        parts.append(Xb); names.extend(binn)

    X = np.column_stack(parts)
    y = data["action"].values.astype(int)
    return X, y, names


def fit_ridge_logistic(X, y, C=RIDGE_C):
    m = LogisticRegression(penalty="l2", C=C, max_iter=1000, solver="lbfgs")
    m.fit(X, y)
    return m


def _one_bootstrap_fit(X, y, row_indices, C):
    m = fit_ridge_logistic(X[row_indices], y[row_indices], C=C)
    return m.coef_[0]


def bootstrap_coefficients(data, X, y, n_boot=N_BOOT, C=RIDGE_C,
                           random_state=RANDOM_STATE, n_jobs=N_JOBS):
    """Session-level bootstrap on the (already-built) design matrix X and y.

    Uses `date` as the session identifier. Ridge C is held fixed across resamples
    (matches existing pipeline; understates uncertainty only slightly).
    """
    rng = np.random.default_rng(random_state)
    sessions = data["date"].unique()
    n_sessions = len(sessions)
    s2rows = {s: np.where(data["date"].values == s)[0] for s in sessions}

    boot_choice = rng.integers(0, n_sessions, size=(n_boot, n_sessions))

    def build_rows(choice):
        return np.concatenate([s2rows[sessions[s]] for s in choice])

    coefs = Parallel(n_jobs=n_jobs)(
        delayed(_one_bootstrap_fit)(X, y, build_rows(boot_choice[i]), C)
        for i in range(n_boot)
    )
    return np.asarray(coefs)


# ============================================================
# FIT WRAPPER
# ============================================================

def fit_on_subset(rat_df: pd.DataFrame, modalities_to_include: list[int],
                  do_bootstrap: bool = True,
                  n_boot: int = N_BOOT,
                  random_state: int = RANDOM_STATE,
                  n_jobs: int = N_JOBS):
    """Fit ridge logistic on the rat's trials restricted to given modalities.

    Returns a dict with coefficients and (optional) bootstrap CIs, or None if
    the subset has fewer than MIN_TRIALS after dropping NaNs.
    """
    subset = rat_df[rat_df["mod"].isin(modalities_to_include)].copy()
    subset = subset.dropna(subset=FEATURES + ["action", "date"])
    n = len(subset)
    if n < MIN_TRIALS:
        return None

    X, y, ordered_names = build_design_matrix(subset, FEATURES)
    model = fit_ridge_logistic(X, y)

    out = {
        "feature_names": ordered_names,
        "coefficients": model.coef_[0],
        "n_trials": n,
        "n_sessions": subset["date"].nunique(),
    }

    if do_bootstrap:
        boot = bootstrap_coefficients(subset, X, y, n_boot=n_boot,
                                      random_state=random_state, n_jobs=n_jobs)
        alpha = (100 - CI_LEVEL) / 2
        out["ci_low"] = np.percentile(boot, alpha, axis=0)
        out["ci_high"] = np.percentile(boot, 100 - alpha, axis=0)

    return out


# ============================================================
# BASELINE SIGN FITS (pooled + leave-one-out)
# ============================================================

def run_baseline_fits(df: pd.DataFrame,
                      n_boot: int = N_BOOT,
                      random_state: int = RANDOM_STATE,
                      n_jobs: int = N_JOBS) -> pd.DataFrame:
    """For each rat, fit 4 baseline models with bootstrap CIs:
        - pooled          (all 3 modalities)
        - loo_tactile     (visual + visuotactile)
        - loo_visual      (tactile + visuotactile)
        - loo_visuotactile (tactile + visual)
    """
    all_mods = list(MODALITY_CODES.values())
    rats = sorted(df["rat"].unique())
    rows = []

    for rat_id in rats:
        rat_df = df[df["rat"] == rat_id].copy()

        methods = [("pooled", "none", all_mods)]
        for name, code in MODALITY_CODES.items():
            methods.append((f"loo_{name}", name,
                            [m for m in all_mods if m != code]))

        for label, held_out, include in methods:
            print(f"  Rat {rat_id} [{label:<20}] ...", end=" ", flush=True)
            fit = fit_on_subset(rat_df, include, do_bootstrap=True,
                                n_boot=n_boot, random_state=random_state,
                                n_jobs=n_jobs)
            if fit is None:
                print(f"skipped (< {MIN_TRIALS} trials)")
                continue
            print(f"n={fit['n_trials']:>6d}, sessions={fit['n_sessions']}")

            for reg in ALIGN_REGRESSORS:
                j = fit["feature_names"].index(reg)
                beta = float(fit["coefficients"][j])
                lo, hi = float(fit["ci_low"][j]), float(fit["ci_high"][j])
                rows.append({
                    "rat": int(rat_id),
                    "sign_method": label,
                    "modality_held_out": held_out,
                    "regressor": reg,
                    "beta": beta,
                    "ci_low": lo,
                    "ci_high": hi,
                    "sign": int(np.sign(beta)) if beta != 0 else 0,
                    "ci_crosses_zero": bool(lo <= 0 <= hi),
                    "n_trials": fit["n_trials"],
                    "n_sessions": fit["n_sessions"],
                })

    return pd.DataFrame(rows)


# ============================================================
# PER-MODALITY FITS
# ============================================================

def run_per_modality_fits(df: pd.DataFrame,
                          do_bootstrap: bool = BOOTSTRAP_PER_MODALITY,
                          n_boot: int = N_BOOT,
                          random_state: int = RANDOM_STATE,
                          n_jobs: int = N_JOBS) -> pd.DataFrame:
    """For each (rat, modality) subset, fit one ridge logistic.

    Point estimates only unless do_bootstrap=True.
    """
    rats = sorted(df["rat"].unique())
    rows = []

    for rat_id in rats:
        rat_df = df[df["rat"] == rat_id].copy()
        for mod_name, mod_code in MODALITY_CODES.items():
            print(f"  Rat {rat_id} [{mod_name:<12}] ...", end=" ", flush=True)
            fit = fit_on_subset(rat_df, [mod_code], do_bootstrap=do_bootstrap,
                                n_boot=n_boot, random_state=random_state,
                                n_jobs=n_jobs)
            if fit is None:
                print(f"skipped (< {MIN_TRIALS} trials)")
                continue
            print(f"n={fit['n_trials']:>6d}, sessions={fit['n_sessions']}")

            for reg in ALIGN_REGRESSORS:
                j = fit["feature_names"].index(reg)
                row = {
                    "rat": int(rat_id),
                    "modality": mod_name,
                    "regressor": reg,
                    "beta": float(fit["coefficients"][j]),
                    "n_trials": fit["n_trials"],
                    "n_sessions": fit["n_sessions"],
                }
                if do_bootstrap:
                    row["ci_low"] = float(fit["ci_low"][j])
                    row["ci_high"] = float(fit["ci_high"][j])
                rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# SIGN ALIGNMENT
# ============================================================

def compute_sign_aligned(per_modality_df: pd.DataFrame,
                         baseline_df: pd.DataFrame) -> pd.DataFrame:
    """For each (rat, modality, regressor), attach the LOO baseline sign
    (holding out that modality) and multiply by the raw beta.

    Also attaches the pooled sign for a consistency diagnostic.
    """
    rows = []
    for _, row in per_modality_df.iterrows():
        rat_id = row["rat"]
        modality = row["modality"]
        regressor = row["regressor"]
        beta_raw = row["beta"]

        loo_sel = ((baseline_df["rat"] == rat_id) &
                   (baseline_df["sign_method"] == f"loo_{modality}") &
                   (baseline_df["regressor"] == regressor))
        pool_sel = ((baseline_df["rat"] == rat_id) &
                    (baseline_df["sign_method"] == "pooled") &
                    (baseline_df["regressor"] == regressor))

        loo_rows = baseline_df[loo_sel]
        pool_rows = baseline_df[pool_sel]

        if len(loo_rows) == 0 or len(pool_rows) == 0:
            continue

        sign_loo = int(loo_rows["sign"].iloc[0])
        sign_pooled = int(pool_rows["sign"].iloc[0])
        loo_beta = float(loo_rows["beta"].iloc[0])
        loo_ci_crosses_zero = bool(loo_rows["ci_crosses_zero"].iloc[0])

        rows.append({
            "rat": int(rat_id),
            "modality": modality,
            "regressor": regressor,
            "beta_raw": float(beta_raw),
            "sign_loo": sign_loo,
            "sign_pooled": sign_pooled,
            "sign_agrees_loo_vs_pooled": bool(sign_loo == sign_pooled),
            "loo_baseline_beta": loo_beta,
            "loo_ci_crosses_zero": loo_ci_crosses_zero,
            "beta_aligned": float(sign_loo * beta_raw),
        })

    return pd.DataFrame(rows)


# ============================================================
# WILCOXON TESTS
# ============================================================

def wilcoxon_modality_contrasts(aligned_df: pd.DataFrame) -> pd.DataFrame:
    """For each regressor and modality pair, compute per-rat contrasts of the
    sign-aligned beta and run a two-sided Wilcoxon signed-rank test.
    """
    rows = []
    for reg in ALIGN_REGRESSORS:
        sub = aligned_df[aligned_df["regressor"] == reg]
        wide = sub.pivot_table(index="rat", columns="modality",
                               values="beta_aligned")

        for m1, m2 in combinations(MODALITY_NAMES, 2):
            if m1 not in wide.columns or m2 not in wide.columns:
                continue
            paired = wide[[m1, m2]].dropna()
            if len(paired) < 3:
                continue

            diff = (paired[m1] - paired[m2]).values
            n_nonzero = int(np.sum(diff != 0))
            if n_nonzero < 3:
                continue

            # scipy.wilcoxon default zero_method='wilcox' drops exact zeros
            result = wilcoxon(diff, alternative="two-sided",
                              zero_method="wilcox")

            rows.append({
                "regressor": reg,
                "modality_1": m1,
                "modality_2": m2,
                "n_rats": len(paired),
                "n_nonzero": n_nonzero,
                "median_diff_m1_minus_m2": float(np.median(diff)),
                "mean_diff_m1_minus_m2": float(np.mean(diff)),
                "W": float(result.statistic),
                "p_value": float(result.pvalue),
            })

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    # -----------------------------------------------------
    # Data loading + preprocessing (adapt to your pipeline)
    # -----------------------------------------------------
    import sys
    sys.path.append(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy")
    from serial_dependence_analysis import SerialDependenceAnalyzer

    an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
    an.load_and_preprocess_data()
    df = an.create_lagged_features()

    df["action_n-1"] = df["action_n-1"] * 2 - 1
    df["success_n-1"] = df["hit_n-1"] * df["action_n-1"]
    df["failure_n-1"] = (1 - df["hit_n-1"]) * df["action_n-1"]
    df["angle"] = df["angle"] - 45
    df["angle_n-1"] = df["angle_n-1"] - 45

    # -----------------------------------------------------
    # 1. Per-modality fits (point estimates by default)
    # -----------------------------------------------------
    print("=" * 60)
    print("Per-modality fits")
    print("=" * 60)
    per_mod_df = run_per_modality_fits(df)
    per_mod_path = os.path.join(OUT_DIR, "per_modality_coefficients.csv")
    per_mod_df.to_csv(per_mod_path, index=False)
    print(f"Saved {per_mod_path}: {len(per_mod_df)} rows")

    # -----------------------------------------------------
    # 2. Baseline sign fits (pooled + LOO, with bootstrap)
    # -----------------------------------------------------
    print()
    print("=" * 60)
    print("Baseline sign fits (pooled + LOO, session-level bootstrap)")
    print("=" * 60)
    baseline_df = run_baseline_fits(df)
    baseline_path = os.path.join(OUT_DIR, "baseline_signs.csv")
    baseline_df.to_csv(baseline_path, index=False)
    print(f"Saved {baseline_path}: {len(baseline_df)} rows")

    # -----------------------------------------------------
    # 3. Sign alignment
    # -----------------------------------------------------
    print()
    print("=" * 60)
    print("Sign-aligned coefficients")
    print("=" * 60)
    aligned_df = compute_sign_aligned(per_mod_df, baseline_df)
    aligned_path = os.path.join(OUT_DIR, "sign_aligned_coefficients.csv")
    aligned_df.to_csv(aligned_path, index=False)
    print(f"Saved {aligned_path}: {len(aligned_df)} rows")

    # sign stability diagnostic
    n_total = len(aligned_df)
    n_disagree = int((~aligned_df["sign_agrees_loo_vs_pooled"]).sum())
    n_crosses = int(aligned_df["loo_ci_crosses_zero"].sum())
    print()
    print("Sign stability diagnostics "
          f"({n_total} rat x modality x regressor combos):")
    print(f"  LOO vs pooled sign disagreements: "
          f"{n_disagree} ({100 * n_disagree / n_total:.1f}%)")
    print(f"  LOO baseline CI crosses zero:     "
          f"{n_crosses} ({100 * n_crosses / n_total:.1f}%)")

    # per-regressor breakdown
    for reg in ALIGN_REGRESSORS:
        sub = aligned_df[aligned_df["regressor"] == reg]
        nd = int((~sub["sign_agrees_loo_vs_pooled"]).sum())
        nc = int(sub["loo_ci_crosses_zero"].sum())
        print(f"    {reg}: {nd} disagreements, {nc} CI-crosses-zero "
              f"(out of {len(sub)})")

    # -----------------------------------------------------
    # 4. Wilcoxon signed-rank tests
    # -----------------------------------------------------
    print()
    print("=" * 60)
    print("Wilcoxon signed-rank tests on modality contrasts of aligned beta")
    print("=" * 60)
    wilcoxon_df = wilcoxon_modality_contrasts(aligned_df)
    wilcoxon_path = os.path.join(OUT_DIR, "wilcoxon_modality_contrasts.csv")
    wilcoxon_df.to_csv(wilcoxon_path, index=False)
    print(f"Saved {wilcoxon_path}: {len(wilcoxon_df)} rows\n")
    print(wilcoxon_df.to_string(index=False))