"""
GLM analysis with modality interactions for Section 6.

Modification of the per-modality script: instead of fitting one GLM per
(rat, modality), fits ONE GLM per rat on all trials, with modality dummies
fully interacted with every predictor. This handles the mixed-session
design correctly (sessions contain trials from all three modalities), and
lets contrasts (Delta_extreme, Delta_multi, etc.) be extracted directly
from the fitted coefficient vector with the correct joint session-level
bootstrap variance.

Parameterization
----------------
Modality is coded as two dummy variables:
    mod_V  = 1 if visual,        else 0
    mod_VT = 1 if visuotactile,  else 0
Tactile is the reference (mod_V = mod_VT = 0).

Every predictor in FEATURES_FULL is interacted with both dummies. So the
model has:
    intercept                                (tactile intercept)
    mod_V,  mod_VT                           (intercept shifts)
    angle,               mod_V:angle,               mod_VT:angle
    angle_n-1,           mod_V:angle_n-1,           mod_VT:angle_n-1
    reward_n-1,          mod_V:reward_n-1,          mod_VT:reward_n-1
    success_n-1,         mod_V:success_n-1,         mod_VT:success_n-1
    failure_n-1,         mod_V:failure_n-1,         mod_VT:failure_n-1
= 18 coefficients per rat.

Modality-specific coefficients are recovered as
    beta_visual        = beta_tactile + Delta_visual
    beta_visuotactile  = beta_tactile + Delta_VT
and paired contrasts as linear combinations of the Delta terms.

Point-estimate equivalence
--------------------------
For rats whose sessions contain a single modality only, the interaction fit
gives point estimates identical to fitting three separate per-modality GLMs
(the log-likelihood decomposes). For rats with mixed-modality sessions
(the case here), the point estimates coincide only up to shared-column
identifiability, which for the design used here means they still coincide
in the unpenalized limit -- but the bootstrap variances differ, because the
joint session bootstrap correctly handles session-level correlation across
modalities.

LR tests
--------
For each history predictor block Q in {success, failure, reward, angle_n-1},
a per-rat likelihood-ratio test compares the full interaction model against
a restricted model in which the modality interactions on Q are dropped
(Q's main coefficient is retained). Under the null, -2 (ll_restricted -
ll_full) is chi-squared with 2 df.

Output structure
----------------
Two CSVs (in addition to the current per-fit metadata JSON):

    glm_summary_long.csv
        One row per (rat, C, modality, quantity). Modality-specific quantities
        computed from the joint fit's main + interaction coefficients. Layout
        matches the per-modality script's output, so downstream code that
        reads that file continues to work.

    glm_contrasts_long.csv
        One row per (rat, C, contrast_pair, quantity). Contrast pairs are
        V-T, VT-T, VT-V, and multi = VT - (T+V)/2. Point estimate + 95%
        bootstrap CI + significance flag. This is the file Section 6.2
        consumes.

    glm_bootstrap_draws.parquet
        One row per (rat, C, quantity_or_contrast, boot_idx). Both
        per-modality reconstructed values AND contrasts are stored, so the
        bootstrap CIs on any downstream derived quantity can be recomputed
        without refitting.

    glm_lr_tests.csv
        One row per (rat, C, predictor_block). Chi-squared statistic,
        degrees of freedom (2), p-value, and a boolean "rejects at 5%" flag.

    glm_metadata.json
        Config + per-fit diagnostics, same layout as per-modality script.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

from functions_Ale import filter_training_trials


# ============================================================
# CONFIG
# ============================================================

MODE = "pooled"           # NEW MODE. Kept "pooled" / "per_modality" below unused here.

C_VALUES: List[Optional[float]] = [None, 0.1, 1.0, 10.0]
REFERENCE_C: Optional[float] = None

N_BOOT = 5000
CI_LEVEL = 95
RANDOM_STATE = 42
N_JOBS = 4
MIN_TRIALS = 200

STIMULUS_ANGLE_MIN = 0.0
STIMULUS_ANGLE_MAX = 90.0
ANGLE_CENTER_DEG = 45.0
MIN_ABS_ANGLE_BETA = 1e-6

OUT_DIR = "./glm_results_thesis_filtered_interaction"

# Base features (the tactile-reference "main" columns). Modality dummies and
# every interaction with these are added in build_design_matrix_interaction.
FEATURES_FULL = [
    "angle",
    "angle_n-1",
    "reward_n-1",
    "success_n-1",
    "failure_n-1",
]

# History predictor blocks tested by the LR routine. Each block corresponds
# to a set of columns dropped from the design under H0.
LR_BLOCKS = ["angle_n-1", "reward_n-1", "success_n-1", "failure_n-1"]

MODALITY_CODES = {"tactile": 1, "visual": 2, "visuotactile": 3}
CONTINUOUS_REGRESSORS = {"angle", "angle_n-1"}
FILTER_KWARGS = dict(method='criterion', criterion=0.80, min_consec=2)


# ============================================================
# PREPROCESSING (unchanged from per-modality script)
# ============================================================

def prepare_history_predictors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    vals = set(pd.Series(out["action_n-1"].dropna().unique()).astype(float))
    if vals.issubset({0.0, 1.0}):
        out["action_n-1"] = out["action_n-1"] * 2.0 - 1.0
    elif not vals.issubset({-1.0, 1.0}):
        raise ValueError("action_n-1 must be 0/1 or -1/+1.")

    hit_vals = set(pd.Series(out["hit_n-1"].dropna().unique()).astype(float))
    if not hit_vals.issubset({0.0, 1.0}):
        raise ValueError("hit_n-1 must be 0/1.")

    hit01 = out["hit_n-1"].astype(float)
    out["success_n-1"] = hit01 * out["action_n-1"]
    out["failure_n-1"] = (1.0 - hit01) * out["action_n-1"]
    out["reward_n-1"] = hit01 - 0.5

    out["angle"] = out["angle"].astype(float) - ANGLE_CENTER_DEG
    out["angle_n-1"] = out["angle_n-1"].astype(float) - ANGLE_CENTER_DEG

    return out


# ============================================================
# DESIGN MATRIX FOR INTERACTION FIT
# ============================================================

def build_design_matrix_interaction(
    rat_data: pd.DataFrame,
    base_features: List[str],
    exclude_blocks: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, float]]:
    """Build the interaction design matrix for one rat.

    Column layout, in order:
        (continuous base features, scaled to unit SD)
            angle, angle_n-1
        (categorical base features, native scale)
            reward_n-1, success_n-1, failure_n-1
        (modality dummies, native 0/1)
            mod_V, mod_VT
        (interactions of every base feature with each modality dummy)
            mod_V:angle, mod_VT:angle,
            mod_V:angle_n-1, mod_VT:angle_n-1,
            mod_V:reward_n-1, mod_VT:reward_n-1,
            mod_V:success_n-1, mod_VT:success_n-1,
            mod_V:failure_n-1, mod_VT:failure_n-1

    exclude_blocks: list of base feature names whose modality-interaction
    columns should be dropped (used for LR-test restricted models). The
    corresponding main column is retained; only the mod_V:Q and mod_VT:Q
    interaction columns are removed.

    Continuous regressors are scaled to unit SD using a scaler fitted on the
    base column, and the same scale factor is applied to the corresponding
    interaction columns so that Delta coefficients are on the same scale as
    the main coefficient.
    """
    exclude_blocks = set(exclude_blocks or [])

    cont_base = [f for f in base_features if f in CONTINUOUS_REGRESSORS]
    cat_base  = [f for f in base_features if f not in CONTINUOUS_REGRESSORS]

    scale_by_feature: Dict[str, float] = {}
    X_parts = []
    names: List[str] = []

    # --- continuous base features (scaled) ---
    cont_scaled = {}
    if cont_base:
        X_cont = rat_data[cont_base].to_numpy(dtype=float)
        scaler = StandardScaler(with_mean=False)
        X_cont_scaled = scaler.fit_transform(X_cont)
        for i, name in enumerate(cont_base):
            scale_by_feature[name] = float(scaler.scale_[i])
            cont_scaled[name] = X_cont_scaled[:, i]
            X_parts.append(X_cont_scaled[:, [i]])
            names.append(name)

    # --- categorical base features (native scale) ---
    cat_native = {}
    for name in cat_base:
        col = rat_data[name].to_numpy(dtype=float)
        cat_native[name] = col
        scale_by_feature[name] = 1.0
        X_parts.append(col.reshape(-1, 1))
        names.append(name)

    # --- modality dummies ---
    mod_col = rat_data["mod"].to_numpy(dtype=float)
    dummy_V  = (mod_col == MODALITY_CODES["visual"]).astype(float)
    dummy_VT = (mod_col == MODALITY_CODES["visuotactile"]).astype(float)
    X_parts.append(dummy_V.reshape(-1, 1));  names.append("mod_V")
    X_parts.append(dummy_VT.reshape(-1, 1)); names.append("mod_VT")

    # --- interaction columns ---
    for base in base_features:
        if base in exclude_blocks:
            continue
        if base in cont_scaled:
            base_col = cont_scaled[base]
        else:
            base_col = cat_native[base]
        for dummy, dname in [(dummy_V, "mod_V"), (dummy_VT, "mod_VT")]:
            X_parts.append((dummy * base_col).reshape(-1, 1))
            names.append(f"{dname}:{base}")

    X = np.column_stack(X_parts)
    y = rat_data["action"].to_numpy(dtype=int)
    return X, y, names, scale_by_feature


# ============================================================
# FITTING (unchanged core)
# ============================================================

def fit_logistic(X: np.ndarray, y: np.ndarray, C: Optional[float]) -> LogisticRegression:
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=ConvergenceWarning)
        try:
            if C is None:
                model = LogisticRegression(penalty=None, max_iter=2000, solver="lbfgs")
            else:
                model = LogisticRegression(penalty="l2", C=C, max_iter=2000, solver="lbfgs")
            model.fit(X, y)
            return model
        except (ConvergenceWarning, Exception):
            model = LogisticRegression(penalty="l2", C=1.0, max_iter=5000, solver="lbfgs")
            model.fit(X, y)
            return model


def log_likelihood(model, X: np.ndarray, y: np.ndarray) -> float:
    proba = model.predict_proba(X)
    return -log_loss(y, proba, normalize=False)


def compute_aic_bic(model, X: np.ndarray, y: np.ndarray, C: Optional[float]) -> Tuple[float, float, float]:
    n = len(y)
    ll = log_likelihood(model, X, y)
    k = X.shape[1] + 1
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll
    return float(aic), float(bic), float(ll)


# ============================================================
# BOOTSTRAP (unchanged core; resamples full session pool)
# ============================================================

def _one_bootstrap_fit(X, y, row_indices, C):
    model = fit_logistic(X[row_indices], y[row_indices], C=C)
    return np.concatenate(([float(model.intercept_[0])], model.coef_[0]))


def bootstrap_parameters(
    rat_data: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    C: Optional[float],
    n_boot: int,
    random_state: int,
    n_jobs: int,
) -> np.ndarray:
    """Session-level bootstrap on the RAT'S FULL SESSION SET.

    Each resampled session brings all its trials -- across all modalities --
    into the fit. This is the key difference from the per-modality script,
    where each modality's fit resampled its own modality-filtered subset.
    """
    rng = np.random.default_rng(random_state)
    sessions = rat_data["date"].unique()
    n_sessions = len(sessions)
    if n_sessions == 0:
        raise ValueError("No sessions available for bootstrap.")

    date_values = rat_data["date"].to_numpy()
    session_to_rows = {s: np.where(date_values == s)[0] for s in sessions}

    boot_session_idx = rng.integers(0, n_sessions, size=(n_boot, n_sessions))

    def build_rows(session_choice):
        return np.concatenate([session_to_rows[sessions[idx]] for idx in session_choice])

    draws = Parallel(n_jobs=n_jobs)(
        delayed(_one_bootstrap_fit)(X, y, build_rows(boot_session_idx[b]), C)
        for b in range(n_boot)
    )
    return np.asarray(draws, dtype=float)


# ============================================================
# COEFFICIENT UNPACKING AND MODALITY-SPECIFIC RECONSTRUCTION
# ============================================================

def unpack_by_modality(
    intercept: float,
    coefs: np.ndarray,
    column_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """Given the fitted joint model, return per-modality parameter dicts.

    Output:
        {
          "tactile":      {intercept, angle, angle_n-1, reward_n-1, success_n-1, failure_n-1},
          "visual":       same keys, values = tactile + Delta_visual on each,
          "visuotactile": same keys, values = tactile + Delta_VT on each,
        }
    """
    coef_by_name = {name: float(val) for name, val in zip(column_names, coefs)}

    tactile = {"intercept": float(intercept)}
    for base in FEATURES_FULL:
        tactile[base] = coef_by_name.get(base, 0.0)

    visual = {"intercept": tactile["intercept"] + coef_by_name.get("mod_V", 0.0)}
    for base in FEATURES_FULL:
        visual[base] = tactile[base] + coef_by_name.get(f"mod_V:{base}", 0.0)

    visuotactile = {"intercept": tactile["intercept"] + coef_by_name.get("mod_VT", 0.0)}
    for base in FEATURES_FULL:
        visuotactile[base] = tactile[base] + coef_by_name.get(f"mod_VT:{base}", 0.0)

    return {"tactile": tactile, "visual": visual, "visuotactile": visuotactile}


# ============================================================
# QUANTITY RECONSTRUCTION (per-modality, from unpacked params)
# ============================================================

def compute_quantities_from_params(
    params: Dict[str, float],
    angle_scale_deg: float,
) -> Dict[str, float]:
    """Same set of quantities as the per-modality script, computed from one
    modality's parameter dict.

    params keys: intercept, angle, angle_n-1, reward_n-1, success_n-1, failure_n-1.
    """
    out: Dict[str, float] = {}

    out["intercept"] = params["intercept"]
    out["beta_angle"] = params["angle"]
    out["beta_angle_n-1"] = params["angle_n-1"]
    out["beta_reward_n-1"] = params["reward_n-1"]
    out["beta_success_n-1"] = params["success_n-1"]
    out["beta_failure_n-1"] = params["failure_n-1"]

    out["dp_success"] = 0.5 * params["success_n-1"]
    out["dp_failure"] = 0.5 * params["failure_n-1"]
    out["dp_reward"]  = 0.25 * params["reward_n-1"]

    b_angle = params["angle"]
    if abs(b_angle) < MIN_ABS_ANGLE_BETA:
        for key in ["delta_mu_rewarded", "delta_mu_unrewarded",
                    "S_glm", "D_glm", "P_glm",
                    "pse_V_rewarded", "pse_H_rewarded",
                    "pse_V_unrewarded", "pse_H_unrewarded"]:
            out[key] = np.nan
        return out

    b0 = params["intercept"]
    b_reward = params["reward_n-1"]
    b_success = params["success_n-1"]
    b_failure = params["failure_n-1"]

    scale_over_slope = angle_scale_deg / b_angle

    def pse_from_offset(offset: float) -> float:
        centered_pse_deg = -scale_over_slope * offset
        return ANGLE_CENTER_DEG + centered_pse_deg

    pse_v_r = pse_from_offset(b0 + 0.5 * b_reward + b_success)
    pse_h_r = pse_from_offset(b0 + 0.5 * b_reward - b_success)
    pse_v_u = pse_from_offset(b0 - 0.5 * b_reward + b_failure)
    pse_h_u = pse_from_offset(b0 - 0.5 * b_reward - b_failure)

    delta_r = pse_v_r - pse_h_r
    delta_u = pse_v_u - pse_h_u

    out["delta_mu_rewarded"] = float(delta_r)
    out["delta_mu_unrewarded"] = float(delta_u)
    out["S_glm"] = float(delta_r + delta_u)
    out["D_glm"] = float(delta_r - delta_u)
    out["P_glm"] = float(0.5 * (pse_v_u + pse_h_u) - 0.5 * (pse_v_r + pse_h_r))
    out["pse_V_rewarded"] = float(pse_v_r)
    out["pse_H_rewarded"] = float(pse_h_r)
    out["pse_V_unrewarded"] = float(pse_v_u)
    out["pse_H_unrewarded"] = float(pse_h_u)

    return out


# ============================================================
# CONTRAST DEFINITIONS
# ============================================================

CONTRAST_PAIRS = {
    "V_minus_T":     lambda vals: vals["visual"]       - vals["tactile"],
    "VT_minus_T":    lambda vals: vals["visuotactile"] - vals["tactile"],
    "VT_minus_V":    lambda vals: vals["visuotactile"] - vals["visual"],
    "multi":         lambda vals: vals["visuotactile"] - 0.5 * (vals["tactile"] + vals["visual"]),
}


def compute_contrasts_from_quantities(
    per_modality_quantities: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    """Given a dict {modality: {quantity: value}}, return
    {contrast_pair: {quantity: contrast_value}}."""
    out: Dict[str, Dict[str, float]] = {}
    # All modalities share the same key set.
    quantity_keys = list(per_modality_quantities["tactile"].keys())
    for pair_name, fn in CONTRAST_PAIRS.items():
        out[pair_name] = {}
        for q in quantity_keys:
            vals = {m: per_modality_quantities[m].get(q, np.nan) for m in ["tactile", "visual", "visuotactile"]}
            if any(not np.isfinite(vals[m]) for m in vals):
                out[pair_name][q] = np.nan
            else:
                out[pair_name][q] = float(fn(vals))
    return out


# ============================================================
# CI HELPERS
# ============================================================

def _ci(values: np.ndarray, ci_level: float = CI_LEVEL) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    alpha = (100.0 - ci_level) / 2.0
    return tuple(float(v) for v in np.percentile(values, [alpha, 100.0 - alpha]))


def _ci_excludes_zero(ci: Tuple[float, float]) -> bool:
    lo, hi = ci
    return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))


def _quantity_scale(quantity: str) -> str:
    if quantity == "intercept" or quantity.startswith("beta_"):
        return "log_odds"
    if quantity.startswith("dp_"):
        return "probability"
    if quantity in {"delta_mu_rewarded", "delta_mu_unrewarded",
                    "S_glm", "D_glm", "P_glm",
                    "pse_V_rewarded", "pse_H_rewarded",
                    "pse_V_unrewarded", "pse_H_unrewarded"}:
        return "degrees"
    return "other"


def check_pse_on_scale_one_modality(q: Dict[str, float]) -> bool:
    keys = ["pse_V_rewarded", "pse_H_rewarded", "pse_V_unrewarded", "pse_H_unrewarded"]
    if not all(k in q for k in keys):
        return False
    return all(
        np.isfinite(q[k]) and STIMULUS_ANGLE_MIN <= q[k] <= STIMULUS_ANGLE_MAX
        for k in keys
    )


def profile_from_boot_quantities(boot_quantities: Dict[str, np.ndarray]) -> str:
    labels = []
    if "S_glm" in boot_quantities and _ci_excludes_zero(_ci(boot_quantities["S_glm"])):
        labels.append("choice")
    if "D_glm" in boot_quantities and _ci_excludes_zero(_ci(boot_quantities["D_glm"])):
        labels.append("interaction")
    if "P_glm" in boot_quantities and _ci_excludes_zero(_ci(boot_quantities["P_glm"])):
        labels.append("reward")
    return "+".join(labels) if labels else "null"


# ============================================================
# LR TESTS
# ============================================================

def lr_test_block(
    rat_data: pd.DataFrame,
    C: Optional[float],
    block: str,
    ll_full: float,
) -> Dict[str, float]:
    """Fit the model with the modality interactions on `block` removed, and
    return the chi-squared statistic vs. the full model.

    Under H0 (no modality effect on this predictor's coefficient), the two
    removed interaction terms (mod_V:block and mod_VT:block) are zero.
    df = 2.
    """
    X_r, y_r, names_r, _ = build_design_matrix_interaction(
        rat_data, FEATURES_FULL, exclude_blocks=[block],
    )
    model_r = fit_logistic(X_r, y_r, C=C)
    ll_r = log_likelihood(model_r, X_r, y_r)

    chi2_stat = float(-2.0 * (ll_r - ll_full))
    # Chi-squared statistics can go slightly negative when the restricted model
    # accidentally fits better under ridge or numerical noise; clamp at 0.
    chi2_stat = max(0.0, chi2_stat)
    df = 2
    p_value = float(1.0 - stats.chi2.cdf(chi2_stat, df=df))
    return {
        "block": block,
        "chi2": chi2_stat,
        "df": df,
        "p_value": p_value,
        "reject_at_5pct": bool(p_value < 0.05),
        "ll_restricted": float(ll_r),
    }


# ============================================================
# ONE FIT (per rat, per C)
# ============================================================

def analyze_single_fit_interaction(
    rat_data: pd.DataFrame,
    C: Optional[float],
    n_boot: int,
    random_state: int,
    n_jobs: int,
    run_lr_tests: bool,
) -> Dict:
    """Fit the joint interaction model and session-bootstrap it. Also runs
    LR tests for each history predictor block if requested."""
    X, y, column_names, scale_by_feature = build_design_matrix_interaction(
        rat_data, FEATURES_FULL, exclude_blocks=None,
    )
    model = fit_logistic(X, y, C=C)
    aic, bic, ll = compute_aic_bic(model, X, y, C=C)

    boot_params = bootstrap_parameters(
        rat_data, X, y, C=C,
        n_boot=n_boot, random_state=random_state, n_jobs=n_jobs,
    )

    angle_scale = scale_by_feature.get("angle", 1.0)

    # ---- point-estimate quantities per modality
    per_mod_params_point = unpack_by_modality(
        model.intercept_[0], model.coef_[0], column_names
    )
    per_mod_quantities_point: Dict[str, Dict[str, float]] = {}
    for m, params in per_mod_params_point.items():
        per_mod_quantities_point[m] = compute_quantities_from_params(params, angle_scale)

    # ---- point-estimate contrasts
    point_contrasts = compute_contrasts_from_quantities(per_mod_quantities_point)

    # ---- bootstrap quantities per modality, and bootstrap contrasts
    # For each boot draw, unpack -> compute quantities per modality -> compute contrasts.
    quantity_keys = list(per_mod_quantities_point["tactile"].keys())
    boot_quantities_by_mod: Dict[str, Dict[str, List[float]]] = {
        m: {q: [] for q in quantity_keys} for m in ["tactile", "visual", "visuotactile"]
    }
    boot_contrasts: Dict[str, Dict[str, List[float]]] = {
        pair: {q: [] for q in quantity_keys} for pair in CONTRAST_PAIRS.keys()
    }

    for row in boot_params:
        per_mod_p = unpack_by_modality(row[0], row[1:], column_names)
        per_mod_q = {m: compute_quantities_from_params(p, angle_scale)
                     for m, p in per_mod_p.items()}
        cont = compute_contrasts_from_quantities(per_mod_q)
        for m in ["tactile", "visual", "visuotactile"]:
            for q in quantity_keys:
                boot_quantities_by_mod[m][q].append(per_mod_q[m].get(q, np.nan))
        for pair in CONTRAST_PAIRS.keys():
            for q in quantity_keys:
                boot_contrasts[pair][q].append(cont[pair].get(q, np.nan))

    # Convert to arrays.
    for m in boot_quantities_by_mod:
        for q in quantity_keys:
            boot_quantities_by_mod[m][q] = np.asarray(boot_quantities_by_mod[m][q], dtype=float)
    for pair in boot_contrasts:
        for q in quantity_keys:
            boot_contrasts[pair][q] = np.asarray(boot_contrasts[pair][q], dtype=float)

    # ---- CIs
    ci_by_mod_by_q = {m: {q: _ci(boot_quantities_by_mod[m][q]) for q in quantity_keys}
                      for m in ["tactile", "visual", "visuotactile"]}
    ci_by_pair_by_q = {pair: {q: _ci(boot_contrasts[pair][q]) for q in quantity_keys}
                       for pair in CONTRAST_PAIRS.keys()}

    # ---- diagnostics per modality (pse_on_scale, profile)
    pse_on_scale_by_mod = {
        m: check_pse_on_scale_one_modality(per_mod_quantities_point[m])
        for m in ["tactile", "visual", "visuotactile"]
    }
    profile_by_mod = {
        m: profile_from_boot_quantities(boot_quantities_by_mod[m])
        for m in ["tactile", "visual", "visuotactile"]
    }

    # ---- beta_angle diagnostics per modality
    beta_angle_boot_min_by_mod = {}
    beta_angle_boot_frac_pos_by_mod = {}
    for m in ["tactile", "visual", "visuotactile"]:
        vals = boot_quantities_by_mod[m]["beta_angle"]
        beta_angle_boot_min_by_mod[m] = float(np.min(vals)) if len(vals) else np.nan
        beta_angle_boot_frac_pos_by_mod[m] = float(np.mean(vals > 0)) if len(vals) else np.nan

    # ---- per-modality trial counts (for the metadata)
    mod_col = rat_data["mod"].to_numpy()
    n_trials_by_mod = {
        "tactile":      int((mod_col == MODALITY_CODES["tactile"]).sum()),
        "visual":       int((mod_col == MODALITY_CODES["visual"]).sum()),
        "visuotactile": int((mod_col == MODALITY_CODES["visuotactile"]).sum()),
    }
    # Sessions containing at least one trial of that modality
    date_col = rat_data["date"].to_numpy()
    n_sessions_by_mod = {
        m: int(pd.Series(date_col[mod_col == MODALITY_CODES[m]]).nunique())
        for m in ["tactile", "visual", "visuotactile"]
    }

    # ---- LR tests (fit 4 restricted models, chi2 vs full)
    lr_results = None
    if run_lr_tests:
        lr_results = []
        for block in LR_BLOCKS:
            lr_results.append(lr_test_block(rat_data, C, block, ll_full=ll))

    return {
        "column_names": column_names,
        "coefficients_full": model.coef_[0].copy(),
        "intercept_full": float(model.intercept_[0]),
        "angle_scale_deg": angle_scale,
        "aic_approx": aic,
        "bic_approx": bic,
        "log_lik": ll,
        "n_trials_total": int(len(y)),
        "n_sessions_total": int(rat_data["date"].nunique()),
        "n_trials_by_mod": n_trials_by_mod,
        "n_sessions_by_mod": n_sessions_by_mod,
        # Per-modality reconstructed point estimates and CIs
        "per_mod_quantities_point": per_mod_quantities_point,
        "per_mod_ci": ci_by_mod_by_q,
        # Per-modality bootstrap arrays (needed for parquet dump)
        "per_mod_boot": boot_quantities_by_mod,
        # Contrasts
        "point_contrasts": point_contrasts,
        "contrast_ci": ci_by_pair_by_q,
        "boot_contrasts": boot_contrasts,
        # Diagnostics
        "pse_on_scale_by_mod": pse_on_scale_by_mod,
        "profile_by_mod": profile_by_mod,
        "beta_angle_boot_min_by_mod": beta_angle_boot_min_by_mod,
        "beta_angle_boot_frac_positive_by_mod": beta_angle_boot_frac_pos_by_mod,
        # LR tests
        "lr_tests": lr_results,
    }


# ============================================================
# TOP-LEVEL ORCHESTRATION
# ============================================================

def _c_label(C: Optional[float]) -> str:
    return "unpenalized" if C is None else f"C={C}"


def run_analysis(
    df: pd.DataFrame,
    c_values: List[Optional[float]],
    reference_c: Optional[float],
    n_boot: int,
    random_state: int,
    n_jobs: int,
    min_trials: int,
) -> Dict:
    """Fit the interaction model for every rat at every C value.

    LR tests are computed only at the reference C to keep compute manageable
    (they add 4 extra fits per rat per C).
    """
    required = ["rat", "date", "action", "mod", *FEATURES_FULL, "action_n-1"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rats = sorted(df["rat"].dropna().unique())
    all_results = {}
    t0 = time.time()

    for rat_idx, rat_id in enumerate(rats):
        rat_df = df[df["rat"] == rat_id].dropna(
            subset=["action", "date", "mod", *FEATURES_FULL, "action_n-1"]
        ).copy()

        if len(rat_df) < min_trials:
            print(f"  Rat {rat_id}: skip ({len(rat_df)} trials)")
            continue

        n_sessions = rat_df["date"].nunique()
        print(f"  Rat {rat_id}: {len(rat_df)} trials, {n_sessions} sessions"
              f"  (elapsed {(time.time()-t0)/60:.1f} min)")

        all_results[rat_id] = {}
        for C in c_values:
            is_reference = (C == reference_c)
            fit_seed = random_state + rat_idx  # deterministic per-rat seed
            result = analyze_single_fit_interaction(
                rat_df, C=C,
                n_boot=n_boot, random_state=fit_seed, n_jobs=n_jobs,
                run_lr_tests=is_reference,
            )
            all_results[rat_id][_c_label(C)] = result

    return all_results


# ============================================================
# OUTPUT BUILDERS
# ============================================================

def build_summary_long(all_results: Dict) -> pd.DataFrame:
    """One row per (rat, C, modality, quantity). Layout matches the
    per-modality script's output so downstream code doesn't need to change."""
    rows = []
    for rat_id, per_C in all_results.items():
        for c_label, res in per_C.items():
            for m in ["tactile", "visual", "visuotactile"]:
                pse = res["pse_on_scale_by_mod"][m]
                profile = res["profile_by_mod"][m]
                q_dict = res["per_mod_quantities_point"][m]
                ci_dict = res["per_mod_ci"][m]
                n_trials = res["n_trials_by_mod"][m]
                n_sessions = res["n_sessions_by_mod"][m]
                for quantity, value in q_dict.items():
                    lo, hi = ci_dict.get(quantity, (np.nan, np.nan))
                    rows.append({
                        "rat_id": int(rat_id),
                        "subset": m,
                        "C": c_label,
                        "n_trials": n_trials,
                        "n_sessions": n_sessions,
                        "quantity": quantity,
                        "scale": _quantity_scale(quantity),
                        "value": value,
                        "ci_low": lo,
                        "ci_high": hi,
                        "pse_on_scale": pse,
                        "profile": profile,
                    })
    return pd.DataFrame(rows)


def build_contrasts_long(all_results: Dict) -> pd.DataFrame:
    """One row per (rat, C, contrast_pair, quantity)."""
    rows = []
    for rat_id, per_C in all_results.items():
        for c_label, res in per_C.items():
            for pair, q_dict in res["point_contrasts"].items():
                for quantity, value in q_dict.items():
                    lo, hi = res["contrast_ci"][pair].get(quantity, (np.nan, np.nan))
                    excludes_zero = _ci_excludes_zero((lo, hi))
                    rows.append({
                        "rat_id": int(rat_id),
                        "C": c_label,
                        "contrast_pair": pair,
                        "quantity": quantity,
                        "scale": _quantity_scale(quantity),
                        "value": value,
                        "ci_low": lo,
                        "ci_high": hi,
                        "ci_excludes_zero": excludes_zero,
                    })
    return pd.DataFrame(rows)


def build_bootstrap_long(all_results: Dict) -> pd.DataFrame:
    """One row per (rat, C, kind, subset_or_pair, quantity, boot_idx).

    kind = "modality"  -> subset_or_pair is a modality name
    kind = "contrast"  -> subset_or_pair is a contrast pair name
    """
    frames = []
    for rat_id, per_C in all_results.items():
        for c_label, res in per_C.items():
            # Per-modality reconstructed bootstrap arrays.
            for m, q_dict in res["per_mod_boot"].items():
                if not q_dict:
                    continue
                n_boot = len(next(iter(q_dict.values())))
                for quantity, values in q_dict.items():
                    frames.append(pd.DataFrame({
                        "rat_id": int(rat_id),
                        "C": c_label,
                        "kind": "modality",
                        "subset_or_pair": m,
                        "quantity": quantity,
                        "boot_idx": np.arange(n_boot, dtype=np.int32),
                        "value": values.astype(np.float32),
                    }))
            # Bootstrap contrasts.
            for pair, q_dict in res["boot_contrasts"].items():
                if not q_dict:
                    continue
                n_boot = len(next(iter(q_dict.values())))
                for quantity, values in q_dict.items():
                    frames.append(pd.DataFrame({
                        "rat_id": int(rat_id),
                        "C": c_label,
                        "kind": "contrast",
                        "subset_or_pair": pair,
                        "quantity": quantity,
                        "boot_idx": np.arange(n_boot, dtype=np.int32),
                        "value": values.astype(np.float32),
                    }))
    if not frames:
        return pd.DataFrame(columns=["rat_id","C","kind","subset_or_pair","quantity","boot_idx","value"])
    return pd.concat(frames, ignore_index=True)


def build_lr_tests(all_results: Dict) -> pd.DataFrame:
    rows = []
    for rat_id, per_C in all_results.items():
        for c_label, res in per_C.items():
            lr = res.get("lr_tests")
            if lr is None:
                continue
            for entry in lr:
                rows.append({
                    "rat_id": int(rat_id),
                    "C": c_label,
                    "block": entry["block"],
                    "chi2": entry["chi2"],
                    "df": entry["df"],
                    "p_value": entry["p_value"],
                    "reject_at_5pct": entry["reject_at_5pct"],
                    "ll_full": res["log_lik"],
                    "ll_restricted": entry["ll_restricted"],
                })
    return pd.DataFrame(rows)


def build_metadata(all_results: Dict, config: Dict) -> Dict:
    per_fit = []
    for rat_id, per_C in all_results.items():
        for c_label, res in per_C.items():
            entry = {
                "rat_id": int(rat_id),
                "C": c_label,
                "n_trials_total": res["n_trials_total"],
                "n_sessions_total": res["n_sessions_total"],
                "n_trials_by_mod": res["n_trials_by_mod"],
                "n_sessions_by_mod": res["n_sessions_by_mod"],
                "angle_scale_deg": res["angle_scale_deg"],
                "aic_approx": res["aic_approx"],
                "bic_approx": res["bic_approx"],
                "log_lik": res["log_lik"],
                "pse_on_scale_by_mod": {m: bool(v) for m, v in res["pse_on_scale_by_mod"].items()},
                "profile_by_mod": res["profile_by_mod"],
                "beta_angle_boot_min_by_mod": res["beta_angle_boot_min_by_mod"],
                "beta_angle_boot_frac_positive_by_mod": res["beta_angle_boot_frac_positive_by_mod"],
            }
            per_fit.append(entry)
    return {"config": config, "per_fit": per_fit}


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import sys

    sys.path.append(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy")
    from serial_dependence_analysis import SerialDependenceAnalyzer

    an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
    an.load_and_preprocess_data()
    df = an.create_lagged_features()
    df = filter_training_trials(df, **FILTER_KWARGS)
    df = prepare_history_predictors(df)

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Running GLM analysis (INTERACTION mode)")
    print(f"  C values:    {C_VALUES}")
    print(f"  Reference C: {REFERENCE_C}")
    print(f"  N bootstrap: {N_BOOT}")
    print(f"  Output dir:  {OUT_DIR}")
    print()

    t_start = time.time()
    results = run_analysis(
        df,
        c_values=C_VALUES,
        reference_c=REFERENCE_C,
        n_boot=N_BOOT,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        min_trials=MIN_TRIALS,
    )
    t_fit = time.time() - t_start
    print(f"\nFitting done in {t_fit/60:.1f} minutes")

    print("Building output tables...")
    summary_long = build_summary_long(results)
    contrasts_long = build_contrasts_long(results)
    boot_long = build_bootstrap_long(results)
    lr_tests = build_lr_tests(results)

    config = {
        "mode": "interaction",
        "c_values": [str(c) if c is not None else "unpenalized" for c in C_VALUES],
        "reference_c": "unpenalized" if REFERENCE_C is None else str(REFERENCE_C),
        "n_boot": N_BOOT,
        "ci_level": CI_LEVEL,
        "random_state": RANDOM_STATE,
        "min_trials": MIN_TRIALS,
        "stimulus_angle_min": STIMULUS_ANGLE_MIN,
        "stimulus_angle_max": STIMULUS_ANGLE_MAX,
        "angle_center_deg": ANGLE_CENTER_DEG,
        "timestamp": datetime.now().isoformat(),
        "fitting_seconds": t_fit,
    }
    metadata = build_metadata(results, config)

    summary_path   = os.path.join(OUT_DIR, "glm_summary_long.csv")
    contrasts_path = os.path.join(OUT_DIR, "glm_contrasts_long.csv")
    boot_path      = os.path.join(OUT_DIR, "glm_bootstrap_draws.parquet")
    lr_path        = os.path.join(OUT_DIR, "glm_lr_tests.csv")
    meta_path      = os.path.join(OUT_DIR, "glm_metadata.json")

    summary_long.to_csv(summary_path, index=False)
    print(f"  Saved {summary_path} ({len(summary_long)} rows)")

    contrasts_long.to_csv(contrasts_path, index=False)
    print(f"  Saved {contrasts_path} ({len(contrasts_long)} rows)")

    try:
        boot_long.to_parquet(boot_path, index=False, compression="snappy")
        print(f"  Saved {boot_path} ({len(boot_long):,} rows)")
    except Exception as e:
        print(f"  Parquet save failed ({e}); falling back to CSV")
        boot_path_csv = os.path.join(OUT_DIR, "glm_bootstrap_draws.csv.gz")
        boot_long.to_csv(boot_path_csv, index=False, compression="gzip")
        print(f"  Saved {boot_path_csv}")

    lr_tests.to_csv(lr_path, index=False)
    print(f"  Saved {lr_path} ({len(lr_tests)} rows)")

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Saved {meta_path}")

    print(f"\nTotal time: {(time.time() - t_start)/60:.1f} minutes")