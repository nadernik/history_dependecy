"""
NEW WITH NON C VALUES: 0.1 1 10 

GLM analysis for previous-choice, previous-outcome, and sensory-history effects
in rat psychophysics, aligned conceptually with the Section 3 psychometric
S / D / P decomposition.

Main model (per rat, optionally per modality):

    logit P(action_n = vertical) =
        intercept
        + beta_angle       * angle_n
        + beta_angle_n1    * angle_{n-1}
        + beta_reward      * reward_{n-1}
        + beta_success     * success_{n-1}
        + beta_failure     * failure_{n-1}

Coding:
    action:       1 = vertical, 0 = horizontal
    action_n-1:  +1 = vertical, -1 = horizontal
    hit_n-1:      1 = rewarded/correct, 0 = unrewarded/error
    reward_n-1:  +0.5 = rewarded, -0.5 = unrewarded
    success_n-1 = hit_n-1 * action_n-1
    failure_n-1 = (1-hit_n-1) * action_n-1

Angles are first centered around 45 degrees in the dataframe, then continuous
regressors are divided by their SD using StandardScaler(with_mean=False).
Categoricals are left on their native scale to preserve interpretability of
the coefficients.

Fitting:
    By default fits are unpenalized MLE. Ridge fits at multiple C values can
    be requested via the C_VALUES config. C=None means unpenalized. Ridge is
    also used as a fallback if unpenalized MLE fails to converge.

Outputs (all written to OUT_DIR):
    - glm_summary_long.csv       point + CI for every quantity, one row per
                                 (rat, subset, C, quantity)
    - glm_bootstrap_draws.parquet raw bootstrap samples for every quantity;
                                 columns: rat_id, subset, C, quantity, boot_idx, value
    - glm_metadata.json          run config, timestamp, and per-fit metadata

Quantities stored per fit:
    Log-odds coefficients (6):
        intercept, beta_angle, beta_angle_n-1, beta_reward_n-1,
        beta_success_n-1, beta_failure_n-1
    Probability-scale summaries at boundary (3):
        dp_success  = 0.5  * beta_success_n-1
        dp_failure  = 0.5  * beta_failure_n-1
        dp_reward   = 0.25 * beta_reward_n-1
    Degree-scale contrasts (5):
        delta_mu_rewarded, delta_mu_unrewarded, S_glm, D_glm, P_glm
    Individual PSE cells (4):
        pse_V_rewarded, pse_H_rewarded, pse_V_unrewarded, pse_H_unrewarded
    Diagnostics (per fit, in metadata JSON, not long table):
        aic_approx, bic_approx, log_lik, n_trials, n_sessions,
        beta_angle_boot_min, beta_angle_boot_frac_positive,
        pse_on_scale, profile

Physical-realizability gate:
    Fits are flagged pse_on_scale=False if any of the four fitted PSE cells
    (point estimate) fall outside the physical stimulus range [0, 90] degrees.
    Degree-scale outputs are still stored for such fits, but downstream
    plotting and reporting should suppress the degree quantities and rely on
    log-odds and probability-scale summaries.

Model comparisons:
    Reduced-vs-full AIC/BIC comparisons are computed only for the reference
    C value (REFERENCE_C, unpenalized by default). Other C values fit the
    full model only, as a ridge sensitivity check.

    NOTE ON AIC/BIC UNDER RIDGE:
        The AIC/BIC values reported here use k = raw parameter count, which is
        only exact for unpenalized MLE. Under ridge the effective degrees of
        freedom is smaller (trace of the hat matrix), so ridge AIC/BIC are
        approximate. They are stored with the suffix _approx and should be
        treated as secondary/descriptive. The unpenalized reference-C values
        are exact.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple
from functions_Ale import filter_training_trials
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
import sklearn
print(sklearn.__version__)

# ============================================================
# CONFIG
# ============================================================

MODE = "pooled"           # "pooled" or "per_modality" do not consider the per modality one, it was an older version of the analysis, we only consider the pooled one

# List of ridge C values to fit. None means unpenalized MLE.
# The reference C (below) is used for reduced-model comparisons.
# Other C values fit only the full model (ridge sensitivity check).
C_VALUES: List[Optional[float]] = [None, 0.1, 1.0, 10.0]
REFERENCE_C: Optional[float] = None      # reduced-model comparisons at this C

N_BOOT = 5000
CI_LEVEL = 95
RANDOM_STATE = 42
N_JOBS = 4
MIN_TRIALS = 200

# Physical stimulus range for the PSE-cell gate.
STIMULUS_ANGLE_MIN = 0.0
STIMULUS_ANGLE_MAX = 90.0
ANGLE_CENTER_DEG = 45.0

# Numerical guard for the degree reconstruction.
MIN_ABS_ANGLE_BETA = 1e-6

# Output directory.
OUT_DIR = "./glm_results_thesis_filtered_per_modality"

# Features for each model.
FEATURES_FULL = [
    "angle",
    "angle_n-1",
    "reward_n-1",
    "success_n-1",
    "failure_n-1",
]
FEATURES_ADDITIVE_CHOICE = [
    "angle",
    "angle_n-1",
    "reward_n-1",
    "action_n-1",
]
FEATURES_NO_REWARD = [
    "angle",
    "angle_n-1",
    "success_n-1",
    "failure_n-1",
]
FEATURES_NO_ANGLE_N1 = [
    "angle",
    "reward_n-1",
    "success_n-1",
    "failure_n-1",
]

MODALITY_CODES = {"tactile": 1, "visual": 2, "visuotactile": 3}
CONTINUOUS_REGRESSORS = {"angle", "angle_n-1"}
FILTER_KWARGS = dict(method='criterion', criterion=0.80, min_consec=2)

# ============================================================
# PREPROCESSING
# ============================================================

def prepare_history_predictors(df: pd.DataFrame) -> pd.DataFrame:
    """Create the agreed history coding without modifying the input dataframe."""
    out = df.copy()

    vals = set(pd.Series(out["action_n-1"].dropna().unique()).astype(float))
    if vals.issubset({0.0, 1.0}):
        out["action_n-1"] = out["action_n-1"] * 2.0 - 1.0
    elif not vals.issubset({-1.0, 1.0}):
        raise ValueError(
            "action_n-1 must be coded either 0/1 or -1/+1 before preparation."
        )

    hit_vals = set(pd.Series(out["hit_n-1"].dropna().unique()).astype(float))
    if not hit_vals.issubset({0.0, 1.0}):
        raise ValueError("hit_n-1 must be coded 0/1.")

    hit01 = out["hit_n-1"].astype(float)
    out["success_n-1"] = hit01 * out["action_n-1"]
    out["failure_n-1"] = (1.0 - hit01) * out["action_n-1"]
    out["reward_n-1"] = hit01 - 0.5

    out["angle"] = out["angle"].astype(float) - ANGLE_CENTER_DEG
    out["angle_n-1"] = out["angle_n-1"].astype(float) - ANGLE_CENTER_DEG

    return out


# ============================================================
# DESIGN MATRIX / FITTING
# ============================================================

def build_design_matrix(
    rat_data: pd.DataFrame,
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler], List[str], Dict[str, float]]:
    """Build X and y. Continuous regressors are scaled to unit SD (no centering).

    Categoricals are left on their native scale so their coefficients are
    directly interpretable in log-odds units.
    """
    cont_cols = [f for f in feature_names if f in CONTINUOUS_REGRESSORS]
    other_cols = [f for f in feature_names if f not in CONTINUOUS_REGRESSORS]

    X_parts = []
    ordered_names: List[str] = []
    scale_by_feature: Dict[str, float] = {}
    scaler = None

    if cont_cols:
        X_cont = rat_data[cont_cols].to_numpy(dtype=float)
        scaler = StandardScaler(with_mean=False)
        X_cont = scaler.fit_transform(X_cont)
        X_parts.append(X_cont)
        ordered_names.extend(cont_cols)
        scale_by_feature.update(
            {name: float(scale) for name, scale in zip(cont_cols, scaler.scale_)}
        )

    if other_cols:
        X_other = rat_data[other_cols].to_numpy(dtype=float)
        X_parts.append(X_other)
        ordered_names.extend(other_cols)
        # Categoricals are on native scale.
        for name in other_cols:
            scale_by_feature[name] = 1.0

    X = np.column_stack(X_parts)
    y = rat_data["action"].to_numpy(dtype=int)
    return X, y, scaler, ordered_names, scale_by_feature
"""

def fit_logistic(X: np.ndarray, y: np.ndarray, C: Optional[float]) -> LogisticRegression:
  
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=ConvergenceWarning)
        try:
            if C is None:
                model = LogisticRegression(
                    penalty=None,
                    max_iter=2000,
                    solver="lbfgs",
                )
            else:
                model = LogisticRegression(
                    penalty="l2",
                    C=C,
                    max_iter=2000,
                    solver="lbfgs",
                )
            model.fit(X, y)
            return model
        except (ConvergenceWarning, Exception):
            # Fallback: ridge C=1.0
            
            model = LogisticRegression(
                penalty="l2",
                C=1.0,
                max_iter=5000,
                solver="lbfgs",
            )
            model.fit(X, y)
            return model
"""
            

def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    C: Optional[float],
) -> LogisticRegression:
    """
    Fit the requested logistic model.

    C=None requests unpenalised maximum likelihood.
    A convergence failure may trigger a retry with more iterations,
    but never a change in penalty or C.
    """

    if C is None:
        model_kwargs = {
            "penalty": None,
            "solver": "lbfgs",
        }
        fit_label = "unpenalised"
    else:
        model_kwargs = {
            "penalty": "l2",
            "C": C,
            "solver": "lbfgs",
        }
        fit_label = f"ridge C={C}"

    last_error = None

    # Retrying with more iterations does not change the requested estimator.
    for max_iter in (2000, 10000):
        model = LogisticRegression(
            **model_kwargs,
            max_iter=max_iter,
            tol=1e-8,
        )

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(X, y)

            if not np.all(np.isfinite(model.coef_)):
                raise FloatingPointError("Non-finite coefficient estimate.")

            if not np.all(np.isfinite(model.intercept_)):
                raise FloatingPointError("Non-finite intercept estimate.")

            return model

        except ConvergenceWarning as exc:
            last_error = exc
        except (ValueError, FloatingPointError) as exc:
            last_error = exc
            break

    raise RuntimeError(
        f"{fit_label} logistic regression failed after retrying "
        f"with up to 10,000 iterations."
    ) from last_error


def compute_aic_bic(model, X: np.ndarray, y: np.ndarray, C: Optional[float]) -> Tuple[float, float, float]:
    """AIC/BIC using k = raw parameter count.

    Exact for unpenalized MLE (C=None). Approximate under ridge because the
    effective degrees of freedom is smaller than the parameter count.
    Reported values carry the suffix _approx everywhere they are stored.
    """
    n = len(y)
    proba = model.predict_proba(X)
    ll = -log_loss(y, proba, normalize=False)
    k = X.shape[1] + 1
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll
    return float(aic), float(bic), float(ll)


# ============================================================
# BOOTSTRAP
# ============================================================

#def _one_bootstrap_fit(X, y, row_indices, C):
#    """Return [intercept, coefficients...] for one bootstrap sample."""
#    model = fit_logistic(X[row_indices], y[row_indices], C=C)
#    return np.concatenate(([float(model.intercept_[0])], model.coef_[0]))

def _one_bootstrap_fit(X, y, row_indices, C):
    try:
        model = fit_logistic(X[row_indices], y[row_indices], C=C)
        params = np.concatenate(
            ([float(model.intercept_[0])], model.coef_[0])
        )

        if not np.all(np.isfinite(params)):
            return None

        return params

    except RuntimeError:
        return None


def bootstrap_parameters(
    rat_data: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    C: Optional[float],
    n_boot: int,
    random_state: int,
    n_jobs: int,
) -> np.ndarray:
    """Session-level bootstrap. Returns array shape (n_boot, 1 + n_features)."""
    rng = np.random.default_rng(random_state)
    sessions = rat_data["date"].unique()
    n_sessions = len(sessions)
    if n_sessions == 0:
        raise ValueError("No sessions available for bootstrap.")

    date_values = rat_data["date"].to_numpy()
    session_to_rows = {s: np.where(date_values == s)[0] for s in sessions}

    boot_session_idx = rng.integers(0, n_sessions, size=(n_boot, n_sessions))

    def build_rows(session_choice):
        return np.concatenate(
            [session_to_rows[sessions[idx]] for idx in session_choice]
        )

        draws = Parallel(n_jobs=n_jobs)(
        delayed(_one_bootstrap_fit)(
            X,
            y,
            build_rows(boot_session_idx[b]),
            C,
        )
        for b in range(n_boot)
    )

    valid_draws = [draw for draw in draws if draw is not None]
    n_valid = len(valid_draws)
    valid_fraction = n_valid / n_boot

    minimum_valid = max(
        100,
        int(np.ceil(0.80 * n_boot)),
    )

    if n_valid < minimum_valid:
        raise RuntimeError(
            f"Too few valid bootstrap fits: {n_valid}/{n_boot} "
            f"({100 * valid_fraction:.1f}%). "
            f"Required at least {minimum_valid}."
        )

    if n_valid < n_boot:
        print(
            f"Warning: retained {n_valid}/{n_boot} valid bootstrap fits "
            f"({100 * valid_fraction:.1f}%)."
        )

    return np.asarray(valid_draws, dtype=float)

    #draws = Parallel(n_jobs=n_jobs)(
    #    delayed(_one_bootstrap_fit)(X, y, build_rows(boot_session_idx[b]), C)
    #    for b in range(n_boot)
    #)
    #return np.asarray(draws, dtype=float)


# ============================================================
# QUANTITY RECONSTRUCTION
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


def _extract_named_params(intercept, coefficients, feature_names):
    params = {name: float(value) for name, value in zip(feature_names, coefficients)}
    params["intercept"] = float(intercept)
    return params


def compute_all_quantities(
    intercept: float,
    coefficients: np.ndarray,
    feature_names: List[str],
    angle_scale_deg: float,
) -> Dict[str, float]:
    """Compute log-odds, probability-scale, and degree-scale quantities for one param vector."""
    p = _extract_named_params(intercept, coefficients, feature_names)
    out: Dict[str, float] = {}

    # Log-odds coefficients (always present for features in the model).
    out["intercept"] = p["intercept"]
    for name in ["angle", "angle_n-1", "reward_n-1", "success_n-1",
                 "failure_n-1", "action_n-1"]:
        if name in p:
            out[f"beta_{name}"] = p[name]

    # Probability-scale summaries at the boundary (logit = 0, dP/dlogit = 0.25).
    if "success_n-1" in p:
        out["dp_success"] = 0.5 * p["success_n-1"]
    if "failure_n-1" in p:
        out["dp_failure"] = 0.5 * p["failure_n-1"]
    if "reward_n-1" in p:
        out["dp_reward"] = 0.25 * p["reward_n-1"]

    # Degree-scale reconstruction requires the full choice-by-outcome model.
    required = {"angle", "reward_n-1", "success_n-1", "failure_n-1"}
    if not required.issubset(p):
        return out

    b_angle = p["angle"]
    if abs(b_angle) < MIN_ABS_ANGLE_BETA:
        for key in ["delta_mu_rewarded", "delta_mu_unrewarded",
                    "S_glm", "D_glm", "P_glm",
                    "pse_V_rewarded", "pse_H_rewarded",
                    "pse_V_unrewarded", "pse_H_unrewarded"]:
            out[key] = np.nan
        return out

    b0 = p["intercept"]
    b_reward = p["reward_n-1"]
    b_success = p["success_n-1"]
    b_failure = p["failure_n-1"]

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


def compute_bootstrap_quantities(
    boot_params: np.ndarray,
    feature_names: List[str],
    angle_scale_deg: float,
) -> Dict[str, np.ndarray]:
    """Compute all quantities for every bootstrap draw."""
    # Determine which quantities exist by computing on the first row.
    first = compute_all_quantities(
        boot_params[0, 0], boot_params[0, 1:], feature_names, angle_scale_deg
    )
    keys = list(first.keys())
    out = {key: [] for key in keys}

    for row in boot_params:
        vals = compute_all_quantities(row[0], row[1:], feature_names, angle_scale_deg)
        for key in keys:
            out[key].append(vals.get(key, np.nan))

    return {key: np.asarray(vals, dtype=float) for key, vals in out.items()}


def check_pse_on_scale(quantities: Dict[str, float]) -> bool:
    """Return True if all four PSE cells (point estimates) fall within [0, 90] deg."""
    pse_keys = ["pse_V_rewarded", "pse_H_rewarded",
                "pse_V_unrewarded", "pse_H_unrewarded"]
    if not all(k in quantities for k in pse_keys):
        return False
    return all(
        np.isfinite(quantities[k]) and
        STIMULUS_ANGLE_MIN <= quantities[k] <= STIMULUS_ANGLE_MAX
        for k in pse_keys
    )


def profile_from_boot_quantities(
    boot_quantities: Dict[str, np.ndarray],
) -> str:
    """Statistical profile based on bootstrap CIs of S, D, P."""
    labels = []
    if "S_glm" in boot_quantities and _ci_excludes_zero(_ci(boot_quantities["S_glm"])):
        labels.append("choice")
    if "D_glm" in boot_quantities and _ci_excludes_zero(_ci(boot_quantities["D_glm"])):
        labels.append("interaction")
    if "P_glm" in boot_quantities and _ci_excludes_zero(_ci(boot_quantities["P_glm"])):
        labels.append("reward")
    return "+".join(labels) if labels else "null"


# ============================================================
# ONE FIT
# ============================================================

def analyze_single_fit(
    rat_data: pd.DataFrame,
    feature_names: List[str],
    C: Optional[float],
    compute_reduced_stats: bool,
    n_boot: int,
    random_state: int,
    n_jobs: int,
) -> Dict:
    """Fit one logistic model and session-bootstrap. Returns rich dict."""
    X, y, scaler, ordered_names, scale_by_feature = build_design_matrix(
        rat_data, feature_names
    )
    model = fit_logistic(X, y, C=C)
    aic, bic, ll = compute_aic_bic(model, X, y, C=C)

    boot_params = bootstrap_parameters(
        rat_data, X, y, C=C,
        n_boot=n_boot, random_state=random_state, n_jobs=n_jobs,
    )

    angle_scale = scale_by_feature.get("angle", 1.0)

    # Point-estimate quantities.
    point_quantities = compute_all_quantities(
        model.intercept_[0], model.coef_[0], ordered_names, angle_scale
    )
    # Bootstrap quantities.
    boot_quantities = compute_bootstrap_quantities(
        boot_params, ordered_names, angle_scale
    )
    # CIs from bootstrap.
    ci_by_quantity = {key: _ci(vals) for key, vals in boot_quantities.items()}

    # Diagnostics.
    i_angle = ordered_names.index("angle") if "angle" in ordered_names else None
    beta_angle_boot = boot_params[:, 1 + i_angle] if i_angle is not None else np.array([])
    beta_angle_boot_min = float(np.min(beta_angle_boot)) if len(beta_angle_boot) else np.nan
    beta_angle_boot_frac_pos = (
        float(np.mean(beta_angle_boot > 0)) if len(beta_angle_boot) else np.nan
    )

    pse_on_scale = check_pse_on_scale(point_quantities)
    profile = profile_from_boot_quantities(boot_quantities)

    result = {
        "feature_names": ordered_names,
        "coefficients": model.coef_[0].copy(),
        "intercept": float(model.intercept_[0]),
        "angle_scale_deg": angle_scale,
        "point_quantities": point_quantities,
        "boot_quantities": boot_quantities,
        "ci_by_quantity": ci_by_quantity,
        "aic_approx": aic,
        "bic_approx": bic,
        "log_lik": ll,
        "n_trials": len(y),
        "n_sessions": int(rat_data["date"].nunique()),
        "beta_angle_boot_min": beta_angle_boot_min,
        "beta_angle_boot_frac_positive": beta_angle_boot_frac_pos,
        "pse_on_scale": pse_on_scale,
        "profile": profile,
    }
    return result


# ============================================================
# TOP-LEVEL ANALYSIS
# ============================================================

def _c_label(C: Optional[float]) -> str:
    return "unpenalized" if C is None else f"C={C}"


def run_analysis(
    df: pd.DataFrame,
    mode: str,
    c_values: List[Optional[float]],
    reference_c: Optional[float],
    n_boot: int,
    random_state: int,
    n_jobs: int,
    min_trials: int,
) -> Dict:
    """Run all fits and return a nested dict keyed by (rat, subset, C)."""
    if mode not in ("pooled", "per_modality"):
        raise ValueError("mode must be 'pooled' or 'per_modality'")

    required_for_full = [
        "rat", "date", "action", "mod",
        *FEATURES_FULL,
        "action_n-1",
    ]
    missing = [col for col in required_for_full if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rats = sorted(df["rat"].dropna().unique())
    all_results = {}
    subset_counter = 0
    t0 = time.time()

    for rat_id in rats:
        rat_df = df[df["rat"] == rat_id].copy()
        all_results[rat_id] = {}

        if mode == "pooled":
            subsets = {"pooled": rat_df}
        else:
            subsets = {
                name: rat_df[rat_df["mod"] == code].copy()
                for name, code in MODALITY_CODES.items()
            }

        for subset_name, subset_df in subsets.items():
            subset_df = subset_df.dropna(
                subset=["action", "date", *FEATURES_FULL, "action_n-1"]
            ).copy()

            if len(subset_df) < min_trials:
                print(f"  Rat {rat_id} [{subset_name}]: skip ({len(subset_df)} trials)")
                continue

            elapsed = time.time() - t0
            print(
                f"  Rat {rat_id} [{subset_name}]: "
                f"{len(subset_df)} trials, {subset_df['date'].nunique()} sessions"
                f"  (elapsed {elapsed/60:.1f} min)"
            )

            fit_seed = random_state + subset_counter
            subset_counter += 1

            all_results[rat_id][subset_name] = {}

            for C in c_values:
                is_reference = (C == reference_c)
                fits = {}
                fits["full"] = analyze_single_fit(
                    subset_df, FEATURES_FULL, C=C,
                    compute_reduced_stats=is_reference,
                    n_boot=n_boot, random_state=fit_seed, n_jobs=n_jobs,
                )

                # Reduced-model comparisons only at reference C.
                if is_reference:
                    fits["additive_choice"] = analyze_single_fit(
                        subset_df, FEATURES_ADDITIVE_CHOICE, C=C,
                        compute_reduced_stats=False,
                        n_boot=n_boot, random_state=fit_seed, n_jobs=n_jobs,
                    )
                    fits["no_reward"] = analyze_single_fit(
                        subset_df, FEATURES_NO_REWARD, C=C,
                        compute_reduced_stats=False,
                        n_boot=n_boot, random_state=fit_seed, n_jobs=n_jobs,
                    )
                    fits["no_angle_n1"] = analyze_single_fit(
                        subset_df, FEATURES_NO_ANGLE_N1, C=C,
                        compute_reduced_stats=False,
                        n_boot=n_boot, random_state=fit_seed, n_jobs=n_jobs,
                    )

                    full = fits["full"]
                    for restricted_name in ["additive_choice", "no_reward", "no_angle_n1"]:
                        restricted = fits[restricted_name]
                        fits[f"delta_aic_approx_{restricted_name}_minus_full"] = (
                            restricted["aic_approx"] - full["aic_approx"]
                        )
                        fits[f"delta_bic_approx_{restricted_name}_minus_full"] = (
                            restricted["bic_approx"] - full["bic_approx"]
                        )

                all_results[rat_id][subset_name][_c_label(C)] = fits

    return all_results


# ============================================================
# TABLE BUILDERS
# ============================================================

def build_long_summary(all_results: Dict) -> pd.DataFrame:
    """One row per (rat, subset, C, quantity)."""
    rows = []
    for rat_id, per_subset in all_results.items():
        for subset_name, per_C in per_subset.items():
            for c_label, fits in per_C.items():
                full = fits["full"]
                n_trials = full["n_trials"]
                n_sessions = full["n_sessions"]
                pse_on_scale = full["pse_on_scale"]
                profile = full["profile"]

                for quantity, value in full["point_quantities"].items():
                    ci_lo, ci_hi = full["ci_by_quantity"].get(quantity, (np.nan, np.nan))
                    scale = _quantity_scale(quantity)
                    rows.append({
                        "rat_id": rat_id,
                        "subset": subset_name,
                        "C": c_label,
                        "n_trials": n_trials,
                        "n_sessions": n_sessions,
                        "quantity": quantity,
                        "scale": scale,
                        "value": value,
                        "ci_low": ci_lo,
                        "ci_high": ci_hi,
                        "pse_on_scale": pse_on_scale,
                        "profile": profile,
                    })
    return pd.DataFrame(rows)


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


def build_bootstrap_long(all_results: Dict) -> pd.DataFrame:
    """One row per (rat, subset, C, quantity, boot_idx). Parquet-only."""
    frames = []
    for rat_id, per_subset in all_results.items():
        for subset_name, per_C in per_subset.items():
            for c_label, fits in per_C.items():
                full = fits["full"]
                boot = full["boot_quantities"]
                if not boot:
                    continue
                first_key = next(iter(boot.keys()))
                n_boot = len(boot[first_key])
                # Build a dataframe per fit, then concatenate.
                for quantity, values in boot.items():
                    frames.append(pd.DataFrame({
                        "rat_id": rat_id,
                        "subset": subset_name,
                        "C": c_label,
                        "quantity": quantity,
                        "boot_idx": np.arange(n_boot, dtype=np.int32),
                        "value": values.astype(np.float32),
                    }))
    if not frames:
        return pd.DataFrame(columns=["rat_id","subset","C","quantity","boot_idx","value"])
    return pd.concat(frames, ignore_index=True)


def build_metadata(all_results: Dict, config: Dict) -> Dict:
    """Per-fit diagnostics and run config, saved as JSON."""
    per_fit = []
    for rat_id, per_subset in all_results.items():
        for subset_name, per_C in per_subset.items():
            for c_label, fits in per_C.items():
                full = fits["full"]
                entry = {
                    "rat_id": int(rat_id),
                    "subset": subset_name,
                    "C": c_label,
                    "n_trials": full["n_trials"],
                    "n_sessions": full["n_sessions"],
                    "angle_scale_deg": full["angle_scale_deg"],
                    "aic_approx": full["aic_approx"],
                    "bic_approx": full["bic_approx"],
                    "log_lik": full["log_lik"],
                    "beta_angle_boot_min": full["beta_angle_boot_min"],
                    "beta_angle_boot_frac_positive": full["beta_angle_boot_frac_positive"],
                    "pse_on_scale": bool(full["pse_on_scale"]),
                    "profile": full["profile"],
                }
                # Add reduced-model comparisons if present.
                for k, v in fits.items():
                    if k.startswith("delta_aic_approx_") or k.startswith("delta_bic_approx_"):
                        entry[k] = float(v)
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

    print(f"Running GLM analysis")
    print(f"  Mode:        {MODE}")
    print(f"  C values:    {C_VALUES}")
    print(f"  Reference C: {REFERENCE_C}")
    print(f"  N bootstrap: {N_BOOT}")
    print(f"  Output dir:  {OUT_DIR}")
    print()

    t_start = time.time()
    results = run_analysis(
        df,
        mode=MODE,
        c_values=C_VALUES,
        reference_c=REFERENCE_C,
        n_boot=N_BOOT,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        min_trials=MIN_TRIALS,
    )
    t_fit = time.time() - t_start
    print(f"\nFitting done in {t_fit/60:.1f} minutes")

    # Build outputs.
    print("Building output tables...")
    summary_long = build_long_summary(results)
    boot_long = build_bootstrap_long(results)

    config = {
        "mode": MODE,
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

    # Save.
    summary_path = os.path.join(OUT_DIR, "glm_summary_long.csv")
    boot_path = os.path.join(OUT_DIR, "glm_bootstrap_draws.parquet")
    meta_path = os.path.join(OUT_DIR, "glm_metadata.json")

    summary_long.to_csv(summary_path, index=False)
    print(f"  Saved {summary_path} ({len(summary_long)} rows)")

    try:
        boot_long.to_parquet(boot_path, index=False, compression="snappy")
        print(f"  Saved {boot_path} ({len(boot_long):,} rows)")
    except Exception as e:
        print(f"  Parquet save failed ({e}); falling back to CSV")
        boot_path_csv = os.path.join(OUT_DIR, "glm_bootstrap_draws.csv.gz")
        boot_long.to_csv(boot_path_csv, index=False, compression="gzip")
        print(f"  Saved {boot_path_csv}")

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  Saved {meta_path}")

    print(f"\nTotal time: {(time.time() - t_start)/60:.1f} minutes")