"""
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
Thus 45 degrees remains exactly 0 in the model, while ridge regularization acts
on reasonably comparable predictor scales.

Primary outputs per rat / subset:
    1) Native ridge-logistic coefficients in log-odds with session-bootstrap CIs.
    2) Reconstructed GLM-implied PSE effects in degrees, evaluated at
       angle_{n-1} = 45 degrees:

         delta_mu_rewarded = -2 * sigma_angle * beta_success / beta_angle
         delta_mu_unrewarded = -2 * sigma_angle * beta_failure / beta_angle
         S_glm = delta_mu_rewarded + delta_mu_unrewarded
         D_glm = delta_mu_rewarded - delta_mu_unrewarded
         P_glm = sigma_angle * beta_reward / beta_angle

       Because action=1 means vertical and beta_angle should be positive,
       a positive beta_success/beta_failure corresponds to a negative PSE shift
       (less vertical evidence needed): the classical attractive choice bias.

    3) Four GLM-predicted PSE cells in degrees:
         mu(V, rewarded), mu(H, rewarded),
         mu(V, unrewarded), mu(H, unrewarded)
       with angle_{n-1}=45 degrees.

    4) Section-3-style statistical profile based on bootstrap CIs of S, D, P:
         choice / interaction / reward / combinations / null
       The profile is a statistical decomposition, not a behavioral strategy label.

Model comparisons (secondary/descriptive because fits are ridge-regularized):
    full vs additive_choice : tests whether success/failure-specific choice slopes
                              improve over one common previous-choice slope
    full vs no_reward       : tests independent reward term
    full vs no_angle_n1     : tests previous sensory-angle term

Bootstrap:
    Sessions are resampled with replacement. The original fit's scaling is held
    fixed across bootstrap iterations. Each bootstrap draw is converted to degrees
    before percentile CIs are calculated, so uncertainty in beta_angle is propagated.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss


# ============================================================
# CONFIG
# ============================================================

MODE = "pooled"           # "pooled" or "per_modality"
N_BOOT = 5000                   # use 5000 for final publication analysis
CI_LEVEL = 95
RANDOM_STATE = 42
N_JOBS = 4
RIDGE_C = 1.0
MIN_TRIALS = 200

# The theoretical neutral orientation used throughout Sections 2-4.
ANGLE_CENTER_DEG = 45.0

# Protects degree reconstruction against a numerically near-zero psychometric slope.
MIN_ABS_ANGLE_BETA = 1e-6

# Main model agreed for Section 4.
FEATURES_FULL = [
    "angle",
    "angle_n-1",
    "reward_n-1",
    "success_n-1",
    "failure_n-1",
]

# Restricted model: same previous-choice coefficient after success and failure.
FEATURES_ADDITIVE_CHOICE = [
    "angle",
    "angle_n-1",
    "reward_n-1",
    "action_n-1",
]

# Full outcome-specific choice effects, but no pure reward translation.
FEATURES_NO_REWARD = [
    "angle",
    "angle_n-1",
    "success_n-1",
    "failure_n-1",
]

# Full history-choice/reward structure, but no previous sensory angle.
FEATURES_NO_ANGLE_N1 = [
    "angle",
    "reward_n-1",
    "success_n-1",
    "failure_n-1",
]

MODALITY_CODES = {"tactile": 1, "visual": 2, "visuotactile": 3}
CONTINUOUS_REGRESSORS = {"angle", "angle_n-1"}


# ============================================================
# PREPROCESSING HELPERS
# ============================================================

def prepare_history_predictors(df: pd.DataFrame) -> pd.DataFrame:
    """Create the agreed history coding without modifying the input dataframe.

    Assumptions:
        action_n-1 entering this function is coded 0/1 (H/V), as produced by
        create_lagged_features(). If it is already -1/+1, it is left unchanged.
        hit_n-1 is coded 0/1.
        angle and angle_n-1 are in physical degrees before centering.
    """
    out = df.copy()

    # Previous choice: -1 = horizontal, +1 = vertical.
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

    # Outcome-specific choice branches.
    out["success_n-1"] = hit01 * out["action_n-1"]
    out["failure_n-1"] = (1.0 - hit01) * out["action_n-1"]

    # Pure reward predictor: symmetric coding, but success-failure distance = 1.
    # Therefore beta_reward is the full success-vs-failure log-odds shift.
    out["reward_n-1"] = hit01 - 0.5

    # Preserve the theoretically meaningful 45-degree origin.
    out["angle"] = out["angle"].astype(float) - ANGLE_CENTER_DEG
    out["angle_n-1"] = out["angle_n-1"].astype(float) - ANGLE_CENTER_DEG

    return out


# ============================================================
# DESIGN MATRIX / FITTING
# ============================================================

def build_design_matrix(
    rat_data: pd.DataFrame,
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, StandardScaler | None, List[str], Dict[str, float]]:
    """Build X and y.

    Continuous regressors are divided by their SD with
    StandardScaler(with_mean=False), so their already meaningful zero is retained.

    Returns
    -------
    X, y, scaler, ordered_feature_names, scale_by_feature
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

    X = np.column_stack(X_parts)
    y = rat_data["action"].to_numpy(dtype=int)
    return X, y, scaler, ordered_names, scale_by_feature


def fit_ridge_logistic(X: np.ndarray, y: np.ndarray, C: float = RIDGE_C):
    model = LogisticRegression(
        penalty="l2",
        C=C,
        max_iter=1000,
        solver="lbfgs",
    )
    model.fit(X, y)
    return model


def compute_aic_bic(model, X: np.ndarray, y: np.ndarray):
    """Conventional AIC/BIC approximation for a ridge-logistic fit.

    IMPORTANT: because ridge uses penalized rather than ordinary maximum
    likelihood, k = raw parameter count is not the exact effective degrees of
    freedom. Treat these values as secondary/descriptive model-comparison aids,
    not as the primary inferential result.
    """
    n = len(y)
    proba = model.predict_proba(X)
    ll = -log_loss(y, proba, normalize=False)
    k = X.shape[1] + 1  # coefficients + intercept, approximate for ridge
    aic = 2 * k - 2 * ll
    bic = k * np.log(n) - 2 * ll
    return float(aic), float(bic), float(ll)


# ============================================================
# SESSION BOOTSTRAP
# ============================================================

def _one_bootstrap_fit(X, y, row_indices, C):
    """Return [intercept, coefficients...] for one bootstrap sample."""
    model = fit_ridge_logistic(X[row_indices], y[row_indices], C=C)
    return np.concatenate(([float(model.intercept_[0])], model.coef_[0]))


def bootstrap_parameters(
    rat_data: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    n_boot: int = N_BOOT,
    C: float = RIDGE_C,
    random_state: int = RANDOM_STATE,
    n_jobs: int = N_JOBS,
) -> np.ndarray:
    """Session-level bootstrap.

    Returns an array with shape (n_boot, 1 + n_features):
        column 0 = intercept
        columns 1.. = coefficients in the design-matrix order.

    The design matrix (and therefore its scaling) is fixed from the original fit.
    """
    rng = np.random.default_rng(random_state)
    sessions = rat_data["date"].unique()
    n_sessions = len(sessions)
    if n_sessions == 0:
        raise ValueError("No sessions available for bootstrap.")

    date_values = rat_data["date"].to_numpy()
    session_to_rows = {
        s: np.where(date_values == s)[0]
        for s in sessions
    }

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
    return np.asarray(draws, dtype=float)


# ============================================================
# DEGREE RECONSTRUCTION
# ============================================================

def _ci(values: np.ndarray, ci_level: float = CI_LEVEL) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.array([np.nan, np.nan])
    alpha = (100.0 - ci_level) / 2.0
    return np.percentile(values, [alpha, 100.0 - alpha])


def _ci_excludes_zero(ci: Iterable[float]) -> bool:
    lo, hi = ci
    return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))


def _extract_named_params(intercept, coefficients, feature_names):
    params = {name: float(value) for name, value in zip(feature_names, coefficients)}
    params["intercept"] = float(intercept)
    return params


def reconstruct_degree_metrics(
    intercept: float,
    coefficients: np.ndarray,
    feature_names: List[str],
    angle_scale_deg: float,
) -> Dict[str, float]:
    """Convert one fitted parameter vector to Section-3-like quantities in degrees.

    Four-cell PSEs are evaluated at previous angle = 45 degrees, i.e.
    standardized angle_n-1 = 0.
    """
    p = _extract_named_params(intercept, coefficients, feature_names)

    required = {"angle", "reward_n-1", "success_n-1", "failure_n-1"}
    missing = required.difference(p)
    if missing:
        raise ValueError(f"Degree reconstruction requires features: {sorted(missing)}")

    b_angle = p["angle"]
    if abs(b_angle) < MIN_ABS_ANGLE_BETA:
        return {key: np.nan for key in [
            "delta_mu_rewarded", "delta_mu_unrewarded",
            "S_glm", "D_glm", "P_glm",
            "pse_V_rewarded", "pse_H_rewarded",
            "pse_V_unrewarded", "pse_H_unrewarded",
        ]}

    b0 = p["intercept"]
    b_reward = p["reward_n-1"]
    b_success = p["success_n-1"]
    b_failure = p["failure_n-1"]

    scale_over_slope = angle_scale_deg / b_angle

    # Rewarded state: reward=+0.5, success=choice, failure=0.
    # Unrewarded state: reward=-0.5, success=0, failure=choice.
    def pse_from_offset(offset: float) -> float:
        # angle variable was centered as theta - 45 before standardization.
        centered_pse_deg = -scale_over_slope * offset
        return ANGLE_CENTER_DEG + centered_pse_deg

    pse_v_r = pse_from_offset(b0 + 0.5 * b_reward + b_success)
    pse_h_r = pse_from_offset(b0 + 0.5 * b_reward - b_success)
    pse_v_u = pse_from_offset(b0 - 0.5 * b_reward + b_failure)
    pse_h_u = pse_from_offset(b0 - 0.5 * b_reward - b_failure)

    delta_r = pse_v_r - pse_h_r
    delta_u = pse_v_u - pse_h_u
    S_glm = delta_r + delta_u
    D_glm = delta_r - delta_u
    P_glm = 0.5 * (pse_v_u + pse_h_u) - 0.5 * (pse_v_r + pse_h_r)

    return {
        "delta_mu_rewarded": float(delta_r),
        "delta_mu_unrewarded": float(delta_u),
        "S_glm": float(S_glm),
        "D_glm": float(D_glm),
        "P_glm": float(P_glm),
        "pse_V_rewarded": float(pse_v_r),
        "pse_H_rewarded": float(pse_h_r),
        "pse_V_unrewarded": float(pse_v_u),
        "pse_H_unrewarded": float(pse_h_u),
    }


def reconstruct_bootstrap_degree_metrics(
    boot_params: np.ndarray,
    feature_names: List[str],
    angle_scale_deg: float,
) -> Dict[str, np.ndarray]:
    """Convert every bootstrap draw to degrees before calculating CIs."""
    keys = [
        "delta_mu_rewarded", "delta_mu_unrewarded",
        "S_glm", "D_glm", "P_glm",
        "pse_V_rewarded", "pse_H_rewarded",
        "pse_V_unrewarded", "pse_H_unrewarded",
    ]
    out = {key: [] for key in keys}

    for row in boot_params:
        metrics = reconstruct_degree_metrics(
            intercept=row[0],
            coefficients=row[1:],
            feature_names=feature_names,
            angle_scale_deg=angle_scale_deg,
        )
        for key in keys:
            out[key].append(metrics[key])

    return {key: np.asarray(values, dtype=float) for key, values in out.items()}


def profile_from_degree_boot(degree_boot: Dict[str, np.ndarray]) -> str:
    """Section-3-style inferential profile based on S, D, P bootstrap CIs."""
    labels = []
    if _ci_excludes_zero(_ci(degree_boot["S_glm"])):
        labels.append("choice")
    if _ci_excludes_zero(_ci(degree_boot["D_glm"])):
        labels.append("interaction")
    if _ci_excludes_zero(_ci(degree_boot["P_glm"])):
        labels.append("reward")
    return "+".join(labels) if labels else "null"


# ============================================================
# ONE FIT
# ============================================================

def analyze_single_fit(
    rat_data: pd.DataFrame,
    feature_names: List[str],
    n_boot: int = N_BOOT,
    C: float = RIDGE_C,
    random_state: int = RANDOM_STATE,
    n_jobs: int = N_JOBS,
    reconstruct_degrees: bool = False,
):
    """Fit one ridge logistic model and session-bootstrap its parameters."""
    X, y, scaler, ordered_names, scale_by_feature = build_design_matrix(
        rat_data, feature_names
    )
    model = fit_ridge_logistic(X, y, C=C)
    aic, bic, ll = compute_aic_bic(model, X, y)

    boot_params = bootstrap_parameters(
        rat_data,
        X,
        y,
        n_boot=n_boot,
        C=C,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    boot_intercepts = boot_params[:, 0]
    boot_coefs = boot_params[:, 1:]
    alpha = (100.0 - CI_LEVEL) / 2.0
    ci_low, ci_high = np.percentile(
        boot_coefs, [alpha, 100.0 - alpha], axis=0
    )
    intercept_ci = np.percentile(
        boot_intercepts, [alpha, 100.0 - alpha]
    )

    result = {
        "feature_names": ordered_names,
        "coefficients": model.coef_[0].copy(),
        "intercept": float(model.intercept_[0]),
        "boot_params": boot_params,
        "boot_draws": boot_coefs,
        "boot_intercepts": boot_intercepts,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "intercept_ci": intercept_ci,
        "scale_by_feature": scale_by_feature,
        "aic": aic,
        "bic": bic,
        "log_lik": ll,
        "n_trials": len(y),
        "n_sessions": int(rat_data["date"].nunique()),
    }

    if "angle" in ordered_names:
        i_angle = ordered_names.index("angle")
        result["p_angle_positive"] = float(np.mean(boot_coefs[:, i_angle] > 0))

    if reconstruct_degrees:
        angle_scale = scale_by_feature["angle"]
        degree_point = reconstruct_degree_metrics(
            model.intercept_[0],
            model.coef_[0],
            ordered_names,
            angle_scale,
        )
        degree_boot = reconstruct_bootstrap_degree_metrics(
            boot_params,
            ordered_names,
            angle_scale,
        )
        degree_ci = {key: _ci(values) for key, values in degree_boot.items()}
        valid_fraction = float(
            np.mean(np.isfinite(degree_boot["delta_mu_rewarded"]))
        )

        result["degree_point"] = degree_point
        result["degree_boot"] = degree_boot
        result["degree_ci"] = degree_ci
        result["degree_valid_fraction"] = valid_fraction
        result["profile"] = profile_from_degree_boot(degree_boot)

    return result


# ============================================================
# TOP-LEVEL ANALYSIS
# ============================================================

def run_analysis(
    df: pd.DataFrame,
    mode: str = MODE,
    n_boot: int = N_BOOT,
    random_state: int = RANDOM_STATE,
    n_jobs: int = N_JOBS,
    min_trials: int = MIN_TRIALS,
):
    """Run all agreed models per rat and per requested subset."""
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
                print(
                    f"  Rat {rat_id} [{subset_name}]: "
                    f"skip ({len(subset_df)} trials)"
                )
                continue

            print(
                f"  Rat {rat_id} [{subset_name}]: "
                f"{len(subset_df)} trials, {subset_df['date'].nunique()} sessions"
            )

            # Different deterministic bootstrap stream across rat/subset, while
            # using the same stream across nested models within a subset.
            fit_seed = random_state + subset_counter
            subset_counter += 1

            fits = {}
            fits["full"] = analyze_single_fit(
                subset_df,
                FEATURES_FULL,
                n_boot=n_boot,
                random_state=fit_seed,
                n_jobs=n_jobs,
                reconstruct_degrees=True,
            )
            fits["additive_choice"] = analyze_single_fit(
                subset_df,
                FEATURES_ADDITIVE_CHOICE,
                n_boot=n_boot,
                random_state=fit_seed,
                n_jobs=n_jobs,
            )
            fits["no_reward"] = analyze_single_fit(
                subset_df,
                FEATURES_NO_REWARD,
                n_boot=n_boot,
                random_state=fit_seed,
                n_jobs=n_jobs,
            )
            fits["no_angle_n1"] = analyze_single_fit(
                subset_df,
                FEATURES_NO_ANGLE_N1,
                n_boot=n_boot,
                random_state=fit_seed,
                n_jobs=n_jobs,
            )

            full = fits["full"]
            for restricted_name in ["additive_choice", "no_reward", "no_angle_n1"]:
                restricted = fits[restricted_name]
                fits[f"delta_aic_{restricted_name}_minus_full"] = (
                    restricted["aic"] - full["aic"]
                )
                fits[f"delta_bic_{restricted_name}_minus_full"] = (
                    restricted["bic"] - full["bic"]
                )

            all_results[rat_id][subset_name] = fits

    return all_results


# ============================================================
# SUMMARY TABLE
# ============================================================

def build_summary_table(all_results) -> pd.DataFrame:
    """One row per rat/subset with native and degree-scale results."""
    rows = []

    for rat_id, per_subset in all_results.items():
        for subset_name, fits in per_subset.items():
            full = fits["full"]
            names = full["feature_names"]

            row = {
                "rat_id": rat_id,
                "subset": subset_name,
                "n_trials": full["n_trials"],
                "n_sessions": full["n_sessions"],
                "profile": full["profile"],
                "angle_scale_deg": full["scale_by_feature"]["angle"],
                "p_angle_positive": full.get("p_angle_positive", np.nan),
                "degree_valid_fraction": full["degree_valid_fraction"],
                "intercept": full["intercept"],
                "intercept_ci_low": full["intercept_ci"][0],
                "intercept_ci_high": full["intercept_ci"][1],
            }

            # Native coefficients and CIs.
            for feat in FEATURES_FULL:
                j = names.index(feat)
                row[f"beta_{feat}"] = full["coefficients"][j]
                row[f"ci_low_{feat}"] = full["ci_low"][j]
                row[f"ci_high_{feat}"] = full["ci_high"][j]

            # Reconstructed point estimates + CIs in degrees.
            for metric, value in full["degree_point"].items():
                row[metric] = value
                row[f"ci_low_{metric}"] = full["degree_ci"][metric][0]
                row[f"ci_high_{metric}"] = full["degree_ci"][metric][1]

            # Secondary/descriptive model comparisons.
            for restricted_name in ["additive_choice", "no_reward", "no_angle_n1"]:
                row[f"delta_aic_{restricted_name}_minus_full"] = fits[
                    f"delta_aic_{restricted_name}_minus_full"
                ]
                row[f"delta_bic_{restricted_name}_minus_full"] = fits[
                    f"delta_bic_{restricted_name}_minus_full"
                ]

            rows.append(row)

    return pd.DataFrame(rows)


def profile_counts(summary_df: pd.DataFrame, subset: str | None = None) -> pd.Series:
    """Descriptive counts only; no arbitrary binomial null is imposed."""
    d = summary_df if subset is None else summary_df[summary_df["subset"] == subset]
    return d["profile"].value_counts(dropna=False)


# ============================================================
# PLOTTING: PER FIT
# ============================================================

def plot_coefficient_forest(full_fit, title=""):
    """Native GLM coefficients in log-odds with bootstrap percentile CIs."""
    names = full_fit["feature_names"]
    betas = full_fit["coefficients"]
    lo = full_fit["ci_low"]
    hi = full_fit["ci_high"]

    y = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(7, 0.48 * len(names) + 1.7))

    # Draw CI independently from the original-fit point.
    # This is valid even if the fitted beta lies outside the percentile CI.
    ax.hlines(y, lo, hi, linewidth=1.5)
    ax.scatter(betas, y, zorder=3)

    # CI caps
    ax.plot(lo, y, "|", markersize=8)
    ax.plot(hi, y, "|", markersize=8)

    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Coefficient (log-odds; continuous predictors standardized)")
    ax.set_title(title)
    ax.invert_yaxis()

    fig.tight_layout()
    return fig


def plot_degree_effects(full_fit, title=""):
    """Section-3-like reconstructed history quantities in degrees."""
    metrics = [
        "delta_mu_rewarded",
        "delta_mu_unrewarded",
        "S_glm",
        "D_glm",
        "P_glm",
    ]

    labels = [
        r"$\Delta\mu_R$",
        r"$\Delta\mu_U$",
        r"$S_{GLM}$",
        r"$D_{GLM}$",
        r"$P_{GLM}$",
    ]

    points = np.array([
        full_fit["degree_point"][m]
        for m in metrics
    ])

    cis = np.array([
        full_fit["degree_ci"][m]
        for m in metrics
    ])

    y = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(7, 4.0))

    ax.hlines(y, cis[:, 0], cis[:, 1], linewidth=1.5)
    ax.scatter(points, y, zorder=3)

    ax.plot(cis[:, 0], y, "|", markersize=8)
    ax.plot(cis[:, 1], y, "|", markersize=8)

    ax.axvline(0, linestyle="--", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("GLM-implied PSE shift (degrees)")
    ax.set_title(title)
    ax.invert_yaxis()

    fig.tight_layout()
    return fig

def plot_four_pse_cells(full_fit, title=""):
    """Four GLM-predicted PSE cells at previous angle = 45 degrees."""
    metrics = [
        "pse_V_rewarded",
        "pse_H_rewarded",
        "pse_V_unrewarded",
        "pse_H_unrewarded",
    ]

    labels = [
        "V, rewarded",
        "H, rewarded",
        "V, unrewarded",
        "H, unrewarded",
    ]

    points = np.array([
        full_fit["degree_point"][m]
        for m in metrics
    ])

    cis = np.array([
        full_fit["degree_ci"][m]
        for m in metrics
    ])

    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(7, 4.2))

    # Vertical percentile CI
    ax.vlines(x, cis[:, 0], cis[:, 1], linewidth=1.5)
    ax.scatter(x, points, zorder=3)

    # CI caps
    ax.plot(x, cis[:, 0], "_", markersize=8)
    ax.plot(x, cis[:, 1], "_", markersize=8)

    ax.axhline(ANGLE_CENTER_DEG, linestyle="--", linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("GLM-predicted PSE (degrees)")
    ax.set_title(title + "\n(previous angle fixed at 45°)")

    fig.tight_layout()
    return fig


# ============================================================
# PLOTTING: COHORT
# ============================================================

def plot_choice_interaction_geometry(summary_df: pd.DataFrame, subset: str):
    """Cohort scatter analogous to the Section 3 delta-mu plane."""
    d = summary_df[summary_df["subset"] == subset].copy()
    if d.empty:
        return None

    x = d["delta_mu_rewarded"].to_numpy(dtype=float)
    y = d["delta_mu_unrewarded"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    d = d.loc[finite]
    x = x[finite]
    y = y[finite]

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(x, y)

    for _, row in d.iterrows():
        ax.annotate(
            str(row["rat_id"]),
            (row["delta_mu_rewarded"], row["delta_mu_unrewarded"]),
            xytext=(4, 4),
            textcoords="offset points",
        )

    if len(x):
        lim = max(np.max(np.abs(np.concatenate([x, y]))), 1.0) * 1.15
    else:
        lim = 1.0
    line = np.array([-lim, lim])
    ax.plot(line, line, linestyle="--", linewidth=1, label="stable choice: y=x")
    ax.plot(line, -line, linestyle=":", linewidth=1, label="reversal: y=-x")
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$\Delta\mu_R^{GLM}$ (degrees)")
    ax.set_ylabel(r"$\Delta\mu_U^{GLM}$ (degrees)")
    ax.set_title(f"GLM choice–interaction geometry — {subset}")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_reward_effects(summary_df: pd.DataFrame, subset: str):
    """Per-rat pure reward translation P_GLM with reconstructed bootstrap CIs."""
    d = summary_df[summary_df["subset"] == subset].copy()

    if d.empty:
        return None

    d = d.sort_values("rat_id")

    y = np.arange(len(d))

    p = d["P_glm"].to_numpy(dtype=float)
    lo = d["ci_low_P_glm"].to_numpy(dtype=float)
    hi = d["ci_high_P_glm"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 0.42 * len(d) + 1.8))

    ax.hlines(y, lo, hi, linewidth=1.5)
    ax.scatter(p, y, zorder=3)

    ax.plot(lo, y, "|", markersize=8)
    ax.plot(hi, y, "|", markersize=8)

    ax.axvline(0, linestyle="--", linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels([f"Rat {r}" for r in d["rat_id"]])

    ax.set_xlabel(
        r"$P_{GLM}$: unrewarded − rewarded PSE (degrees)"
    )
    ax.set_title(f"Pure reward translation — {subset}")

    ax.invert_yaxis()

    fig.tight_layout()
    return fig


def plot_all(all_results, summary_df, out_dir="./glm_plots"):
    """Save per-rat and cohort plots."""
    os.makedirs(out_dir, exist_ok=True)

    for rat_id, per_subset in all_results.items():
        for subset_name, fits in per_subset.items():
            full = fits["full"]
            title = f"Rat {rat_id} [{subset_name}] — {full['profile']}"

            figs = {
                "coefficients": plot_coefficient_forest(full, title),
                "degree_effects": plot_degree_effects(full, title),
                "four_pse_cells": plot_four_pse_cells(full, title),
            }
            for stem, fig in figs.items():
                fig.savefig(
                    os.path.join(out_dir, f"{stem}_rat{rat_id}_{subset_name}.svg"),
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)

    for subset in sorted(summary_df["subset"].dropna().unique()):
        fig = plot_choice_interaction_geometry(summary_df, subset)
        if fig is not None:
            fig.savefig(
                os.path.join(out_dir, f"geometry_{subset}.svg"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

        fig = plot_reward_effects(summary_df, subset)
        if fig is not None:
            fig.savefig(
                os.path.join(out_dir, f"reward_P_{subset}.svg"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    import sys

    sys.path.append(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy")
    from serial_dependence_analysis import SerialDependenceAnalyzer

    # from functions_Ale import filter_training_trials

    an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
    an.load_and_preprocess_data()
    df = an.create_lagged_features()

    # Optional training criterion, if/when you want it:
    # df = filter_training_trials(df, method="criterion", criterion=0.80, min_consec=2)

    df = prepare_history_predictors(df)

    results = run_analysis(
        df,
        mode=MODE,
        n_boot=N_BOOT,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        min_trials=MIN_TRIALS,
    )

    summary = build_summary_table(results)
    print(summary.to_string(index=False))

    out_csv = f"history_bias_summary_revised_{MODE}.csv"
    summary.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(summary)} rows)")

    if not summary.empty:
        print("\nProfile counts:")
        for subset in sorted(summary["subset"].unique()):
            print(f"\n{subset}")
            print(profile_counts(summary, subset=subset).to_string())

        plot_all(results, summary, out_dir="./glm_plots_revised")
        print("Saved plots to ./glm_plots_revised")
