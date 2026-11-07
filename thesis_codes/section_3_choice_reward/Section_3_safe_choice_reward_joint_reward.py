"""
Section_3_choice_reward_joint_bootstrap.py
==========================================

Revised Section 3 analysis of previous-choice effects conditional on the
previous outcome.  The analysis is built around the four cells of the
previous-choice x previous-outcome design:

    H,R   V,R   H,U   V,U

where H/V denote the previous choice and R/U denote rewarded/unrewarded.

Why this version differs from the earlier script
------------------------------------------------
1. All four cells are fitted in the SAME bootstrap replicate.  The resulting
   parameter samples are aligned, so covariance between every derived
   contrast is retained.
2. The direct simple effects are formal tests:

       delta_mu_rew   = mu_V,R - mu_H,R
       delta_mu_unrew = mu_V,U - mu_H,U

   Thus a rewarded-only or unrewarded-only effect does not have to be inferred
   indirectly through the rotated S/D coordinates.
3. The additional contrasts are retained:

       S = delta_mu_rew + delta_mu_unrew
       D = delta_mu_rew - delta_mu_unrew
       P = (mu_H,U + mu_V,U)/2 - (mu_H,R + mu_V,R)/2

   D and P are formal planned contrasts by default.  S is descriptive by
   default because it can be small when an effect is restricted to one outcome
   or reverses across outcomes.  This choice is configurable.
4. Two complementary omnibus Wald tests are saved: a two-degree-of-freedom
   choice-plane test of (delta_mu_rew, delta_mu_unrew), and a three-degree-of-
   freedom full-factorial test that additionally includes P.  The script also
   verifies that each statistic is unchanged after the invertible rotation to
   S/D coordinates.  Neither omnibus is used as a gatekeeper for direct tests.
5. Benjamini-Hochberg FDR correction is applied to the configured primary
   family.  By default this is 12 rats x 4 contrasts (rewarded simple effect,
   unrewarded simple effect, D, and P) at the pooled scope.  The 12 pooled
   omnibus p-values form a separate complementary family.
6. Point estimates come from the complete available dataset.  Bootstrap
   samples quantify uncertainty; both the point estimate and bootstrap median
   are exported.
7. The psychometric sigma bound is 1--50 degrees.
8. DATA_FILTER_MODE can be 'filtered', 'unfiltered', or 'both'.  When 'both'
   is selected, the two pipelines run consecutively and save to separate
   directories.
9. The bootstrap unit is configurable.  'session' is the default and resamples
   whole sessions once per joint replicate.  'trial_stratified' independently
   resamples trials within each of the four cells.  The script never silently
   switches between them.
10. USE_CACHED_BOOTSTRAP lets a rerun resume cheaply: if a rat already has a
    rat_<id>_joint_bootstrap.npz on disk whose data_condition, scope,
    bootstrap_unit, n_bootstrap, and derived seed match this run's
    configuration exactly, its saved bootstrap parameter samples are loaded
    instead of being refit.  Every rat present in the data is still processed
    in the same pass, so every downstream table -- composition, empirical
    points, contrasts, omnibus tests, FDR correction, geometry, figures -- is
    recomputed fresh across the FULL set of rats every run, keeping the
    multiplicity corrections correct even though only the newly-added rats
    pay for the expensive resample-and-refit step.  See load_cached_bootstrap.

Saved outputs
-------------
Each data-condition/scope directory contains plot-ready CSV tables, per-rat
compressed NPZ files with all parameter and contrast bootstrap samples, fit
and bound diagnostics, cell composition, empirical points, omnibus results,
geometry summaries, and figures.  Raw p-values are retained so alternative
multiplicity corrections can be applied without refitting psychometric curves.

Important interpretation note
-----------------------------
The fake-45 design uses previous angles flanking 45 degrees.  Because previous
reward, previous choice, and which flank was presented can be structurally
related, D should be described conservatively as an outcome-dependent choice
contrast unless the design identifies a pure causal reward x choice effect.
The exported cell_composition table makes this relation inspectable.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import zipfile
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize
from scipy.stats import chi2, norm


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Change this if the project modules are elsewhere.
PROJECT_ROOT = Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy")

OUTPUT_ROOT = Path(__file__).resolve().parent / "choice_reward_joint_results"

# 'filtered' | 'unfiltered' | 'both'
# "Unfiltered" still receives the loader's preprocessing and the validity
# filters below; it means that the training-performance criterion is not used.
DATA_FILTER_MODE = "unfiltered"

# 'pooled' | 'per_modality' | 'both'
ANALYSIS_SCOPE = "pooled"

# 'session' (preferred cluster bootstrap) | 'trial_stratified'
BOOTSTRAP_UNIT = "session"

N_BOOTSTRAP = 5000 # 5000
MIN_VALID_BOOT_COUNT = 100 # 100
MIN_VALID_BOOT_FRACTION = 0.80
RANDOM_SEED = 20260912

# When True (default), a rerun of this script looks for an existing
# rat_<id>_joint_bootstrap.npz under bootstraps/ for each rat and, if it
# matches the current data_condition/scope/bootstrap_unit/n_bootstrap/seed
# exactly, loads the saved bootstrap parameter samples instead of refitting
# them. This is what lets you resume an interrupted run cheaply: rerun the
# same command, and rats that already have a compatible cached file are
# skipped for the expensive resample-and-refit step while every downstream
# table (contrasts, omnibus tests, FDR correction, geometry, figures) is
# still recomputed fresh for ALL rats found in the data, keeping multiplicity
# corrections correct. Set to False, or pass --recompute-all on the command
# line, to force every rat to be refit from scratch. If you change anything
# about the underlying data (loader, filters, PSE_FLANKS) between runs,
# delete the affected rat(s)' npz file(s) so the stale cache isn't reused --
# the compatibility check below only covers run configuration, not data
# content.
USE_CACHED_BOOTSTRAP = True

HISTORY_DEPTH = 1
FILTER_KWARGS = dict(method="criterion", criterion=0.80, min_consec=2)

MIN_TRIALS_PER_ANGLE = 5
MIN_TRIALS_TOTAL = 30
MIN_UNIQUE_ANGLES = 4
MIN_SESSIONS = 3

MU_BOUNDS = (0.0, 90.0)
SIGMA_BOUNDS = (1.0, 50.0)  # requested bound
GAMMA_BOUNDS = (0.0, 0.30)
LAPSE_BOUNDS = (0.0, 0.30)
BOUND_TOLERANCE = 1e-3

PSE_FLANKS = {
    1: (40, 50), 2: (40, 50), 3: (40, 50), 5: (40, 50),
    6: (40, 50), 7: (40, 50), 9: (40, 50), 10: (40, 50),
    11: (40, 50), 12: (40, 50), 13: (40, 50), 16: (40, 50),
}

MODALITY_CODES = {
    "tactile": 1,
    "visual": 2,
    "visuotactile": 3,
}

# These contrasts form one BH family across all rats in each inferential scope.
PRIMARY_CONTRASTS = ("delta_mu_rew", "delta_mu_unrew", "D", "P")
INFERENTIAL_SCOPES = ("pooled",)
FDR_ALPHA = 0.05

# Alternative q-values are also exported for all five core contrasts and for
# each contrast separately.  They are sensitivity summaries, not the declared
# primary analysis.
CORE_CONTRASTS = ("delta_mu_rew", "delta_mu_unrew", "S", "D", "P")

GEOMETRY_NULL_RADIUS_DEG = 2.0
PLOT_MAX_BOOT_POINTS = 5000
PLOTS_ENABLED = True


# =============================================================================
# COLUMN AND ARRAY DEFINITIONS
# =============================================================================

COL_STIM = "angle"
COL_STIM_NM1 = "angle_n-1"
COL_CHOICE = "action"
COL_CHOICE_NM1 = "action_n-1"
COL_REWARD = "hitmiss_n-1"
COL_RAT = "rat"
COL_SESSION = "date"
COL_MODALITY = "mod"

CELL_ORDER = ("h_rew", "v_rew", "h_unrew", "v_unrew")
CELL_DEFINITION = {
    "h_rew": (0, 1),
    "v_rew": (1, 1),
    "h_unrew": (0, 0),
    "v_unrew": (1, 0),
}
CELL_LABELS = {
    "h_rew": "prev-H, rewarded",
    "v_rew": "prev-V, rewarded",
    "h_unrew": "prev-H, unrewarded",
    "v_unrew": "prev-V, unrewarded",
}

PARAMETER_ORDER = ("mu", "sigma", "gamma", "lapse")
PARAMETER_BOUNDS = {
    "mu": MU_BOUNDS,
    "sigma": SIGMA_BOUNDS,
    "gamma": GAMMA_BOUNDS,
    "lapse": LAPSE_BOUNDS,
}

CONTRAST_ORDER = (
    "delta_mu_rew",
    "delta_mu_unrew",
    "S",
    "D",
    "P",
    "choice_main_half",
    "choice_by_outcome_half",
    "reward_effect_after_h",
    "reward_effect_after_v",
)


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def stable_seed(*parts: object) -> int:
    """Deterministic 32-bit seed, stable across Python processes."""
    payload = "|".join(map(str, (RANDOM_SEED,) + parts)).encode("utf-8")
    return zlib.crc32(payload) & 0xFFFFFFFF


def json_ready(value):
    """Convert common NumPy/Path objects into JSON-compatible objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)


def save_csv(rows, path: Path) -> pd.DataFrame:
    """Save a row list/DataFrame and return the resulting DataFrame."""
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.8g")
    return frame


def validate_configuration() -> None:
    if DATA_FILTER_MODE not in {"filtered", "unfiltered", "both"}:
        raise ValueError("DATA_FILTER_MODE must be filtered, unfiltered, or both")
    if ANALYSIS_SCOPE not in {"pooled", "per_modality", "both"}:
        raise ValueError("ANALYSIS_SCOPE must be pooled, per_modality, or both")
    if BOOTSTRAP_UNIT not in {"session", "trial_stratified"}:
        raise ValueError("BOOTSTRAP_UNIT must be session or trial_stratified")
    unknown = set(PRIMARY_CONTRASTS) - set(CONTRAST_ORDER)
    if unknown:
        raise ValueError(f"Unknown PRIMARY_CONTRASTS: {sorted(unknown)}")


def config_snapshot() -> dict:
    return {
        "data_filter_mode": DATA_FILTER_MODE,
        "analysis_scope": ANALYSIS_SCOPE,
        "bootstrap_unit": BOOTSTRAP_UNIT,
        "n_bootstrap": N_BOOTSTRAP,
        "min_valid_boot_count": MIN_VALID_BOOT_COUNT,
        "min_valid_boot_fraction": MIN_VALID_BOOT_FRACTION,
        "random_seed": RANDOM_SEED,
        "use_cached_bootstrap": USE_CACHED_BOOTSTRAP,
        "history_depth": HISTORY_DEPTH,
        "filter_kwargs": FILTER_KWARGS,
        "min_trials_per_angle": MIN_TRIALS_PER_ANGLE,
        "min_trials_total": MIN_TRIALS_TOTAL,
        "min_unique_angles": MIN_UNIQUE_ANGLES,
        "min_sessions": MIN_SESSIONS,
        "parameter_bounds": PARAMETER_BOUNDS,
        "pse_flanks": PSE_FLANKS,
        "modality_codes": MODALITY_CODES,
        "primary_contrasts": PRIMARY_CONTRASTS,
        "inferential_scopes": INFERENTIAL_SCOPES,
        "fdr_alpha": FDR_ALPHA,
        "geometry_null_radius_deg": GEOMETRY_NULL_RADIUS_DEG,
        "plots_enabled": PLOTS_ENABLED,
    }


# =============================================================================
# DATA LOADING AND FILTER MODES
# =============================================================================

def load_base_data() -> pd.DataFrame:
    """Load data lazily so this module remains importable for testing."""
    project_text = str(PROJECT_ROOT)
    if project_text not in sys.path:
        sys.path.append(project_text)

    from serial_dependence_analysis import SerialDependenceAnalyzer

    analyzer = SerialDependenceAnalyzer(
        history_depth=HISTORY_DEPTH,
        save_figures=False,
    )
    analyzer.load_and_preprocess_data(fold = False)
    data = analyzer.create_lagged_features()
    data = data[(data['angle'] >= 0) & (data['angle'] <= 90)].copy()
    data = data[(data['angle_n-1'] >= 0) & (data['angle_n-1'] <= 90)].copy()
    required = {
        COL_STIM, COL_STIM_NM1, COL_CHOICE, COL_CHOICE_NM1,
        COL_REWARD, COL_RAT, COL_SESSION, COL_MODALITY,
    }
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    data = data.dropna(subset=list(required)).copy()
    data = data[data[COL_STIM].between(0, 90)].copy()
    data = data[data[COL_STIM_NM1].between(0, 90)].copy()
    data = data[data[COL_CHOICE].isin([0, 1])].copy()
    data = data[data[COL_CHOICE_NM1].isin([0, 1])].copy()
    data = data[data[COL_REWARD].isin([0, 1])].copy()
    return data


def make_filter_datasets(base_data: pd.DataFrame, mode: str) -> dict[str, pd.DataFrame]:
    """Return named data variants; filtered and unfiltered never overwrite."""
    if mode not in {"filtered", "unfiltered", "both"}:
        raise ValueError("mode must be filtered, unfiltered, or both")

    variants = {}
    if mode in {"unfiltered", "both"}:
        variants["unfiltered"] = base_data.copy()
    if mode in {"filtered", "both"}:
        project_text = str(PROJECT_ROOT)
        if project_text not in sys.path:
            sys.path.append(project_text)
        from functions_Ale import filter_training_trials

        variants["filtered"] = filter_training_trials(
            base_data.copy(), **FILTER_KWARGS
        )
    return variants


def make_scopes(data: pd.DataFrame, scope_mode: str) -> dict[str, pd.DataFrame]:
    scopes = {}
    if scope_mode in {"pooled", "both"}:
        scopes["pooled"] = data.copy()
    if scope_mode in {"per_modality", "both"}:
        for label, code in MODALITY_CODES.items():
            scopes[label] = data[data[COL_MODALITY] == code].copy()
    return {name: frame for name, frame in scopes.items() if not frame.empty}


# =============================================================================
# PSYCHOMETRIC FITTING
# =============================================================================

def neg_log_likelihood(params, x, choices):
    mu, sigma, gamma, lapse = params
    probability = gamma + (1.0 - gamma - lapse) * norm.cdf(
        (x - mu) / sigma
    )
    probability = np.clip(probability, 1e-10, 1.0 - 1e-10)
    return -np.sum(
        choices * np.log(probability)
        + (1.0 - choices) * np.log(1.0 - probability)
    )


def evaluate_psychometric(x, params):
    mu, sigma, gamma, lapse = params
    return gamma + (1.0 - gamma - lapse) * norm.cdf((x - mu) / sigma)


def fit_psychometric(cell_data: pd.DataFrame) -> dict:
    """Fit one cell and always return diagnostics, including failure reason."""
    counts = cell_data.groupby(COL_STIM)[COL_CHOICE].count()
    valid_angles = counts[counts >= MIN_TRIALS_PER_ANGLE].index
    used = cell_data[cell_data[COL_STIM].isin(valid_angles)].copy()

    diagnostic = {
        "success": False,
        "reason": "unknown",
        "n_input": int(len(cell_data)),
        "n_used": int(len(used)),
        "n_unique_angles_input": int(cell_data[COL_STIM].nunique()),
        "n_unique_angles_used": int(used[COL_STIM].nunique()),
        "params": None,
        "objective": np.nan,
        "optimizer_message": "",
    }

    if len(used) < MIN_TRIALS_TOTAL:
        diagnostic["reason"] = "insufficient_total_trials"
        return diagnostic
    if used[COL_STIM].nunique() < MIN_UNIQUE_ANGLES:
        diagnostic["reason"] = "insufficient_unique_angles"
        return diagnostic

    x = used[COL_STIM].to_numpy(dtype=float)
    choices = used[COL_CHOICE].to_numpy(dtype=float)
    x0 = [45.0, 10.0, 0.05, 0.05]
    bounds = [MU_BOUNDS, SIGMA_BOUNDS, GAMMA_BOUNDS, LAPSE_BOUNDS]

    try:
        result = minimize(
            neg_log_likelihood,
            x0,
            args=(x, choices),
            bounds=bounds,
            method="L-BFGS-B",
        )
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        diagnostic["reason"] = f"optimizer_exception:{type(exc).__name__}"
        diagnostic["optimizer_message"] = str(exc)
        return diagnostic

    diagnostic["objective"] = float(result.fun)
    diagnostic["optimizer_message"] = str(result.message)
    if not result.success or not np.all(np.isfinite(result.x)):
        diagnostic["reason"] = "optimizer_nonconvergence"
        return diagnostic

    diagnostic["success"] = True
    diagnostic["reason"] = "ok"
    diagnostic["params"] = np.asarray(result.x, dtype=float)
    return diagnostic


def bound_flags(params: np.ndarray) -> dict[str, bool]:
    flags = {}
    for value, name in zip(params, PARAMETER_ORDER):
        lower, upper = PARAMETER_BOUNDS[name]
        flags[f"{name}_at_lower"] = bool(abs(value - lower) <= BOUND_TOLERANCE)
        flags[f"{name}_at_upper"] = bool(abs(value - upper) <= BOUND_TOLERANCE)
    return flags


# =============================================================================
# FOUR-CELL DATA AND JOINT BOOTSTRAP
# =============================================================================

def build_factorial_pool(rat_data: pd.DataFrame, flanks: tuple[int, int]) -> pd.DataFrame:
    """Use all valid fake-45 trials from the two specified previous-angle flanks."""
    pool = rat_data[rat_data[COL_STIM_NM1].isin(flanks)].copy()
    pool = pool[
        pool[COL_CHOICE_NM1].isin([0, 1])
        & pool[COL_REWARD].isin([0, 1])
    ].copy()
    return pool


def split_cells(pool: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cells = {}
    for cell_name, (previous_choice, previous_reward) in CELL_DEFINITION.items():
        cells[cell_name] = pool[
            (pool[COL_CHOICE_NM1] == previous_choice)
            & (pool[COL_REWARD] == previous_reward)
        ].copy()
    return cells


def fit_all_cells(pool: pd.DataFrame) -> dict[str, dict]:
    return {name: fit_psychometric(frame) for name, frame in split_cells(pool).items()}


def resample_sessions(pool: pd.DataFrame, rng: np.random.Generator):
    sessions = pool[COL_SESSION].dropna().unique()
    sampled = rng.choice(sessions, size=len(sessions), replace=True)
    return pd.concat(
        [pool[pool[COL_SESSION] == session] for session in sampled],
        ignore_index=True,
    )


def resample_trials_by_cell(pool: pd.DataFrame, rng: np.random.Generator):
    pieces = []
    for frame in split_cells(pool).values():
        if frame.empty:
            return frame.copy()
        indices = rng.integers(0, len(frame), size=len(frame))
        pieces.append(frame.iloc[indices])
    return pd.concat(pieces, ignore_index=True)


def bootstrap_joint_cells(
    pool: pd.DataFrame,
    n_bootstrap: int,
    bootstrap_unit: str,
    seed: int,
) -> dict:
    """Generate aligned four-cell parameter arrays with all-or-none replicates."""
    if bootstrap_unit not in {"session", "trial_stratified"}:
        raise ValueError("Unknown bootstrap unit")

    sessions = pool[COL_SESSION].dropna().unique()
    if bootstrap_unit == "session" and len(sessions) < MIN_SESSIONS:
        return {
            "params": np.empty((0, len(CELL_ORDER), len(PARAMETER_ORDER))),
            "attempted": n_bootstrap,
            "valid": 0,
            "valid_fraction": 0.0,
            "ci_valid": False,
            "failure_counts": {"insufficient_sessions": n_bootstrap},
        }

    rng = np.random.default_rng(seed)
    parameter_samples = []
    failures = Counter()

    for _ in range(n_bootstrap):
        if bootstrap_unit == "session":
            bootstrap_pool = resample_sessions(pool, rng)
        else:
            bootstrap_pool = resample_trials_by_cell(pool, rng)

        fits = fit_all_cells(bootstrap_pool)
        failed = [
            f"{cell}:{fit['reason']}"
            for cell, fit in fits.items()
            if not fit["success"]
        ]
        if failed:
            failures.update(failed)
            failures["replicate_failed"] += 1
            continue

        parameter_samples.append(
            np.stack([fits[cell]["params"] for cell in CELL_ORDER], axis=0)
        )

    params = (
        np.stack(parameter_samples, axis=0)
        if parameter_samples
        else np.empty((0, len(CELL_ORDER), len(PARAMETER_ORDER)))
    )
    n_valid = int(len(params))
    valid_fraction = n_valid / n_bootstrap if n_bootstrap else 0.0
    threshold = max(
        MIN_VALID_BOOT_COUNT,
        int(np.ceil(MIN_VALID_BOOT_FRACTION * n_bootstrap)),
    )
    return {
        "params": params,
        "attempted": int(n_bootstrap),
        "valid": n_valid,
        "valid_fraction": float(valid_fraction),
        "valid_threshold": int(threshold),
        "ci_valid": bool(n_valid >= threshold),
        "failure_counts": dict(failures),
    }


def load_cached_bootstrap(
    bootstrap_dir: Path,
    rat,
    data_condition: str,
    scope: str,
    bootstrap_unit: str,
    n_bootstrap: int,
    expected_seed: int,
) -> dict | None:
    """Load a previously saved joint bootstrap for this rat, if one exists on
    disk and matches the current run configuration exactly.

    This is what makes a rerun after an interrupted run cheap: for a rat
    whose rat_<id>_joint_bootstrap.npz already has parameter samples fit
    under the identical data_condition, scope, bootstrap_unit, n_bootstrap,
    and derived seed, we reuse those samples instead of repeating the
    expensive resample-and-refit loop. Everything downstream (full-data cell
    fits, contrasts, omnibus tests, FDR correction, geometry, figures) is
    still computed fresh from this loaded array, exactly as it would be for a
    freshly bootstrapped rat -- only the resampling/refitting step is
    skipped.

    Returns a dict shaped like bootstrap_joint_cells' return value, or None
    if no compatible cached file exists (the caller should then fall back to
    bootstrap_joint_cells as normal). Per-failure-reason counts are not
    retained across a cache load, since they were not persisted in the npz;
    the returned "failure_counts" says so explicitly rather than fabricating
    numbers.
    """
    npz_path = bootstrap_dir / f"rat_{int(rat)}_joint_bootstrap.npz"
    if not npz_path.exists():
        return None
    try:
        with np.load(npz_path) as cached:
            if (
                cached["data_condition"].item() != data_condition
                or cached["scope"].item() != scope
                or cached["bootstrap_unit"].item() != bootstrap_unit
                or int(cached["n_bootstrap_attempted"].item()) != int(n_bootstrap)
                or int(cached["seed"].item()) != int(expected_seed)
            ):
                return None
            params = np.array(cached["parameter_bootstrap"], dtype=float)
    except (KeyError, ValueError, OSError, zipfile.BadZipFile) as exc:
        print(
            f"  Rat {rat}: cached bootstrap at {npz_path.name} unreadable or "
            f"incompatible ({exc}); recomputing."
        )
        return None

    attempted = int(n_bootstrap)
    valid = int(len(params))
    valid_fraction = valid / attempted if attempted else 0.0
    threshold = max(
        MIN_VALID_BOOT_COUNT,
        int(np.ceil(MIN_VALID_BOOT_FRACTION * attempted)),
    )
    return {
        "params": params,
        "attempted": attempted,
        "valid": valid,
        "valid_fraction": float(valid_fraction),
        "valid_threshold": int(threshold),
        "ci_valid": bool(valid >= threshold),
        "failure_counts": {
            "note": "loaded_from_cache; per-failure-reason counts were not "
                    "persisted by the run that created this cache file"
        },
        "loaded_from_cache": True,
    }


# =============================================================================
# CONTRASTS, P-VALUES, FDR, AND OMNIBUS
# =============================================================================

def contrasts_from_parameter_array(parameter_array: np.ndarray) -> np.ndarray:
    """Map (..., 4 cells, 4 parameters) to (..., n contrasts)."""
    mu = parameter_array[..., :, PARAMETER_ORDER.index("mu")]
    h_rew, v_rew, h_unrew, v_unrew = [mu[..., i] for i in range(4)]

    delta_rew = v_rew - h_rew
    delta_unrew = v_unrew - h_unrew
    S = delta_rew + delta_unrew
    D = delta_rew - delta_unrew
    P = 0.5 * (h_unrew + v_unrew) - 0.5 * (h_rew + v_rew)

    values = {
        "delta_mu_rew": delta_rew,
        "delta_mu_unrew": delta_unrew,
        "S": S,
        "D": D,
        "P": P,
        "choice_main_half": 0.5 * S,
        "choice_by_outcome_half": 0.5 * D,
        "reward_effect_after_h": h_unrew - h_rew,
        "reward_effect_after_v": v_unrew - v_rew,
    }
    return np.stack([values[name] for name in CONTRAST_ORDER], axis=-1)


def equal_tail_bootstrap_p(samples: np.ndarray, null_value: float = 0.0) -> float:
    """Two-sided p-value corresponding to inversion of an equal-tail bootstrap CI."""
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return np.nan
    lower = (np.count_nonzero(samples <= null_value) + 1.0) / (len(samples) + 1.0)
    upper = (np.count_nonzero(samples >= null_value) + 1.0) / (len(samples) + 1.0)
    return float(min(1.0, 2.0 * min(lower, upper)))


def benjamini_hochberg(p_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return BH q-values and rejection flags while preserving NaNs."""
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full_like(p_values, np.nan)
    rejects = np.zeros(len(p_values), dtype=bool)
    valid_indices = np.flatnonzero(np.isfinite(p_values))
    if len(valid_indices) == 0:
        return q_values, rejects

    p = p_values[valid_indices]
    order = np.argsort(p)
    ranked = p[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty(m, dtype=float)
    restored[order] = adjusted
    q_values[valid_indices] = restored
    rejects[valid_indices] = restored <= FDR_ALPHA
    return q_values, rejects


def add_contrast_fdr_columns(contrast_table: pd.DataFrame, scope: str) -> pd.DataFrame:
    table = contrast_table.copy()
    table["formal_primary"] = table["contrast"].isin(PRIMARY_CONTRASTS)
    table["scope_declared_inferential"] = scope in INFERENTIAL_SCOPES

    table["q_bh_primary_family"] = np.nan
    table["reject_bh_primary_family"] = False
    primary_mask = table["formal_primary"]
    primary_p = table.loc[primary_mask, "p_boot_equal_tail"].fillna(1.0).to_numpy()
    q_values, rejects = benjamini_hochberg(
        primary_p
    )
    table.loc[primary_mask, "q_bh_primary_family"] = q_values
    table.loc[primary_mask, "reject_bh_primary_family"] = rejects

    table["q_bh_all_core_sensitivity"] = np.nan
    table["reject_bh_all_core_sensitivity"] = False
    core_mask = table["contrast"].isin(CORE_CONTRASTS)
    core_p = table.loc[core_mask, "p_boot_equal_tail"].fillna(1.0).to_numpy()
    q_values, rejects = benjamini_hochberg(
        core_p
    )
    table.loc[core_mask, "q_bh_all_core_sensitivity"] = q_values
    table.loc[core_mask, "reject_bh_all_core_sensitivity"] = rejects

    table["q_bh_within_contrast_sensitivity"] = np.nan
    table["reject_bh_within_contrast_sensitivity"] = False
    for contrast_name, indices in table.groupby("contrast").groups.items():
        indices = np.asarray(list(indices), dtype=int)
        q_values, rejects = benjamini_hochberg(
            table.loc[indices, "p_boot_equal_tail"].fillna(1.0).to_numpy()
        )
        table.loc[indices, "q_bh_within_contrast_sensitivity"] = q_values
        table.loc[indices, "reject_bh_within_contrast_sensitivity"] = rejects

    # Only these flags are to be interpreted as the declared confirmatory result.
    table["declared_fdr_reject"] = (
        table["reject_bh_primary_family"]
        & table["scope_declared_inferential"]
    )
    return table


def omnibus_wald(theta_hat: np.ndarray, theta_boot: np.ndarray) -> dict:
    """Approximate bootstrap-calibrated and chi-square Wald omnibus tests."""
    theta_hat = np.asarray(theta_hat, dtype=float)
    theta_boot = np.asarray(theta_boot, dtype=float)
    dimension = len(theta_hat)
    if (
        dimension not in {2, 3}
        or theta_boot.ndim != 2
        or theta_boot.shape[1] != dimension
        or len(theta_boot) < 4
    ):
        return {
            "T": np.nan,
            "p_boot": np.nan,
            "p_chi2": np.nan,
            "rank": 0,
            "condition_number": np.nan,
            "covariance": np.full((dimension, dimension), np.nan),
            "T_boot": np.empty(0),
            "rotation_T_abs_difference": np.nan,
        }

    covariance = np.cov(theta_boot, rowvar=False, ddof=1)
    rank = int(np.linalg.matrix_rank(covariance))
    inverse = np.linalg.pinv(covariance, rcond=1e-10)
    T_observed = float(theta_hat @ inverse @ theta_hat)

    # Centering the bootstrap cloud estimates the null distribution of error.
    deviations = theta_boot - np.mean(theta_boot, axis=0)
    T_boot = np.einsum("bi,ij,bj->b", deviations, inverse, deviations)
    p_boot = float(
        (np.count_nonzero(T_boot >= T_observed) + 1.0) / (len(T_boot) + 1.0)
    )
    p_chi2 = float(chi2.sf(T_observed, df=rank)) if rank > 0 else np.nan

    # Exact linear rotation: (x, y[, P]) -> (S=x+y, D=x-y[, P]).
    if dimension == 2:
        rotation = np.array([[1.0, 1.0], [1.0, -1.0]])
    else:
        rotation = np.array(
            [[1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
        )
    rotated_hat = rotation @ theta_hat
    rotated_covariance = rotation @ covariance @ rotation.T
    rotated_inverse = np.linalg.pinv(rotated_covariance, rcond=1e-10)
    rotated_T = float(rotated_hat @ rotated_inverse @ rotated_hat)

    finite_condition = np.linalg.cond(covariance)
    return {
        "T": T_observed,
        "p_boot": p_boot,
        "p_chi2": p_chi2,
        "rank": rank,
        "condition_number": float(finite_condition),
        "covariance": covariance,
        "T_boot": T_boot,
        "rotation_T_abs_difference": float(abs(T_observed - rotated_T)),
    }


# =============================================================================
# GEOMETRY AND DESCRIPTIVE DIAGNOSTICS
# =============================================================================

def angular_line_distance(angle: float, axis_angle: float) -> float:
    """Smallest angular distance to a bidirectional line, in radians."""
    return float(abs((angle - axis_angle + np.pi / 2) % np.pi - np.pi / 2))


def geometry_class(x: float, y: float, null_radius: float) -> str:
    radius = float(np.hypot(x, y))
    if radius < null_radius:
        return "null"
    angle = float(np.arctan2(y, x))
    axes = {
        "rew_only": 0.0,
        "S_dominant": np.pi / 4,
        "unrew_only": np.pi / 2,
        "D_dominant": -np.pi / 4,
    }
    return min(axes, key=lambda name: angular_line_distance(angle, axes[name]))


def quadrant_label(x: float, y: float) -> str:
    if x < 0 and y < 0:
        return "win_stay_and_lose_stay"
    if x < 0 and y >= 0:
        return "win_stay_and_lose_switch"
    if x >= 0 and y < 0:
        return "win_switch_and_lose_stay"
    return "win_switch_and_lose_switch"


def geometry_summary(x_hat: float, y_hat: float, x_boot: np.ndarray, y_boot: np.ndarray) -> dict:
    point_class = geometry_class(x_hat, y_hat, GEOMETRY_NULL_RADIUS_DEG)
    boot_classes = [
        geometry_class(x, y, GEOMETRY_NULL_RADIUS_DEG)
        for x, y in zip(x_boot, y_boot)
    ]
    class_counts = Counter(boot_classes)
    quadrants = Counter(quadrant_label(x, y) for x, y in zip(x_boot, y_boot))
    n = max(len(x_boot), 1)
    result = {
        "delta_mu_rew": float(x_hat),
        "delta_mu_unrew": float(y_hat),
        "radius": float(np.hypot(x_hat, y_hat)),
        "angle_degrees": float(np.degrees(np.arctan2(y_hat, x_hat))),
        "geometry_class": point_class,
        "quadrant": quadrant_label(x_hat, y_hat),
        "bootstrap_same_sign_fraction": float(np.mean(np.sign(x_boot) == np.sign(y_boot))),
        "bootstrap_geometry_agreement": class_counts[point_class] / n,
        "prob_delta_mu_rew_positive": float(np.mean(x_boot > 0)),
        "prob_delta_mu_unrew_positive": float(np.mean(y_boot > 0)),
    }
    for name in ("null", "rew_only", "unrew_only", "S_dominant", "D_dominant"):
        result[f"bootstrap_prob_{name}"] = class_counts[name] / n
    for name in (
        "win_stay_and_lose_stay",
        "win_stay_and_lose_switch",
        "win_switch_and_lose_stay",
        "win_switch_and_lose_switch",
    ):
        result[f"bootstrap_prob_{name}"] = quadrants[name] / n
    return result


# =============================================================================
# EXPORTABLE TABLE ROWS
# =============================================================================

def cell_composition_rows(pool, data_condition, scope, rat, flanks):
    rows = []
    for cell_name, frame in split_cells(pool).items():
        previous_choice, previous_reward = CELL_DEFINITION[cell_name]
        for flank in flanks:
            flank_frame = frame[frame[COL_STIM_NM1] == flank]
            rows.append({
                "data_condition": data_condition,
                "scope": scope,
                "rat": int(rat),
                "cell": cell_name,
                "previous_choice": previous_choice,
                "previous_reward": previous_reward,
                "previous_angle": flank,
                "n_trials": int(len(flank_frame)),
                "n_sessions": int(flank_frame[COL_SESSION].nunique()),
                "n_current_angles": int(flank_frame[COL_STIM].nunique()),
            })
    return rows


def empirical_point_rows(pool, data_condition, scope, rat):
    rows = []
    for cell_name, frame in split_cells(pool).items():
        grouped = frame.groupby(COL_STIM)[COL_CHOICE].agg(["count", "mean"])
        for angle, values in grouped.iterrows():
            rows.append({
                "data_condition": data_condition,
                "scope": scope,
                "rat": int(rat),
                "cell": cell_name,
                "current_angle": float(angle),
                "n_trials": int(values["count"]),
                "p_vertical": float(values["mean"]),
            })
    return rows


def full_fit_rows(fits, data_condition, scope, rat):
    rows = []
    for cell_name in CELL_ORDER:
        fit = fits[cell_name]
        row = {
            "data_condition": data_condition,
            "scope": scope,
            "rat": int(rat),
            "cell": cell_name,
            "success": fit["success"],
            "failure_reason": fit["reason"],
            "n_input": fit["n_input"],
            "n_used": fit["n_used"],
            "n_unique_angles_input": fit["n_unique_angles_input"],
            "n_unique_angles_used": fit["n_unique_angles_used"],
            "objective": fit["objective"],
            "optimizer_message": fit["optimizer_message"],
        }
        if fit["success"]:
            for name, value in zip(PARAMETER_ORDER, fit["params"]):
                row[name] = float(value)
            row.update(bound_flags(fit["params"]))
        rows.append(row)
    return rows


def bootstrap_bound_rows(parameter_bootstrap, data_condition, scope, rat):
    rows = []
    if len(parameter_bootstrap) == 0:
        return rows
    for cell_index, cell_name in enumerate(CELL_ORDER):
        for parameter_index, parameter_name in enumerate(PARAMETER_ORDER):
            values = parameter_bootstrap[:, cell_index, parameter_index]
            lower, upper = PARAMETER_BOUNDS[parameter_name]
            rows.append({
                "data_condition": data_condition,
                "scope": scope,
                "rat": int(rat),
                "cell": cell_name,
                "parameter": parameter_name,
                "lower_bound": lower,
                "upper_bound": upper,
                "fraction_at_lower": float(np.mean(np.abs(values - lower) <= BOUND_TOLERANCE)),
                "fraction_at_upper": float(np.mean(np.abs(values - upper) <= BOUND_TOLERANCE)),
                "bootstrap_min": float(np.min(values)),
                "bootstrap_max": float(np.max(values)),
            })
    return rows


def bootstrap_parameter_rows(
    full_parameters, parameter_bootstrap, ci_valid, data_condition, scope, rat
):
    """Plot-ready cell-parameter estimates and bootstrap summaries."""
    rows = []
    for cell_index, cell_name in enumerate(CELL_ORDER):
        for parameter_index, parameter_name in enumerate(PARAMETER_ORDER):
            samples = parameter_bootstrap[:, cell_index, parameter_index]
            if ci_valid and len(samples):
                low, high = np.percentile(samples, [2.5, 97.5])
                standard_deviation = float(np.std(samples, ddof=1))
            else:
                low = high = standard_deviation = np.nan
            full_estimate = float(full_parameters[cell_index, parameter_index])
            rows.append({
                "data_condition": data_condition,
                "scope": scope,
                "rat": int(rat),
                "cell": cell_name,
                "parameter": parameter_name,
                "estimate_full_data": full_estimate,
                "bootstrap_median": float(np.median(samples)) if len(samples) else np.nan,
                "bootstrap_mean": float(np.mean(samples)) if len(samples) else np.nan,
                "bootstrap_bias": float(np.mean(samples) - full_estimate) if len(samples) else np.nan,
                "bootstrap_sd": standard_deviation,
                "ci_low": float(low),
                "ci_high": float(high),
                "n_boot_valid": int(len(samples)),
                "ci_valid": bool(ci_valid),
            })
    return rows


def contrast_covariance_rows(contrast_bootstrap, data_condition, scope, rat):
    """Save the full covariance and correlation matrices in tidy form."""
    rows = []
    if len(contrast_bootstrap) < 2:
        return rows
    covariance = np.cov(contrast_bootstrap, rowvar=False, ddof=1)
    correlation = np.corrcoef(contrast_bootstrap, rowvar=False)
    for i, row_name in enumerate(CONTRAST_ORDER):
        for j, column_name in enumerate(CONTRAST_ORDER):
            rows.append({
                "data_condition": data_condition,
                "scope": scope,
                "rat": int(rat),
                "contrast_row": row_name,
                "contrast_column": column_name,
                "covariance": float(covariance[i, j]),
                "correlation": float(correlation[i, j]),
            })
    return rows


def contrast_rows(
    estimate_array,
    bootstrap_array,
    ci_valid,
    data_condition,
    scope,
    rat,
):
    rows = []
    for index, name in enumerate(CONTRAST_ORDER):
        samples = bootstrap_array[:, index]
        estimate = float(estimate_array[index])
        if ci_valid and len(samples):
            ci_low, ci_high = np.percentile(samples, [2.5, 97.5])
            bootstrap_sd = float(np.std(samples, ddof=1))
            p_boot = equal_tail_bootstrap_p(samples)
            p_normal = float(2.0 * norm.sf(abs(estimate / bootstrap_sd))) if bootstrap_sd > 0 else np.nan
            excludes = bool(not (ci_low <= 0 <= ci_high))
        else:
            ci_low = ci_high = bootstrap_sd = p_boot = p_normal = np.nan
            excludes = False
        rows.append({
            "data_condition": data_condition,
            "scope": scope,
            "rat": int(rat),
            "contrast": name,
            "estimate_full_data": estimate,
            "bootstrap_median": float(np.median(samples)) if len(samples) else np.nan,
            "bootstrap_mean": float(np.mean(samples)) if len(samples) else np.nan,
            "bootstrap_bias": float(np.mean(samples) - estimate) if len(samples) else np.nan,
            "bootstrap_sd": bootstrap_sd,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "ci_excludes_zero": excludes,
            "p_boot_equal_tail": p_boot,
            "p_wald_normal_sensitivity": p_normal,
            "n_boot_valid": int(len(samples)),
            "ci_valid": bool(ci_valid),
            "role": "formal_primary" if name in PRIMARY_CONTRASTS else "descriptive_or_derived",
        })
    return rows


def unavailable_contrast_rows(data_condition, scope, rat, reason):
    """Keep unavailable planned tests in multiplicity families as p=1 rows."""
    rows = []
    for name in CONTRAST_ORDER:
        rows.append({
            "data_condition": data_condition,
            "scope": scope,
            "rat": int(rat),
            "contrast": name,
            "estimate_full_data": np.nan,
            "bootstrap_median": np.nan,
            "bootstrap_mean": np.nan,
            "bootstrap_bias": np.nan,
            "bootstrap_sd": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "ci_excludes_zero": False,
            "p_boot_equal_tail": np.nan,
            "p_wald_normal_sensitivity": np.nan,
            "n_boot_valid": 0,
            "ci_valid": False,
            "role": "formal_primary" if name in PRIMARY_CONTRASTS else "descriptive_or_derived",
            "unavailable_reason": reason,
        })
    return rows


# =============================================================================
# PLOTTING
# =============================================================================

def plot_rat_summary(bundle: dict, contrast_table: pd.DataFrame, output_path: Path) -> None:
    parameter_bootstrap = bundle["parameter_bootstrap"]
    full_parameters = bundle["full_parameters"]
    contrast_bootstrap = bundle["contrast_bootstrap"]
    full_contrasts = bundle["full_contrasts"]
    rat = bundle["rat"]
    scope = bundle["scope"]
    data_condition = bundle["data_condition"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)

    # Four-cell PSEs.
    mu_index = PARAMETER_ORDER.index("mu")
    for index, cell_name in enumerate(CELL_ORDER):
        point = full_parameters[index, mu_index]
        samples = parameter_bootstrap[:, index, mu_index]
        low, high = np.percentile(samples, [2.5, 97.5])
        axes[0].vlines(index, low, high, linewidth=2)
        axes[0].hlines([low, high], index - 0.08, index + 0.08, linewidth=2)
        axes[0].scatter(index, point, s=55, zorder=3)
    axes[0].axhline(45, color="gray", linestyle=":")
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels([CELL_LABELS[name] for name in CELL_ORDER], rotation=20, ha="right")
    axes[0].set_ylabel("PSE mu [degrees]")
    axes[0].set_title("Four joint cell fits")

    # Direct-effect plane.
    x_index = CONTRAST_ORDER.index("delta_mu_rew")
    y_index = CONTRAST_ORDER.index("delta_mu_unrew")
    x_boot = contrast_bootstrap[:, x_index]
    y_boot = contrast_bootstrap[:, y_index]
    if len(x_boot) > PLOT_MAX_BOOT_POINTS:
        rng = np.random.default_rng(stable_seed(data_condition, scope, rat, "plot"))
        keep = rng.choice(len(x_boot), PLOT_MAX_BOOT_POINTS, replace=False)
        x_plot, y_plot = x_boot[keep], y_boot[keep]
    else:
        x_plot, y_plot = x_boot, y_boot
    axes[1].scatter(x_plot, y_plot, s=5, alpha=0.12, color="#444444")
    x_hat, y_hat = full_contrasts[x_index], full_contrasts[y_index]
    axes[1].scatter([x_hat], [y_hat], s=75, color="crimson", zorder=4)
    limit = max(2.0, 1.05 * np.max(np.abs(np.r_[x_plot, y_plot, x_hat, y_hat])))
    axes[1].plot([-limit, limit], [-limit, limit], ":", label="S axis: y=x")
    axes[1].plot([-limit, limit], [limit, -limit], ":", label="D axis: y=-x")
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].axvline(0, color="black", linewidth=0.6)
    axes[1].set_xlim(-limit, limit)
    axes[1].set_ylim(-limit, limit)
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("delta mu rewarded [degrees]")
    axes[1].set_ylabel("delta mu unrewarded [degrees]")
    axes[1].set_title("Direct-effect geometry")
    axes[1].legend(fontsize=8)

    # Five core contrasts.
    rat_rows = contrast_table[
        (contrast_table["rat"] == int(rat))
        & contrast_table["contrast"].isin(CORE_CONTRASTS)
    ].set_index("contrast")
    for index, name in enumerate(CORE_CONTRASTS):
        row = rat_rows.loc[name]
        estimate = row["estimate_full_data"]
        low, high = row["ci_low"], row["ci_high"]
        declared = bool(row["declared_fdr_reject"])
        color = "crimson" if declared else ("steelblue" if row["ci_excludes_zero"] else "gray")
        axes[2].vlines(index, low, high, linewidth=2, color=color)
        axes[2].hlines([low, high], index - 0.08, index + 0.08, linewidth=2, color=color)
        axes[2].scatter(index, estimate, s=55, color=color, zorder=3)
    axes[2].axhline(0, color="gray", linestyle="--")
    axes[2].set_xticks(range(len(CORE_CONTRASTS)))
    axes[2].set_xticklabels(CORE_CONTRASTS, rotation=25, ha="right")
    axes[2].set_ylabel("Contrast [degrees]")
    axes[2].set_title("Gray: unresolved; blue: pointwise; red: declared FDR")

    fig.suptitle(f"Rat {rat} | {data_condition} | {scope}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_forest(contrast_table: pd.DataFrame, output_path: Path) -> None:
    core = contrast_table[contrast_table["contrast"].isin(CORE_CONTRASTS)].copy()
    if core.empty:
        return
    rats = sorted(core["rat"].unique())
    fig, axes = plt.subplots(1, len(CORE_CONTRASTS), figsize=(20, 8), sharey=True, constrained_layout=True)
    y_positions = {rat: i for i, rat in enumerate(rats)}
    for axis, contrast_name in zip(axes, CORE_CONTRASTS):
        rows = core[core["contrast"] == contrast_name].set_index("rat")
        for rat in rats:
            if rat not in rows.index:
                continue
            row = rows.loc[rat]
            estimate, low, high = row["estimate_full_data"], row["ci_low"], row["ci_high"]
            color = "crimson" if row["declared_fdr_reject"] else ("steelblue" if row["ci_excludes_zero"] else "gray")
            y = y_positions[rat]
            axis.hlines(y, low, high, color=color, linewidth=2)
            axis.vlines([low, high], y - 0.08, y + 0.08, color=color, linewidth=1.5)
            axis.scatter(estimate, y, color=color, s=32, zorder=3)
        axis.axvline(0, color="black", linestyle="--", linewidth=0.7)
        axis.set_title(contrast_name)
        axis.set_xlabel("degrees")
    axes[0].set_yticks(range(len(rats)))
    axes[0].set_yticklabels([f"Rat {rat}" for rat in rats])
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# ONE SCOPE AND COMPLETE PIPELINE
# =============================================================================

def run_scope(
    scope_data: pd.DataFrame,
    data_condition: str,
    scope: str,
    output_directory: Path,
    n_bootstrap: int,
    bootstrap_unit: str,
    plots_enabled: bool,
) -> dict[str, pd.DataFrame]:
    table_dir = output_directory / "tables"
    figure_dir = output_directory / "figures"
    bootstrap_dir = output_directory / "bootstraps"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_dir.mkdir(parents=True, exist_ok=True)

    composition = []
    empirical = []
    full_fits_table = []
    bound_diagnostics = []
    parameter_summaries = []
    contrast_covariances = []
    bootstrap_qc = []
    all_contrasts = []
    all_omnibus = []
    all_geometry = []
    bundles = {}

    rats = sorted(scope_data[COL_RAT].unique())
    print(f"\n[{data_condition} | {scope}] rats={len(rats)}, bootstrap={n_bootstrap}, unit={bootstrap_unit}")
    n_cache_hits = 0

    for rat in rats:
        rat_data = scope_data[scope_data[COL_RAT] == rat].copy()
        flanks = PSE_FLANKS.get(int(rat), (40, 50))
        pool = build_factorial_pool(rat_data, flanks)
        composition.extend(cell_composition_rows(pool, data_condition, scope, rat, flanks))
        empirical.extend(empirical_point_rows(pool, data_condition, scope, rat))

        full_fits = fit_all_cells(pool)
        full_fits_table.extend(full_fit_rows(full_fits, data_condition, scope, rat))
        failed_full = [name for name, fit in full_fits.items() if not fit["success"]]
        if failed_full:
            failure_reason = "full_fit_failed:" + ",".join(failed_full)
            bootstrap_qc.append({
                "data_condition": data_condition,
                "scope": scope,
                "rat": int(rat),
                "bootstrap_unit": bootstrap_unit,
                "attempted": 0,
                "valid": 0,
                "valid_fraction": 0.0,
                "valid_threshold": np.nan,
                "ci_valid": False,
                "failure_counts_json": json.dumps({"full_fit_failed": failed_full}),
            })
            all_contrasts.extend(
                unavailable_contrast_rows(data_condition, scope, rat, failure_reason)
            )
            all_omnibus.append({
                "data_condition": data_condition,
                "scope": scope,
                "rat": int(rat),
                "omnibus": "choice_plane_2df",
                "delta_mu_rew": np.nan,
                "delta_mu_unrew": np.nan,
                "P": np.nan,
                "wald_T": np.nan,
                "covariance_rank": 0,
                "covariance_condition_number": np.nan,
                "p_omnibus_bootstrap": np.nan,
                "p_omnibus_chi2_sensitivity": np.nan,
                "rotation_T_abs_difference": np.nan,
                "n_boot_valid": 0,
                "ci_valid": False,
                "unavailable_reason": failure_reason,
            })
            all_omnibus.append({
                "data_condition": data_condition,
                "scope": scope,
                "rat": int(rat),
                "omnibus": "full_factorial_3df",
                "delta_mu_rew": np.nan,
                "delta_mu_unrew": np.nan,
                "P": np.nan,
                "wald_T": np.nan,
                "covariance_rank": 0,
                "covariance_condition_number": np.nan,
                "p_omnibus_bootstrap": np.nan,
                "p_omnibus_chi2_sensitivity": np.nan,
                "rotation_T_abs_difference": np.nan,
                "n_boot_valid": 0,
                "ci_valid": False,
                "unavailable_reason": failure_reason,
            })
            print(f"  Rat {rat}: skipped; full fits failed for {failed_full}")
            continue

        full_parameters = np.stack([full_fits[name]["params"] for name in CELL_ORDER])
        full_contrasts = contrasts_from_parameter_array(full_parameters)
        seed = stable_seed(data_condition, scope, rat, bootstrap_unit)

        bootstrap = None
        if USE_CACHED_BOOTSTRAP:
            bootstrap = load_cached_bootstrap(
                bootstrap_dir, rat, data_condition, scope, bootstrap_unit, n_bootstrap, seed
            )
        if bootstrap is None:
            bootstrap = bootstrap_joint_cells(pool, n_bootstrap, bootstrap_unit, seed)
        else:
            n_cache_hits += 1
        parameter_bootstrap = bootstrap["params"]
        contrast_bootstrap = contrasts_from_parameter_array(parameter_bootstrap) if len(parameter_bootstrap) else np.empty((0, len(CONTRAST_ORDER)))

        bound_diagnostics.extend(
            bootstrap_bound_rows(parameter_bootstrap, data_condition, scope, rat)
        )
        parameter_summaries.extend(
            bootstrap_parameter_rows(
                full_parameters,
                parameter_bootstrap,
                bootstrap["ci_valid"],
                data_condition,
                scope,
                rat,
            )
        )
        contrast_covariances.extend(
            contrast_covariance_rows(
                contrast_bootstrap, data_condition, scope, rat
            )
        )
        bootstrap_qc.append({
            "data_condition": data_condition,
            "scope": scope,
            "rat": int(rat),
            "bootstrap_unit": bootstrap_unit,
            "seed": int(seed),
            "attempted": bootstrap["attempted"],
            "valid": bootstrap["valid"],
            "valid_fraction": bootstrap["valid_fraction"],
            "valid_threshold": bootstrap.get("valid_threshold", np.nan),
            "ci_valid": bootstrap["ci_valid"],
            "failure_counts_json": json.dumps(bootstrap["failure_counts"], sort_keys=True),
            "loaded_from_cache": bool(bootstrap.get("loaded_from_cache", False)),
        })

        all_contrasts.extend(
            contrast_rows(
                full_contrasts,
                contrast_bootstrap,
                bootstrap["ci_valid"],
                data_condition,
                scope,
                rat,
            )
        )

        if len(contrast_bootstrap):
            x_idx = CONTRAST_ORDER.index("delta_mu_rew")
            y_idx = CONTRAST_ORDER.index("delta_mu_unrew")
            p_idx = CONTRAST_ORDER.index("P")
            theta_choice_hat = full_contrasts[[x_idx, y_idx]]
            theta_choice_boot = contrast_bootstrap[:, [x_idx, y_idx]]
            theta_full_hat = full_contrasts[[x_idx, y_idx, p_idx]]
            theta_full_boot = contrast_bootstrap[:, [x_idx, y_idx, p_idx]]
            if bootstrap["ci_valid"]:
                choice_omnibus = omnibus_wald(theta_choice_hat, theta_choice_boot)
                full_omnibus = omnibus_wald(theta_full_hat, theta_full_boot)
            else:
                choice_omnibus = omnibus_wald(theta_choice_hat, np.empty((0, 2)))
                full_omnibus = omnibus_wald(theta_full_hat, np.empty((0, 3)))
            geometry = geometry_summary(
                full_contrasts[x_idx],
                full_contrasts[y_idx],
                contrast_bootstrap[:, x_idx],
                contrast_bootstrap[:, y_idx],
            )
        else:
            theta_choice_hat = np.full(2, np.nan)
            theta_choice_boot = np.empty((0, 2))
            theta_full_hat = np.full(3, np.nan)
            theta_full_boot = np.empty((0, 3))
            choice_omnibus = omnibus_wald(theta_choice_hat, theta_choice_boot)
            full_omnibus = omnibus_wald(theta_full_hat, theta_full_boot)
            geometry = {}

        for omnibus_name, theta_hat, omnibus in (
            ("choice_plane_2df", theta_choice_hat, choice_omnibus),
            ("full_factorial_3df", theta_full_hat, full_omnibus),
        ):
            omnibus_row = {
                "data_condition": data_condition,
                "scope": scope,
                "rat": int(rat),
                "omnibus": omnibus_name,
                "delta_mu_rew": theta_hat[0],
                "delta_mu_unrew": theta_hat[1],
                "P": theta_hat[2] if len(theta_hat) == 3 else np.nan,
                "wald_T": omnibus["T"],
                "covariance_rank": omnibus["rank"],
                "covariance_condition_number": omnibus["condition_number"],
                "p_omnibus_bootstrap": omnibus["p_boot"],
                "p_omnibus_chi2_sensitivity": omnibus["p_chi2"],
                "rotation_T_abs_difference": omnibus["rotation_T_abs_difference"],
                "n_boot_valid": int(len(parameter_bootstrap)),
                "ci_valid": bootstrap["ci_valid"],
            }
            covariance = omnibus["covariance"]
            for row_index in range(covariance.shape[0]):
                for column_index in range(covariance.shape[1]):
                    omnibus_row[f"cov_{row_index}_{column_index}"] = covariance[row_index, column_index]
            all_omnibus.append(omnibus_row)

        geometry.update({
            "data_condition": data_condition,
            "scope": scope,
            "rat": int(rat),
            "n_boot_valid": int(len(parameter_bootstrap)),
            "ci_valid": bootstrap["ci_valid"],
        })
        all_geometry.append(geometry)

        npz_path = bootstrap_dir / f"rat_{int(rat)}_joint_bootstrap.npz"
        np.savez_compressed(
            npz_path,
            parameter_bootstrap=parameter_bootstrap,
            contrast_bootstrap=contrast_bootstrap,
            theta_bootstrap_x_y=theta_choice_boot,
            theta_bootstrap_x_y_P=theta_full_boot,
            omnibus_choice_T_bootstrap=choice_omnibus["T_boot"],
            omnibus_full_T_bootstrap=full_omnibus["T_boot"],
            full_cell_parameters=full_parameters,
            full_contrast_estimates=full_contrasts,
            covariance_x_y=choice_omnibus["covariance"],
            covariance_x_y_P=full_omnibus["covariance"],
            cell_order=np.asarray(CELL_ORDER),
            parameter_order=np.asarray(PARAMETER_ORDER),
            contrast_order=np.asarray(CONTRAST_ORDER),
            data_condition=np.asarray(data_condition),
            scope=np.asarray(scope),
            rat=np.asarray(int(rat)),
            bootstrap_unit=np.asarray(bootstrap_unit),
            seed=np.asarray(seed),
            n_bootstrap_attempted=np.asarray(n_bootstrap),
            ci_valid=np.asarray(bootstrap["ci_valid"]),
        )

        bundles[int(rat)] = {
            "parameter_bootstrap": parameter_bootstrap,
            "contrast_bootstrap": contrast_bootstrap,
            "full_parameters": full_parameters,
            "full_contrasts": full_contrasts,
            "rat": int(rat),
            "scope": scope,
            "data_condition": data_condition,
            "ci_valid": bootstrap["ci_valid"],
        }
        source = "cache" if bootstrap.get("loaded_from_cache") else "computed"
        print(
            f"  Rat {rat}: {bootstrap['valid']}/{n_bootstrap} valid "
            f"(ci_valid={bootstrap['ci_valid']}, source={source})"
        )

    if USE_CACHED_BOOTSTRAP:
        print(
            f"  -> {n_cache_hits}/{len(rats)} rats reused a cached bootstrap; "
            f"{len(rats) - n_cache_hits} were (re)computed this run."
        )

    contrast_table = add_contrast_fdr_columns(pd.DataFrame(all_contrasts), scope) if all_contrasts else pd.DataFrame()
    omnibus_table = pd.DataFrame(all_omnibus)
    if not omnibus_table.empty:
        omnibus_table["q_bh_omnibus_family"] = np.nan
        omnibus_table["reject_bh_omnibus_family"] = False
        for omnibus_name, indices in omnibus_table.groupby("omnibus").groups.items():
            indices = np.asarray(list(indices), dtype=int)
            omnibus_p = omnibus_table.loc[
                indices, "p_omnibus_bootstrap"
            ].fillna(1.0).to_numpy()
            q_values, rejects = benjamini_hochberg(omnibus_p)
            omnibus_table.loc[indices, "q_bh_omnibus_family"] = q_values
            omnibus_table.loc[indices, "reject_bh_omnibus_family"] = rejects
        omnibus_table["scope_declared_inferential"] = scope in INFERENTIAL_SCOPES
        omnibus_table["declared_omnibus_fdr_reject"] = (
            omnibus_table["reject_bh_omnibus_family"]
            & omnibus_table["scope_declared_inferential"]
        )

    tables = {
        "cell_composition": save_csv(composition, table_dir / "cell_composition.csv"),
        "empirical_points": save_csv(empirical, table_dir / "empirical_points.csv"),
        "full_data_cell_fits": save_csv(full_fits_table, table_dir / "full_data_cell_fits.csv"),
        "bootstrap_parameter_summary": save_csv(parameter_summaries, table_dir / "bootstrap_parameter_summary.csv"),
        "bootstrap_bound_diagnostics": save_csv(bound_diagnostics, table_dir / "bootstrap_bound_diagnostics.csv"),
        "contrast_covariance_long": save_csv(contrast_covariances, table_dir / "contrast_covariance_long.csv"),
        "bootstrap_qc": save_csv(bootstrap_qc, table_dir / "bootstrap_qc.csv"),
        "contrast_summary_long": save_csv(contrast_table, table_dir / "contrast_summary_long.csv"),
        "omnibus_summary": save_csv(omnibus_table, table_dir / "omnibus_summary.csv"),
        "geometry_summary": save_csv(all_geometry, table_dir / "geometry_summary.csv"),
    }

    write_json(
        output_directory / "array_manifest.json",
        {
            "cell_order": CELL_ORDER,
            "parameter_order": PARAMETER_ORDER,
            "contrast_order": CONTRAST_ORDER,
            "theta_choice_order": ["delta_mu_rew", "delta_mu_unrew"],
            "theta_full_order": ["delta_mu_rew", "delta_mu_unrew", "P"],
            "npz_description": {
                "parameter_bootstrap": "valid replicates x four cells x four parameters",
                "contrast_bootstrap": "valid replicates x all named contrasts",
                "full_cell_parameters": "four cells x four parameters",
                "full_contrast_estimates": "all named contrasts from full data",
            },
        },
    )

    if plots_enabled and not contrast_table.empty:
        for rat, bundle in bundles.items():
            if len(bundle["parameter_bootstrap"]) and bundle["ci_valid"]:
                plot_rat_summary(
                    bundle,
                    contrast_table,
                    figure_dir / f"rat_{rat}_contrast_summary.png",
                )
        plot_forest(contrast_table, figure_dir / "forest_core_contrasts.png")

    return tables


def run_pipeline(
    data_filter_mode: str = DATA_FILTER_MODE,
    analysis_scope: str = ANALYSIS_SCOPE,
    bootstrap_unit: str = BOOTSTRAP_UNIT,
    n_bootstrap: int = N_BOOTSTRAP,
    plots_enabled: bool = PLOTS_ENABLED,
    use_cached_bootstrap: bool = USE_CACHED_BOOTSTRAP,
) -> dict:
    global DATA_FILTER_MODE, ANALYSIS_SCOPE, BOOTSTRAP_UNIT, N_BOOTSTRAP, PLOTS_ENABLED, USE_CACHED_BOOTSTRAP
    DATA_FILTER_MODE = data_filter_mode
    ANALYSIS_SCOPE = analysis_scope
    BOOTSTRAP_UNIT = bootstrap_unit
    N_BOOTSTRAP = n_bootstrap
    PLOTS_ENABLED = plots_enabled
    USE_CACHED_BOOTSTRAP = use_cached_bootstrap
    validate_configuration()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(
        OUTPUT_ROOT / "run_manifest.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "package_versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "matplotlib": matplotlib.__version__,
            },
            "configuration": config_snapshot(),
            "interpretation": {
                "primary_family": "configured direct/planned contrasts across rats",
                "omnibus": "complementary 2df and 3df tests, each with a separate BH family; not gatekeepers",
                "S": "descriptive by default",
                "D": "outcome-dependent choice contrast; causal interaction requires design identification",
                "unfiltered": "no training-performance criterion; basic loader/validity filters remain",
            },
        },
    )

    base_data = load_base_data()
    datasets = make_filter_datasets(base_data, data_filter_mode)
    results = {}
    for data_condition, condition_data in datasets.items():
        results[data_condition] = {}
        print(f"\n{'#' * 76}\nDATA CONDITION: {data_condition} (n={len(condition_data)})\n{'#' * 76}")
        for scope, scope_data in make_scopes(condition_data, analysis_scope).items():
            scope_directory = OUTPUT_ROOT / data_condition / scope
            results[data_condition][scope] = run_scope(
                scope_data,
                data_condition,
                scope,
                scope_directory,
                n_bootstrap,
                bootstrap_unit,
                plots_enabled,
            )

    # Convenience copies spanning all requested data conditions and scopes.
    combined_directory = OUTPUT_ROOT / "combined_tables"
    combined_directory.mkdir(parents=True, exist_ok=True)
    table_names = sorted({
        table_name
        for condition_results in results.values()
        for scope_results in condition_results.values()
        for table_name in scope_results
    })
    for table_name in table_names:
        frames = [
            scope_results[table_name]
            for condition_results in results.values()
            for scope_results in condition_results.values()
            if table_name in scope_results and not scope_results[table_name].empty
        ]
        if frames:
            save_csv(
                pd.concat(frames, ignore_index=True, sort=False),
                combined_directory / f"{table_name}_all_runs.csv",
            )
    return results


# =============================================================================
# BUILT-IN SELF TESTS (do not require project data)
# =============================================================================

def run_self_tests() -> None:
    rng = np.random.default_rng(123)

    # Algebra and array shape.
    parameters = np.zeros((200, 4, 4), dtype=float)
    parameters[:, :, 0] = rng.normal(45, 3, size=(200, 4))
    parameters[:, :, 1] = 10
    parameters[:, :, 2:] = 0.05
    contrasts = contrasts_from_parameter_array(parameters)
    x = contrasts[:, CONTRAST_ORDER.index("delta_mu_rew")]
    y = contrasts[:, CONTRAST_ORDER.index("delta_mu_unrew")]
    S = contrasts[:, CONTRAST_ORDER.index("S")]
    D = contrasts[:, CONTRAST_ORDER.index("D")]
    assert np.allclose(S, x + y)
    assert np.allclose(D, x - y)
    assert np.allclose(x, 0.5 * (S + D))
    assert np.allclose(y, 0.5 * (S - D))

    # Coordinate invariance of the Wald statistic.
    theta_boot = np.column_stack([x, y, contrasts[:, CONTRAST_ORDER.index("P")]])
    theta_hat = np.mean(theta_boot, axis=0) + np.array([0.5, -0.2, 0.1])
    omnibus = omnibus_wald(theta_hat, theta_boot)
    assert omnibus["rotation_T_abs_difference"] < 1e-8

    # BH known example: smallest two are discoveries at q=.05.
    q_values, rejects = benjamini_hochberg(np.array([0.001, 0.01, 0.2, 0.8]))
    assert np.array_equal(rejects, np.array([True, True, False, False]))
    assert np.all((q_values >= 0) & (q_values <= 1))

    # Geometry axes.
    assert geometry_class(5, 0, 2) == "rew_only"
    assert geometry_class(0, -5, 2) == "unrew_only"
    assert geometry_class(-5, -5, 2) == "S_dominant"
    assert geometry_class(-5, 5, 2) == "D_dominant"
    assert geometry_class(0.5, 0.5, 2) == "null"

    print("All self-tests passed.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filter-mode",
        choices=("filtered", "unfiltered", "both"),
        default=DATA_FILTER_MODE,
    )
    parser.add_argument(
        "--analysis-scope",
        choices=("pooled", "per_modality", "both"),
        default=ANALYSIS_SCOPE,
    )
    parser.add_argument(
        "--bootstrap-unit",
        choices=("session", "trial_stratified"),
        default=BOOTSTRAP_UNIT,
    )
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--recompute-all",
        action="store_true",
        help=(
            "Ignore any cached per-rat bootstrap NPZ files under "
            "bootstraps/ and refit every rat's joint bootstrap from "
            "scratch, even if a compatible cached file exists."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.self_test:
        run_self_tests()
    else:
        run_pipeline(
            data_filter_mode=arguments.filter_mode,
            analysis_scope=arguments.analysis_scope,
            bootstrap_unit=arguments.bootstrap_unit,
            n_bootstrap=arguments.n_bootstrap,
            plots_enabled=not arguments.no_plots,
            use_cached_bootstrap=not arguments.recompute_all,
        )