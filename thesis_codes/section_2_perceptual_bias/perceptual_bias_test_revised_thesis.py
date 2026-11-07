"""
perceptual_bias_test_thesis.py
========================
Tests whether the difficulty of the n-1 stimulus biases the current
perceptual boundary (PSE shift) — i.e. whether history-dependent
PERCEPTUAL bias is present. Changes in sigma are retained as secondary
descriptive output.

For every (rat, modality):
    - fit a 4-parameter cumulative Gaussian P(repeat) vs
      evidence-for-repetition, per difficulty_n-1 bin, via MLE
    - jointly resample sessions across all three bins, preserving the
      within-session dependence between bin-specific fits
    - compute full-data ΔPSE estimates plus aligned-bootstrap 95% CIs
      and two-sided bootstrap p-values
    - apply Benjamini-Hochberg FDR correction across the primary
      hard-to-easy contrasts only
    - describe, without additional discovery claims, whether the primary
      shift is monotonic and how it is divided across the two half-steps

Outputs include per-bin fits, all pairwise contrasts, the FDR-corrected
primary summary, descriptive shape diagnostics, publication tables, and
a compressed archive of all bootstrap distributions.

Fixes relative to the previous per-rat inspection script
--------------------------------------------------------
    ▸ MLE (Bernoulli log-likelihood) replaces scipy.curve_fit
      (LS is wrong for binary data and inconsistent with the
       choice-effect script; both scripts now fit the same way).
    ▸ Balanced binning via np.array_split (was uneven with //).
    ▸ Bootstrap stores fitted parameters, not just curves,
      so ΔPSE and Δslope CIs can be computed downstream.
    ▸ Adds ΔPSE / Δslope tests (per-pair CIs + primary-pair
      classification) — the actual statistical answer to
      "does n-1 difficulty shift the psychometric curve?".
    ▸ Loops over ALL rats (previous script hard-coded TARGET_RAT).
    ▸ Deduplicated legend code in the curves panel.
    ▸ Per-(rat, modality) 2-panel figure + group-level summary table.

Inferential hierarchy
----------------------
    ▸ Stage 1 (formal): one primary hard-to-easy ΔPSE test per rat ×
      modality, followed by BH-FDR across the complete primary family.
    ▸ Stage 2 (descriptive/diagnostic): hard-to-medium and medium-to-easy
      shifts, curvature Q, share R, and monotonicity. These quantities do
      not generate a second set of discoveries.
"""

import numpy as np
import pandas as pd
import os
import json
from datetime import datetime, timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import re
import sys
sys.path.append(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy")
from functions_Ale import filter_training_trials
from serial_dependence_analysis import SerialDependenceAnalyzer


# =============================================================================
# COLUMN NAMES
# =============================================================================
ANGLE_COL          = 'angle'
RAT_COL            = 'rat'
SESSION_COL        = 'date'
EVIDENCE_COL       = 'evidence_4_repetition'
REP_COL            = 'repeated_action'
STATE_COL          = 'hitmiss_n-1'          # 1 = prev correct
DIFFICULTY_NM1_COL = 'difficulty_n-1'
BOUNDARY           = 45
RESP_COL           = 'action'
MODALITY_COL       = 'mod'


# =============================================================================
# TUNABLE PARAMETERS
# =============================================================================
TARGET_RATS           = None      # None = all rats; else list like [9, 11, 13]
MIN_TRIALS_PER_BIN    = 20        # per (rat, mod, bin)
MIN_SESSIONS          = 3         # min sessions for reliable session-bootstrap
N_BOOT                = 1000      # final session-bootstrap iterations
BOOTSTRAP_SEED         = 20260912
MIN_BOOT_VALID_FRACTION = 0.80    # flag cells with excessive fit failures
FDR_ALPHA              = 0.05
N_DIFFICULTY_BINS     = 3
X_GRID                = np.linspace(-45, 45, 200)
BUBBLE_SCALE          = 0.5         # scatter marker area = n_trials * this
N_EMP_POINTS          = 5         # empirical dots per bin curve
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "figures_perceptual_bias_fail_safe")
TABLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "2.0_tables_perceptual_bias_fail_safe")
BOOTSTRAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "bootstrap_perceptual_bias_fail_safe")


# =============================================================================
# LOAD + PREPROCESS  (identical to your previous script)
# =============================================================================
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data(fold=False)
df = an.create_lagged_features()
df = df[(df[ANGLE_COL] >= 0) & (df[ANGLE_COL] <= 90)].copy()
df = df[(df['angle_n-1'] >= 0) & (df['angle_n-1'] <= 90)].copy()
df = df[(df['difficulty_n-1'] >= -46) & (df['difficulty_n-1'] <= 45)].copy()
df = filter_training_trials(df, method='criterion', criterion=0.80, min_consec=2)

# derive P(repeat) and signed evidence for repetition
df[REP_COL]      = (df['action_n-1'] == df[RESP_COL]).astype(int)
df[EVIDENCE_COL] = (df[ANGLE_COL] - BOUNDARY) * (2 * df['action_n-1'] - 1)

# key filters: n-1 correct only; exclude ambiguous n-1 (difficulty code 0)
df = df[df[STATE_COL] == 0].copy()
df = df[df[DIFFICULTY_NM1_COL] != 0].copy()

MODALITIES = sorted(df[MODALITY_COL].dropna().unique().tolist())

# =============================================================================
# MLE psychometric fit (Bernoulli negative log-likelihood)
# =============================================================================

def _nll(params, x, c):
    """4-param cum. Gaussian NLL: p = γ + (1 − γ − λ)·Φ((x − μ)/σ)."""
    mu, sigma, gamma, lapse = params
    p = gamma + (1 - gamma - lapse) * norm.cdf(x, loc=mu, scale=sigma)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -np.sum(c * np.log(p) + (1 - c) * np.log(1 - p))


def fit_psychometric(x, y):
    """MLE fit. Returns [mu, sigma, gamma, lapse] or None on failure."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 10 or len(np.unique(x)) < 3:
        return None

    x0     = [0.0, 15.0, 0.05, 0.05]
    bounds = [(-45, 45), (1, 50), (0, 0.3), (0, 0.3)]

    try:
        result = minimize(_nll, x0, args=(x, y),
                          bounds=bounds, method='L-BFGS-B')
        return result.x if result.success else None
    except (RuntimeError, ValueError, FloatingPointError):
        return None


def psychometric(x, mu, sigma, gamma, lapse):
    return gamma + (1 - gamma - lapse) * norm.cdf(x, loc=mu, scale=sigma)


# =============================================================================
# Balanced difficulty binning
# =============================================================================
BIN_NAMES = {3: 'easy', 2: 'medium', 1: 'hard'}
"""
def infer_difficulty_names(df,
                            difficulty_col=DIFFICULTY_NM1_COL,
                            angle_col=ANGLE_COL,
                            boundary=BOUNDARY,
                            names=('easy', 'medium', 'hard')):

    # angle for the n-th trial paired with the n-th trial's difficulty code
    valid = df[[difficulty_col, angle_col]].dropna()
    valid = valid[valid[difficulty_col] != 0]
    if valid.empty:
        return {}

    dist = (valid[angle_col] - boundary).abs()
    mean_dist = dist.groupby(valid[difficulty_col]).mean()

    # rank codes from EASIEST (largest distance) to HARDEST (smallest distance)
    codes_sorted = mean_dist.sort_values(ascending=False).index.tolist()

    n = len(codes_sorted)
    mapping = {}
    if n == len(names):
        for code, name in zip(codes_sorted, names):
            mapping[int(code)] = name
    elif n < len(names):
        # only extremes
        mapping[int(codes_sorted[0])]  = names[0]      # easiest
        mapping[int(codes_sorted[-1])] = names[-1]     # hardest
    else:
        # more codes than names: label extremes, number the middle
        mapping[int(codes_sorted[0])]  = names[0]
        mapping[int(codes_sorted[-1])] = names[-1]
        for code in codes_sorted[1:-1]:
            mapping[int(code)] = f"diff {int(code)}"

    return mapping
"""
# after the filters that build the final df, before run_analysis
#DIFFICULTY_NAMES = infer_difficulty_names(df)
print(f"difficulty naming: {BIN_NAMES}")

def make_bin_labels(bin_map):
    bin_to_levels = {}
    for level, bin_idx in bin_map.items():
        bin_to_levels.setdefault(bin_idx, []).append(int(level))

    labels = {}
    for bin_idx, levels in bin_to_levels.items():
        lv = sorted(levels)
        name = BIN_NAMES.get(bin_idx, f"bin {bin_idx}")
        rng  = f"{lv[0]}°" if len(lv) == 1 else f"{lv[0]}–{lv[-1]}°"
        labels[bin_idx] = f"{name} ({rng})"
    return labels

def make_bin_labels_short(bin_map):
    """Short labels for tick marks: just the semantic name, no numeric range."""
    bin_to_levels = {}
    for level, bin_idx in bin_map.items():
        bin_to_levels.setdefault(bin_idx, []).append(int(level))
    return {bin_idx: BIN_NAMES.get(bin_idx, f"bin {bin_idx}")
            for bin_idx in bin_to_levels}

def bin_difficulty(rat_df, min_trials=MIN_TRIALS_PER_BIN, n_bins=N_DIFFICULTY_BINS):
    """
    Split n-1 difficulty levels into n_bins as evenly as possible using
    np.array_split (bin sizes differ by at most 1).

    Bin 1 = lowest difficulty codes, bin n = highest.  If your convention
    is that low code = hardest, then bin 1 = hard, bin n = easy; rename in
    the figures if you prefer.

    Returns {level: bin} or None if fewer than n_bins valid levels.
    """
    counts       = rat_df[DIFFICULTY_NM1_COL].value_counts().sort_index()
    valid_levels = counts[counts >= min_trials].index.tolist()
    if len(valid_levels) < n_bins:
        return None
    splits = np.array_split(sorted(valid_levels), n_bins)
    return {lv: bin_idx + 1
            for bin_idx, group in enumerate(splits) for lv in group}


# =============================================================================
# Joint session bootstrap — one resampled session set feeds every bin
# =============================================================================

def _two_sided_bootstrap_p(samples):
    """Equal-tail two-sided bootstrap p-value with a finite-sample correction."""
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    lower = (np.count_nonzero(x <= 0) + 1) / (len(x) + 1)
    upper = (np.count_nonzero(x >= 0) + 1) / (len(x) + 1)
    return float(min(1.0, 2.0 * min(lower, upper)))


def test_linear_slope(bin_data, bin_map):
    """
    Bootstrap test for a linear trend (tilt) of mu across n-1 difficulty,
    computed *within* each already-stored joint bootstrap iteration, so
    whatever covariance exists across bins (they share resampled sessions)
    and any non-normality in the per-bin mu distributions are carried
    through automatically -- no covariance matrix or normal-theory
    approximation needed, unlike a CI-derived-SE version.

    H0: slope = 0 (no net linear trend in PSE across previous-trial
    difficulty). This tests whether a trend exists, not whether it is
    *purely* linear.
    """
    bins = sorted(bin_data.keys())
    x = np.array([np.mean([lv for lv, b in bin_map.items() if b == bn])
                  for bn in bins])
    X = np.column_stack([np.ones_like(x), x])

    def _slope(mu_vec, w):
        W = np.diag(w)
        beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ mu_vec)
        return beta[1]

    mu_hat = np.array([bin_data[bn]['popt'][0] for bn in bins])
    se_hat = np.array([np.std(bin_data[bn]['param_boot'][:, 0], ddof=1)
                       for bn in bins])
    w = 1.0 / se_hat**2   # fixed weights from the full-data SEs, reused
                          # across bootstrap iterations for stability

    slope_hat = _slope(mu_hat, w)
    n_boot = bin_data[bins[0]]['param_boot'].shape[0]
    slope_boot = np.array([
        _slope(np.array([bin_data[bn]['param_boot'][b, 0] for bn in bins]), w)
        for b in range(n_boot)
    ])

    lo, hi = np.percentile(slope_boot, [2.5, 97.5])
    return {
        'slope_hat': float(slope_hat),
        'slope_ci_lo': float(lo),
        'slope_ci_hi': float(hi),
        'slope_excl_0': bool(not (lo <= 0 <= hi)),
        'p_boot_slope': _two_sided_bootstrap_p(slope_boot),
        'n_boot_valid': int(n_boot),
    }


def bootstrap_bins_joint(df_modality, bins, n_boot=N_BOOT, seed=0):
    """
    Fit every bin to the full data, then jointly bootstrap sessions.

    In bootstrap iteration b the same sampled session multiset is used for
    every difficulty bin. An iteration is retained only when every bin fit
    succeeds, so parameter rows are aligned and may be subtracted directly.
    """
    bins = sorted(bins)
    bin_data = {}

    for lv in bins:
        sub = df_modality[df_modality['difficulty_n-1_bin'] == lv].copy()
        popt = fit_psychometric(sub[EVIDENCE_COL].values, sub[REP_COL].values)
        bin_data[lv] = {
            'popt': popt,
            'param_boot': None,
            'n_trials': len(sub),
            'n_sessions': sub[SESSION_COL].nunique(),
            'df': sub,
        }

    failed_full = [lv for lv in bins if bin_data[lv]['popt'] is None]
    if failed_full:
        return {
            'bin_data': bin_data,
            'ci_valid': False,
            'n_valid': 0,
            'n_requested': n_boot,
            'reason': f"full-data fit failed for bins {failed_full}",
        }

    too_few_trials = [lv for lv in bins
                      if bin_data[lv]['n_trials'] < MIN_TRIALS_PER_BIN]
    too_few_sessions = [lv for lv in bins
                        if bin_data[lv]['n_sessions'] < MIN_SESSIONS]
    if too_few_trials or too_few_sessions:
        pieces = []
        if too_few_trials:
            pieces.append(f"too few trials in bins {too_few_trials}")
        if too_few_sessions:
            pieces.append(f"too few sessions in bins {too_few_sessions}")
        return {
            'bin_data': bin_data,
            'ci_valid': False,
            'n_valid': 0,
            'n_requested': n_boot,
            'reason': '; '.join(pieces),
        }

    session_groups = {
        session: group for session, group in df_modality.groupby(SESSION_COL, sort=False)
    }
    sessions = np.asarray(list(session_groups.keys()), dtype=object)
    if len(sessions) < MIN_SESSIONS:
        return {
            'bin_data': bin_data,
            'ci_valid': False,
            'n_valid': 0,
            'n_requested': n_boot,
            'reason': f"only {len(sessions)} sessions in rat-modality cell",
        }

    rng = np.random.default_rng(seed)
    boot_by_bin = {lv: [] for lv in bins}

    for _ in range(n_boot):
        sampled_sessions = rng.choice(sessions, size=len(sessions), replace=True)
        boot_df = pd.concat(
            [session_groups[s] for s in sampled_sessions], ignore_index=True
        )

        fitted = {}
        iteration_valid = True
        for lv in bins:
            boot_bin = boot_df[boot_df['difficulty_n-1_bin'] == lv]
            p = fit_psychometric(
                boot_bin[EVIDENCE_COL].values,
                boot_bin[REP_COL].values,
            )
            if p is None:
                iteration_valid = False
                break
            fitted[lv] = p

        if iteration_valid:
            for lv in bins:
                boot_by_bin[lv].append(fitted[lv])

    n_valid = len(boot_by_bin[bins[0]])
    min_valid = max(10, int(np.ceil(MIN_BOOT_VALID_FRACTION * n_boot)))
    ci_valid = n_valid >= min_valid
    if ci_valid:
        for lv in bins:
            bin_data[lv]['param_boot'] = np.asarray(boot_by_bin[lv], dtype=float)

    return {
        'bin_data': bin_data,
        'ci_valid': ci_valid,
        'n_valid': n_valid,
        'n_requested': n_boot,
        'reason': ('' if ci_valid else
                   f"only {n_valid}/{n_boot} joint bootstrap fits succeeded; "
                   f"required at least {min_valid}"),
    }


# =============================================================================
# Pairwise effects, multiplicity correction, and shape diagnostics
# =============================================================================

def compute_bin_differences(bin_data):
    """Compute full-data effects and aligned-bootstrap uncertainty for all pairs."""
    bins = sorted(lv for lv, d in bin_data.items() if d['param_boot'] is not None)
    results = {}

    for i, bin_a in enumerate(bins):
        for bin_b in bins[i + 1:]:
            popt_a = bin_data[bin_a]['popt']
            popt_b = bin_data[bin_b]['popt']
            boot_a = bin_data[bin_a]['param_boot']
            boot_b = bin_data[bin_b]['param_boot']

            d_mu = boot_b[:, 0] - boot_a[:, 0]
            d_sigma = boot_b[:, 1] - boot_a[:, 1]
            mu_lo, mu_hi = np.percentile(d_mu, [2.5, 97.5])
            si_lo, si_hi = np.percentile(d_sigma, [2.5, 97.5])

            results[(bin_a, bin_b)] = {
                'delta_mu_hat': float(popt_b[0] - popt_a[0]),
                'delta_mu_boot_median': float(np.median(d_mu)),
                'delta_mu_ci': (float(mu_lo), float(mu_hi)),
                'delta_mu_excl_0': bool(not (mu_lo <= 0 <= mu_hi)),
                'p_boot': _two_sided_bootstrap_p(d_mu),
                'p_fdr': np.nan,
                'fdr_reject': False,
                'delta_sigma_hat': float(popt_b[1] - popt_a[1]),
                'delta_sigma_boot_median': float(np.median(d_sigma)),
                'delta_sigma_ci': (float(si_lo), float(si_hi)),
                'delta_sigma_excl_0': bool(not (si_lo <= 0 <= si_hi)),
                'delta_mu_boot': d_mu,
                'delta_sigma_boot': d_sigma,
                'n_boot_valid': int(len(d_mu)),
            }
    return results


def compute_shape_diagnostics(bin_data):
    """Describe three-bin shape; no additional discovery test is defined here."""
    bins = sorted(bin_data)
    if len(bins) != 3 or any(bin_data[lv]['param_boot'] is None for lv in bins):
        return None

    hard, medium, easy = bins
    mu_h, mu_m, mu_e = [float(bin_data[lv]['popt'][0]) for lv in bins]
    b_h, b_m, b_e = [bin_data[lv]['param_boot'][:, 0] for lv in bins]

    d1_boot = b_m - b_h
    d2_boot = b_e - b_m
    total_boot = b_e - b_h
    q_boot = b_h - 2.0 * b_m + b_e
    valid_r = np.isfinite(total_boot) & (np.abs(total_boot) > 1e-8)
    r_boot = d1_boot[valid_r] / total_boot[valid_r]

    d1_hat = mu_m - mu_h
    d2_hat = mu_e - mu_m
    total_hat = mu_e - mu_h
    q_hat = mu_h - 2.0 * mu_m + mu_e
    r_hat = d1_hat / total_hat if abs(total_hat) > 1e-8 else np.nan

    def ci(x):
        lo, hi = np.percentile(x, [2.5, 97.5])
        return float(lo), float(hi)

    if total_hat > 0:
        both_same_direction = d1_hat >= 0 and d2_hat >= 0
    elif total_hat < 0:
        both_same_direction = d1_hat <= 0 and d2_hat <= 0
    else:
        both_same_direction = False

    return {
        'hard_bin': hard,
        'medium_bin': medium,
        'easy_bin': easy,
        'mu_hard_hat': mu_h,
        'mu_medium_hat': mu_m,
        'mu_easy_hat': mu_e,
        'delta1_hat': d1_hat,
        'delta1_ci': ci(d1_boot),
        'delta2_hat': d2_hat,
        'delta2_ci': ci(d2_boot),
        'delta_total_hat': total_hat,
        'delta_total_ci': ci(total_boot),
        'q_hat': q_hat,
        'q_ci': ci(q_boot),
        'r_hat': float(r_hat),
        'r_boot_median': float(np.median(r_boot)) if len(r_boot) else np.nan,
        'r_ci': ci(r_boot) if len(r_boot) else (np.nan, np.nan),
        'r_n_boot_valid': int(len(r_boot)),
        'medium_between_extremes': bool(min(mu_h, mu_e) <= mu_m <= max(mu_h, mu_e)),
        'both_steps_same_direction': bool(both_same_direction),
        'delta1_boot': d1_boot,
        'delta2_boot': d2_boot,
        'delta_total_boot': total_boot,
        'q_boot': q_boot,
        'r_boot': r_boot,
    }


def benjamini_hochberg(p_values, alpha=FDR_ALPHA):
    """Return BH-adjusted p-values and rejection flags, preserving input order."""
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    reject = np.zeros(p.shape, dtype=bool)
    valid = np.isfinite(p)
    if not valid.any():
        return q, reject

    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adjusted_ranked = ranked * m / np.arange(1, m + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_ranked
    q[valid] = adjusted
    reject[valid] = adjusted <= alpha
    return q, reject


def classify_bias(differences, primary_pair):
    """Create pointwise primary labels; FDR fields are filled cohort-wide later."""
    if primary_pair not in differences:
        raise KeyError(f"Primary pair {primary_pair} is unavailable")

    primary = differences[primary_pair]
    half_w = 0.5 * (primary['delta_mu_ci'][1] - primary['delta_mu_ci'][0])
    max_hw = max(0.5 * (d['delta_mu_ci'][1] - d['delta_mu_ci'][0])
                 for d in differences.values())
    pointwise = bool(primary['delta_mu_excl_0'])
    return {
        'primary_pair': primary_pair,
        'primary_delta_mu': primary['delta_mu_hat'],
        'primary_ci': primary['delta_mu_ci'],
        'primary_ci_halfwidth': float(half_w),
        'max_ci_halfwidth': float(max_hw),
        'primary_p_boot': primary['p_boot'],
        'primary_p_fdr': np.nan,
        'pointwise_detected': pointwise,
        'fdr_reject': False,
        'classification_pointwise': ('pointwise_detected' if pointwise
                                     else 'pointwise_not_detected'),
        'classification_fdr': 'pending',
        'any_pair_excl_0': any(d['delta_mu_excl_0']
                              for d in differences.values()),
    }


def apply_primary_fdr(all_results, alpha=FDR_ALPHA):
    """Apply BH-FDR once across all available primary rat × modality tests."""
    entries = []
    for rat, mods in all_results.items():
        for mod, result in mods.items():
            c = result['classification']
            entries.append((rat, mod, c['primary_p_boot']))

    q_values, rejected = benjamini_hochberg(
        [entry[2] for entry in entries], alpha=alpha
    )
    for (rat, mod, _), q_value, is_rejected in zip(entries, q_values, rejected):
        result = all_results[rat][mod]
        c = result['classification']
        pp = c['primary_pair']
        c['primary_p_fdr'] = float(q_value)
        c['fdr_reject'] = bool(is_rejected)
        c['classification_fdr'] = ('fdr_detected' if is_rejected
                                   else 'not_fdr_detected')
        result['differences'][pp]['p_fdr'] = float(q_value)
        result['differences'][pp]['fdr_reject'] = bool(is_rejected)

    return len(entries)


# =============================================================================
# Empirical points helper
# =============================================================================

def empirical_points(df_bin, n_points=N_EMP_POINTS):
    """P(repeat) and n at n_points evidence levels evenly spaced by quantile."""
    grp        = df_bin.groupby(EVIDENCE_COL)[REP_COL]
    p_rep      = grp.mean()
    n_trials   = grp.count()
    xs, ys, ns = p_rep.index.to_numpy(), p_rep.to_numpy(), n_trials.to_numpy()
    if len(xs) <= n_points:
        return xs, ys, ns
    targets = np.quantile(xs, np.linspace(0, 1, n_points))
    idx     = sorted(set(np.argmin(np.abs(xs - t)) for t in targets))
    return xs[idx], ys[idx], ns[idx]

# =============================================================================
# FAST DIAGNOSTIC VIEW — finer difficulty binning, no bootstrap, no stats
# =============================================================================
# Full-data MLE fit only, restricted to the cells that were significant at
# 3-bin resolution, so this runs in seconds instead of minutes.

FINE_BIN_TARGET_CELLS = [
    #(3,  2),   # visual         — pointwise significant, not FDR
    (4,  3),   # visuo-tactile  — FDR significant
    #(13, 2),   # visual         — pointwise significant, not FDR
    (16, 1),   # tactile        — FDR significant
    (16, 3),   # visuo-tactile  — FDR significant
]
# =============================================================================
# SECTION 2, FINAL ANALYSIS — fine-grained PSE curve with uncertainty
# =============================================================================
# Same descriptive question as the fast fine-binning view, but now with
# proper joint session-bootstrap CIs on each bin's PSE, so the reader can
# judge which steps exceed the noise band. No new hypothesis tests are
# introduced — this stays in the "Stage 2 descriptive" category, same as
# the shape diagnostics above. Restricted to the 5 cells that were
# pointwise/FDR significant in the primary 3-bin analysis.

FINE_N_BINS_FINAL = 9
FINE_FINAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "2.2_fine_binning_uncertainty")


def run_fine_binning_with_uncertainty(df, target_cells=FINE_BIN_TARGET_CELLS,
                                       n_bins=FINE_N_BINS_FINAL,
                                       n_boot=N_BOOT, seed_base=BOOTSTRAP_SEED):
    os.makedirs(FINE_FINAL_DIR, exist_ok=True)
    table_rows = []
    slope_rows = [] 

    for cell_idx, (rat, modality) in enumerate(target_cells):
        print(f"\n--- rat {rat}, modality {modality}, n_bins={n_bins}, "
              f"n_boot={n_boot} ---")
        sub = df[(df[RAT_COL] == rat) & (df[MODALITY_COL] == modality)].dropna(
            subset=[EVIDENCE_COL, REP_COL, SESSION_COL, DIFFICULTY_NM1_COL]
        ).copy()
        if sub.empty:
            print("  no trials found — check modality dtype")
            continue

        bin_map = bin_difficulty(sub, min_trials=MIN_TRIALS_PER_BIN, n_bins=n_bins)
        if bin_map is None:
            print(f"  fewer than {n_bins} difficulty levels with "
                  f">= {MIN_TRIALS_PER_BIN} trials")
            continue

        sub = sub.copy()
        sub['difficulty_n-1_bin'] = sub[DIFFICULTY_NM1_COL].map(bin_map)
        bins_sorted = sorted(sub['difficulty_n-1_bin'].dropna().unique())

        seed = int(np.random.SeedSequence(
            [seed_base, int(rat), int(cell_idx), n_bins]
        ).generate_state(1)[0])
        joint = bootstrap_bins_joint(sub, bins=bins_sorted, n_boot=n_boot, seed=seed)
        bin_data = joint['bin_data']

        if not joint['ci_valid']:
            print(f"  \u26a0 bootstrap not valid: {joint['reason']}")
            continue
        print(f"  joint bootstrap: {joint['n_valid']}/{joint['n_requested']} valid")

        slope = test_linear_slope(bin_data, bin_map)                        # <-- add
        slope_rows.append({'rat': rat, 'modality': modality, **slope})      # <-- add
        print(f"  linear slope: {slope['slope_hat']:+.4f}\u00b0 PSE/\u00b0 difficulty "
              f"[{slope['slope_ci_lo']:+.4f}, {slope['slope_ci_hi']:+.4f}], "
              f"p_boot = {slope['p_boot_slope']:.4g}")                      # <-- add

        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        cmap = plt.cm.viridis
        colors = {b: cmap(x) for b, x in
                  zip(bins_sorted, np.linspace(0.05, 0.95, len(bins_sorted)))}

        # ---------- Panel 1: curves only, no CI shading (kept clean) --------
        for b in bins_sorted:
            d = bin_data[b]
            levels = sorted(k for k, v in bin_map.items() if v == b)
            label = "/".join(f"{int(lv)}" for lv in levels) + "\u00b0"
            c = colors[b]
            axes[0].plot(X_GRID, psychometric(X_GRID, *d['popt']), color=c, lw=1.6,
                        label=f"{label} (n={d['n_trials']})")
            xs, ys, ns = empirical_points(d['df'])
            axes[0].scatter(xs, ys, s=ns * BUBBLE_SCALE, color=c, alpha=0.5,
                            edgecolors='white', linewidths=0.4)

        axes[0].axhline(0.5, color='k', ls='--', alpha=0.4)
        axes[0].axvline(0, color='k', ls='--', alpha=0.4)
        axes[0].set_xlim(-45, 45); axes[0].set_ylim(0, 1)
        axes[0].set_xlabel("Evidence for repetition [\u00b0]")
        axes[0].set_ylabel("P(repeat)")
        axes[0].set_title(f"Rat {rat} | mod {modality} \u2014 full-data fits, {n_bins} bins")
        axes[0].legend(fontsize=7, title="n-1 difficulty")

        # ---------- Panel 2: PSE vs. bin, with 95% joint-bootstrap CI --------
        pse_rows = []
        for b in bins_sorted:
            d = bin_data[b]
            mu_hat = d['popt'][0]
            mu_boot = d['param_boot'][:, 0]
            lo, hi = np.percentile(mu_boot, [2.5, 97.5])
            levels = sorted(k for k, v in bin_map.items() if v == b)
            label = "/".join(f"{int(lv)}" for lv in levels) + "\u00b0"
            pse_rows.append({'bin': b, 'label': label, 'mu_hat': mu_hat,
                              'ci_lo': lo, 'ci_hi': hi,
                              'n_trials': d['n_trials'], 'n_sessions': d['n_sessions'],
                              'n_boot_valid': len(mu_boot)})
            table_rows.append({'rat': rat, 'modality': modality, 'bin': int(b),
                                'bin_label': label, 'n_trials': d['n_trials'],
                                'n_sessions': d['n_sessions'],
                                'n_boot_valid': len(mu_boot),
                                'mu_hat': mu_hat, 'mu_ci_lo': lo, 'mu_ci_hi': hi})

        xs_b = [r['bin'] for r in pse_rows]
        ys_b = [r['mu_hat'] for r in pse_rows]
        err_lo = [r['mu_hat'] - r['ci_lo'] for r in pse_rows]
        err_hi = [r['ci_hi'] - r['mu_hat'] for r in pse_rows]
        axes[1].errorbar(xs_b, ys_b, yerr=[err_lo, err_hi], fmt='o-',
                         color='crimson', ecolor='crimson', elinewidth=1.3,
                         capsize=3, markersize=6)
        for r in pse_rows:
            axes[1].annotate(r['label'], (r['bin'], r['ci_hi']),
                             textcoords="offset points", xytext=(0, 5),
                             ha='center', fontsize=7)
        axes[1].axhline(0, color='gray', ls='--', alpha=0.5)
        axes[1].set_xlabel(f"difficulty bin (1 = hardest \u2026 {n_bins} = easiest)")
        axes[1].set_ylabel("PSE (\u03bc) [\u00b0], full-data estimate \u00b1 95% joint bootstrap CI")
        axes[1].set_title("PSE vs. n-1 difficulty, with uncertainty")

        stem = f"rat_{rat}_mod_{modality}_fine{n_bins}_uncertainty"
        fig.savefig(os.path.join(FINE_FINAL_DIR, f"{stem}.svg"),
                    dpi=150, bbox_inches='tight')
        for r in pse_rows:
            print(f"    bin {r['bin']} ({r['label']}): mu={r['mu_hat']:+.2f} "
                  f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}], "
                  f"n={r['n_trials']}, sessions={r['n_sessions']}")
        plt.close(fig)
        print(f"  saved {stem}.svg")

    if table_rows:
        out_df = pd.DataFrame(table_rows)
        table_path = os.path.join(FINE_FINAL_DIR, 'fine_binning_pse_with_ci.csv')
        out_df.to_csv(table_path, index=False, float_format='%.4f')
        print(f"\n\u2713 Table written to {table_path}")


    if slope_rows:                                                          # <-- add
        slope_df = pd.DataFrame(slope_rows)                                 # <-- add
        slope_path = os.path.join(FINE_FINAL_DIR,                           # <-- add
                                   'fine_binning_linear_slope_test.csv')     # <-- add
        slope_df.to_csv(slope_path, index=False, float_format='%.6f')       # <-- add
        print(f"\u2713 Slope-test table written to {slope_path}")           # <-- add

    return table_rows, slope_rows                                          # <-- change 


# =============================================================================
# Plotting — 2-panel per (rat, modality)
# =============================================================================
def plot_bin_curves_with_delta(bin_data, differences, classification,
                                rat_id, modality, bin_labels,bin_labels_short, savefig=True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)

    bins   = sorted(bin_data.keys())
    cmap   = plt.cm.viridis
    colors = {lv: cmap(np.linspace(0.1, 0.9, len(bins))[i])
              for i, lv in enumerate(bins)}

    # ---------- Panel 1 : curves per bin ----------
    ax = axes[0]
    curve_handles = []
    for lv in bins:
        d    = bin_data[lv]
        popt = d['popt']
        c    = colors[lv]
        label = bin_labels.get(lv, f"bin {lv}")

        if d['param_boot'] is not None:
            curves = np.array([psychometric(X_GRID, *p) for p in d['param_boot']])
            y_lo = np.percentile(curves,  2.5, axis=0)
            y_hi = np.percentile(curves, 97.5, axis=0)
            ax.fill_between(X_GRID, y_lo, y_hi, alpha=0.15, color=c, linewidth=0)
            ci_tag = ""
        else:
            ci_tag = " (no CI)"

        h, = ax.plot(X_GRID, psychometric(X_GRID, *popt),
                     color=c, lw=2,
                     label=f"{label}{ci_tag}  (n={d['n_trials']})")
        curve_handles.append(h)

        xs, ys, ns = empirical_points(d['df'])
        ax.scatter(xs, ys, s=ns * BUBBLE_SCALE, color=c, alpha=0.6,
                   edgecolors='white', linewidths=0.5, zorder=3)
        ax.axvline(popt[0], color=c, ls=':', lw=1.2, alpha=0.7)

    ax.axhline(0.5, color='k', ls='--', alpha=0.4)
    ax.axvline(0,   color='k', ls='--', alpha=0.4)
    ax.set_xlim(-45, 45)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Evidence for repetition [°]")
    ax.set_ylabel("P(repeat)")
    ax.set_title(f"Rat {rat_id}  |  {modality}\n"
                 f"P(repeat) fits by difficulty n-1 bin")

    leg_curves = ax.legend(handles=curve_handles, title="difficulty n-1",
                           fontsize=9, title_fontsize=9, loc='upper left')
    ax.add_artist(leg_curves)

    bubble_handles = [
        plt.scatter([], [], s=n * BUBBLE_SCALE, color='grey', alpha=0.6,
                    edgecolors='white', linewidths=0.5, label=f'n={n}')
        for n in (20, 50, 100)
    ]
    ax.legend(handles=bubble_handles, title="trials / level",
              fontsize=8, title_fontsize=8, loc='lower right', framealpha=0.7)

    # ---------- Panel 2 : ΔPSE per pair ----------
    ax     = axes[1]
    pairs  = list(differences.keys())
    xpos   = np.arange(len(pairs))
    primary_pair = classification['primary_pair']

    # data extent for ylim; include 0 so the reference line is always visible
    all_lo = [differences[p]['delta_mu_ci'][0] for p in pairs] + [0.0]
    all_hi = [differences[p]['delta_mu_ci'][1] for p in pairs] + [0.0]
    data_min, data_max = min(all_lo), max(all_hi)
    data_range = max(data_max - data_min, 1.0)   # avoid divide-by-zero / tiny range

    for i, pair in enumerate(pairs):
        d      = differences[pair]
        estimate = d['delta_mu_hat']
        lo, hi = d['delta_mu_ci']
        is_primary = (pair == primary_pair)

        color = 'crimson' if is_primary else 'steelblue'
        lw    = 3 if is_primary else 2
        # Draw the CI and full-data estimate separately. This remains valid even
        # in the rare case that a skewed percentile CI does not contain the MLE.
        ax.vlines(i, lo, hi, color=color, lw=lw)
        ax.hlines([lo, hi], i - 0.09, i + 0.09, color=color, lw=lw)
        ax.plot(i, estimate, 'o', color=color, markersize=10)

        mark    = ' *' if d['delta_mu_excl_0'] else ''
        y_annot = hi + 0.04 * data_range
        ax.text(i, y_annot,
                f"{estimate:+.2f}°{mark}\n[{lo:+.2f}, {hi:+.2f}]",
                ha='center', va='bottom', fontsize=8.5,
                bbox=dict(facecolor='white', edgecolor='none',
                          alpha=0.7, pad=1.5))

    # explicit limits: headroom above for annotations, padding on sides
    ax.set_ylim(data_min - 0.10 * data_range,
                data_max + 0.32 * data_range)
    ax.set_xlim(-0.6, len(pairs) - 0.4)

    ax.axhline(0, color='gray', ls='--', alpha=0.6)
    ax.set_xticks(xpos)
    ax.set_xticklabels(
        [f"{bin_labels_short.get(a, f'bin {a}')} → {bin_labels_short.get(b, f'bin {b}')}"
         + ('\n(primary)' if (a, b) == primary_pair else '')
         for (a, b) in pairs],
        fontsize=9,
    )
    ax.set_ylabel("ΔPSE  (μ_bin_b − μ_bin_a) [°]")
    pointwise_txt = ('pointwise CI excludes 0' if classification['pointwise_detected']
                     else 'pointwise CI includes 0')
    fdr_txt = ('FDR significant' if classification['fdr_reject']
               else 'not FDR significant')
    color_txt = 'crimson' if classification['fdr_reject'] else 'seagreen'
    ax.set_title(
        f"ΔPSE: full-data estimate + joint-bootstrap 95% CI\n"
        f"Primary: {pointwise_txt}; {fdr_txt} "
        f"(q={classification['primary_p_fdr']:.4g})",
        color=color_txt,
    )

    if savefig:
        os.makedirs(FIGURE_DIR, exist_ok=True)
        stem = f"rat_{rat_id}_{modality}_perceptual_bias"
        for ext in ('png', 'svg'):
            path = os.path.join(FIGURE_DIR, f"{stem}.{ext}")
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"      ✓ Figure saved: {path}")
        plt.close(fig)

# =============================================================================
# Group-level summary
# =============================================================================

def summarise_bias(all_results):
    rows = []
    for rat, mods in all_results.items():
        for mod, r in mods.items():
            c   = r['classification']
            lo, hi = c['primary_ci']
            rows.append({
                'rat':                rat,
                'modality':           mod,
                'primary_pair':       f"{c['primary_pair'][0]}→{c['primary_pair'][1]}",
                'ΔPSE_primary':       c['primary_delta_mu'],
                'CI_primary':         f"[{lo:+.2f}, {hi:+.2f}]",
                'CI_halfwidth':       c['primary_ci_halfwidth'],
                'p_boot':             c['primary_p_boot'],
                'p_fdr':              c['primary_p_fdr'],
                'pointwise_detected': c['pointwise_detected'],
                'fdr_reject':         c['fdr_reject'],
                'any_pair_excl_0':    c['any_pair_excl_0'],
            })
    if not rows:
        print("  (no results to summarise)")
        return None

    df_out = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("Per-(rat × modality) primary perceptual-bias inference")
    print("=" * 90)
    print(df_out.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nPointwise and FDR-controlled counts by modality:")
    for mod in sorted(df_out['modality'].unique()):
        sub    = df_out[df_out['modality'] == mod]
        n_point = int(sub['pointwise_detected'].sum())
        n_fdr = int(sub['fdr_reject'].sum())
        n_tot  = len(sub)
        print(f"  {mod:<15}: pointwise {n_point:>2}/{n_tot}; "
              f"FDR {n_fdr:>2}/{n_tot}; "
              f"— median CI half-width across nulls: "
              f"{sub[~sub['pointwise_detected']]['CI_halfwidth'].median():.2f}°"
              if (~sub['pointwise_detected']).any() else
              f"  {mod:<15}: pointwise {n_point:>2}/{n_tot}; FDR {n_fdr:>2}/{n_tot}")

    n_total = len(df_out)
    n_point = int(df_out['pointwise_detected'].sum())
    n_fdr = int(df_out['fdr_reject'].sum())
    print(f"\nOverall: pointwise {n_point}/{n_total}; "
          f"BH-FDR(q={FDR_ALPHA:.2f}) {n_fdr}/{n_total}")
    if n_total != 36:
        print(f"  ⚠ FDR family contains {n_total} valid primary tests, not 36.")
    print("=" * 90)
    return df_out

# =============================================================================
# Table export — LaTeX (booktabs) + CSV
# =============================================================================


def _fmt_pm(x):
    """Signed float with 2 decimals, e.g. '+0.32', '−1.05'."""
    return f"{x:+.2f}".replace('-', '−')


def _df_to_booktabs(df, column_format, header_labels, caption, label):
    """
    Minimal, dependency-free booktabs LaTeX table.
    Requires \\usepackage{booktabs} in the preamble.
    """
    n_cols = len(df.columns)
    assert len(column_format) >= n_cols
    assert len(header_labels) == n_cols

    lines = [
        r'\begin{table}[htbp]',
        r'  \centering',
        r'  \caption{' + caption + r'}',
        r'  \label{' + label + r'}',
        r'  \begin{tabular}{' + column_format + r'}',
        r'    \toprule',
        '    ' + ' & '.join(header_labels) + r' \\',
        r'    \midrule',
    ]
    for _, row in df.iterrows():
        lines.append('    ' + ' & '.join(str(v) for v in row) + r' \\')
    lines += [
        r'    \bottomrule',
        r'  \end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


def export_tables(all_results, analysis_df, qc_rows, bin_mapping_rows):
    """
    Export complete machine-readable results and compact LaTeX tables.

    The compressed NPZ archive retains the distributions themselves, so later
    summaries need not rerun thousands of psychometric fits.
    """
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(BOOTSTRAP_DIR, exist_ok=True)

    # ---- REPRODUCIBILITY METADATA ---------------------------------------
    metadata = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'analysis_script': os.path.basename(os.path.abspath(__file__)),
        'n_trials_in_final_filtered_dataset': int(len(analysis_df)),
        'rats_requested': ('all' if TARGET_RATS is None
                           else [int(r) for r in TARGET_RATS]),
        'rats_in_final_filtered_dataset': [
            int(r) for r in sorted(analysis_df[RAT_COL].dropna().unique())
        ],
        'modalities': [str(m) for m in MODALITIES],
        'column_names': {
            'rat': RAT_COL,
            'session': SESSION_COL,
            'modality': MODALITY_COL,
            'current_angle': ANGLE_COL,
            'previous_difficulty': DIFFICULTY_NM1_COL,
            'evidence_for_repetition': EVIDENCE_COL,
            'repeat_response': REP_COL,
            'previous_outcome': STATE_COL,
        },
        'filters': {
            'current_angle_range_inclusive': [0, 90],
            'previous_angle_range_inclusive': [0, 90],
            'previous_difficulty_range_inclusive': [-46, 45],
            'previous_trial_correct_only': True,
            'ambiguous_previous_difficulty_zero_excluded': True,
            'training_filter_method': 'criterion',
            'training_filter_criterion': 0.80,
            'training_filter_min_consecutive': 2,
        },
        'fit': {
            'model': ('four_parameter_cumulative_gaussian: '
                      'gamma + (1-gamma-lapse)*Phi((x-mu)/sigma)'),
            'estimation': 'Bernoulli maximum likelihood (L-BFGS-B)',
            'parameter_order': ['mu', 'sigma', 'gamma', 'lapse'],
            'parameter_bounds': {
                'mu': [-45, 45],
                'sigma': [1, 50],
                'gamma': [0, 0.4],
                'lapse': [0, 0.4],
            },
            'starting_values': [0.0, 15.0, 0.05, 0.05],
        },
        'binning': {
            'n_difficulty_bins': int(N_DIFFICULTY_BINS),
            'method': 'sorted valid difficulty levels split with numpy.array_split',
            'minimum_trials_per_original_difficulty_level': int(MIN_TRIALS_PER_BIN),
            'semantic_bin_names': {str(k): v for k, v in BIN_NAMES.items()},
        },
        'bootstrap': {
            'unit': 'session',
            'joint_across_difficulty_bins': True,
            'n_requested': int(N_BOOT),
            'seed': int(BOOTSTRAP_SEED),
            'minimum_sessions': int(MIN_SESSIONS),
            'minimum_valid_fraction': float(MIN_BOOT_VALID_FRACTION),
            'confidence_interval_percentiles': [2.5, 97.5],
        },
        'multiple_comparisons': {
            'method': 'Benjamini-Hochberg',
            'alpha': float(FDR_ALPHA),
            'family': 'available primary extreme-bin contrasts across rat x modality',
        },
        'plotting_defaults_used_during_run': {
            'x_grid_min': float(X_GRID.min()),
            'x_grid_max': float(X_GRID.max()),
            'x_grid_n': int(len(X_GRID)),
            'n_empirical_points_displayed': int(N_EMP_POINTS),
            'bubble_scale': float(BUBBLE_SCALE),
        },
    }
    with open(os.path.join(TABLE_DIR, 'analysis_metadata.json'), 'w',
              encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # ---- QC + EXACT DIFFICULTY-BIN MAPPING ------------------------------
    pd.DataFrame(qc_rows).to_csv(
        os.path.join(TABLE_DIR, 'analysis_qc.csv'), index=False)
    pd.DataFrame(bin_mapping_rows).to_csv(
        os.path.join(TABLE_DIR, 'difficulty_bin_mapping.csv'), index=False)

    # ---- FULL EMPIRICAL COUNTS ------------------------------------------
    # Session-level aggregation is compact but preserves everything needed
    # to reconstruct empirical psychometric points at any desired resolution.
    empirical_rows = []
    for rat, mods in all_results.items():
        for mod, r in mods.items():
            for lv, d in sorted(r['bin_data'].items()):
                grouped = (
                    d['df']
                    .groupby([SESSION_COL, DIFFICULTY_NM1_COL, EVIDENCE_COL],
                             dropna=False)[REP_COL]
                    .agg(n_trials='size', n_repeat='sum')
                    .reset_index()
                )
                grouped['p_repeat'] = grouped['n_repeat'] / grouped['n_trials']
                for _, row in grouped.iterrows():
                    empirical_rows.append({
                        'rat': int(rat),
                        'modality': str(mod),
                        'bin': int(lv),
                        'bin_label': r['bin_labels'].get(lv, str(lv)),
                        'session': row[SESSION_COL],
                        'difficulty_n_minus_1': row[DIFFICULTY_NM1_COL],
                        'evidence_for_repetition': row[EVIDENCE_COL],
                        'n_trials': int(row['n_trials']),
                        'n_repeat': int(row['n_repeat']),
                        'p_repeat': float(row['p_repeat']),
                    })
    pd.DataFrame(empirical_rows).to_csv(
        os.path.join(TABLE_DIR, 'empirical_points_full.csv'),
        index=False, float_format='%.8g')

    # ---- SUMMARY ---------------------------------------------------------
    summary_rows = []
    for rat, mods in all_results.items():
        for mod, r in mods.items():
            c        = r['classification']
            lo, hi   = c['primary_ci']
            pp       = c['primary_pair']
            bl       = r.get('bin_labels', {})
            primary  = r['differences'][pp]
            pp_label = f"{bl.get(pp[0], pp[0])} → {bl.get(pp[1], pp[1])}"
            summary_rows.append({
                'rat':                       int(rat),
                'modality':                  str(mod),
                'primary_pair':              pp_label,
                'delta_pse_hat':             c['primary_delta_mu'],
                'delta_pse_boot_median':     primary['delta_mu_boot_median'],
                'ci_lo':                     lo,
                'ci_hi':                     hi,
                'ci_halfwidth':              c['primary_ci_halfwidth'],
                'p_boot':                    c['primary_p_boot'],
                'p_fdr_bh':                  c['primary_p_fdr'],
                'pointwise_ci_excludes_0':   c['pointwise_detected'],
                'fdr_reject':                c['fdr_reject'],
                'n_boot_valid':              primary['n_boot_valid'],
                'any_pair_ci_excludes_0':    bool(c['any_pair_excl_0']),
            })
    summary_raw = pd.DataFrame(summary_rows)
    summary_raw.to_csv(os.path.join(TABLE_DIR, 'summary_perceptual_bias.csv'),
                        index=False, float_format='%.4f')

    # LaTeX-formatted view
    summary_tex = pd.DataFrame({
        'Rat':               summary_raw['rat'],
        'Modality':          summary_raw['modality'],
        'Primary pair':      summary_raw['primary_pair'],
        r'$\Delta$PSE (°)':  summary_raw['delta_pse_hat'].map(_fmt_pm),
        r'95\% CI':          [f"[{_fmt_pm(lo)}, {_fmt_pm(hi)}]"
                              for lo, hi in zip(summary_raw['ci_lo'],
                                                 summary_raw['ci_hi'])],
        r'$p_{boot}$':       summary_raw['p_boot'].map(lambda v: f"{v:.4g}"),
        r'$p_{FDR}$':        summary_raw['p_fdr_bh'].map(lambda v: f"{v:.4g}"),
        'FDR reject':        summary_raw['fdr_reject'].map(lambda v: 'yes' if v else 'no'),
    })
    tex = _df_to_booktabs(
        summary_tex,
        column_format='rlllllll',
        header_labels=list(summary_tex.columns),
        caption=(r"Primary per-rat $\times$ modality perceptual-bias results. "
                 r"Point estimates are full-data MLE differences; intervals "
                 r"come from the joint session bootstrap. $p_{FDR}$ is the "
                 r"Benjamini--Hochberg adjusted value across primary contrasts."),
        label='tab:perceptual_bias_summary',
    )
    with open(os.path.join(TABLE_DIR, 'summary_perceptual_bias.tex'), 'w',
              encoding='utf-8') as f:
        f.write(tex + '\n')

    # ---- DETAILS ---------------------------------------------------------
    detail_rows = []
    for rat, mods in all_results.items():
        for mod, r in mods.items():
            bl        = r.get('bin_labels', {})
            prim_pair = r['classification']['primary_pair']
            for pair, d in r['differences'].items():
                a, b = pair
                detail_rows.append({
                    'rat':                int(rat),
                    'modality':           str(mod),
                    'pair':               f"{bl.get(a, a)} → {bl.get(b, b)}",
                    'is_primary':         (pair == prim_pair),
                    'delta_pse_hat':      d['delta_mu_hat'],
                    'delta_pse_boot_median': d['delta_mu_boot_median'],
                    'delta_pse_ci_lo':    d['delta_mu_ci'][0],
                    'delta_pse_ci_hi':    d['delta_mu_ci'][1],
                    'delta_pse_excl_0':   bool(d['delta_mu_excl_0']),
                    'p_boot_unadjusted':  d['p_boot'],
                    'p_fdr_bh_primary_only': d['p_fdr'],
                    'fdr_reject_primary_only': d['fdr_reject'],
                    'delta_sigma_hat':    d['delta_sigma_hat'],
                    'delta_sigma_boot_median': d['delta_sigma_boot_median'],
                    'delta_sigma_ci_lo':  d['delta_sigma_ci'][0],
                    'delta_sigma_ci_hi':  d['delta_sigma_ci'][1],
                    'delta_sigma_excl_0': bool(d['delta_sigma_excl_0']),
                    'n_boot_valid':       d['n_boot_valid'],
                })
    detail_raw = pd.DataFrame(detail_rows)
    detail_raw.to_csv(os.path.join(TABLE_DIR, 'details_perceptual_bias.csv'),
                       index=False, float_format='%.4f')

    detail_tex = pd.DataFrame({
        'Rat':                    detail_raw['rat'],
        'Modality':               detail_raw['modality'],
        'Pair':                   detail_raw['pair'] + detail_raw['is_primary'].map(
                                    lambda p: r' $\star$' if p else ''),
        r'$\Delta$PSE (°)':       detail_raw['delta_pse_hat'].map(_fmt_pm),
        r'95\% CI $\Delta$PSE':   [f"[{_fmt_pm(lo)}, {_fmt_pm(hi)}]"
                                    for lo, hi in zip(detail_raw['delta_pse_ci_lo'],
                                                       detail_raw['delta_pse_ci_hi'])],
        r'$\Delta\sigma$ (°)':    detail_raw['delta_sigma_hat'].map(_fmt_pm),
        r'95\% CI $\Delta\sigma$':[f"[{_fmt_pm(lo)}, {_fmt_pm(hi)}]"
                                    for lo, hi in zip(detail_raw['delta_sigma_ci_lo'],
                                                       detail_raw['delta_sigma_ci_hi'])],
    })
    tex = _df_to_booktabs(
        detail_tex,
        column_format='rllrlrl',
        header_labels=list(detail_tex.columns),
        caption=(r"Per bin-pair $\Delta$PSE and $\Delta\sigma$ with 95\% "
                 r"bootstrap CIs. The primary (extreme-bin) pair is marked "
                 r"with $\star$."),
        label='tab:perceptual_bias_details',
    )
    with open(os.path.join(TABLE_DIR, 'details_perceptual_bias.tex'), 'w',
              encoding='utf-8') as f:
        f.write(tex + '\n')

    # ---- PER-BIN FITS + BOUND DIAGNOSTICS -------------------------------
    bin_rows = []
    for rat, mods in all_results.items():
        for mod, r in mods.items():
            for lv, d in sorted(r['bin_data'].items()):
                boot = d['param_boot']
                popt = d['popt']
                ci_param = np.percentile(boot, [2.5, 97.5], axis=0)
                bin_rows.append({
                    'rat': int(rat),
                    'modality': str(mod),
                    'bin': int(lv),
                    'bin_label': r['bin_labels'].get(lv, str(lv)),
                    'n_trials': d['n_trials'],
                    'n_sessions': d['n_sessions'],
                    'n_boot_valid_joint': len(boot),
                    'mu_hat': popt[0],
                    'mu_ci_lo': ci_param[0, 0],
                    'mu_ci_hi': ci_param[1, 0],
                    'sigma_hat': popt[1],
                    'sigma_ci_lo': ci_param[0, 1],
                    'sigma_ci_hi': ci_param[1, 1],
                    'gamma_hat': popt[2],
                    'lapse_hat': popt[3],
                    'sigma_hat_at_lower_bound': bool(np.isclose(popt[1], 1.0, atol=1e-4)),
                    'sigma_hat_at_upper_bound': bool(np.isclose(popt[1], 50.0, atol=1e-4)),
                    'sigma_boot_fraction_lower_bound': float(np.mean(
                        np.isclose(boot[:, 1], 1.0, atol=1e-4))),
                    'sigma_boot_fraction_upper_bound': float(np.mean(
                        np.isclose(boot[:, 1], 50.0, atol=1e-4))),
                })
    bin_raw = pd.DataFrame(bin_rows)
    bin_raw.to_csv(os.path.join(TABLE_DIR, 'bin_fits_perceptual_bias.csv'),
                   index=False, float_format='%.6g')

    # ---- DESCRIPTIVE SHAPE DIAGNOSTICS ----------------------------------
    shape_rows = []
    for rat, mods in all_results.items():
        for mod, r in mods.items():
            s = r.get('shape_diagnostics')
            if s is None:
                continue
            c = r['classification']
            shape_rows.append({
                'rat': int(rat),
                'modality': str(mod),
                'pointwise_primary_detected': c['pointwise_detected'],
                'fdr_primary_reject': c['fdr_reject'],
                'delta_hard_to_medium_hat': s['delta1_hat'],
                'delta_hard_to_medium_ci_lo': s['delta1_ci'][0],
                'delta_hard_to_medium_ci_hi': s['delta1_ci'][1],
                'delta_medium_to_easy_hat': s['delta2_hat'],
                'delta_medium_to_easy_ci_lo': s['delta2_ci'][0],
                'delta_medium_to_easy_ci_hi': s['delta2_ci'][1],
                'delta_total_hat': s['delta_total_hat'],
                'delta_total_ci_lo': s['delta_total_ci'][0],
                'delta_total_ci_hi': s['delta_total_ci'][1],
                'q_curvature_hat': s['q_hat'],
                'q_curvature_ci_lo': s['q_ci'][0],
                'q_curvature_ci_hi': s['q_ci'][1],
                'r_first_step_share_hat': s['r_hat'],
                'r_first_step_share_boot_median': s['r_boot_median'],
                'r_first_step_share_ci_lo': s['r_ci'][0],
                'r_first_step_share_ci_hi': s['r_ci'][1],
                'r_n_boot_valid': s['r_n_boot_valid'],
                'medium_between_extremes': s['medium_between_extremes'],
                'both_steps_same_direction': s['both_steps_same_direction'],
            })
    shape_raw = pd.DataFrame(shape_rows)
    shape_raw.to_csv(os.path.join(TABLE_DIR, 'shape_diagnostics_all_cells.csv'),
                     index=False, float_format='%.6g')
    shape_raw[shape_raw['pointwise_primary_detected']].to_csv(
        os.path.join(TABLE_DIR, 'shape_diagnostics_pointwise_cells.csv'),
        index=False, float_format='%.6g')
    shape_raw[shape_raw['fdr_primary_reject']].to_csv(
        os.path.join(TABLE_DIR, 'shape_diagnostics_fdr_cells.csv'),
        index=False, float_format='%.6g')

    shape_selected = shape_raw[shape_raw['pointwise_primary_detected']].copy()
    if not shape_selected.empty:
        shape_tex = pd.DataFrame({
            'Rat': shape_selected['rat'],
            'Modality': shape_selected['modality'],
            r'$\Delta_1$ (°)': shape_selected['delta_hard_to_medium_hat'].map(_fmt_pm),
            r'$\Delta_2$ (°)': shape_selected['delta_medium_to_easy_hat'].map(_fmt_pm),
            r'$Q$ (°)': shape_selected['q_curvature_hat'].map(_fmt_pm),
            r'$R$': shape_selected['r_first_step_share_hat'].map(
                lambda v: 'NA' if not np.isfinite(v) else f"{v:.2f}"),
            'Monotonic': shape_selected['medium_between_extremes'].map(
                lambda v: 'yes' if v else 'no'),
            'FDR reject': shape_selected['fdr_primary_reject'].map(
                lambda v: 'yes' if v else 'no'),
        })
        tex = _df_to_booktabs(
            shape_tex,
            column_format='rlllllll',
            header_labels=list(shape_tex.columns),
            caption=(r"Descriptive shape diagnostics for cells whose pointwise "
                     r"primary 95\% bootstrap CI excluded zero. $\Delta_1$ is "
                     r"hard-to-medium, $\Delta_2$ is medium-to-easy, "
                     r"$Q=\Delta_2-\Delta_1$, and "
                     r"$R=\Delta_1/(\Delta_1+\Delta_2)$."),
            label='tab:perceptual_bias_shape_diagnostics',
        )
        with open(os.path.join(TABLE_DIR, 'shape_diagnostics_pointwise_cells.tex'),
                  'w', encoding='utf-8') as f:
            f.write(tex + '\n')

    # ---- BOOTSTRAP DISTRIBUTION ARCHIVE ---------------------------------
    archive = {}
    manifest_rows = []

    def safe_key(value):
        return re.sub(r'[^A-Za-z0-9_]+', '_', str(value)).strip('_')

    for rat, mods in all_results.items():
        for mod, r in mods.items():
            prefix = f"rat_{safe_key(rat)}__mod_{safe_key(mod)}"
            for lv, d in sorted(r['bin_data'].items()):
                key = f"{prefix}__bin_{safe_key(lv)}__params"
                archive[key] = d['param_boot']
                manifest_rows.append({
                    'key': key,
                    'rat': int(rat),
                    'modality': str(mod),
                    'quantity': 'bin_parameters_mu_sigma_gamma_lapse',
                    'bin_or_pair': str(lv),
                    'n_rows': len(d['param_boot']),
                })
            for pair, d in r['differences'].items():
                pair_tag = f"{safe_key(pair[0])}_to_{safe_key(pair[1])}"
                for suffix, quantity in (
                        ('delta_pse', 'delta_pse'), ('delta_sigma', 'delta_sigma')):
                    key = f"{prefix}__pair_{pair_tag}__{suffix}"
                    archive[key] = d[f"{suffix.replace('pse', 'mu')}_boot"]
                    manifest_rows.append({
                        'key': key,
                        'rat': int(rat),
                        'modality': str(mod),
                        'quantity': quantity,
                        'bin_or_pair': f"{pair[0]}->{pair[1]}",
                        'n_rows': len(archive[key]),
                    })
            s = r.get('shape_diagnostics')
            if s is not None:
                for field in ('delta1_boot', 'delta2_boot', 'delta_total_boot',
                              'q_boot', 'r_boot'):
                    key = f"{prefix}__shape__{field}"
                    archive[key] = s[field]
                    manifest_rows.append({
                        'key': key,
                        'rat': int(rat),
                        'modality': str(mod),
                        'quantity': field,
                        'bin_or_pair': 'three_bin_shape',
                        'n_rows': len(s[field]),
                    })

    archive_path = os.path.join(
        BOOTSTRAP_DIR, 'bootstrap_distributions_perceptual_bias.npz')
    np.savez_compressed(archive_path, **archive)
    pd.DataFrame(manifest_rows).to_csv(
        os.path.join(BOOTSTRAP_DIR, 'bootstrap_archive_manifest.csv'), index=False)

    print(f"\n✓ Tables written to {TABLE_DIR}")
    print(f"✓ Bootstrap distributions written to {archive_path}")


# =============================================================================
# MAIN LOOP — all rats × all modalities
# =============================================================================

def run_analysis(df, modalities, target_rats=TARGET_RATS):
    all_results = {}
    qc_rows = []
    bin_mapping_rows = []

    rats = sorted(df[RAT_COL].dropna().unique())
    if target_rats is not None:
        rats = [r for r in rats if r in target_rats]

    print(f"\n{'=' * 70}")
    print(f"Perceptual bias test — joint session bootstrap + primary BH-FDR")
    print(f"  Rats             : {len(rats)}  ({rats})")
    print(f"  Modalities       : {list(modalities)}")
    print(f"  Bootstrap iters  : {N_BOOT}")
    print(f"  Sigma bounds     : 1–50°")
    print(f"  Primary FDR      : Benjamini-Hochberg q={FDR_ALPHA:.2f}")
    print(f"  Difficulty bins  : {N_DIFFICULTY_BINS}")
    print(f"{'=' * 70}")

    for rat in rats:
        print(f"\n{'-' * 70}\nRat {rat}\n{'-' * 70}")
        rat_df = df[df[RAT_COL] == rat].dropna(
            subset=[EVIDENCE_COL, REP_COL, SESSION_COL, DIFFICULTY_NM1_COL]
        ).copy()

        bin_map = bin_difficulty(rat_df)
        if bin_map is None:
            print(f"  ⚠ Skipping rat {rat}: too few valid difficulty levels "
                  f"(need ≥ {N_DIFFICULTY_BINS} with ≥ {MIN_TRIALS_PER_BIN} trials)")
            for mod in modalities:
                qc_rows.append({
                    'rat': int(rat), 'modality': str(mod),
                    'status': 'skipped',
                    'reason': 'too few valid original difficulty levels',
                    'n_trials': int(len(rat_df)),
                    'n_sessions': int(rat_df[SESSION_COL].nunique()),
                    'n_bins': 0, 'n_boot_requested': int(N_BOOT),
                    'n_boot_valid': 0,
                })
            continue
        for difficulty_level, bin_index in sorted(bin_map.items()):
            bin_mapping_rows.append({
                'rat': int(rat),
                'difficulty_n_minus_1': int(difficulty_level),
                'bin': int(bin_index),
                'bin_name': BIN_NAMES.get(bin_index, f"bin {bin_index}"),
            })
        rat_df.loc[:, 'difficulty_n-1_bin'] = rat_df[DIFFICULTY_NM1_COL].map(bin_map)
        print(f"  Difficulty bin map: {bin_map}")

        all_results[rat] = {}

        for mod_index, mod in enumerate(modalities):
            print(f"\n  Modality: {mod}")
            sub_md = rat_df[rat_df[MODALITY_COL] == mod].copy()
            if len(sub_md) < MIN_TRIALS_PER_BIN * N_DIFFICULTY_BINS:
                print(f"    ⚠ Skipping: too few trials ({len(sub_md)})")
                qc_rows.append({
                    'rat': int(rat), 'modality': str(mod),
                    'status': 'skipped', 'reason': 'too few trials in cell',
                    'n_trials': int(len(sub_md)),
                    'n_sessions': int(sub_md[SESSION_COL].nunique()),
                    'n_bins': int(sub_md['difficulty_n-1_bin'].nunique()),
                    'n_boot_requested': int(N_BOOT), 'n_boot_valid': 0,
                })
                continue

            bins = sorted(sub_md['difficulty_n-1_bin'].dropna().unique())
            if len(bins) != N_DIFFICULTY_BINS:
                print(f"    ⚠ Skipping: found {len(bins)} usable bins; "
                      f"expected {N_DIFFICULTY_BINS}")
                qc_rows.append({
                    'rat': int(rat), 'modality': str(mod),
                    'status': 'skipped',
                    'reason': f'found {len(bins)} usable bins; expected {N_DIFFICULTY_BINS}',
                    'n_trials': int(len(sub_md)),
                    'n_sessions': int(sub_md[SESSION_COL].nunique()),
                    'n_bins': int(len(bins)),
                    'n_boot_requested': int(N_BOOT), 'n_boot_valid': 0,
                })
                continue

            seed = int(np.random.SeedSequence(
                [BOOTSTRAP_SEED, int(rat), int(mod_index)]
            ).generate_state(1)[0])
            joint = bootstrap_bins_joint(sub_md, bins=bins, seed=seed)
            bin_data = joint['bin_data']

            for lv in bins:
                d = bin_data[lv]
                if d['popt'] is None:
                    print(f"    bin {lv}: full-data fit failed")
                    continue
                mu, sigma, gamma, lapse = d['popt']
                print(f"    bin {lv}: n={d['n_trials']}, sessions={d['n_sessions']}, "
                      f"μ={mu:+.2f}°, σ={sigma:.2f}°, "
                      f"γ={gamma:.3f}, λ={lapse:.3f}")

            if not joint['ci_valid']:
                print(f"    ⚠ No joint-bootstrap inference: {joint['reason']}")
                qc_rows.append({
                    'rat': int(rat), 'modality': str(mod),
                    'status': 'skipped', 'reason': joint['reason'],
                    'n_trials': int(len(sub_md)),
                    'n_sessions': int(sub_md[SESSION_COL].nunique()),
                    'n_bins': int(len(bins)),
                    'n_boot_requested': int(joint['n_requested']),
                    'n_boot_valid': int(joint['n_valid']),
                })
                continue

            print(f"    joint bootstrap: {joint['n_valid']}/{joint['n_requested']} "
                  f"iterations valid")

            bin_labels_short = make_bin_labels_short(bin_map)
            bin_labels = make_bin_labels(bin_map)
            differences    = compute_bin_differences(bin_data)
            bins_sorted    = sorted(bin_data.keys())
            primary_pair   = (bins_sorted[0], bins_sorted[-1])   # extreme pair
            classification = classify_bias(differences, primary_pair=primary_pair)
            shape_diagnostics = compute_shape_diagnostics(bin_data)

            for pair, d in differences.items():
                mark = ' *' if d['delta_mu_excl_0'] else ''
                tag  = ' (primary)' if pair == primary_pair else ''
                print(f"    ΔPSE bin {pair[0]}→{pair[1]}: "
                      f"{d['delta_mu_hat']:+.2f}° "
                      f"[{d['delta_mu_ci'][0]:+.2f}, "
                      f"{d['delta_mu_ci'][1]:+.2f}], "
                      f"p_boot={d['p_boot']:.4g}{mark}{tag}")

            all_results[rat][mod] = {
                'bin_data':       bin_data,
                'differences':    differences,
                'classification': classification,
                'shape_diagnostics': shape_diagnostics,
                'bin_labels':     bin_labels,
                'bin_labels_short': bin_labels_short,
                'n_boot_requested': joint['n_requested'],
                'n_boot_valid': joint['n_valid'],
            }
            qc_rows.append({
                'rat': int(rat), 'modality': str(mod),
                'status': 'success', 'reason': '',
                'n_trials': int(len(sub_md)),
                'n_sessions': int(sub_md[SESSION_COL].nunique()),
                'n_bins': int(len(bins)),
                'n_boot_requested': int(joint['n_requested']),
                'n_boot_valid': int(joint['n_valid']),
            })

    n_primary = apply_primary_fdr(all_results)
    print(f"\nApplied BH-FDR at q={FDR_ALPHA:.2f} across "
          f"{n_primary} available primary contrasts.")

    # Plot only after cohort-wide correction, so each title can report FDR.
    for rat, mods in all_results.items():
        for mod, r in mods.items():
            c = r['classification']
            print(f"  Rat {rat}, modality {mod}: "
                  f"ΔPSE={c['primary_delta_mu']:+.2f}°, "
                  f"p={c['primary_p_boot']:.4g}, "
                  f"p_FDR={c['primary_p_fdr']:.4g}, "
                  f"reject={c['fdr_reject']}")
            plot_bin_curves_with_delta(
                r['bin_data'], r['differences'], c,
                rat_id=rat, modality=mod,
                bin_labels=r['bin_labels'],
                bin_labels_short=r['bin_labels_short'],
            )

    summary_df = summarise_bias(all_results)
    export_tables(all_results, df, qc_rows, bin_mapping_rows)
    return all_results, summary_df


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    for rat in sorted(df['rat'].dropna().unique()):
        rat_df = df[df['rat'] == rat].dropna(subset=['difficulty_n-1']).copy()
        bin_map = bin_difficulty(rat_df)
        print(f"\n--- Rat {int(rat)} ---")
        if bin_map is None:
            print("  skipped (too few difficulty levels)")
            continue
        print(f"  bin_map: {bin_map}")
        rat_df['bin'] = rat_df['difficulty_n-1'].map(bin_map)
        per_bin = rat_df.groupby(['bin', 'difficulty_n-1'])['angle_n-1'].agg(
            n_trials='count',
            angles=lambda s: sorted(s.dropna().unique().tolist())
        )
        print(per_bin)
    RUN_MODE = 'full'   # 'full' | 'fine_view' | 'fine_uncertainty'

    if RUN_MODE == 'full':
        results, summary = run_analysis(df, MODALITIES)
    elif RUN_MODE == 'fine_uncertainty':
        run_fine_binning_with_uncertainty(df, FINE_BIN_TARGET_CELLS,
                                           n_bins=FINE_N_BINS_FINAL, n_boot=N_BOOT)

