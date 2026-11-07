"""
Psychometric Curve Analysis — Choice Effects (harmonised)
=========================================================
Analyses the effect of n-1 choice on trial n choice,
conditioned on n-1 stimulus ≈ 45°.

Four analyses:
    1) Real 45° trials, split by n-1 choice (vertical vs horizontal)
    2) Fake 45° (from flanking angles), split by n-1 choice
    3) Fake 45°, split by n-1 choice, conditioned on n-1 rewarded
    4) Fake 45°, split by n-1 choice, conditioned on n-1 unrewarded

Then combines analyses 3 and 4 into three orthogonal contrasts of the
(prev-choice × prev-reward) 2×2 factorial:
    S  = Δμ_rew + Δμ_unrew         (main effect of previous choice)
    D  = Δμ_rew − Δμ_unrew         (reward × choice interaction)
    P  = μ_unrew_avg − μ_rew_avg   (main effect of previous outcome)

Harmonisation with task-performance and perceptual-bias scripts:
    ▸ Preprocessing: adds criterion training filter and 0–90° angle
      filter on both angle and angle_n-1.
    ▸ Fit bounds: μ ∈ [0, 90], σ ∈ [1, 90], γ, λ ∈ [0, 0.4].
    ▸ Fit acceptance: ≥ 30 trials, ≥ 4 unique angles, plus the existing
      per-angle floor of MIN_TRIALS_PER_ANGLE trials.
    ▸ Non-converged fits are rejected (return None), not kept with a warning.
    ▸ Deterministic seeding for both bootstrap_psychometric and compute_P.
    ▸ CIs suppressed (invalid) below MIN_VALID_BOOT successful iterations.

Bootstrap protocol note: trial-level balancing is kept here (unlike the
session-level resampling used by the other two psychometric scripts)
because the fake-45° 2×2 factorial cells contain a median of 1–2 trials
per session, so session-level resampling cannot identify the psychometric
fits within a typical resampled session set. See methods section.

Column name mapping:
    angle       → stimulus angle at trial n
    angle_n-1   → stimulus angle at trial n-1
    action      → choice at trial n       (1=vertical, 0=horizontal)
    action_n-1  → choice at trial n-1
    hitmiss_n-1 → reward at trial n-1     (1=rewarded, 0=not)
    rat         → subject ID
"""
import sys
sys.path.append(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from serial_dependence_analysis import SerialDependenceAnalyzer
from functions_Ale import filter_training_trials  # HARMONISED
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(BASE_DIR, "choice_effects_psychometric_results")

# Backward-compatible defaults. In normal use, run_requested_scopes() below
# creates scope-specific directories and passes them explicitly.
FIGURE_DIR = os.path.join(OUTPUT_ROOT, "pooled", "figures")
TABLE_DIR  = os.path.join(OUTPUT_ROOT, "pooled", "tables")
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(TABLE_DIR,  exist_ok=True)

# =============================================================================
# TUNABLE PARAMETERS
# =============================================================================

HISTORY_DEPTH        = 1
N_BOOTSTRAP          = 1000   # bootstrap iterations
N_RESAMPLES          = 10000  # independent resamples for S, D, P
X_PLOT               = np.linspace(0, 90, 200)
MIN_TRIALS_PER_ANGLE = 5      # per-angle floor inside fit_psychometric
MIN_TRIALS_TOTAL     = 30     # HARMONISED with task-performance
MIN_UNIQUE_ANGLES    = 4      # HARMONISED with task-performance
MIN_VALID_BOOT       = 10     # HARMONISED with task-performance/perceptual-bias
MINIMAL_FIT          = False

# Filter parameters — HARMONISED with perceptual-bias script
FILTER_KWARGS = dict(method='criterion', criterion=0.80, min_consec=2)

# Which level(s) to run.
#   "pooled"       -> one analysis per rat using all modalities together
#   "per_modality" -> separate tactile / visual / visuotactile analyses
#   "both"         -> run pooled first, then all three modalities
ANALYSIS_SCOPE = "per_modality"  # "pooled", "per_modality", or "both"

COL_MODALITY = "mod"
MODALITY_CODES = {
    "tactile":      1,
    "visual":       2,
    "visuotactile": 3,
}

# Analysis 3 only: which reward condition to condition on
REWARD_CONDITION = 1          # 1 = conditioned on rewarded, 0 = unrewarded

# Per-rat PSE flanking angles for fake 45°
PSE_FLANKS = {
    2:  [(35, 55), (40, 50)], 3:  [(40, 50)], 4:  [(40, 50)], 5:  [(40, 50)],
    6:  [(40, 50)], 7:  [(40, 50)], 9:  [(40, 50)], 10: [(40, 50)],
    11: [(40, 50)], 12: [(40, 50)], 13: [(40, 50)], 16: [(40, 50)],
}

# =============================================================================
# COLUMN NAMES
# =============================================================================

COL_STIM       = 'angle'
COL_STIM_NM1   = 'angle_n-1'
COL_CHOICE     = 'action'
COL_CHOICE_NM1 = 'action_n-1'
COL_REWARD     = 'hitmiss_n-1'
COL_RAT        = 'rat'

# =============================================================================
# LOAD + PREPROCESS  (HARMONISED)
# =============================================================================
an = SerialDependenceAnalyzer(history_depth=HISTORY_DEPTH, save_figures=False)
an.load_and_preprocess_data()
df = an.create_lagged_features()

# HARMONISED: apply criterion training filter (was missing entirely)
df = filter_training_trials(df, **FILTER_KWARGS)

# HARMONISED: apply angle filters on both current and previous trial
# (previous-trial filter is essential because this analysis conditions on angle_n-1)
df = df[(df[COL_STIM]     >= 0) & (df[COL_STIM]     <= 90)].copy()
df = df[(df[COL_STIM_NM1] >= 0) & (df[COL_STIM_NM1] <= 90)].copy()

# =============================================================================
# SECTION 2 — Core fitting functions  (HARMONISED bounds and acceptance)
# =============================================================================

def neg_log_likelihood(params, x, c):
    """
    Negative log-likelihood for cumulative Gaussian with lapse and guess rate.
    Parameter order: (mu, sigma, gamma, lapse)  — HARMONISED across scripts.
    Model: P(vertical) = gamma + (1 - gamma - lapse) * Φ((x - mu) / sigma)
    """
    mu, sigma, gamma, lapse = params
    p = gamma + (1 - gamma - lapse) * norm.cdf((x - mu) / sigma)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -np.sum(c * np.log(p) + (1 - c) * np.log(1 - p))


def neg_log_likelihood_minimal(params, x, c):
    """2-parameter version with gamma and lapse fixed at 0.05."""
    mu, sigma = params
    return neg_log_likelihood([mu, sigma, 0.05, 0.05], x, c)


def fit_psychometric(df, min_trials=MIN_TRIALS_PER_ANGLE, minimal_fit=MINIMAL_FIT):
    """
    Fit cumulative Gaussian with lapses to binary choice data using MLE.

    HARMONISED acceptance criteria:
        - drop angles with fewer than `min_trials` trials (per-angle floor)
        - after that: require ≥ MIN_TRIALS_TOTAL trials AND ≥ MIN_UNIQUE_ANGLES
          unique angles (matches task-performance)
        - reject fits where the optimizer did not converge (return None)

    HARMONISED bounds:
        μ ∈ [0, 90],  σ ∈ [1, 90],  γ, λ ∈ [0, 0.4]

    Returns dict {mu, sigma, gamma, lapse, success} or None if insufficient
    data or fit failed.
    """
    counts       = df.groupby(COL_STIM)[COL_CHOICE].count()
    valid_angles = counts[counts >= min_trials].index
    df           = df[df[COL_STIM].isin(valid_angles)]

    # HARMONISED: global acceptance floor
    if len(df) < MIN_TRIALS_TOTAL or df[COL_STIM].nunique() < MIN_UNIQUE_ANGLES:
        return None

    x = df[COL_STIM].values
    c = df[COL_CHOICE].values

    try:
        if minimal_fit:
            x0     = [45, 10]
            bounds = [(0, 90), (1, 90)]  # HARMONISED σ upper bound 50 → 90
            result = minimize(neg_log_likelihood_minimal, x0, args=(x, c),
                              bounds=bounds, method='L-BFGS-B')
            mu, sigma    = result.x
            gamma, lapse = 0.05, 0.05
        else:
            x0     = [45, 10, 0.05, 0.05]
            # HARMONISED: σ upper bound 50 → 90; γ, λ upper bound 0.1 → 0.4
            bounds = [(0, 90), (1, 90), (0, 0.4), (0, 0.4)]
            result = minimize(neg_log_likelihood, x0, args=(x, c),
                              bounds=bounds, method='L-BFGS-B')
            mu, sigma, gamma, lapse = result.x
    except (RuntimeError, ValueError, FloatingPointError):
        return None

    # HARMONISED: reject non-converged fits (was: warn but keep)
    if not result.success:
        return None

    return {'mu': mu, 'sigma': sigma, 'gamma': gamma,
            'lapse': lapse, 'success': result.success}


def eval_psychometric(x, mu, sigma, gamma, lapse):
    """Evaluate the psychometric function at points x given parameters."""
    return gamma + (1 - gamma - lapse) * norm.cdf((x - mu) / sigma)


# =============================================================================
# SECTION 3 — Trial preparation  (unchanged, seeding added downstream)
# =============================================================================

def prepare_trials(df, use_fake_45, split_by,
                   condition_on=None,
                   pse_flanks=[(40, 50)],   # was: pse_flanks=(40, 50)
                   rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if use_fake_45:
        sub_pools = []
        for low, high in pse_flanks:
            pool_low  = df[df[COL_STIM_NM1] == low]
            pool_high = df[df[COL_STIM_NM1] == high]
            n1 = min(len(pool_low), len(pool_high))
            if n1 == 0:
                continue                          # skip empty layers, don't abort
            sub_pools.append(pool_low.sample(n=n1, replace=False,
                                             random_state=rng.integers(0, 2**31)))
            sub_pools.append(pool_high.sample(n=n1, replace=False,
                                              random_state=rng.integers(0, 2**31)))
        if not sub_pools:
            return None, None
        df_45 = pd.concat(sub_pools, ignore_index=True)
    else:
        df_45 = df[df[COL_STIM_NM1] == 45].copy()

    if condition_on is not None:
        df_45 = df_45[df_45[condition_on['col']] == condition_on['val']]

    group_0 = df_45[df_45[split_by] == 0]
    group_1 = df_45[df_45[split_by] == 1]
    n2 = min(len(group_0), len(group_1))
    if n2 == 0:
        return None, None
    return (group_0.sample(n=n2, replace=True, random_state=rng.integers(0, 2**31)),
            group_1.sample(n=n2, replace=True, random_state=rng.integers(0, 2**31)))


# =============================================================================
# SECTION 4 — Bootstrap  (HARMONISED seeding + convergence handling)
# =============================================================================

def bootstrap_psychometric(df_rat, use_fake_45, split_by,
                            condition_on=None, pse_flanks=[(40, 50)],
                            n_bootstrap=N_BOOTSTRAP, minimal_fit=MINIMAL_FIT,
                            seed=0):
    """
    Run bootstrap to estimate parameter distributions for both groups.

    HARMONISED:
        - deterministic seeding via `seed` (was: no seed)
        - iterations with failed fits (including non-convergence) are skipped

    Returns
    -------
    dict with keys 'group_0' and 'group_1', each a DataFrame of shape
    (n_valid_iters, 4) with columns [mu, sigma, gamma, lapse], plus
    'ci_valid' (bool) indicating whether ≥ MIN_VALID_BOOT iterations
    succeeded.
    """
    rng       = np.random.default_rng(seed)
    params_0  = []
    params_1  = []
    n_skipped = 0

    for i in range(n_bootstrap):
        df_g0, df_g1 = prepare_trials(
            df_rat,
            use_fake_45=use_fake_45,
            split_by=split_by,
            condition_on=condition_on,
            pse_flanks= [(35, 55), (40, 50)],
            rng=rng,
        )

        if df_g0 is None or df_g1 is None:
            n_skipped += 1
            continue

        fit_0 = fit_psychometric(df_g0, minimal_fit=minimal_fit)
        fit_1 = fit_psychometric(df_g1, minimal_fit=minimal_fit)

        # HARMONISED: fit_psychometric now returns None on non-convergence,
        # so no separate success check is needed here.
        if fit_0 is None or fit_1 is None:
            n_skipped += 1
            continue

        fit_0.pop('success')
        fit_1.pop('success')

        params_0.append(fit_0)
        params_1.append(fit_1)

    if n_skipped > 0:
        print(f"  ℹ {n_skipped}/{n_bootstrap} iterations skipped")

    n_valid = len(params_0)
    ci_valid = n_valid >= MIN_VALID_BOOT
    if not ci_valid:
        print(f"  ⚠ Only {n_valid} valid iterations "
              f"(< MIN_VALID_BOOT={MIN_VALID_BOOT}) — CIs will be suppressed")

    return {
        'group_0':  pd.DataFrame(params_0),
        'group_1':  pd.DataFrame(params_1),
        'ci_valid': ci_valid,
        'n_valid':  n_valid,
    }


# =============================================================================
# SECTION 5 — Plotting (unchanged except for CI-validity gating)
# =============================================================================

def plot_psychometric(boot_results, df_45_full, rat_id,
                      split_by, condition_on=None,
                      labels=('group 0', 'group 1'),
                      analysis_label='',
                      x_plot=X_PLOT,
                      show_reference=False,
                      subset_name='pooled',
                      figure_dir=FIGURE_DIR):
    """
    Plot psychometric curves with bootstrap CI and Δμ.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors    = {'group_0': "#5C36F4", 'group_1': "#F32128"}
    label_map = {'group_0': labels[0], 'group_1': labels[1]}

    condition_str = (
        f"conditioned on {condition_on['col']}={condition_on['val']}"
        if condition_on is not None else 'no conditioning'
    )
    ci_valid = boot_results.get('ci_valid', True)

    # --- panel 1: psychometric curves ---
    for group in ['group_0', 'group_1']:
        params = boot_results[group]
        color  = colors[group]
        if len(params) == 0:
            continue

        curves = np.array([
            eval_psychometric(
                x_plot,
                params['mu'].values[i],
                params['sigma'].values[i],
                params['gamma'].values[i],
                params['lapse'].values[i]
            )
            for i in range(len(params))
        ])

        median_curve = np.median(curves, axis=0)
        axes[0].plot(x_plot, median_curve, color=color,
                     label=label_map[group], lw=2)

        if ci_valid:
            ci_low  = np.percentile(curves,  2.5, axis=0)
            ci_high = np.percentile(curves, 97.5, axis=0)
            axes[0].fill_between(x_plot, ci_low, ci_high,
                                 alpha=0.25, color=color)

    # reference curves on full unbalanced dataset (dotted)
    if show_reference:
        for group_val, color in [(0, colors['group_0']),
                                (1, colors['group_1'])]:
            ref_df = df_45_full[df_45_full[split_by] == group_val].copy()
            if condition_on is not None:
                ref_df = ref_df[ref_df[condition_on['col']] == condition_on['val']]
            ref_fit = fit_psychometric(ref_df)
            if ref_fit is not None:
                axes[0].plot(
                    x_plot,
                    eval_psychometric(x_plot, ref_fit['mu'], ref_fit['sigma'],
                                    ref_fit['gamma'], ref_fit['lapse']),
                    color=color, lw=1.5, linestyle='--', alpha=0.6
                )

    axes[0].axvline(45,  color='gray', linestyle=':', alpha=0.5)
    axes[0].axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    axes[0].set_xlabel('Stimulus angle (°)', fontsize=12)
    axes[0].set_ylabel('P(vertical)', fontsize=12)
    ref_note = ', dashed = full-data reference' if show_reference else ''
    ci_note  = '' if ci_valid else '  [CI suppressed: too few valid iterations]'
    axes[0].set_title(f'Rat {rat_id} [{subset_name}] — psychometric curves\n'
                      f'({condition_str}{ref_note}){ci_note}', fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(-0.05, 1.05)

    # --- panel 2: Δμ ---
    if len(boot_results['group_0']) > 0 and len(boot_results['group_1']) > 0:
        delta_mu  = (boot_results['group_1']['mu'].values -
                     boot_results['group_0']['mu'].values)
        med_delta = np.median(delta_mu)
        if ci_valid:
            ci_low_d  = np.percentile(delta_mu,  2.5)
            ci_high_d = np.percentile(delta_mu, 97.5)
            axes[1].errorbar(
                x=0, y=med_delta,
                yerr=[[med_delta - ci_low_d], [ci_high_d - med_delta]],
                fmt='o', color='black', capsize=8, markersize=10, lw=2
            )
            annot = (f'Δμ = {med_delta:.2f}°\n'
                     f'95% CI [{ci_low_d:.2f}, {ci_high_d:.2f}]')
        else:
            axes[1].plot(0, med_delta, 'o', color='black', markersize=10)
            annot = f'Δμ = {med_delta:.2f}°\n(CI suppressed)'
            ci_low_d, ci_high_d = med_delta, med_delta

        axes[1].axhline(0, color='gray', linestyle='--', alpha=0.6)
        axes[1].set_xlim(-1, 1)
        axes[1].set_xticks([])
        axes[1].set_ylabel(f'Δμ  ({labels[1]} − {labels[0]}) [°]', fontsize=12)
        axes[1].set_title(f'Rat {rat_id} — shift in perceptual boundary\n'
                          f'95% bootstrap CI', fontsize=11)

        data_min = min(ci_low_d,  0.0)
        data_max = max(ci_high_d, 0.0)
        data_range = max(data_max - data_min, 1.0)
        axes[1].set_ylim(data_min - 0.15 * data_range,
                         data_max + 0.35 * data_range)
        axes[1].text(
            0, data_max + 0.20 * data_range, annot,
            ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.85)
        )

        plt.suptitle(
            f'Rat {rat_id} [{subset_name}]  |  {analysis_label}  |  '
            f'{condition_str}\n{len(delta_mu)} bootstrap iterations',
            fontsize=12, y=1.01
        )

    stem = f'rat_{rat_id}_{analysis_label}'
    for ext in ('png', 'svg'):
        path = os.path.join(figure_dir, f'{stem}.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Figure saved: {path}")
    plt.close(fig)


# =============================================================================
# SECTION 6 — Analysis definitions
# =============================================================================

ANALYSES = [
    {
        'label':        'analysis_1_real45',
        'use_fake_45':  False,
        'split_by':     COL_CHOICE_NM1,
        'condition_on': None,
        'labels':       ('n-1 horizontal', 'n-1 vertical'),
        'description':  'Real 45° | split by n-1 choice',
    },
    {
        'label':        'analysis_2_fake45',
        'use_fake_45':  True,
        'split_by':     COL_CHOICE_NM1,
        'condition_on': None,
        'labels':       ('n-1 horizontal', 'n-1 vertical'),
        'description':  'Fake 45° | split by n-1 choice',
    },
    {
        'label':        'analysis_3_fake45_reward_controlled',
        'use_fake_45':  True,
        'split_by':     COL_CHOICE_NM1,
        'condition_on': {'col': COL_REWARD, 'val': REWARD_CONDITION},
        'labels':       ('n-1 horizontal', 'n-1 vertical'),
        'description':  f'Fake 45° | split by n-1 choice | n-1 reward={REWARD_CONDITION}',
    },
    {
        'label':        'analysis_4_fake45_reward_controlled_opposite',
        'use_fake_45':  True,
        'split_by':     COL_CHOICE_NM1,
        'condition_on': {'col': COL_REWARD, 'val': 1 - REWARD_CONDITION},
        'labels':       ('n-1 horizontal', 'n-1 vertical'),
        'description':  f'Fake 45° | split by n-1 choice | n-1 reward={1 - REWARD_CONDITION}',
    },
]


# =============================================================================
# Orthogonal contrasts (S, D) via independent resampling  (HARMONISED seeding)
# =============================================================================

def compute_S_and_D(delta_mu_rew, delta_mu_unrew,
                     n_samples=N_RESAMPLES, seed=0):
    """
    Build the sum (S) and difference (D) distributions via *independent*
    resampling of the two bootstrap distributions. Independence is the
    right assumption because the reward=1 and reward=0 fits use disjoint
    trials and independent bootstrap iterations — no pairing is justified.

    HARMONISED: seed now defaults to 0 rather than None, and is set from
    a per-rat value in run_analysis.
    """
    rng          = np.random.default_rng(seed)
    rew_paired   = rng.choice(delta_mu_rew,   size=n_samples, replace=True)
    unrew_paired = rng.choice(delta_mu_unrew, size=n_samples, replace=True)

    S = rew_paired + unrew_paired
    D = rew_paired - unrew_paired

    p_same_sign = float(np.mean(np.sign(rew_paired) == np.sign(unrew_paired)))

    return {
        'S':            S,
        'D':            D,
        'rew_paired':   rew_paired,
        'unrew_paired': unrew_paired,
        'p_same_sign':  p_same_sign,
    }


def classify_choice_pattern(delta_mu_rew, delta_mu_unrew, joint):
    """
    Independent tests of S (main effect of choice) and D (reward × choice
    interaction). Each is a 2×2 factorial contrast, orthogonal by construction,
    so each is tested against 0 on its own — no sequential gating.
    """
    rew_lo,   rew_hi   = np.percentile(delta_mu_rew,   [2.5, 97.5])
    unrew_lo, unrew_hi = np.percentile(delta_mu_unrew, [2.5, 97.5])
    S_lo,     S_hi     = np.percentile(joint['S'],     [2.5, 97.5])
    D_lo,     D_hi     = np.percentile(joint['D'],     [2.5, 97.5])

    return {
        'rew_excludes_0':   not (rew_lo   <= 0 <= rew_hi),
        'unrew_excludes_0': not (unrew_lo <= 0 <= unrew_hi),
        'S_excludes_0':     not (S_lo     <= 0 <= S_hi),
        'D_excludes_0':     not (D_lo     <= 0 <= D_hi),
        'rew_ci':           (rew_lo, rew_hi),
        'unrew_ci':         (unrew_lo, unrew_hi),
        'S_ci':             (S_lo, S_hi),
        'D_ci':             (D_lo, D_hi),
        'p_same_sign':      joint['p_same_sign'],
    }


def build_contrast_profile(choice_result, pure_reward_result=None):
    """Compact string listing which orthogonal contrasts exclude 0."""
    parts = []
    if choice_result['S_excludes_0']:                     parts.append('choice')
    if choice_result['D_excludes_0']:                     parts.append('interaction')
    if pure_reward_result and pure_reward_result['P_excludes_0']:
        parts.append('reward')
    return '+'.join(parts) if parts else 'null'


# =============================================================================
# Choice-distribution diagnostic + pure-reward test (P)  (HARMONISED seeding)
# =============================================================================

def report_choice_distribution(df_rat, rat_id, pse_flanks):
    """Print prev-choice distribution within each reward state at fake 45°."""
    flank_angles = [a for pair in pse_flanks for a in pair]   # [40, 50, 35, 55]
    df_45 = df_rat[df_rat[COL_STIM_NM1].isin(flank_angles)]
    print(f"\n  Choice distribution at fake 45°:")
    for rew_val, rew_label in [(1, 'rewarded'), (0, 'unrewarded')]:
        subset = df_45[df_45[COL_REWARD] == rew_val]
        n_total = len(subset)
        if n_total == 0:
            print(f"    prev-{rew_label:>10s}: no trials")
            continue
        n_v = (subset[COL_CHOICE_NM1] == 1).sum()
        n_h = (subset[COL_CHOICE_NM1] == 0).sum()
        pct_v = 100 * n_v / n_total
        print(f"    prev-{rew_label:>10s}: n={n_total:4d}  "
              f"prev-V={n_v:4d} ({pct_v:5.1f}%)  "
              f"prev-H={n_h:4d} ({100-pct_v:5.1f}%)  "
              f"|imbalance|={abs(pct_v - 50):4.1f}pp")


def compute_P(mu_v_rew, mu_h_rew, mu_v_unrew, mu_h_unrew,
              n_samples=N_RESAMPLES, seed=0):
    """Compute P (pure-reward effect) via independent resampling.

    HARMONISED: seed now defaults to 0 rather than None, and is set from
    a per-rat value in run_analysis. Previously used np.random.default_rng()
    with no seed, making runs non-reproducible.

    P = μ_unrew_avg − μ_rew_avg
      = [(μ_prev-H,unrew + μ_prev-V,unrew) / 2]
      − [(μ_prev-H,rew   + μ_prev-V,rew  ) / 2]

    Equal weighting inside each outcome pool algebraically cancels the
    reward × choice component (which is antisymmetric in previous choice),
    so P isolates the choice-agnostic outcome-driven bias regardless of
    how imbalanced the actual previous-choice counts are within each pool.
    """
    rng = np.random.default_rng(seed)
    n = n_samples
    idx_vr = rng.integers(0, len(mu_v_rew),   n)
    idx_hr = rng.integers(0, len(mu_h_rew),   n)
    idx_vu = rng.integers(0, len(mu_v_unrew), n)
    idx_hu = rng.integers(0, len(mu_h_unrew), n)
    mu_rew_avg   = (mu_v_rew[idx_vr]   + mu_h_rew[idx_hr])   / 2
    mu_unrew_avg = (mu_v_unrew[idx_vu] + mu_h_unrew[idx_hu]) / 2
    return {
        'P':            mu_unrew_avg - mu_rew_avg,
        'mu_rew_avg':   mu_rew_avg,
        'mu_unrew_avg': mu_unrew_avg,
    }


def classify_pure_reward(P_dist, ci_level=0.95):
    """Return classification dict for the pure-reward P statistic."""
    alpha = 1 - ci_level
    ci_low  = float(np.percentile(P_dist, 100 * alpha / 2))
    ci_high = float(np.percentile(P_dist, 100 * (1 - alpha / 2)))
    excludes_zero = not (ci_low <= 0 <= ci_high)
    if excludes_zero:
        label = ('pure_reward_toward_vertical' if np.median(P_dist) > 0
                 else 'pure_reward_toward_horizontal')
    else:
        label = 'no_pure_reward'
    return {
        'classification': label,
        'P_ci':           (ci_low, ci_high),
        'P_excludes_0':   excludes_zero,
    }


# =============================================================================
# Contrast summary plot  (unchanged)
# =============================================================================

def plot_contrast_summary(mu_dists, delta_mus, joint, P_result,
                          choice_result, pure_result, profile, rat_id,
                          n_bootstrap_rew, n_bootstrap_unrew,
                          subset_name='pooled',
                          figure_dir=None):
    if figure_dir is None:
        figure_dir = FIGURE_DIR
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Panel 1: 4-cell μ (raw cell means the eye reads from) ---
    keys   = ['mu_h_rew', 'mu_v_rew', 'mu_h_unrew', 'mu_v_unrew']
    labels = ['prev-H\nrew', 'prev-V\nrew', 'prev-H\nunrew', 'prev-V\nunrew']
    cols   = ['#21F33A', '#0FA023', '#F436D1', '#B0289F']
    for i, (k, c) in enumerate(zip(keys, cols)):
        d = mu_dists[k]
        med, lo, hi = np.median(d), np.percentile(d, 2.5), np.percentile(d, 97.5)
        axes[0].errorbar(i, med, yerr=[[med - lo], [hi - med]],
                         fmt='o', color=c, capsize=8, markersize=10, lw=2)
    axes[0].axhline(45, color='gray', linestyle=':', alpha=0.5, label='45°')
    axes[0].set_xticks(range(4)); axes[0].set_xticklabels(labels, fontsize=10)
    axes[0].set_ylabel('μ (PSE) [°]', fontsize=12)
    axes[0].set_title(f'Rat {rat_id} [{subset_name}] — 4-cell PSE means', fontsize=11)
    axes[0].legend(fontsize=9)

    # --- Panel 2: (Δμ_rew, Δμ_unrew) scatter with diagonals ---
    rew_paired   = joint['rew_paired']
    unrew_paired = joint['unrew_paired']
    axes[1].scatter(rew_paired, unrew_paired, s=4, alpha=0.15, color='#444444')
    lim = np.max(np.abs([rew_paired, unrew_paired])) * 1.1

    col_pure = "#E68722"
    col_rxc  = "#2922E6"

    axes[1].plot([-lim, lim], [-lim, lim], ':', color=col_pure, lw=1.8,
                 label='y = x   (pure choice, reward-independent)')
    axes[1].plot([-lim, lim], [ lim, -lim], ':', color=col_rxc, lw=1.8,
                 label='y = −x  (reward × choice, flips with reward)')
    axes[1].axhline(0, color='k', lw=0.5)
    axes[1].axvline(0, color='k', lw=0.5)
    axes[1].set_xlim(-lim, lim)
    axes[1].set_ylim(-lim, lim)
    axes[1].set_aspect('equal')

    box = dict(boxstyle='round,pad=0.25', facecolor='white',
               alpha=0.85, edgecolor='none')
    axes[1].text(0.97, 0.97, 'switch\n(win-switch\n+ lose-switch)',
                 transform=axes[1].transAxes, ha='right', va='top',
                 fontsize=8, color=col_pure, bbox=box, linespacing=1.1)
    axes[1].text(0.03, 0.03, 'repeat\n(win-stay\n+ lose-stay)',
                 transform=axes[1].transAxes, ha='left', va='bottom',
                 fontsize=8, color=col_pure, bbox=box, linespacing=1.1)
    axes[1].text(0.03, 0.97, 'win-stay\n+ lose-switch',
                 transform=axes[1].transAxes, ha='left', va='top',
                 fontsize=8, color=col_rxc, bbox=box, linespacing=1.1)
    axes[1].text(0.97, 0.03, 'win-switch\n+ lose-stay',
                 transform=axes[1].transAxes, ha='right', va='bottom',
                 fontsize=8, color=col_rxc, bbox=box, linespacing=1.1)

    axes[1].set_xlabel('Δμ_rew [°]', fontsize=12)
    axes[1].set_ylabel('Δμ_unrew [°]', fontsize=12)
    axes[1].set_title(f'p(same sign) = {joint["p_same_sign"]:.2f}', fontsize=11)
    axes[1].legend(fontsize=8, loc='upper center',
                   bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False)

    # --- Panel 3: three orthogonal contrasts as CI bars ---
    contrasts = [
        ('S\n(choice)',       np.median(joint['S']),      choice_result['S_ci'], choice_result['S_excludes_0']),
        ('D\n(interaction)',  np.median(joint['D']),      choice_result['D_ci'], choice_result['D_excludes_0']),
        ('P\n(reward)',       np.median(P_result['P']),   pure_result['P_ci'],   pure_result['P_excludes_0']),
    ]
    for i, (name, med, ci, excl) in enumerate(contrasts):
        color = '#c0392b' if excl else 'gray'
        axes[2].errorbar(i, med, yerr=[[med - ci[0]], [ci[1] - med]],
                         fmt='o', color=color, capsize=8, markersize=10, lw=2)
    axes[2].axhline(0, color='gray', linestyle='--', alpha=0.6)
    axes[2].set_xticks(range(3)); axes[2].set_xticklabels([c[0] for c in contrasts], fontsize=10)
    axes[2].set_ylabel('Contrast value [°]', fontsize=12)
    axes[2].set_title(f'Orthogonal contrasts — profile: {profile}', fontsize=11)

    plt.suptitle(
        f'Rat {rat_id} [{subset_name}]  |  profile: {profile}  |  '
        f'rew: {n_bootstrap_rew} iters, unrew: {n_bootstrap_unrew} iters',
        fontsize=12, y=1.02,
    )
    stem = f'rat_{rat_id}_contrast_summary'
    for ext in ('png', 'svg'):
        path = os.path.join(figure_dir, f'{stem}.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Figure saved: {path}")
    plt.close(fig)


# =============================================================================
# Console summary + table export  (unchanged)
# =============================================================================

def summarise_classifications(all_results, key='contrast_summary'):
    rows = []
    for rat, res in all_results.items():
        if key not in res:
            continue
        r = res[key]
        rows.append({
            'rat':          rat,
            'profile':      r['profile'],
            'Δμ_rew':       r['delta_mu_rew_median'],
            'Δμ_unrew':     r['delta_mu_unrew_median'],
            'S':            r['S_median'],
            'S_CI':         f"[{r['S_ci_low']:.2f}, {r['S_ci_high']:.2f}]",
            'D':            r['D_median'],
            'D_CI':         f"[{r['D_ci_low']:.2f}, {r['D_ci_high']:.2f}]",
            'P':            r['P_median'],
            'P_CI':         f"[{r['P_ci_low']:.2f}, {r['P_ci_high']:.2f}]",
            'p(same sign)': r['p_same_sign'],
        })
    if not rows:
        print("  (no contrast summaries to report)")
        return
    df = pd.DataFrame(rows).set_index('rat')
    print("\n" + "=" * 70)
    print("Per-rat orthogonal-contrast summary")
    print("=" * 70)
    print(df.to_string(float_format='{:.2f}'.format))
    print("\nGroup-level profile counts:")
    for label, n in df['profile'].value_counts().items():
        print(f"  {label:35s}: {n:2d}/{len(df)}  ({100 * n / len(df):.0f}%)")
    print("=" * 70)


def _fmt_pm(x):
    return f"{x:+.2f}".replace('-', '−')


def _df_to_booktabs(df, column_format, header_labels, caption, label):
    n_cols = len(df.columns)
    assert len(column_format) >= n_cols and len(header_labels) == n_cols
    lines = [
        r'\begin{table}[htbp]', r'  \centering',
        r'  \caption{' + caption + r'}', r'  \label{' + label + r'}',
        r'  \begin{tabular}{' + column_format + r'}', r'    \toprule',
        '    ' + ' & '.join(header_labels) + r' \\', r'    \midrule',
    ]
    for _, row in df.iterrows():
        lines.append('    ' + ' & '.join(str(v) for v in row) + r' \\')
    lines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def export_tables(all_results, table_dir=TABLE_DIR, subset_name='pooled'):
    # (body unchanged — omitted here for brevity; identical to previous)
    rows = []
    for rat, rat_res in all_results.items():
        for label, res in rat_res.items():
            if not label.startswith('analysis_'):
                continue
            rows.append({
                'rat':               int(rat),
                'subset':            subset_name,
                'analysis':          label,
                'delta_mu':          res['delta_mu_median'],
                'ci_lo':             res['delta_mu_ci_low'],
                'ci_hi':             res['delta_mu_ci_high'],
                'n_bootstrap_valid': res['n_bootstrap_valid'],
                'ci_valid':          res.get('ci_valid', True),
            })
    if rows:
        raw = pd.DataFrame(rows)
        raw.to_csv(os.path.join(table_dir, 'delta_mu_per_analysis.csv'),
                   index=False, float_format='%.4f')

    rows = []
    for rat, rat_res in all_results.items():
        c = rat_res.get('contrast_summary')
        if c is None:
            continue
        rows.append({
            'rat':            int(rat),
            'subset':         subset_name,
            'delta_mu_rew':   c['delta_mu_rew_median'],
            'delta_mu_unrew': c['delta_mu_unrew_median'],
            'S':              c['S_median'],
            'S_ci_lo':        c['S_ci_low'],
            'S_ci_hi':        c['S_ci_high'],
            'S_excludes_0':   c['S_excludes_0'],
            'D':              c['D_median'],
            'D_ci_lo':        c['D_ci_low'],
            'D_ci_hi':        c['D_ci_high'],
            'D_excludes_0':   c['D_excludes_0'],
            'P':              c['P_median'],
            'P_ci_lo':        c['P_ci_low'],
            'P_ci_hi':        c['P_ci_high'],
            'P_excludes_0':   c['P_excludes_0'],
            'p_same_sign':    c['p_same_sign'],
            'profile':        c['profile'],
        })
    if rows:
        raw = pd.DataFrame(rows)
        raw.to_csv(os.path.join(table_dir, 'orthogonal_contrasts.csv'),
                   index=False, float_format='%.4f')

    print(f"\n✓ Tables written to {table_dir} [{subset_name}]")


# =============================================================================
# SECTION 7 — Main loop  (HARMONISED seeding)
# =============================================================================

def _make_seed(rat_id, analysis_idx, subset_name):
    """Deterministic seed per (rat, analysis, subset). HARMONISED addition."""
    subset_offset = {'pooled': 0, 'tactile': 1, 'visual': 2, 'visuotactile': 3}
    return int(10_000 * int(rat_id) + 100 * analysis_idx
               + subset_offset.get(subset_name, 0))


def run_analysis(df, subset_name='pooled', figure_dir=FIGURE_DIR, table_dir=TABLE_DIR):
    """Main entry point for one analysis scope (pooled or one modality)."""
    os.makedirs(figure_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

    rats = df[COL_RAT].unique()
    print(f"\n{'='*60}")
    print(f"Choice effect psychometric analysis [{subset_name}]")
    print(f"  Bootstrap iterations : {N_BOOTSTRAP}")
    print(f"  Rats found           : {len(rats)}")
    print(f"  Analyses to run      : {len(ANALYSES)}")
    print(f"  Fit bounds           : μ∈[0,90], σ∈[1,90], γ,λ∈[0,0.4]")
    print(f"  Filter               : {FILTER_KWARGS}")
    print(f"{'='*60}\n")

    all_results = {}
    for rat in sorted(rats):
        delta_mus  = {}
        boot_valid = {}
        pse_flanks = PSE_FLANKS.get(rat, [(40, 50)])
        print(f"{'='*60}")
        print(f"Rat {rat}  |  PSE flanks: {pse_flanks}")
        print(f"{'='*60}")

        df_rat = df[df[COL_RAT] == rat].copy()
        all_results[rat] = {}
        mu_dists = {}
        report_choice_distribution(df_rat, rat, pse_flanks)

        for analysis_idx, analysis in enumerate(ANALYSES):
            print(f"\n  [{analysis['label']}] {analysis['description']}")

            # build the appropriate 45° pool
            if analysis['use_fake_45']:
                flank_angles = [a for pair in pse_flanks for a in pair]   # [40, 50, 35, 55]
                df_45_full = df_rat[df_rat[COL_STIM_NM1].isin(flank_angles)]
                counts = {a: int((df_rat[COL_STIM_NM1] == a).sum()) for a in flank_angles}
                pool_desc = ", ".join(f"{a}°={n}" for a, n in sorted(counts.items()))
            else:
                df_45_full = df_rat[df_rat[COL_STIM_NM1] == 45].copy()
                pool_desc  = f"real 45°={len(df_45_full)}"

            df_check = df_45_full.copy()
            if analysis['condition_on'] is not None:
                df_check = df_check[
                    df_check[analysis['condition_on']['col']] ==
                    analysis['condition_on']['val']
                ]

            n_g0 = (df_check[analysis['split_by']] == 0).sum()
            n_g1 = (df_check[analysis['split_by']] == 1).sum()
            print(f"  Pool: {pool_desc}")
            print(f"  group_0 ({analysis['labels'][0]}): {n_g0} trials")
            print(f"  group_1 ({analysis['labels'][1]}): {n_g1} trials")

            if min(n_g0, n_g1) < 10:
                print(f"  ⚠ Skipping: too few trials in one group\n")
                continue

            # HARMONISED: deterministic seed per (rat, analysis, subset)
            seed = _make_seed(rat, analysis_idx, subset_name)
            print(f"  Running {N_BOOTSTRAP} bootstrap iterations (seed={seed})...")
            boot_results = bootstrap_psychometric(
                df_rat,
                use_fake_45=analysis['use_fake_45'],
                split_by=analysis['split_by'],
                condition_on=analysis['condition_on'],
                pse_flanks=pse_flanks,
                seed=seed,
            )
            n_valid = boot_results['n_valid']
            print(f"  ✓ {n_valid} valid iterations completed "
                  f"(ci_valid={boot_results['ci_valid']})")

            plot_psychometric(
                boot_results,
                df_45_full=df_45_full,
                rat_id=rat,
                split_by=analysis['split_by'],
                condition_on=analysis['condition_on'],
                labels=analysis['labels'],
                analysis_label=analysis['label'],
                subset_name=subset_name,
                figure_dir=figure_dir,
            )

            if n_valid == 0:
                print(f"  ⚠ No valid iterations, skipping delta_mu\n")
                continue

            delta_mu = (boot_results['group_1']['mu'].values -
                        boot_results['group_0']['mu'].values)
            all_results[rat][analysis['label']] = {
                'delta_mu_median':   np.median(delta_mu),
                'delta_mu_ci_low':   np.percentile(delta_mu,  2.5),
                'delta_mu_ci_high':  np.percentile(delta_mu, 97.5),
                'n_bootstrap_valid': n_valid,
                'ci_valid':          boot_results['ci_valid'],
            }
            print(f"  Δμ = {np.median(delta_mu):.2f}° "
                  f"[{np.percentile(delta_mu, 2.5):.2f}, "
                  f"{np.percentile(delta_mu, 97.5):.2f}]")

            if analysis['label'] == 'analysis_3_fake45_reward_controlled':
                delta_mus['rew']       = delta_mu
                boot_valid['rew']      = n_valid
                mu_dists['mu_h_rew']   = boot_results['group_0']['mu'].values
                mu_dists['mu_v_rew']   = boot_results['group_1']['mu'].values
            elif analysis['label'] == 'analysis_4_fake45_reward_controlled_opposite':
                delta_mus['unrew']     = delta_mu
                boot_valid['unrew']    = n_valid
                mu_dists['mu_h_unrew'] = boot_results['group_0']['mu'].values
                mu_dists['mu_v_unrew'] = boot_results['group_1']['mu'].values

        # --- Combined orthogonal-contrast summary (S, D, P) ---
        required = ['mu_v_rew', 'mu_h_rew', 'mu_v_unrew', 'mu_h_unrew']
        if 'rew' in delta_mus and 'unrew' in delta_mus and all(k in mu_dists for k in required):
            print(f"\n  --- Orthogonal contrast summary (S, D, P) ---")

            # HARMONISED: seed S/D and P from the rat id
            sd_seed = int(10_000 * int(rat)
                          + {'pooled':0,'tactile':1,'visual':2,'visuotactile':3}
                          .get(subset_name, 0) + 90)
            p_seed  = sd_seed + 1

            joint         = compute_S_and_D(delta_mus['rew'], delta_mus['unrew'],
                                            seed=sd_seed)
            choice_result = classify_choice_pattern(delta_mus['rew'],
                                                    delta_mus['unrew'], joint)
            P_result      = compute_P(mu_dists['mu_v_rew'],   mu_dists['mu_h_rew'],
                                      mu_dists['mu_v_unrew'], mu_dists['mu_h_unrew'],
                                      seed=p_seed)
            pure_result   = classify_pure_reward(P_result['P'])

            profile = build_contrast_profile(choice_result, pure_result)

            plot_contrast_summary(
                mu_dists=mu_dists, delta_mus=delta_mus, joint=joint,
                P_result=P_result, choice_result=choice_result,
                pure_result=pure_result, profile=profile, rat_id=rat,
                n_bootstrap_rew=boot_valid.get('rew', 0),
                n_bootstrap_unrew=boot_valid.get('unrew', 0),
                subset_name=subset_name, figure_dir=figure_dir,
            )

            all_results[rat]['contrast_summary'] = {
                'profile':               profile,
                'delta_mu_rew_median':   float(np.median(delta_mus['rew'])),
                'delta_mu_unrew_median': float(np.median(delta_mus['unrew'])),
                'S_median':              float(np.median(joint['S'])),
                'S_ci_low':              choice_result['S_ci'][0],
                'S_ci_high':             choice_result['S_ci'][1],
                'S_excludes_0':          choice_result['S_excludes_0'],
                'D_median':              float(np.median(joint['D'])),
                'D_ci_low':              choice_result['D_ci'][0],
                'D_ci_high':             choice_result['D_ci'][1],
                'D_excludes_0':          choice_result['D_excludes_0'],
                'P_median':              float(np.median(P_result['P'])),
                'P_ci_low':              pure_result['P_ci'][0],
                'P_ci_high':             pure_result['P_ci'][1],
                'P_excludes_0':          pure_result['P_excludes_0'],
                'p_same_sign':           choice_result['p_same_sign'],
            }
            print(f"  S = {np.median(joint['S']):+.2f}° [{choice_result['S_ci'][0]:+.2f}, {choice_result['S_ci'][1]:+.2f}]"
                  f"  ({'✓' if choice_result['S_excludes_0'] else '·'})")
            print(f"  D = {np.median(joint['D']):+.2f}° [{choice_result['D_ci'][0]:+.2f}, {choice_result['D_ci'][1]:+.2f}]"
                  f"  ({'✓' if choice_result['D_excludes_0'] else '·'})")
            print(f"  P = {np.median(P_result['P']):+.2f}° [{pure_result['P_ci'][0]:+.2f}, {pure_result['P_ci'][1]:+.2f}]"
                  f"  ({'✓' if pure_result['P_excludes_0'] else '·'})")
            print(f"  → profile: {profile}\n")

    print(f"\n{'='*60}")
    print("Summary across rats and analyses:")
    rows = []
    for rat, rat_res in all_results.items():
        for label, res in rat_res.items():
            rows.append({'rat': rat, 'analysis': label, **res})
    if rows:
        summary = pd.DataFrame(rows).set_index(['rat', 'analysis'])
        print(summary.to_string(float_format='{:.2f}'.format))
    print(f"{'='*60}\n")

    summarise_classifications(all_results)
    export_tables(all_results, table_dir=table_dir, subset_name=subset_name)

    return all_results


def contrast_results_to_dataframe(all_results, subset_name):
    """Flatten only the primary S/D/P contrast summary for cross-scope comparison."""
    rows = []
    for rat, rat_res in all_results.items():
        c = rat_res.get('contrast_summary')
        if c is None:
            continue
        rows.append({
            'rat':            int(rat),
            'subset':         subset_name,
            'delta_mu_rew':   c['delta_mu_rew_median'],
            'delta_mu_unrew': c['delta_mu_unrew_median'],
            'S':              c['S_median'],
            'S_ci_lo':        c['S_ci_low'],
            'S_ci_hi':        c['S_ci_high'],
            'S_excludes_0':   c['S_excludes_0'],
            'D':              c['D_median'],
            'D_ci_lo':        c['D_ci_low'],
            'D_ci_hi':        c['D_ci_high'],
            'D_excludes_0':   c['D_excludes_0'],
            'P':              c['P_median'],
            'P_ci_lo':        c['P_ci_low'],
            'P_ci_hi':        c['P_ci_high'],
            'P_excludes_0':   c['P_excludes_0'],
            'p_same_sign':    c['p_same_sign'],
            'profile':        c['profile'],
        })
    return pd.DataFrame(rows)


def run_requested_scopes(df, analysis_scope=ANALYSIS_SCOPE):
    """Run pooled, per-modality, or both without duplicating the analysis code."""
    if analysis_scope not in ('pooled', 'per_modality', 'both'):
        raise ValueError("ANALYSIS_SCOPE must be 'pooled', 'per_modality', or 'both'.")

    if analysis_scope in ('per_modality', 'both') and COL_MODALITY not in df.columns:
        raise ValueError(f"Per-modality analysis requested but column "
                         f"{COL_MODALITY!r} is not present in the dataframe.")

    scopes = {}
    if analysis_scope in ('pooled', 'both'):
        scopes['pooled'] = df.copy()
    if analysis_scope in ('per_modality', 'both'):
        for name, code in MODALITY_CODES.items():
            scopes[name] = df[df[COL_MODALITY] == code].copy()

    results_by_scope = {}
    combined = []

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for subset_name, subset_df in scopes.items():
        if subset_df.empty:
            print(f"\n⚠ Skipping [{subset_name}]: no trials")
            continue

        figure_dir = os.path.join(OUTPUT_ROOT, subset_name, "figures")
        table_dir  = os.path.join(OUTPUT_ROOT, subset_name, "tables")

        print("\n" + "#" * 72)
        print(f"RUNNING PSYCHOMETRIC SCOPE: {subset_name}")
        print(f"Trials in scope: {len(subset_df)}")
        print("#" * 72)

        res = run_analysis(subset_df, subset_name=subset_name,
                           figure_dir=figure_dir, table_dir=table_dir)
        results_by_scope[subset_name] = res

        flat = contrast_results_to_dataframe(res, subset_name)
        if not flat.empty:
            combined.append(flat)

    if combined:
        combined_df = pd.concat(combined, ignore_index=True)
        combined_path = os.path.join(OUTPUT_ROOT,
                                      "orthogonal_contrasts_all_scopes.csv")
        combined_df.to_csv(combined_path, index=False, float_format="%.4f")
        print(f"\n✓ Combined cross-scope table written to: {combined_path}")

    return results_by_scope


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    results = run_requested_scopes(df, analysis_scope=ANALYSIS_SCOPE)