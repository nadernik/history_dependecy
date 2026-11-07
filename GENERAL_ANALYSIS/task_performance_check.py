"""
task_performance_check.py
==========================
Quantitative answer to "are the rats doing the task?".

For every (rat, condition), where condition = pooled | tactile | visual | visuotactile:
    - Fit a 4-parameter cumulative Gaussian psychometric (μ, σ, γ, λ) via
      MLE on single-trial data (Bernoulli negative log-likelihood).
      Matches the fitting used in the other pipeline scripts.
    - Session-level bootstrap (500 iters) → 95% CIs on all four params.
    - Accuracy on unambiguous trials (|angle − 45| ≥ EXTREME_MARGIN)
      with Wilson 95% CI and one-sided binomial test vs 50%.

Classification per (rat, condition):
    doing_task     : all 3 criteria passed
    borderline     : 1 criterion failed
    not_doing_task : ≥ 2 criteria failed

    Criteria:
      1. accuracy at extremes  ≥ ACC_THRESHOLD (default 0.70)
      2. slope σ 97.5-percentile < SIGMA_MAX_THRESH (default 30°)
      3. total lapse γ + λ      < LAPSE_MAX_THRESH (default 0.30)

Group-level summary table printed at the end.

What was fixed relative to the previous per-rat inspection script
-----------------------------------------------------------------
    ▸ MLE (Bernoulli NLL) instead of curve_fit on binned proportions
      — same fitting as choice_effects_psychometric.py and
      perceptual_bias_test.py.
    ▸ Loop over all rats (was hard-coded to RAT_ID).
    ▸ Cleaned nested loop structure (was: modality loop with orphan
      indentation and commented-out transition loop).
    ▸ Session-level bootstrap for CIs on all fitted parameters.
    ▸ Binomial + Wilson CI on accuracy at extremes.
    ▸ Per-rat classification and group summary table.
"""

import sys
sys.path.append(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm, binomtest
from serial_dependence_analysis import SerialDependenceAnalyzer
from functions_Ale import filter_training_trials


# =============================================================================
# COLUMN NAMES
# =============================================================================
ANGLE_COL    = 'angle'
RESP_COL     = 'action'
RAT_COL      = 'rat'
SESSION_COL  = 'date'
MODALITY_COL = 'mod'
BOUNDARY     = 45


# =============================================================================
# TUNABLE PARAMETERS
# =============================================================================
TARGET_RATS         = None       # None = all rats; else list, e.g. [4, 9, 11]
N_BOOT              = 2
MIN_TRIALS          = 100        # per (rat, condition)
MIN_SESSIONS        = 3
X_GRID              = np.linspace(0, 90, 200)

# task-performance thresholds
EXTREME_MARGIN      = 15         # trials with |angle - 45| >= 15° = "unambiguous"
ACC_THRESHOLD       = 0.70       # accuracy at extremes must exceed this
SIGMA_MAX_THRESH    = 30.0       # σ 97.5-percentile must be below this (steepness)
LAPSE_MAX_THRESH    = 0.30       # γ + λ (median fit) must be below this


# =============================================================================
# LOAD + PREPROCESS
# =============================================================================
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data(fold=False)
df = an.create_lagged_features()
df = filter_training_trials(df, method='criterion', criterion=0.80, min_consec=2)

df = df[(df[ANGLE_COL] >= 0) & (df[ANGLE_COL] <= 90)].copy()

MODALITIES = df[MODALITY_COL].dropna().unique()


# =============================================================================
# MLE psychometric fit
# =============================================================================

def psychometric(x, mu, sigma, gamma, lapse):
    return gamma + (1 - gamma - lapse) * norm.cdf(x, loc=mu, scale=sigma)


def _nll(params, x, c):
    mu, sigma, gamma, lapse = params
    p = psychometric(x, mu, sigma, gamma, lapse)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -np.sum(c * np.log(p) + (1 - c) * np.log(1 - p))


def fit_psychometric(x, y):
    """MLE fit of 4-param cum. Gaussian. Returns [μ, σ, γ, λ] or None."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 30 or len(np.unique(x)) < 4:
        return None

    x0     = [45.0, 15.0, 0.05, 0.05]
    bounds = [(0, 90), (1, 60), (0, 0.3), (0, 0.3)]
    try:
        r = minimize(_nll, x0, args=(x, y), bounds=bounds, method='L-BFGS-B')
        return r.x if r.success else None
    except (RuntimeError, ValueError, FloatingPointError):
        return None


# =============================================================================
# Session bootstrap
# =============================================================================

def bootstrap_fit(df_cond, n_boot=N_BOOT, seed=0):
    """Session bootstrap: refit on resampled sessions. Stores all fit params."""
    x = df_cond[ANGLE_COL].values
    y = df_cond[RESP_COL].values
    popt = fit_psychometric(x, y)
    if popt is None:
        return {'popt': None, 'param_boot': None, 'ci_valid': False, 'n_valid': 0}

    sessions = df_cond[SESSION_COL].dropna().unique()
    if len(sessions) < MIN_SESSIONS:
        return {'popt': popt, 'param_boot': None,
                'ci_valid': False, 'n_valid': 0}

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        sampled = rng.choice(sessions, size=len(sessions), replace=True)
        boot_df = pd.concat([df_cond[df_cond[SESSION_COL] == s] for s in sampled],
                            ignore_index=True)
        p = fit_psychometric(boot_df[ANGLE_COL].values, boot_df[RESP_COL].values)
        if p is not None:
            boot.append(p)

    if len(boot) < 10:
        return {'popt': popt, 'param_boot': None,
                'ci_valid': False, 'n_valid': len(boot)}
    return {'popt': popt, 'param_boot': np.array(boot),
            'ci_valid': True, 'n_valid': len(boot)}


# =============================================================================
# Task-performance metrics: accuracy at unambiguous angles
# =============================================================================

def performance_metrics(df_cond, extreme_margin=EXTREME_MARGIN):
    """
    On trials where |angle - 45| >= extreme_margin (the "unambiguous" trials):
        - accuracy = P(action == (angle > 45))
        - Wilson 95% CI
        - one-sided binomial test: H0 p = 0.5, H1 p > 0.5
    """
    x = df_cond[ANGLE_COL].values
    y = df_cond[RESP_COL].values
    truth = (x > BOUNDARY).astype(int)

    mask = np.abs(x - BOUNDARY) >= extreme_margin
    n_ext = int(mask.sum())
    n_correct = int((y[mask] == truth[mask]).sum())
    if n_ext == 0:
        return {'n_extreme': 0, 'n_correct': 0, 'accuracy': np.nan,
                'ci': (np.nan, np.nan), 'binom_p': np.nan}

    acc = n_correct / n_ext
    result = binomtest(n_correct, n_ext, p=0.5, alternative='greater')
    ci = result.proportion_ci(confidence_level=0.95, method='wilson')
    return {
        'n_extreme': n_ext,
        'n_correct': n_correct,
        'accuracy':  float(acc),
        'ci':        (float(ci.low), float(ci.high)),
        'binom_p':   float(result.pvalue),
    }


# =============================================================================
# Classification
# =============================================================================

def classify_task_performance(perf, boot):
    """
    3 criteria — all must pass for `doing_task`:
        1. accuracy at extremes >= ACC_THRESHOLD
        2. σ bootstrap 97.5-percentile < SIGMA_MAX_THRESH  (slope not flat)
        3. γ + λ (median fit) < LAPSE_MAX_THRESH           (attentive)
    Failing 1 criterion = borderline; 2+ = not_doing_task.
    """
    out = {}

    # criterion 1: accuracy
    out['accuracy']      = perf['accuracy']
    out['accuracy_pass'] = (not np.isnan(perf['accuracy'])
                            and perf['accuracy'] >= ACC_THRESHOLD)

    # criterion 2: slope
    if boot['param_boot'] is not None:
        sigma_hi = float(np.percentile(boot['param_boot'][:, 1], 97.5))
    elif boot['popt'] is not None:
        sigma_hi = float(boot['popt'][1])
    else:
        sigma_hi = np.nan
    out['sigma_97_5']    = sigma_hi
    out['slope_pass']    = (not np.isnan(sigma_hi)
                            and sigma_hi < SIGMA_MAX_THRESH)

    # criterion 3: lapses
    if boot['popt'] is not None:
        _, _, gamma, lapse = boot['popt']
        out['lapse_sum']  = float(gamma + lapse)
        out['lapse_pass'] = (gamma + lapse) < LAPSE_MAX_THRESH
    else:
        out['lapse_sum']  = np.nan
        out['lapse_pass'] = False

    n_failed = sum(not out[k] for k in ('accuracy_pass', 'slope_pass', 'lapse_pass'))
    if n_failed == 0:
        label = 'doing_task'
    elif n_failed == 1:
        label = 'borderline'
    else:
        label = 'not_doing_task'
    out['classification'] = label
    out['n_failed']       = n_failed
    return out


# =============================================================================
# Plotting
# =============================================================================

_LABEL_COLORS = {'doing_task':     'seagreen',
                 'borderline':     'goldenrod',
                 'not_doing_task': 'crimson'}


def plot_psychometric(df_cond, boot, perf, classification,
                       rat_id, condition, savefig=True):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # binned empirical points, size ∝ n_trials
    bin_edges   = np.linspace(0, 90, 20)                 # 19 bins ≈ 4.7°
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx     = np.clip(np.digitize(df_cond[ANGLE_COL].values, bin_edges) - 1,
                          0, len(bin_centers) - 1)
    obs_x, obs_p, obs_n = [], [], []
    for b, center in enumerate(bin_centers):
        mask = bin_idx == b
        if mask.sum() < 3:
            continue
        obs_x.append(center)
        obs_p.append(df_cond[RESP_COL].values[mask].mean())
        obs_n.append(int(mask.sum()))
    obs_x, obs_p, obs_n = np.array(obs_x), np.array(obs_p), np.array(obs_n)

    sizes = (obs_n / obs_n.max()) * 350 + 40
    ax.scatter(obs_x, obs_p, s=sizes, color='steelblue',
               edgecolors='white', linewidths=0.8, zorder=5,
               label='Observed P(vertical)')

    # fitted curve + bootstrap CI band
    popt = boot['popt']
    if popt is not None:
        y_fit = psychometric(X_GRID, *popt)
        if boot['param_boot'] is not None:
            curves = np.array([psychometric(X_GRID, *p) for p in boot['param_boot']])
            y_lo = np.percentile(curves, 2.5,  axis=0)
            y_hi = np.percentile(curves, 97.5, axis=0)
            ax.fill_between(X_GRID, y_lo, y_hi, alpha=0.2, color='tomato',
                             linewidth=0, label='95% bootstrap CI')

        mu, sigma, gamma, lapse = popt
        if boot['param_boot'] is not None:
            pb    = boot['param_boot']
            mu_ci = np.percentile(pb[:, 0], [2.5, 97.5])
            si_ci = np.percentile(pb[:, 1], [2.5, 97.5])
            fit_label = (f'μ={mu:.1f}°  [{mu_ci[0]:.1f}, {mu_ci[1]:.1f}]\n'
                         f'σ={sigma:.1f}°  [{si_ci[0]:.1f}, {si_ci[1]:.1f}]\n'
                         f'γ={gamma:.3f}   λ={lapse:.3f}')
        else:
            fit_label = (f'μ={mu:.1f}°  σ={sigma:.1f}°\n'
                         f'γ={gamma:.3f}   λ={lapse:.3f}\n'
                         f'(no bootstrap CI)')
        ax.plot(X_GRID, y_fit, color='tomato', lw=2.5, zorder=4, label=fit_label)

    ax.axhline(0.5, color='gray', ls='--', lw=0.8, alpha=0.7)
    ax.axvline(45,  color='gray', ls='--', lw=0.8, alpha=0.7)
    ax.set_xlabel('Angle (°)', fontsize=13)
    ax.set_ylabel('P(action = vertical)', fontsize=13)
    ax.set_xlim(0, 90)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.arange(0, 91, 10))

    color = _LABEL_COLORS.get(classification['classification'], 'black')
    verdict = classification['classification'].upper().replace('_', ' ')
    title = (
        f'Rat {rat_id}  |  {condition}  →  {verdict}\n'
        f'Acc @ |angle-45|≥{EXTREME_MARGIN}°: '
        f'{perf["accuracy"]*100:.1f}% '
        f'[{perf["ci"][0]*100:.1f}, {perf["ci"][1]*100:.1f}]  '
        f'(n={perf["n_extreme"]}, binom p={perf["binom_p"]:.1e})   '
        f'σ_97.5={classification["sigma_97_5"]:.1f}°   '
        f'γ+λ={classification["lapse_sum"]:.3f}'
    )
    ax.set_title(title, fontsize=10, color=color, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.annotate('Dot size ∝ trial count', xy=(0.98, 0.04),
                xycoords='axes fraction', ha='right', fontsize=9, color='gray')
    plt.tight_layout()

    if savefig:
        fname = f'rat_{rat_id}_{condition}_task_check.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"    ✓ Figure saved: {fname}")
    plt.show()


# =============================================================================
# Group-level summary
# =============================================================================

def summarise(all_results):
    rows = []
    for (rat, cond), r in all_results.items():
        c = r['classification']
        p = r['perf']
        rows.append({
            'rat':            rat,
            'condition':      cond,
            'n_trials':       r['n_trials'],
            'accuracy':       p['accuracy'],
            'acc_CI':         f"[{p['ci'][0]:.2f}, {p['ci'][1]:.2f}]",
            'binom_p':        p['binom_p'],
            'sigma_97.5':     c['sigma_97_5'],
            'γ+λ':            c['lapse_sum'],
            'n_criteria_fail':c['n_failed'],
            'classification': c['classification'],
        })
    if not rows:
        print("  (no results to summarise)")
        return None

    df_out = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("Task performance classification (rat × condition)")
    print("=" * 100)
    print(df_out.to_string(index=False, float_format=lambda x: f"{x:.3g}"))

    print("\nGroup counts by condition:")
    for cond in ['pooled'] + sorted(c for c in df_out['condition'].unique()
                                      if c != 'pooled'):
        sub = df_out[df_out['condition'] == cond]
        if len(sub) == 0:
            continue
        counts = sub['classification'].value_counts()
        parts = [f"{lbl}: {int(counts.get(lbl, 0))}/{len(sub)}"
                 for lbl in ('doing_task', 'borderline', 'not_doing_task')]
        print(f"  {cond:<15}: " + ",  ".join(parts))
    print("=" * 100)
    return df_out


# =============================================================================
# MAIN LOOP — all rats × (pooled + per modality)
# =============================================================================

def run_analysis(df, modalities, target_rats=TARGET_RATS):
    all_results = {}
    rats = sorted(df[RAT_COL].dropna().unique())
    if target_rats is not None:
        rats = [r for r in rats if r in target_rats]

    conditions = ['pooled'] + list(modalities)

    print(f"\n{'=' * 70}")
    print(f"Task performance check — is each rat doing the task?")
    print(f"  Rats             : {len(rats)}  ({rats})")
    print(f"  Modalities       : {list(modalities)}")
    print(f"  Bootstrap iters  : {N_BOOT}")
    print(f"  Criteria         :")
    print(f"    (1) accuracy @ |angle-45|≥{EXTREME_MARGIN}°  ≥ {ACC_THRESHOLD:.2f}")
    print(f"    (2) σ 97.5-percentile              < {SIGMA_MAX_THRESH:.1f}°")
    print(f"    (3) γ + λ (median)                 < {LAPSE_MAX_THRESH:.2f}")
    print(f"{'=' * 70}")

    for rat in rats:
        print(f"\n{'-' * 70}\nRat {rat}\n{'-' * 70}")
        df_rat = df[df[RAT_COL] == rat].copy()

        for i, cond in enumerate(conditions):
            df_cond = df_rat if cond == 'pooled' else df_rat[df_rat[MODALITY_COL] == cond]
            n = len(df_cond)
            print(f"\n  Condition: {cond}  (n={n})")
            if n < MIN_TRIALS:
                print(f"    ⚠ Skipping: fewer than {MIN_TRIALS} trials")
                continue

            seed = int(1000 * rat + 100 * i)
            boot = bootstrap_fit(df_cond, seed=seed)
            if boot['popt'] is None:
                print(f"    ⚠ Fit failed")
                continue

            perf           = performance_metrics(df_cond)
            classification = classify_task_performance(perf, boot)

            mu, sigma, gamma, lapse = boot['popt']
            print(f"    μ={mu:.1f}°  σ={sigma:.1f}°  γ={gamma:.3f}  λ={lapse:.3f}")
            print(f"    accuracy (extremes) = {perf['accuracy']*100:.1f}%  "
                  f"[{perf['ci'][0]*100:.1f}, {perf['ci'][1]*100:.1f}]  "
                  f"(n={perf['n_extreme']}, binom p={perf['binom_p']:.1e})")
            print(f"    σ 97.5-percentile = {classification['sigma_97_5']:.2f}°  |  "
                  f"γ + λ = {classification['lapse_sum']:.3f}")
            print(f"    → {classification['classification']}  "
                  f"(criteria failed: {classification['n_failed']}/3)")

            plot_psychometric(df_cond, boot, perf, classification,
                              rat_id=rat, condition=cond)

            all_results[(rat, cond)] = {
                'boot':           boot,
                'perf':           perf,
                'classification': classification,
                'n_trials':       n,
            }

    summary_df = summarise(all_results)
    return all_results, summary_df


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    results, summary = run_analysis(df, MODALITIES)