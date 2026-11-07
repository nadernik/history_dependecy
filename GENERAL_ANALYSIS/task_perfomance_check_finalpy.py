"""
task_performance_check_final.py
==========================
Quantitative answer to "are the rats doing the task?".

For every (rat, condition, filter_state):
    condition    ∈ {pooled, <modalities>}  (modality codes come from the data)
    filter_state ∈ {'unfiltered', 'filtered'}  (post-criterion sessions)

Per cell:
    - Fit a 4-parameter cumulative Gaussian psychometric (μ, σ, γ, λ) via
      MLE on single-trial data (Bernoulli negative log-likelihood).
    - Session-level bootstrap → 95% CIs on all four params.
    - Accuracy on unambiguous trials (|angle − 45| ≥ EXTREME_MARGIN) with
      TWO-sided Wilson 95% CI, ONE-sided Wilson 95% lower bound, and
      one-sided binomial test vs 50% (kept as a floor check only).

Three-tier classification per cell: good / medium / poor.
Each of the three metrics (accuracy, σ 97.5-percentile, γ+λ) is scored on
its own good/medium/poor scale; the overall label is the WORST tier.
A `fit_degenerate` flag is raised when σ 97.5-percentile > FIT_DEGENERATE_SIGMA,
indicating an essentially flat psychometric (bounds are now [1, 90]).

RAW OUTPUTS:
    Everything needed to re-classify or re-plot without re-fitting is
    pickled per cell (popt, full param_boot array, perf dict, config).
    Threshold changes, tier changes, CI-scheme changes are all post-processing.

Outputs (all under OUTPUT_DIR):
    raw/<filter>/rat_<rat>_<cond>.pkl          — one pickle per cell
    raw/all_cells.pkl                          — dict[(rat, cond, filter)] → cell
    figures/<filter>/rat_<rat>_<cond>_task_check.svg
    task_performance_summary.csv               — full numeric table
    task_performance_summary.tex               — booktabs LaTeX table
    run_config.json                            — top-level run configuration
"""

import sys
import json
import pickle
from pathlib import Path
from datetime import datetime
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
# OUTPUT LOCATIONS
# =============================================================================
OUTPUT_DIR = Path('task_check_output')
FIG_SUBDIR = OUTPUT_DIR / 'figures'
RAW_SUBDIR = OUTPUT_DIR / 'raw'
FIG_FORMAT = 'svg'


# =============================================================================
# TUNABLE PARAMETERS
# =============================================================================
TARGET_RATS         = None       # None = all rats; else list, e.g. [4, 9, 11]
N_BOOT              = 50        # set to ~50 for a quick sanity pass
MIN_TRIALS          = 100        # per (rat, condition)
MIN_SESSIONS        = 3
X_GRID              = np.linspace(0, 90, 200)

EXTREME_MARGIN      = 15         # trials with |angle - 45| >= 15° = "unambiguous"

# Filter to apply for the 'filtered' state; the 'unfiltered' state uses raw data.
FILTER_KWARGS       = dict(method='criterion', criterion=0.80, min_consec=2)
FILTER_STATES       = ('unfiltered', 'filtered')


# =============================================================================
# FIT BOUNDS
# =============================================================================
# σ widened from (1, 60) to (1, 90) so the value carries information about
# how flat the psychometric is, rather than pinning at an artificial ceiling.
MU_BOUNDS    = (0.0, 90.0)
SIGMA_BOUNDS = (1.0, 90.0)
GAMMA_BOUNDS = (0.0, 0.3)
LAPSE_BOUNDS = (0.0, 0.3)

# Above this σ 97.5-percentile, the fit is considered degenerate
# (essentially flat psychometric — corresponds to the old ceiling).
FIT_DEGENERATE_SIGMA = 60.0


# =============================================================================
# TIERED SCORING THRESHOLDS  (good / medium / poor)
# =============================================================================
# Overall label = worst tier across the three metrics.

ACC_TIERS   = {'good': 0.80, 'medium': 0.65}     # higher is better
SIGMA_TIERS = {'good': 20.0, 'medium': 35.0}     # lower is better
LAPSE_TIERS = {'good': 0.15, 'medium': 0.30}     # lower is better

_TIER_RANK   = {'good': 0, 'medium': 1, 'poor': 2}
_TIER_COLORS = {'good': 'seagreen', 'medium': 'goldenrod', 'poor': 'crimson'}


# =============================================================================
# LOAD + PREPROCESS  (build both filter states)
# =============================================================================

def load_datasets():
    """Return a dict {filter_state: DataFrame} with both filter states."""
    an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
    an.load_and_preprocess_data(fold=False)
    df_raw = an.create_lagged_features()
    df_raw = df_raw[(df_raw[ANGLE_COL] >= 0) & (df_raw[ANGLE_COL] <= 90)].copy()

    df_filt = filter_training_trials(df_raw.copy(), **FILTER_KWARGS)

    return {'unfiltered': df_raw, 'filtered': df_filt}


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
    bounds = [MU_BOUNDS, SIGMA_BOUNDS, GAMMA_BOUNDS, LAPSE_BOUNDS]
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

def _wilson_ci_two_sided(k, n, alpha=0.05):
    """Two-sided Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    z = norm.ppf(1 - alpha / 2)
    denom = 1.0 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n) / denom
    return (float(center - half), float(center + half))


def performance_metrics(df_cond, extreme_margin=EXTREME_MARGIN):
    """
    Accuracy on unambiguous trials with:
        - two-sided Wilson 95% CI (reporting)
        - one-sided Wilson 95% lower bound + binomial p-value (floor check)
    """
    x = df_cond[ANGLE_COL].values
    y = df_cond[RESP_COL].values
    truth = (x > BOUNDARY).astype(int)

    mask = np.abs(x - BOUNDARY) >= extreme_margin
    n_ext = int(mask.sum())
    n_correct = int((y[mask] == truth[mask]).sum())
    if n_ext == 0:
        return {'n_extreme': 0, 'n_correct': 0, 'accuracy': np.nan,
                'ci_two_sided': (np.nan, np.nan),
                'ci_one_sided_lower': np.nan,
                'binom_p': np.nan}

    acc = n_correct / n_ext
    ci_ts = _wilson_ci_two_sided(n_correct, n_ext, alpha=0.05)
    result = binomtest(n_correct, n_ext, p=0.5, alternative='greater')
    ci_os = result.proportion_ci(confidence_level=0.95, method='wilson')

    return {
        'n_extreme':           n_ext,
        'n_correct':           n_correct,
        'accuracy':            float(acc),
        'ci_two_sided':        ci_ts,
        'ci_one_sided_lower':  float(ci_os.low),
        'binom_p':             float(result.pvalue),
    }


# =============================================================================
# Tiered classification
# =============================================================================

def _tier_higher_better(value, tiers):
    if np.isnan(value):
        return 'poor'
    if value >= tiers['good']:
        return 'good'
    if value >= tiers['medium']:
        return 'medium'
    return 'poor'


def _tier_lower_better(value, tiers):
    if np.isnan(value):
        return 'poor'
    if value <= tiers['good']:
        return 'good'
    if value <= tiers['medium']:
        return 'medium'
    return 'poor'


def classify_task_performance(perf, boot):
    """
    Score each metric independently as good / medium / poor:
        - accuracy at extremes  (higher is better)
        - σ 97.5-percentile     (lower is better)
        - γ + λ, total lapses   (lower is better)
    Overall label = WORST tier across the three.

    Also raises `fit_degenerate` when σ 97.5-percentile > FIT_DEGENERATE_SIGMA
    (essentially flat psychometric).
    """
    out = {}

    # metric 1: accuracy
    out['accuracy']   = perf['accuracy']
    out['acc_tier']   = _tier_higher_better(perf['accuracy'], ACC_TIERS)

    # metric 2: slope (σ 97.5-percentile of bootstrap)
    if boot['param_boot'] is not None:
        sigma_hi = float(np.percentile(boot['param_boot'][:, 1], 97.5))
    elif boot['popt'] is not None:
        sigma_hi = float(boot['popt'][1])
    else:
        sigma_hi = np.nan
    out['sigma_97_5']     = sigma_hi
    out['sigma_tier']     = _tier_lower_better(sigma_hi, SIGMA_TIERS)
    out['fit_degenerate'] = bool(
        (not np.isnan(sigma_hi)) and (sigma_hi > FIT_DEGENERATE_SIGMA)
    )

    # metric 3: total lapses γ + λ
    if boot['popt'] is not None:
        _, _, gamma, lapse = boot['popt']
        out['lapse_sum'] = float(gamma + lapse)
    else:
        out['lapse_sum'] = np.nan
    out['lapse_tier'] = _tier_lower_better(out['lapse_sum'], LAPSE_TIERS)

    # overall: worst tier across the three
    tiers = (out['acc_tier'], out['sigma_tier'], out['lapse_tier'])
    out['classification'] = max(tiers, key=lambda t: _TIER_RANK[t])
    out['tier_breakdown'] = tiers
    return out


# =============================================================================
# Plotting
# =============================================================================

def plot_psychometric(df_cond, boot, perf, classification,
                      rat_id, condition, filter_state,
                      output_dir=FIG_SUBDIR, fmt=FIG_FORMAT, savefig=True):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # binned empirical points, size ∝ n_trials
    bin_edges   = np.linspace(0, 90, 20)
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

    color   = _TIER_COLORS.get(classification['classification'], 'black')
    verdict = classification['classification'].upper()
    degen   = '  [fit degenerate]' if classification.get('fit_degenerate') else ''
    ci_ts   = perf['ci_two_sided']
    title = (
        f'Rat {rat_id}  |  cond={condition}  |  {filter_state}  →  {verdict}{degen}\n'
        f'Acc @ |angle-45|≥{EXTREME_MARGIN}°: '
        f'{perf["accuracy"]*100:.1f}% '
        f'[{ci_ts[0]*100:.1f}, {ci_ts[1]*100:.1f}]  '
        f'(n={perf["n_extreme"]}, binom p={perf["binom_p"]:.1e})'
    )
    ax.set_title(title, fontsize=11, color=color, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.annotate('Dot size ∝ trial count', xy=(0.98, 0.04),
                xycoords='axes fraction', ha='right', fontsize=9, color='gray')
    plt.tight_layout()

    if savefig:
        subdir = output_dir / filter_state
        subdir.mkdir(parents=True, exist_ok=True)
        fname = subdir / f'rat_{rat_id}_{condition}_task_check.{fmt}'
        plt.savefig(fname, format=fmt, bbox_inches='tight')
        print(f"    ✓ Figure saved: {fname}")
    plt.close(fig)


# =============================================================================
# Data-availability check (cheap, run BEFORE the fits)
# =============================================================================

def data_availability(datasets, target_rats=TARGET_RATS):
    """Print per-cell trial and session counts so we know what will run."""
    print("\n" + "=" * 90)
    print("DATA AVAILABILITY CHECK  (trials / sessions per cell, before fitting)")
    print("=" * 90)

    for filter_state, df in datasets.items():
        rats = sorted(df[RAT_COL].dropna().unique())
        if target_rats is not None:
            rats = [r for r in rats if r in target_rats]
        modalities = sorted(df[MODALITY_COL].dropna().unique(),
                            key=lambda v: (str(type(v)), v))
        conditions = ['pooled'] + list(modalities)

        print(f"\n[{filter_state}]  rats={len(rats)}  conditions={conditions}")
        header = f"  {'rat':>4}  " + "  ".join(f"{str(c):>14}" for c in conditions)
        print(header)
        print("  " + "-" * (len(header) - 2))

        for rat in rats:
            df_rat = df[df[RAT_COL] == rat]
            cells = []
            for cond in conditions:
                df_c = df_rat if cond == 'pooled' else df_rat[df_rat[MODALITY_COL] == cond]
                n_tr  = len(df_c)
                n_ses = df_c[SESSION_COL].nunique() if n_tr > 0 else 0
                flag = ' '
                if n_tr < MIN_TRIALS:
                    flag = '*'
                elif n_ses < MIN_SESSIONS:
                    flag = '~'
                cells.append(f"{n_tr:>7}/{n_ses:>3}{flag}")
            print(f"  {int(rat):>4}  " + "  ".join(f"{c:>14}" for c in cells))
    print(f"\n  Legend: 'trials/sessions'.  "
          f"*: n_trials < MIN_TRIALS={MIN_TRIALS} (cell skipped).  "
          f"~: n_sessions < MIN_SESSIONS={MIN_SESSIONS} (fit only, no bootstrap CI).")
    print("=" * 90)


# =============================================================================
# Group-level summary + export (CSV + LaTeX)
# =============================================================================

def summarise(all_results, filter_states, output_dir=OUTPUT_DIR):
    rows = []
    for (rat, cond, filter_state), r in all_results.items():
        c = r['classification']
        p = r['perf']
        ci_ts = p['ci_two_sided']
        rows.append({
            'rat':                 int(rat),
            'condition':           cond,
            'filter_state':        filter_state,
            'n_trials':            r['n_trials'],
            'n_sessions':          r['n_sessions'],
            'n_extreme':           p['n_extreme'],
            'n_correct':           p['n_correct'],
            'accuracy':            p['accuracy'],
            'acc_ci_low':          ci_ts[0],
            'acc_ci_high':         ci_ts[1],
            'acc_ci_lower_1sided': p['ci_one_sided_lower'],
            'binom_p':             p['binom_p'],
            'sigma_97_5':          c['sigma_97_5'],
            'lapse_sum':           c['lapse_sum'],
            'fit_degenerate':      c['fit_degenerate'],
            'acc_tier':            c['acc_tier'],
            'sigma_tier':          c['sigma_tier'],
            'lapse_tier':          c['lapse_tier'],
            'classification':      c['classification'],
        })
    if not rows:
        print("  (no results to summarise)")
        return None

    df_out = pd.DataFrame(rows)
    # Stable ordering: filter_state first (unfiltered before filtered), then rat, then condition
    fs_order = {fs: i for i, fs in enumerate(filter_states)}
    df_out['_fs_order'] = df_out['filter_state'].map(fs_order)
    df_out = df_out.sort_values(['_fs_order', 'rat', 'condition']).drop(columns='_fs_order')

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- console print
    disp = df_out.copy()
    disp['acc_CI'] = disp.apply(
        lambda r: f"[{r['acc_ci_low']:.2f}, {r['acc_ci_high']:.2f}]", axis=1)
    disp_cols = ['rat', 'condition', 'filter_state', 'n_trials', 'n_extreme',
                 'accuracy', 'acc_CI', 'binom_p', 'sigma_97_5', 'lapse_sum',
                 'fit_degenerate', 'acc_tier', 'sigma_tier', 'lapse_tier',
                 'classification']
    print("\n" + "=" * 120)
    print("Task performance classification (rat × condition × filter_state)")
    print("=" * 120)
    print(disp[disp_cols].to_string(index=False,
                                    float_format=lambda x: f"{x:.3g}"))

    print("\nGroup counts by (filter_state × condition):")
    for filter_state in filter_states:
        sub_f = df_out[df_out['filter_state'] == filter_state]
        if len(sub_f) == 0:
            continue
        print(f"  [{filter_state}]")
        cond_order = ['pooled'] + sorted(
            [c for c in sub_f['condition'].unique() if c != 'pooled'],
            key=str,
        )
        for cond in cond_order:
            sub = sub_f[sub_f['condition'] == cond]
            if len(sub) == 0:
                continue
            counts = sub['classification'].value_counts()
            parts = [f"{lbl}: {int(counts.get(lbl, 0))}/{len(sub)}"
                     for lbl in ('good', 'medium', 'poor')]
            n_degen = int(sub['fit_degenerate'].sum())
            degen_str = f"   [fit_degenerate: {n_degen}]" if n_degen else ""
            print(f"    {str(cond):<10}: " + ",  ".join(parts) + degen_str)
    print("=" * 120)

    # ---- CSV
    csv_path = output_dir / 'task_performance_summary.csv'
    df_out.to_csv(csv_path, index=False, float_format='%.4g')
    print(f"\n✓ CSV saved:   {csv_path}")

    # ---- LaTeX
    latex_path = output_dir / 'task_performance_summary.tex'
    _write_latex_table(df_out, latex_path)
    print(f"✓ LaTeX saved: {latex_path}")

    return df_out


def _write_latex_table(df_out, path):
    """
    Booktabs LaTeX table, math-mode column headers, ready to \\input{}.
    Requires \\usepackage{booktabs, longtable} (and textcomp for \\textdegree).
    A trailing asterisk on σ marks fit_degenerate cells.
    """
    tex = df_out.copy()
    tex['accuracy_pct'] = tex.apply(
        lambda r: (f"{r['accuracy']*100:.1f} "
                   f"[{r['acc_ci_low']*100:.1f}, {r['acc_ci_high']*100:.1f}]"),
        axis=1)
    tex['sigma_fmt']  = tex.apply(
        lambda r: (f"{r['sigma_97_5']:.1f}$^\\ast$"
                   if r['fit_degenerate']
                   else f"{r['sigma_97_5']:.1f}"),
        axis=1)
    tex['lapse_fmt']  = tex['lapse_sum'].map(lambda v: f"{v:.3f}")
    tex['binom_fmt']  = tex['binom_p'].map(lambda v: f"{v:.1e}")

    cols_out = {
        'rat':            'Rat',
        'condition':      'Cond.',
        'filter_state':   'Filter',
        'n_trials':       '$n$',
        'accuracy_pct':   r'Acc.\ (\%) [95\% CI]',
        'binom_fmt':      r'$p_{\text{binom}}$',
        'sigma_fmt':      r'$\sigma_{97.5}$ (\textdegree)',
        'lapse_fmt':      r'$\gamma+\lambda$',
        'acc_tier':       'Acc.\\ tier',
        'sigma_tier':     r'$\sigma$ tier',
        'lapse_tier':     'Lapse tier',
        'classification': 'Overall',
    }
    tex = tex[list(cols_out.keys())].rename(columns=cols_out)

    latex_str = tex.to_latex(
        index=False,
        escape=False,
        longtable=True,
        column_format='lll' + 'r' * 5 + 'l' * 4,
        caption=('Task-performance classification per rat, condition, and filter '
                 'state. Each metric is scored good / medium / poor; the overall '
                 'label is the worst tier across the three. Two-sided Wilson 95\\% '
                 'CI is reported for accuracy at unambiguous angles ($|\\text{angle} - 45| '
                 '\\geq 15\\degree$). An asterisk ($^\\ast$) on $\\sigma_{97.5}$ marks '
                 'cells with a degenerate fit (essentially flat psychometric).'),
        label='tab:task_performance',
    )

    with open(path, 'w', encoding='utf-8') as f:
        f.write("% Requires \\usepackage{booktabs, longtable} "
                "(and textcomp for \\textdegree).\n")
        f.write(latex_str)


# =============================================================================
# MAIN LOOP — rats × (pooled + per modality) × filter states
# =============================================================================

def run_analysis(datasets, target_rats=TARGET_RATS):
    """
    datasets : dict {filter_state: DataFrame}
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_SUBDIR.mkdir(parents=True, exist_ok=True)
    RAW_SUBDIR.mkdir(parents=True, exist_ok=True)

    filter_states = tuple(datasets.keys())

    # Union of rats across datasets; conditions from the unfiltered set
    # (contains at least as many modalities as the filtered set).
    all_rats = set()
    for df in datasets.values():
        all_rats.update(df[RAT_COL].dropna().unique())
    rats = sorted(all_rats)
    if target_rats is not None:
        rats = [r for r in rats if r in target_rats]

    ref_df = datasets[filter_states[0]]
    modalities = sorted(ref_df[MODALITY_COL].dropna().unique(),
                        key=lambda v: (str(type(v)), v))
    conditions = ['pooled'] + list(modalities)

    # ---- top-level run config
    run_config = {
        'timestamp':           datetime.now().isoformat(),
        'n_boot':              N_BOOT,
        'min_trials':          MIN_TRIALS,
        'min_sessions':        MIN_SESSIONS,
        'extreme_margin':      EXTREME_MARGIN,
        'mu_bounds':           list(MU_BOUNDS),
        'sigma_bounds':        list(SIGMA_BOUNDS),
        'gamma_bounds':        list(GAMMA_BOUNDS),
        'lapse_bounds':        list(LAPSE_BOUNDS),
        'fit_degenerate_sigma': FIT_DEGENERATE_SIGMA,
        'acc_tiers':           ACC_TIERS,
        'sigma_tiers':         SIGMA_TIERS,
        'lapse_tiers':         LAPSE_TIERS,
        'filter_kwargs':       {k: str(v) for k, v in FILTER_KWARGS.items()},
        'filter_states':       list(filter_states),
        'rats':                [int(r) for r in rats],
        'conditions':          [str(c) for c in conditions],
    }
    with open(OUTPUT_DIR / 'run_config.json', 'w') as f:
        json.dump(run_config, f, indent=2)

    print(f"\n{'=' * 90}")
    print(f"Task performance check — is each rat doing the task?")
    print(f"  Rats             : {len(rats)}  ({rats})")
    print(f"  Modalities       : {list(modalities)}")
    print(f"  Filter states    : {list(filter_states)}")
    print(f"  Bootstrap iters  : {N_BOOT}")
    print(f"  σ bounds         : {SIGMA_BOUNDS}   "
          f"(fit_degenerate flag: σ_97.5 > {FIT_DEGENERATE_SIGMA})")
    print(f"  Output directory : {OUTPUT_DIR.resolve()}")
    print(f"  Tier thresholds  :")
    print(f"    accuracy       : good ≥ {ACC_TIERS['good']:.2f},  "
          f"medium ≥ {ACC_TIERS['medium']:.2f}")
    print(f"    σ 97.5-pctile  : good ≤ {SIGMA_TIERS['good']:.1f}°, "
          f"medium ≤ {SIGMA_TIERS['medium']:.1f}°")
    print(f"    γ + λ          : good ≤ {LAPSE_TIERS['good']:.2f}, "
          f"medium ≤ {LAPSE_TIERS['medium']:.2f}")
    print(f"{'=' * 90}")

    all_results = {}

    for filter_state, df in datasets.items():
        print(f"\n{'#' * 90}\n#  FILTER STATE: {filter_state}\n{'#' * 90}")
        for rat in rats:
            df_rat = df[df[RAT_COL] == rat].copy()
            if len(df_rat) == 0:
                print(f"\n  Rat {rat}: no data in this filter state, skipping.")
                continue

            print(f"\n{'-' * 70}\nRat {rat}  [{filter_state}]\n{'-' * 70}")

            for i, cond in enumerate(conditions):
                df_cond = df_rat if cond == 'pooled' else df_rat[df_rat[MODALITY_COL] == cond]
                n = len(df_cond)
                n_ses = int(df_cond[SESSION_COL].nunique()) if n > 0 else 0
                print(f"\n  Condition: {cond}  (n={n} trials, {n_ses} sessions)")
                if n < MIN_TRIALS:
                    print(f"    ⚠ Skipping: fewer than {MIN_TRIALS} trials")
                    continue

                # Distinct seed per (rat, cond, filter_state) → filter-independent bootstraps
                fs_idx = list(filter_states).index(filter_state)
                seed   = int(10000 * rat + 100 * i + fs_idx)

                boot = bootstrap_fit(df_cond, seed=seed)
                if boot['popt'] is None:
                    print(f"    ⚠ Fit failed")
                    continue

                perf           = performance_metrics(df_cond)
                classification = classify_task_performance(perf, boot)

                mu, sigma, gamma, lapse = boot['popt']
                ci_ts = perf['ci_two_sided']
                print(f"    μ={mu:.1f}°  σ={sigma:.1f}°  γ={gamma:.3f}  λ={lapse:.3f}")
                print(f"    accuracy (extremes) = {perf['accuracy']*100:.1f}%  "
                      f"[{ci_ts[0]*100:.1f}, {ci_ts[1]*100:.1f}]  "
                      f"(n={perf['n_extreme']}, binom p={perf['binom_p']:.1e})")
                print(f"    σ 97.5-percentile = {classification['sigma_97_5']:.2f}°"
                      f"{'  [DEGENERATE]' if classification['fit_degenerate'] else ''}  |  "
                      f"γ + λ = {classification['lapse_sum']:.3f}")
                print(f"    tiers → acc: {classification['acc_tier']},  "
                      f"σ: {classification['sigma_tier']},  "
                      f"lapse: {classification['lapse_tier']}")
                print(f"    → OVERALL: {classification['classification']}")

                # Figure
                plot_psychometric(df_cond, boot, perf, classification,
                                  rat_id=rat, condition=cond,
                                  filter_state=filter_state)

                # ---- assemble per-cell raw record and pickle it
                cell = {
                    'rat':            int(rat),
                    'condition':      cond,
                    'filter_state':   filter_state,
                    'n_trials':       int(n),
                    'n_sessions':     n_ses,
                    'popt':           boot['popt'],
                    'param_boot':     boot['param_boot'],
                    'n_valid':        boot['n_valid'],
                    'ci_valid':       boot['ci_valid'],
                    'perf':           perf,
                    'classification': classification,
                    'config': {
                        'seed':                 seed,
                        'n_boot':               N_BOOT,
                        'min_trials':           MIN_TRIALS,
                        'min_sessions':         MIN_SESSIONS,
                        'extreme_margin':       EXTREME_MARGIN,
                        'mu_bounds':            list(MU_BOUNDS),
                        'sigma_bounds':         list(SIGMA_BOUNDS),
                        'gamma_bounds':         list(GAMMA_BOUNDS),
                        'lapse_bounds':         list(LAPSE_BOUNDS),
                        'fit_degenerate_sigma': FIT_DEGENERATE_SIGMA,
                        'acc_tiers':            ACC_TIERS,
                        'sigma_tiers':          SIGMA_TIERS,
                        'lapse_tiers':          LAPSE_TIERS,
                    },
                }

                raw_subdir = RAW_SUBDIR / filter_state
                raw_subdir.mkdir(parents=True, exist_ok=True)
                pkl_path = raw_subdir / f'rat_{rat}_{cond}.pkl'
                with open(pkl_path, 'wb') as f:
                    pickle.dump(cell, f)

                all_results[(rat, cond, filter_state)] = cell

    # ---- master pickle of every cell
    with open(RAW_SUBDIR / 'all_cells.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\n✓ Master raw pickle saved: {RAW_SUBDIR / 'all_cells.pkl'}")

    summary_df = summarise(all_results, filter_states)
    return all_results, summary_df


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    datasets = load_datasets()
    data_availability(datasets)
    results, summary = run_analysis(datasets)