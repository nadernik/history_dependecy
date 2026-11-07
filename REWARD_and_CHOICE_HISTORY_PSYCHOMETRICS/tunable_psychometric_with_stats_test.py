"""
Psychometric Curve Analysis — Choice Effects
=============================================
Analyses the effect of n-1 choice on trial n choice,
conditioned on n-1 stimulus ≈ 45°.

Three analyses:
    1) Real 45° trials, split by n-1 choice (vertical vs horizontal)
    2) Fake 45° (from flanking angles), split by n-1 choice
    3) Fake 45°, split by n-1 choice, conditioned on n-1 reward (tunable)

All three share the same prepare_trials, bootstrap, and plot functions.
The only difference is the arguments passed to each.

Column name mapping:
    angle       → stimulus angle at trial n
    angle_n-1   → stimulus angle at trial n-1
    action      → choice at trial n       (1=vertical, 0=horizontal)
    action_n-1  → choice at trial n-1
    hitmiss_n-1 → reward at trial n-1     (1=rewarded, 0=not)
    rat         → subject ID
"""
from choice_pattern_classification import (
    compute_S_and_D,
    classify_choice_pattern,
    plot_choice_pattern_classification,
    summarise_classifications,
)
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
import os
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "figures_choice_effects_2")
TABLE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "tables_choice_effects_2")
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(TABLE_DIR,  exist_ok=True)

# =============================================================================
# TUNABLE PARAMETERS
# =============================================================================

HISTORY_DEPTH = 1
N_BOOTSTRAP          = 10    # bootstrap iterations (use 1000 for final figures)
N_SYMMETRY_SAMPLES   = 100  # samples for independent resampling of S 100000
X_PLOT               = np.linspace(0, 90, 200)
MIN_TRIALS_PER_ANGLE = 5
MINIMAL_FIT          = False

# Analysis 3 only: which reward condition to condition on
REWARD_CONDITION = 1          # 1 = conditioned on rewarded, 0 = unrewarded

# Per-rat PSE flanking angles for fake 45°
PSE_FLANKS = {
    1: (40, 50),
    2: (40, 50),
    3: (40, 50),
    5: (40, 50),
    6: (40, 50),
    7: (40, 50),
    9: (40, 50),
    10: (40, 50),
    11: (40, 50),
    12: (40, 50),
    13: (40, 50),
    16: (40, 50),
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

# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=HISTORY_DEPTH, save_figures=False)
an.load_and_preprocess_data()
df = an.create_lagged_features()


# =============================================================================
# SECTION 1 — Simulate example data
# =============================================================================



def simulate_data(n_subjects=4, n_trials=1000, seed=42):
    rng    = np.random.default_rng(seed)
    angles = [0, 10, 20, 30, 40, 45, 50, 60, 70, 80, 90]
    dfs    = []
    for sid in range(1, n_subjects + 1):
        stim_n      = rng.choice(angles, size=n_trials)
        stim_nm1    = rng.choice(angles, size=n_trials)
        p_vert      = norm.cdf((stim_n - 45) / 15)
        choice_n    = (rng.random(n_trials) < p_vert).astype(int)
        p_vert_nm1  = norm.cdf((stim_nm1 - 45) / 15)
        choice_nm1  = (rng.random(n_trials) < p_vert_nm1).astype(int)
        correct_nm1 = (stim_nm1 > 45).astype(int)
        reward_nm1  = (choice_nm1 == correct_nm1).astype(int)
        dfs.append(pd.DataFrame({
            COL_RAT:        sid,
            COL_STIM:       stim_n,
            COL_STIM_NM1:   stim_nm1,
            COL_CHOICE:     choice_n,
            COL_CHOICE_NM1: choice_nm1,
            COL_REWARD:     reward_nm1,
        }))
    return pd.concat(dfs, ignore_index=True)

# =============================================================================
# SECTION 2 — Core fitting functions
# =============================================================================

def neg_log_likelihood(params, x, c):
    """
    Negative log-likelihood for cumulative Gaussian with lapse and guess rate.
    Model: P(vertical) = gamma + (1 - gamma - lapse) * Φ((x - mu) / sigma)
    Bernoulli likelihood: p^c * (1-p)^(1-c)
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

    Returns dict {mu, sigma, gamma, lapse, success} or None if insufficient data.
    """
    counts       = df.groupby(COL_STIM)[COL_CHOICE].count()
    valid_angles = counts[counts >= min_trials].index
    df           = df[df[COL_STIM].isin(valid_angles)]

    if df[COL_STIM].nunique() < 3:
        return None

    x = df[COL_STIM].values
    c = df[COL_CHOICE].values

    if minimal_fit:
        x0     = [45, 10]
        bounds = [(0, 90), (1, 50)]
        result = minimize(neg_log_likelihood_minimal, x0, args=(x, c),
                          bounds=bounds, method='L-BFGS-B')
        mu, sigma    = result.x
        gamma, lapse = 0.05, 0.05
    else:
        x0     = [45,   10,   0.05, 0.05]
        bounds = [(0, 90), (1, 50), (0, 0.1), (0, 0.1)]
        result = minimize(neg_log_likelihood, x0, args=(x, c),
                          bounds=bounds, method='L-BFGS-B')
        mu, sigma, gamma, lapse = result.x

    return {'mu': mu, 'sigma': sigma, 'gamma': gamma,
            'lapse': lapse, 'success': result.success}


def eval_psychometric(x, mu, sigma, gamma, lapse):
    """Evaluate the psychometric function at points x given parameters."""
    return gamma + (1 - gamma - lapse) * norm.cdf((x - mu) / sigma)

# =============================================================================
# SECTION 3 — Trial preparation
# =============================================================================

def prepare_trials(df, use_fake_45, split_by,
                   condition_on=None, pse_flanks=(40, 50)):
    """
    Build a balanced trial pool for psychometric curve fitting.

    Parameters
    ----------
    df           : DataFrame for a single rat
    use_fake_45  : bool — True = sample from pse_flanks, False = filter real 45°
    split_by     : str  — column that defines the two output groups (0 and 1)
    condition_on : dict — {'col': col_name, 'val': value} or None
    pse_flanks   : tuple — (low_angle, high_angle) flanking the PSE

    Returns
    -------
    (df_group_0, df_group_1) — balanced with replacement
    or (None, None) if one group is empty
    """
    # step 1: build the 45° dataset (real or fake)
    if use_fake_45:
        pool_low  = df[df[COL_STIM_NM1] == pse_flanks[0]]
        pool_high = df[df[COL_STIM_NM1] == pse_flanks[1]]
        n1        = min(len(pool_low), len(pool_high))
        if n1 == 0:
            return None, None
        df_45 = pd.concat([
            pool_low.sample(n=n1,  replace=False),
            pool_high.sample(n=n1, replace=False)
        ], ignore_index=True)
    else:
        df_45 = df[df[COL_STIM_NM1] == 45].copy()

    # step 2: apply conditioning filter if requested
    if condition_on is not None:
        df_45 = df_45[df_45[condition_on['col']] == condition_on['val']]

    # step 3: split by target column and balance with replacement
    group_0 = df_45[df_45[split_by] == 0]
    group_1 = df_45[df_45[split_by] == 1]
    n2      = min(len(group_0), len(group_1))

    if n2 == 0:
        return None, None

    return group_0.sample(n=n2, replace=True), group_1.sample(n=n2, replace=True)

# =============================================================================
# SECTION 4 — Bootstrap
# =============================================================================

def bootstrap_psychometric(df_rat, use_fake_45, split_by,
                            condition_on=None, pse_flanks=(40, 50),
                            n_bootstrap=N_BOOTSTRAP, minimal_fit=MINIMAL_FIT):
    """
    Run bootstrap to estimate parameter distributions for both groups.

    Returns dict with keys 'group_0' and 'group_1',
    each a DataFrame of shape (n_valid_iters, 4): [mu, sigma, gamma, lapse]
    """
    params_0  = []
    params_1  = []
    n_skipped = 0

    for i in range(n_bootstrap):
        df_g0, df_g1 = prepare_trials(
            df_rat,
            use_fake_45=use_fake_45,
            split_by=split_by,
            condition_on=condition_on,
            pse_flanks=pse_flanks
        )

        if df_g0 is None or df_g1 is None:
            n_skipped += 1
            continue

        fit_0 = fit_psychometric(df_g0, minimal_fit=minimal_fit)
        fit_1 = fit_psychometric(df_g1, minimal_fit=minimal_fit)

        if not fit_0 or not fit_1:
            n_skipped += 1
            continue

        if not fit_0['success'] or not fit_1['success']:
            print(f"  ⚠ Bootstrap iter {i}: optimizer did not converge")

        fit_0.pop('success')
        fit_1.pop('success')

        params_0.append(fit_0)
        params_1.append(fit_1)

    if n_skipped > 0:
        print(f"  ℹ {n_skipped}/{n_bootstrap} iterations skipped")
    if len(params_0) < 20:
        print(f"  ⚠ Only {len(params_0)} valid iterations — CIs may be unreliable")

    return {
        'group_0': pd.DataFrame(params_0),
        'group_1': pd.DataFrame(params_1)
    }

# =============================================================================
# SECTION 5 — Plotting
# =============================================================================

def plot_psychometric(boot_results, df_45_full, rat_id,
                      split_by, condition_on=None,
                      labels=('group 0', 'group 1'),
                      analysis_label='',
                      x_plot=X_PLOT,
                      show_reference=False,
                        ):
    """
    Plot psychometric curves with bootstrap CI and Δμ.

    Panel 1: two curves (group_0 vs group_1) with pointwise 95% CI
             + dotted reference curve on full unbalanced df_45_full
    Panel 2: Δμ = mu_group_1 - mu_group_0 with 95% bootstrap CI

    Parameters
    ----------
    boot_results   : output of bootstrap_psychometric
    df_45_full     : full 45° dataset (unbalanced, for reference curve)
    rat_id         : used in title and filename
    split_by       : column defining the two groups (for reference curve split)
    condition_on   : dict or None (applied to reference curve too)
    labels         : tuple of strings — (label_group_0, label_group_1)
    analysis_label : short string for filename and suptitle
    x_plot         : angles at which to evaluate curves
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors    = {'group_0': "#5C36F4", 'group_1': "#F32128"}
    label_map = {'group_0': labels[0], 'group_1': labels[1]}

    condition_str = (
        f"conditioned on {condition_on['col']}={condition_on['val']}"
        if condition_on is not None else 'no conditioning'
    )

    # --- panel 1: psychometric curves ---
    for group in ['group_0', 'group_1']:
        params = boot_results[group]
        color  = colors[group]

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
        ci_low       = np.percentile(curves,  2.5, axis=0)
        ci_high      = np.percentile(curves, 97.5, axis=0)

        axes[0].plot(x_plot, median_curve, color=color,
                     label=label_map[group], lw=2)
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
    axes[0].set_title(f'Rat {rat_id} — psychometric curves\n'
                      f'({condition_str}{ref_note})', fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(-0.05, 1.05)

    # --- panel 2: Δμ ---
    delta_mu  = (boot_results['group_1']['mu'].values -
                 boot_results['group_0']['mu'].values)
    med_delta = np.median(delta_mu)
    ci_low_d  = np.percentile(delta_mu,  2.5)
    ci_high_d = np.percentile(delta_mu, 97.5)

    axes[1].errorbar(
    x=0, y=med_delta,
    yerr=[[med_delta - ci_low_d], [ci_high_d - med_delta]],
    fmt='o', color='black', capsize=8, markersize=10, lw=2
    )

    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.6)
    axes[1].set_xlim(-1, 1)
    axes[1].set_xticks([])
    axes[1].set_ylabel(f'Δμ  ({labels[1]} − {labels[0]}) [°]', fontsize=12)
    axes[1].set_title(f'Rat {rat_id} — shift in perceptual boundary\n'
                      f'95% bootstrap CI', fontsize=11)

    # explicit y-limits with headroom for the annotation and the title
    data_min = min(ci_low_d,  0.0)
    data_max = max(ci_high_d, 0.0)
    data_range = max(data_max - data_min, 1.0)
    axes[1].set_ylim(data_min - 0.15 * data_range,
                     data_max + 0.35 * data_range)

    # annotation placed relative to data, not axes fraction
    axes[1].text(
        0, data_max + 0.20 * data_range,
        f'Δμ = {med_delta:.2f}°\n95% CI [{ci_low_d:.2f}, {ci_high_d:.2f}]',
        ha='center', va='center', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='gray', alpha=0.85)
    )

    plt.suptitle(
        f'Rat {rat_id}  |  {analysis_label}  |  {condition_str}\n'
        f'{len(delta_mu)} bootstrap iterations',
        fontsize=12, y=1.01
    )
    stem = f'rat_{rat_id}_{analysis_label}'
    for ext in ('png', 'svg'):
        path = os.path.join(FIGURE_DIR, f'{stem}.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Figure saved: {path}")
    plt.close(fig)

# =============================================================================
# SECTION 6 — Analysis definitions
# =============================================================================

# Each analysis is a dict of arguments passed to bootstrap_psychometric
# and plot_psychometric. Adding a new analysis = adding one dict here.
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





def plot_symmetry_test(s_distribution, delta_mus, rat_id,
                       n_bootstrap_1, n_bootstrap_2):
    """
    Plot the side symmetry test for a single rat.

    Panel 1: histogram of S = Δμ_vertical + Δμ_horizontal
             with 95% CI and null line at 0
    Panel 2: Δμ for each choice side separately, for reference

    Parameters
    ----------
    s_distribution     : array of S values from independent resampling
    delta_mus          : dict with keys 'vertical' and 'horizontal'
    rat_id             : used in title and filename
    n_bootstrap_vert   : number of valid bootstrap iterations for vertical
    n_bootstrap_horiz  : number of valid bootstrap iterations for horizontal
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- panel 1: S distribution ---
    med_s     = np.median(s_distribution)
    ci_low_s  = np.percentile(s_distribution,  2.5)
    ci_high_s = np.percentile(s_distribution, 97.5)

    axes[0].hist(s_distribution, bins=80, color='steelblue',
                 alpha=0.7, density=True)
    axes[0].axvline(0,       color='red',   linestyle='--',
                    lw=2,    label='null (S=0)')
    axes[0].axvline(med_s,   color='black', linestyle='-',
                    lw=2,    label=f'median S = {med_s:.2f}°')
    axes[0].axvline(ci_low_s,  color='black', linestyle=':',
                    lw=1.5,  alpha=0.7)
    axes[0].axvline(ci_high_s, color='black', linestyle=':',
                    lw=1.5,  alpha=0.7, label=f'95% CI [{ci_low_s:.2f}, {ci_high_s:.2f}]')
    axes[0].set_xlabel('S = Δμ_vertical + Δμ_horizontal [°]', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title(f'Rat {rat_id} — side symmetry test\n'
                      f'S ≈ 0 → reward effects symmetric across sides',
                      fontsize=11)
    axes[0].legend(fontsize=9)

    # annotate whether CI includes 0
    includes_zero = ci_low_s <= 0 <= ci_high_s
    conclusion = 'CI includes 0 → consistent with symmetry' if includes_zero \
                 else 'CI excludes 0 → asymmetry detected'
    axes[0].annotate(
        conclusion,
        xy=(0.05, 0.97), xycoords='axes fraction', fontsize=10,
        va='top', ha='left',
        color='green' if includes_zero else 'red',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
    )

    # --- panel 2: Δμ per side for reference ---
    for i, (side, color) in enumerate([('rew',   "#21F33A"),
                                        ('unrew', "#F436D1")]):
        d         = delta_mus[side]
        med_d     = np.median(d)
        ci_low_d  = np.percentile(d,  2.5)
        ci_high_d = np.percentile(d, 97.5)
        axes[1].errorbar(
            x=i, y=med_d,
            yerr=[[med_d - ci_low_d], [ci_high_d - med_d]],
            fmt='o', color=color, capsize=8, markersize=10, lw=2,
            label=f'n-1 {side}'
        )

    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.6)
    axes[1].set_xlim(-0.5, 1.5)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['n-1 rewarded', 'n-1 unrewarded'], fontsize=11)
    axes[1].set_ylabel('Δμ  (vertical − horizontal) [°]', fontsize=12)
    axes[1].set_title(f'Rat {rat_id} — Δμ per choice side\n'
                      f'(for reference)', fontsize=11)

    plt.suptitle(
        f'Rat {rat_id}  |  Reward symmetry test  |  '
        f'rew: {n_bootstrap_1} iters, '
        f'unrew: {n_bootstrap_2} iters  |  '
        f'{N_SYMMETRY_SAMPLES} resamples for S',
        fontsize=11, y=1.01
    )
    stem = f'rat_{rat_id}_symmetry_test'
    for ext in ('png', 'svg'):
        path = os.path.join(FIGURE_DIR, f'{stem}.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Figure saved: {path}")
    plt.close(fig)

# =============================================================================
# Table export — LaTeX (booktabs) + CSV
# =============================================================================

def _fmt_pm(x):
    return f"{x:+.2f}".replace('-', '−')


def _df_to_booktabs(df, column_format, header_labels, caption, label):
    n_cols = len(df.columns)
    assert len(column_format) >= n_cols and len(header_labels) == n_cols
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
    lines += [r'    \bottomrule', r'  \end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def export_tables(all_results):
    """
    Writes four tables (CSV + LaTeX) to TABLE_DIR:
      1. delta_mu_per_analysis  — Δμ from analyses 1–4 per rat
      2. choice_pattern         — S, D, classification per rat
      3. pure_reward            — P, classification per rat
      4. summary_combined       — one row per rat combining the above
    """

    # ---- 1. Δμ per analysis --------------------------------------------
    rows = []
    for rat, rat_res in all_results.items():
        for label, res in rat_res.items():
            if not label.startswith('analysis_'):
                continue
            rows.append({
                'rat':               int(rat),
                'analysis':          label,
                'delta_mu':          res['delta_mu_median'],
                'ci_lo':             res['delta_mu_ci_low'],
                'ci_hi':             res['delta_mu_ci_high'],
                'n_bootstrap_valid': res['n_bootstrap_valid'],
            })
    if rows:
        raw = pd.DataFrame(rows)
        raw.to_csv(os.path.join(TABLE_DIR, 'delta_mu_per_analysis.csv'),
                    index=False, float_format='%.4f')
        tex_df = pd.DataFrame({
            'Rat':               raw['rat'],
            'Analysis':          raw['analysis'].str.replace('_', r'\_'),
            r'$\Delta\mu$ (°)':  raw['delta_mu'].map(_fmt_pm),
            r'95\% CI':          [f"[{_fmt_pm(lo)}, {_fmt_pm(hi)}]"
                                   for lo, hi in zip(raw['ci_lo'], raw['ci_hi'])],
            r'$n_{\mathrm{boot}}$': raw['n_bootstrap_valid'],
        })
        tex = _df_to_booktabs(
            tex_df, column_format='rlrlr',
            header_labels=list(tex_df.columns),
            caption=(r"$\Delta\mu$ (shift in perceptual boundary) per analysis "
                     r"and per rat, with 95\% bootstrap CIs."),
            label='tab:choice_delta_mu',
        )
        with open(os.path.join(TABLE_DIR, 'delta_mu_per_analysis.tex'), 'w',
                  encoding='utf-8') as f:
            f.write(tex + '\n')

    # ---- 2. Choice pattern classification (S + D) ----------------------
    rows = []
    for rat, rat_res in all_results.items():
        c = rat_res.get('choice_pattern_classification')
        if c is None:
            continue
        rows.append({
            'rat':                    int(rat),
            'delta_mu_rew':           c['delta_mu_rew_median'],
            'delta_mu_unrew':         c['delta_mu_unrew_median'],
            'S':                      c['S_median'],
            'S_ci_lo':                c['S_ci_low'],
            'S_ci_hi':                c['S_ci_high'],
            'S_excludes_0':           c['S_excludes_0'],
            'D':                      c['D_median'],
            'D_ci_lo':                c['D_ci_low'],
            'D_ci_hi':                c['D_ci_high'],
            'D_excludes_0':           c['D_excludes_0'],
            'p_same_sign':            c['p_same_sign'],
            'classification':         c['classification'],
        })
    if rows:
        raw = pd.DataFrame(rows)
        raw.to_csv(os.path.join(TABLE_DIR, 'choice_pattern.csv'),
                    index=False, float_format='%.4f')
        tex_df = pd.DataFrame({
            'Rat':                        raw['rat'],
            r'$\Delta\mu_{\mathrm{rew}}$':   raw['delta_mu_rew'].map(_fmt_pm),
            r'$\Delta\mu_{\mathrm{unrew}}$': raw['delta_mu_unrew'].map(_fmt_pm),
            r'$S$ (°)':                    raw['S'].map(_fmt_pm),
            r'95\% CI $S$':                [f"[{_fmt_pm(lo)}, {_fmt_pm(hi)}]"
                                             for lo, hi in zip(raw['S_ci_lo'], raw['S_ci_hi'])],
            r'$D$ (°)':                    raw['D'].map(_fmt_pm),
            r'95\% CI $D$':                [f"[{_fmt_pm(lo)}, {_fmt_pm(hi)}]"
                                             for lo, hi in zip(raw['D_ci_lo'], raw['D_ci_hi'])],
            'Class.':                      raw['classification'].str.replace('_', r'\_'),
        })
        tex = _df_to_booktabs(
            tex_df, column_format='rrrrlrlll',
            header_labels=list(tex_df.columns),
            caption=(r"Choice-pattern classification per rat. "
                     r"$S = \Delta\mu_{\mathrm{rew}} + \Delta\mu_{\mathrm{unrew}}$, "
                     r"$D = \Delta\mu_{\mathrm{rew}} - \Delta\mu_{\mathrm{unrew}}$. "
                     r"CIs are 95\% bootstrap intervals."),
            label='tab:choice_pattern',
        )
        with open(os.path.join(TABLE_DIR, 'choice_pattern.tex'), 'w',
                  encoding='utf-8') as f:
            f.write(tex + '\n')

    # ---- 3. Pure-reward test --------------------------------------------
    rows = []
    for rat, rat_res in all_results.items():
        p = rat_res.get('pure_reward_classification')
        if p is None:
            continue
        rows.append({
            'rat':            int(rat),
            'P':              p['P_median'],
            'P_ci_lo':        p['P_ci_low'],
            'P_ci_hi':        p['P_ci_high'],
            'P_excludes_0':   p['P_excludes_0'],
            'classification': p['classification'],
        })
    if rows:
        raw = pd.DataFrame(rows)
        raw.to_csv(os.path.join(TABLE_DIR, 'pure_reward.csv'),
                    index=False, float_format='%.4f')
        tex_df = pd.DataFrame({
            'Rat':          raw['rat'],
            r'$P$ (°)':     raw['P'].map(_fmt_pm),
            r'95\% CI':     [f"[{_fmt_pm(lo)}, {_fmt_pm(hi)}]"
                              for lo, hi in zip(raw['P_ci_lo'], raw['P_ci_hi'])],
            'Excl. 0':      raw['P_excludes_0'].map(lambda b: 'yes' if b else 'no'),
            'Class.':       raw['classification'].str.replace('_', r'\_'),
        })
        tex = _df_to_booktabs(
            tex_df, column_format='rrlll',
            header_labels=list(tex_df.columns),
            caption=(r"Pure-reward test per rat. "
                     r"$P = \bar\mu_{\mathrm{unrew}} - \bar\mu_{\mathrm{rew}}$ "
                     r"averaged over previous choice."),
            label='tab:pure_reward',
        )
        with open(os.path.join(TABLE_DIR, 'pure_reward.tex'), 'w',
                  encoding='utf-8') as f:
            f.write(tex + '\n')

    # ---- 4. Combined summary -------------------------------------------
    combined_rows = []
    for rat, rat_res in all_results.items():
        row = {'rat': int(rat)}
        for label, res in rat_res.items():
            if label.startswith('analysis_'):
                short = label.replace('analysis_', '').replace('_', '\n')
                row[f'Δμ_{short}'] = res['delta_mu_median']
        cp = rat_res.get('choice_pattern_classification', {})
        pr = rat_res.get('pure_reward_classification',   {})
        row['S']              = cp.get('S_median',  np.nan)
        row['D']              = cp.get('D_median',  np.nan)
        row['choice_class']   = cp.get('classification', '—')
        row['P']              = pr.get('P_median',  np.nan)
        row['pure_class']     = pr.get('classification', '—')
        combined_rows.append(row)
    if combined_rows:
        pd.DataFrame(combined_rows).to_csv(
            os.path.join(TABLE_DIR, 'summary_combined.csv'),
            index=False, float_format='%.4f',
        )

    print(f"\n✓ Tables written to {TABLE_DIR}")

# =============================================================================
# SECTION 7 — Main loop
# =============================================================================
def run_analysis(df):
    """
    Main entry point. Runs all three analyses for each rat.
    """
    rats = df[COL_RAT].unique()
    print(f"\n{'='*60}")
    print(f"Choice effect psychometric analysis")
    print(f"  Bootstrap iterations : {N_BOOTSTRAP}")
    print(f"  Rats found           : {len(rats)}")
    print(f"  Analyses to run      : {len(ANALYSES)}")
    print(f"{'='*60}\n")

    all_results = {}
    for rat in sorted(rats):
        delta_mus = {}
        boot_valid ={}
        pse_flanks = PSE_FLANKS.get(rat, (40, 50))
        print(f"{'='*60}")
        print(f"Rat {rat}  |  PSE flanks: {pse_flanks}")
        print(f"{'='*60}")

        df_rat = df[df[COL_RAT] == rat].copy()
        all_results[rat] = {}
        mu_dists = {}                              # NEW
        report_choice_distribution(df_rat, rat, pse_flanks)   # NEW

        for analysis in ANALYSES:
            print(f"\n  [{analysis['label']}] {analysis['description']}")

            # build the appropriate 45° pool for data checks and reference curve
            if analysis['use_fake_45']:
                df_45_full = df_rat[
                    df_rat[COL_STIM_NM1].isin(pse_flanks)
                ].copy()
                pool_desc = f"40°={( df_rat[COL_STIM_NM1] == pse_flanks[0]).sum()}, " \
                            f"50°={(df_rat[COL_STIM_NM1] == pse_flanks[1]).sum()}"
            else:
                df_45_full = df_rat[df_rat[COL_STIM_NM1] == 45].copy()
                pool_desc  = f"real 45°={len(df_45_full)}"

            # apply conditioning to check counts
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

            # run bootstrap
            print(f"  Running {N_BOOTSTRAP} bootstrap iterations...")
            boot_results = bootstrap_psychometric(
                df_rat,
                use_fake_45=analysis['use_fake_45'],
                split_by=analysis['split_by'],
                condition_on=analysis['condition_on'],
                pse_flanks=pse_flanks,
            )
            n_valid = len(boot_results['group_0'])
            print(f"  ✓ {n_valid} valid iterations completed")

            # plot
            plot_psychometric(
                boot_results,
                df_45_full=df_45_full,
                rat_id=rat,
                split_by=analysis['split_by'],
                condition_on=analysis['condition_on'],
                labels=analysis['labels'],
                analysis_label=analysis['label'],
            )

            # store summary
            delta_mu = (boot_results['group_1']['mu'].values -
                        boot_results['group_0']['mu'].values)
            all_results[rat][analysis['label']] = {
                'delta_mu_median':   np.median(delta_mu),
                'delta_mu_ci_low':   np.percentile(delta_mu,  2.5),
                'delta_mu_ci_high':  np.percentile(delta_mu, 97.5),
                'n_bootstrap_valid': n_valid,
            }
            print(f"  Δμ = {np.median(delta_mu):.2f}° "
                  f"[{np.percentile(delta_mu, 2.5):.2f}, "
                  f"{np.percentile(delta_mu, 97.5):.2f}]")
            
            if analysis['label'] == 'analysis_3_fake45_reward_controlled':
                # store Δμ distribution for symmetry test
                delta_mus['rew'] = delta_mu
                boot_valid['rew'] = n_valid
                mu_dists['mu_h_rew'] = boot_results['group_0']['mu'].values   # NEW
                mu_dists['mu_v_rew'] = boot_results['group_1']['mu'].values   # NEW
            elif analysis['label'] == 'analysis_4_fake45_reward_controlled_opposite': 
                # store Δμ distribution for symmetry test
                delta_mus['unrew'] = delta_mu
                boot_valid['unrew'] = n_valid
                mu_dists['mu_h_unrew'] = boot_results['group_0']['mu'].values # NEW
                mu_dists['mu_v_unrew'] = boot_results['group_1']['mu'].values # NEW
            # --- choice pattern classification (S + D, only if both fitted) ---
            if 'rew' in delta_mus and 'unrew' in delta_mus:
                print(f"\n  --- Choice pattern classification (S + D) ---")

                # 1. paired S and D distributions via independent resampling
                joint = compute_S_and_D(delta_mus['rew'], delta_mus['unrew'])

                # 2. classify (mirrors the GLM β_success / β_failure rule)
                classification = classify_choice_pattern(
                    delta_mus['rew'], delta_mus['unrew'], joint,
                )

                # 3. 2×2 summary figure
                plot_choice_pattern_classification(
                    delta_mus['rew'], delta_mus['unrew'],
                    joint, classification,
                    rat_id=rat,
                    n_bootstrap_rew=boot_valid.get('rew',   0),
                    n_bootstrap_unrew=boot_valid.get('unrew', 0),
                )

                # 4. store
                all_results[rat]['choice_pattern_classification'] = {
                    'classification':        classification['classification'],
                    'delta_mu_rew_median':   float(np.median(delta_mus['rew'])),
                    'delta_mu_unrew_median': float(np.median(delta_mus['unrew'])),
                    'S_median':              float(np.median(joint['S'])),
                    'S_ci_low':              classification['S_ci'][0],
                    'S_ci_high':             classification['S_ci'][1],
                    'D_median':              float(np.median(joint['D'])),
                    'D_ci_low':              classification['D_ci'][0],
                    'D_ci_high':             classification['D_ci'][1],
                    'p_same_sign':           classification['p_same_sign'],
                    'D_excludes_0':          classification['D_excludes_0'],
                    'S_excludes_0':          classification['S_excludes_0'],
                    'rew_excludes_0':        classification['rew_excludes_0'],
                    'unrew_excludes_0':      classification['unrew_excludes_0'],
                }
                S_ci = classification['S_ci']
                D_ci = classification['D_ci']
                print(f"  S = {np.median(joint['S']):+.2f}°  "
                    f"[{S_ci[0]:+.2f}, {S_ci[1]:+.2f}]  "
                    f"({'excludes' if classification['S_excludes_0'] else 'includes'} 0)")
                print(f"  D = {np.median(joint['D']):+.2f}°  "
                    f"[{D_ci[0]:+.2f}, {D_ci[1]:+.2f}]  "
                    f"({'excludes' if classification['D_excludes_0'] else 'includes'} 0)")
                print(f"  p(same sign) = {classification['p_same_sign']:.2f}")
                print(f"  → {classification['classification']}\n")
            else:
                print(f"  ⚠ Classification skipped: one or both reward "
                    f"conditions missing\n")

        # --- pure reward test (P) ---
        required_keys = ['mu_v_rew', 'mu_h_rew', 'mu_v_unrew', 'mu_h_unrew']
        if all(k in mu_dists for k in required_keys):
            print(f"\n  --- Pure reward test (P) ---")
            P_result = compute_P(
                mu_dists['mu_v_rew'],   mu_dists['mu_h_rew'],
                mu_dists['mu_v_unrew'], mu_dists['mu_h_unrew'],
            )
            P_class = classify_pure_reward(P_result['P'])

            n_iters = min(boot_valid.get('rew', 0),
                            boot_valid.get('unrew', 0))
            plot_pure_reward(P_result, mu_dists, P_class,
                                rat_id=rat, n_bootstrap_iters=n_iters)

            all_results[rat]['pure_reward_classification'] = {
                'classification': P_class['classification'],
                'P_median':       float(np.median(P_result['P'])),
                'P_ci_low':       P_class['P_ci'][0],
                'P_ci_high':      P_class['P_ci'][1],
                'P_excludes_0':   P_class['P_excludes_0'],
            }
            ci_lo, ci_hi = P_class['P_ci']
            print(f"  P = {np.median(P_result['P']):+.2f}°  "
                    f"[{ci_lo:+.2f}, {ci_hi:+.2f}]  "
                    f"({'excludes' if P_class['P_excludes_0'] else 'includes'} 0)")
            print(f"  → {P_class['classification']}\n")

    # summary table
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
    export_tables(all_results)

    return all_results

# =============================================================================
# Choice-distribution diagnostic + pure-reward test (P)
# =============================================================================

def report_choice_distribution(df_rat, rat_id, pse_flanks):
    """Print prev-choice distribution within each reward state at fake 45°.
    
    If the previous-V vs previous-H split within each reward state is heavily
    imbalanced, the naive pool-all-rewarded psychometric fit would be
    contaminated by any reward × choice component. The weighted-averaging P
    below algebraically cancels that contamination, so this diagnostic is
    mainly informational — but strong imbalance is also a reason to trust
    the weighted P more than a pooled fit."""
    df_45 = df_rat[df_rat[COL_STIM_NM1].isin(pse_flanks)]
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
              n_samples=N_SYMMETRY_SAMPLES):
    """Compute P (pure-reward effect) via independent resampling.
    
    P = μ_unrew_avg − μ_rew_avg
      = [(μ_prev-H,unrew + μ_prev-V,unrew) / 2]
      − [(μ_prev-H,rew   + μ_prev-V,rew  ) / 2]
    
    Positive P → less vertical evidence needed to report vertical after
    reward → pure-reward bias toward vertical (matches positive β_hit).
    
    Equal weighting inside each outcome pool algebraically cancels the
    reward × choice component (which is antisymmetric in previous choice),
    so P isolates the choice-agnostic outcome-driven bias regardless of
    how imbalanced the actual previous-choice counts are within each pool."""
    rng = np.random.default_rng()
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


def plot_pure_reward(P_result, mu_dists, classification, rat_id,
                     n_bootstrap_iters):
    """Two-panel plot: P distribution (left) + per-cell μ estimates (right)."""
    P = P_result['P']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    med_p = np.median(P)
    ci_low, ci_high = classification['P_ci']
    axes[0].hist(P, bins=80, color='darkorange', alpha=0.7, density=True)
    axes[0].axvline(0,      color='red',   linestyle='--', lw=2, label='null (P=0)')
    axes[0].axvline(med_p,  color='black', linestyle='-',  lw=2,
                    label=f'median P = {med_p:.2f}°')
    axes[0].axvline(ci_low,  color='black', linestyle=':', lw=1.5, alpha=0.7)
    axes[0].axvline(ci_high, color='black', linestyle=':', lw=1.5, alpha=0.7,
                    label=f'95% CI [{ci_low:.2f}, {ci_high:.2f}]')
    axes[0].set_xlabel('P = μ_unrew_avg − μ_rew_avg  [°]', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title(f'Rat {rat_id} — pure-reward test\n'
                      f'P > 0 → boundary shifts toward vertical after reward',
                      fontsize=11)
    axes[0].legend(fontsize=9)
    conclusion_color = 'green' if classification['P_excludes_0'] else 'gray'
    axes[0].annotate(f"→ {classification['classification']}",
                     xy=(0.05, 0.97), xycoords='axes fraction',
                     fontsize=10, va='top', ha='left', color=conclusion_color,
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='white', alpha=0.8))

    keys   = ['mu_h_rew', 'mu_v_rew', 'mu_h_unrew', 'mu_v_unrew']
    labels = ['prev-H\nrew', 'prev-V\nrew', 'prev-H\nunrew', 'prev-V\nunrew']
    cols   = ['#21F33A', '#0FA023', '#F436D1', '#B0289F']
    for i, (k, c) in enumerate(zip(keys, cols)):
        d = mu_dists[k]
        med = np.median(d)
        lo  = np.percentile(d,  2.5)
        hi  = np.percentile(d, 97.5)
        axes[1].errorbar(x=i, y=med, yerr=[[med - lo], [hi - med]],
                         fmt='o', color=c, capsize=8, markersize=10, lw=2)
    axes[1].axhline(45, color='gray', linestyle=':', alpha=0.5, label='45°')
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(labels, fontsize=10)
    axes[1].set_ylabel('μ (PSE) [°]', fontsize=12)
    axes[1].set_title(f'Rat {rat_id} — μ per (prev-choice × outcome) cell',
                      fontsize=11)
    axes[1].legend(fontsize=9)

    plt.suptitle(
        f'Rat {rat_id}  |  Pure-reward test  |  '
        f'{n_bootstrap_iters} bootstrap iters, {len(P)} resamples for P',
        fontsize=11, y=1.01,
    )
    stem = f'rat_{rat_id}__symmetry_test_rew'
    for ext in ('png', 'svg'):
        path = os.path.join(FIGURE_DIR, f'{stem}.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Figure saved: {path}")
    plt.close(fig)

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    # replace with your real data:
    # df = pd.read_csv('your_data.csv')

    results = run_analysis(df)
    




   