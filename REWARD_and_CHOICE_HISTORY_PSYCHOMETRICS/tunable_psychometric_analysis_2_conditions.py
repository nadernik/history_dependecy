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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from serial_dependence_analysis import SerialDependenceAnalyzer

# =============================================================================
# TUNABLE PARAMETERS — change these to re-run the analysis
# =============================================================================


HISTORY_DEPTH = 1
N_BOOTSTRAP          = 1000    # bootstrap iterations (use 1000 for final figures)
N_SYMMETRY_SAMPLES   = 10000  # samples for independent resampling of S
X_PLOT               = np.linspace(0, 90, 200)
MIN_TRIALS_PER_ANGLE = 5
MINIMAL_FIT          = False

# Analysis 3 only: which reward condition to condition on
REWARD_CONDITION = 1          # 1 = conditioned on rewarded, 0 = unrewarded

# Per-rat PSE flanking angles for fake 45°
PSE_FLANKS = {
    11: (40, 50),
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
modalities = df["mod"].unique()
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

    return group_0.sample(n=n2, replace=True), group_1.sample(n=n2, replace=True), None
'''


def prepare_trials(df, use_fake_45, split_by,
                   condition_on=None, pse_flanks=(40, 50),
                   col_stim='angle_n-1',
                   col_choice='action_n-1',
                   col_reward='hitmiss_n-1'):
    """
    Build a balanced trial pool for psychometric curve fitting.

    Parameters
    ----------
    df           : DataFrame for a single rat
    use_fake_45  : bool — True = sample from pse_flanks with cell balancing
                          False = filter real 45° trials, balance split_by only
    split_by     : str  — column that defines the two output groups (0 and 1)
    condition_on : dict — {'col': col_name, 'val': value} or None
    pse_flanks   : tuple — (low_angle, high_angle) flanking the PSE
    col_stim     : str  — n-1 stimulus column
    col_choice   : str  — n-1 choice column
    col_reward   : str  — n-1 reward/correct column

    Returns
    -------
    (df_group_0, df_group_1, cell_counts)
        df_group_0/1 : balanced DataFrames split by `split_by`,
                       or (None, None, cell_counts) if any cell is empty
        cell_counts  : dict of n per cell before equalization
                       (None when use_fake_45=False)

    Balancing logic:
        - 8 cells: (stim: low/high) x (choice: 0/1) x (reward: 0/1)
          when condition_on does NOT fix the reward column
        - 4 cells: (stim: low/high) x (choice: 0/1)
          when condition_on fixes reward to a single value
        All cells downsampled to n_min, no replacement.
    """

    # ------------------------------------------------------------------ #
    #  BRANCH A: flank-based (fake 45) — cell balancing                   #
    # ------------------------------------------------------------------ #
    if use_fake_45:
        low, high = pse_flanks

        # step 1: filter to flank stimuli
        df_flanks = df[df[col_stim].isin([low, high])].copy()

        # step 2: apply conditioning filter if requested
        if condition_on is not None:
            df_flanks = df_flanks[df_flanks[condition_on['col']] == condition_on['val']]

        # step 3: build the 4 structurally valid cells only
        # at low flank: correct=horizontal(0), error=vertical(1)
        # at high flank: correct=vertical(1), error=horizontal(0)
        # → these are the only 4 combinations that actually exist in the data
        valid_cells = [
            (low,  0, 1),   # low flank, choice,  correct,
            (low,  1, 0),   # low flank, choice, error, 
            (high, 1, 1),   # high flank, choice, correct, 
            (high, 0, 0),   # high flank, choice, error
        ]

        cells = {}
        for (stim, choice, reward) in valid_cells:
            mask = (
                (df_flanks[col_stim]   == stim)   &
                (df_flanks[col_choice] == choice) &
                (df_flanks[col_reward] == reward)
            )
            cells[(stim, choice, reward)] = df_flanks[mask]

        cell_counts = {k: len(v) for k, v in cells.items()}
        n_min = min(cell_counts.values())

        if n_min == 0:
            empty = [k for k, v in cell_counts.items() if v == 0]
            print(f"[prepare_trials] Empty cells: {empty}")
            return None, None, cell_counts

        if n_min < 15:
            print(f"[prepare_trials] Warning: smallest cell has only {n_min} trials.")

        # step 4: downsample each cell to n_min and concatenate
        # result: 4 * n_min trials, uniform across stim / choice / reward
        balanced = pd.concat(
            [cell_df.sample(n=n_min, replace=False) for cell_df in cells.values()],
            ignore_index=True
        )

        # step 5: split by target column
        group_0 = balanced[balanced[split_by] == 0]
        group_1 = balanced[balanced[split_by] == 1]

        if len(group_0) == 0 or len(group_1) == 0:
            print(f"[prepare_trials] One split_by group is empty after balancing.")
            return None, None, cell_counts

        return group_0, group_1, cell_counts

    # ------------------------------------------------------------------ #
    #  BRANCH B: real 45° trials — original logic unchanged               #
    # ------------------------------------------------------------------ #
    else:
        df_45 = df[df[col_stim] == 45].copy()

        if condition_on is not None:
            df_45 = df_45[df_45[condition_on['col']] == condition_on['val']]

        group_0 = df_45[df_45[split_by] == 0]
        group_1 = df_45[df_45[split_by] == 1]
        n2      = min(len(group_0), len(group_1))

        if n2 == 0:
            return None, None, None

        return group_0.sample(n=n2, replace=True), group_1.sample(n=n2, replace=True), None
'''
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
        df_g0, df_g1, counts = prepare_trials(
            df_rat,
            use_fake_45=use_fake_45,
            split_by=split_by,
            condition_on=condition_on,
            pse_flanks=pse_flanks
        )
        print(counts)  # <-- add this line to print cell counts for each iteration
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
                      x_plot=X_PLOT):
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

    colors    = {'group_0': '#2196F3', 'group_1': '#F44336'}
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
    axes[0].set_title(f'Rat {rat_id} — psychometric curves\n'
                      f'(dashed = reference, {condition_str})', fontsize=11)
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
    axes[1].annotate(
        f'Δμ = {med_delta:.2f}°\n95% CI [{ci_low_d:.2f}, {ci_high_d:.2f}]',
        xy=(0.05, 0.92), xycoords='axes fraction', fontsize=10,
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
    )

    plt.suptitle(
        f'Rat {rat_id}  |  {analysis_label}  |  {condition_str}\n'
        f'{len(delta_mu)} bootstrap iterations',
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    fname = f'rat_{rat_id}_{analysis_label}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Figure saved: {fname}")


def plot_symmetry_test(s_distribution, delta_mus, rat_id,
                       n_bootstrap_1, n_bootstrap_2):
    """
    Plot the side symmetry test for a single rat.

    Panel 1: histogram of S = Δμ_rew + Δμ_unrew
             with 95% CI and null line at 0
    Panel 2: Δμ for each reward value separately, for reference

    Parameters
    ----------
    s_distribution     : array of S values from independent resampling
    delta_mus          : dict with keys 'rew' and 'unrew'
    rat_id             : used in title and filename
    n_bootstrap_rew   : number of valid bootstrap iterations for rew
    n_bootstrap_unrew  : number of valid bootstrap iterations for unrew
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
    axes[0].set_xlabel('S = Δμ_rew + Δμ_unrew [°]', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title(f'Rat {rat_id} — side symmetry test\n'
                      f'S ≈ 0 → choice effects symmetric across rew values',
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
    for i, (side, color) in enumerate([('rew',   "#21F352"),
                                        ('unrew', "#F436E1")]):
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
    axes[1].set_xticklabels(['n-1 rew', 'n-1 unrew'], fontsize=11)
    axes[1].set_ylabel('Δμ  (vertical − horizontal) [°]', fontsize=12)
    axes[1].set_title(f'Rat {rat_id} — Δμ per rew value\n'
                      f'(for reference)', fontsize=11)

    plt.suptitle(
        f'Rat {rat_id}  |  Rew symmetry test  |  '
        f'vertical: {n_bootstrap_1} iters, '
        f'horizontal: {n_bootstrap_2} iters  |  '
        f'{N_SYMMETRY_SAMPLES} resamples for S',
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    fname = f'rat_{rat_id}_symmetry_test_rew.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Figure saved: {fname}")

# =============================================================================
# SECTION 6 — Analysis definitions
# =============================================================================

# Each analysis is a dict of arguments passed to bootstrap_psychometric
# and plot_psychometric. Adding a new analysis = adding one dict here.
#"""
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
    'condition_on': {'col': COL_REWARD, 'val': 1 - REWARD_CONDITION},  # opposite
    'labels':       ('n-1 horizontal', 'n-1 vertical'),
    'description':  f'Fake 45° | split by n-1 choice | n-1 reward={1 - REWARD_CONDITION}',
},
]
"""

ANALYSES = [
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
    'condition_on': {'col': COL_REWARD, 'val': 1 - REWARD_CONDITION},  # opposite
    'labels':       ('n-1 horizontal', 'n-1 vertical'),
    'description':  f'Fake 45° | split by n-1 choice | n-1 reward={1 - REWARD_CONDITION}',
},
]
"""
# =============================================================================
# SECTION 7 — Main loop
# =============================================================================

def run_analysis(df):
    """
    Main entry point. Runs all three analyses for each rat.
    """
    rats = [10]  # for quick testing; change to sorted(df[COL_RAT].unique()) for all rats
    #rats = [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
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
            elif analysis['label'] == 'analysis_4_fake45_reward_controlled_opposite': 
                # store Δμ distribution for symmetry test
                delta_mus['unrew'] = delta_mu
                boot_valid['unrew'] = n_valid
                # --- symmetry test (only if both rew were successfully fitted) ---
                if 'rew' in delta_mus and 'unrew' in delta_mus:
                    print(f"\n  --- Side symmetry test ---")

                    # independent resampling: no pairing justified across separate runs
                    s_distribution = (
                        np.random.choice(delta_mus['rew'],
                                        size=N_SYMMETRY_SAMPLES, replace=True) +
                        np.random.choice(delta_mus['unrew'],
                                        size=N_SYMMETRY_SAMPLES, replace=True)
                    )

                    plot_symmetry_test(
                        s_distribution, delta_mus, rat_id=rat,
                        n_bootstrap_1=boot_valid.get('rew',   0),
                        n_bootstrap_2=boot_valid.get('unrew', 0)
                    )

                    med_s    = np.median(s_distribution)
                    ci_low_s = np.percentile(s_distribution,  2.5)
                    ci_high_s= np.percentile(s_distribution, 97.5)
                    symmetric = ci_low_s <= 0 <= ci_high_s

                    all_results[rat]['symmetry_test'] = {
                        'delta_mu_rew_median':   np.median(delta_mus['rew']),
                        'delta_mu_unrew_median': np.median(delta_mus['unrew']),
                        'S_median':    med_s,
                        'S_ci_low':    ci_low_s,
                        'S_ci_high':   ci_high_s,
                        'symmetric':   symmetric,
                    }
                    print(f"  S = {med_s:.2f}° [{ci_low_s:.2f}, {ci_high_s:.2f}] "
                        f"→ {'symmetric ✓' if symmetric else 'asymmetric ✗'}\n")
                else:
                    print(f"  ⚠ Symmetry test skipped: one or both sides missing\n")

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

    return all_results

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    # replace with your real data:
    # df = pd.read_csv('your_data.csv')

    
#print(f"Simulated dataset: {len(df)} trials, "
          #f"{df[COL_RAT].nunique()} rats\n")
    modalities = sorted(df["mod"].unique())
    for mod in modalities:
        print(f"Modality '{mod}': {len(df[df['mod'] == mod])} trials")
        df_mod = df[df['mod'] == mod].copy()
        results = run_analysis(df_mod)


"""
Test side symmetry using your existing reward analysis (two Δμ values, check if they're equal and opposite)
If side symmetry holds → average of all four groups is a valid neutral reference
With that reference → you can now measure whether reward-yes and reward-no produce symmetric shifts, 
because you have an anchor that isn't contaminated by either reward condition
"""