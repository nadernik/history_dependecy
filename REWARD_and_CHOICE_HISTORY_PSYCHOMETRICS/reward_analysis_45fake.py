"""
Psychometric Curve Analysis
============================
Effect of n-1 trial reward on trial n choice,
conditioned on n-1 stimulus ≈ 45° (fake 45° from 40°+50°)
and n-1 choice (tunable).

Pipeline:
1. Filter by n-1 choice (PREV_CHOICE_CONDITION)
2. Build fake 45° pool (equal N from 40° and 50°, without replacement)
3. Balance rewarded vs unrewarded trial counts (with replacement)
4. Fit cumulative Gaussian per rat per reward condition (MLE)
5. Bootstrap pointwise 95% CI on curves and on Δμ
6. Plot curves + Δμ per rat

Column name mapping (your dataframe):
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
PREV_CHOICE_CONDITION = 0     # 1 = previous choice vertical, 0 = horizontal
N_BOOTSTRAP           = 100   # bootstrap iterations (use 1000 for final figures)
X_PLOT                = np.linspace(0, 90, 200)   # x-axis for curve evaluation
MIN_TRIALS_PER_ANGLE  = 5     # minimum trials per angle to include in fit
MINIMAL_FIT           = False # True = fix gamma & lapse, fit only mu & sigma

# =============================================================================
# COLUMN NAMES — edit here if your dataframe changes
# =============================================================================

COL_STIM       = 'angle'        # stimulus angle at trial n
COL_STIM_NM1   = 'angle_n-1'   # stimulus angle at trial n-1
COL_CHOICE     = 'action'       # choice at trial n
COL_CHOICE_NM1 = 'action_n-1'  # choice at trial n-1
COL_REWARD     = 'hitmiss_n-1'  # reward at trial n-1
COL_RAT        = 'rat'          # subject ID

# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=HISTORY_DEPTH, save_figures=False)
an.load_and_preprocess_data()
df = an.create_lagged_features()

# =============================================================================
# SECTION 1 — Simulate example data (replace df with your real DataFrame)
# =============================================================================

def simulate_data(n_subjects=4, n_trials=1000, seed=42):
    """
    Generate synthetic data mimicking the real structure.
    Replace this with your actual data loading step.
    """
    rng    = np.random.default_rng(seed)
    angles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    dfs    = []
    for sid in range(1, n_subjects + 1):
        stim_n     = rng.choice(angles, size=n_trials)
        stim_nm1   = rng.choice(angles, size=n_trials)
        p_vert     = norm.cdf((stim_n - 45) / 15)
        choice_n   = (rng.random(n_trials) < p_vert).astype(int)
        p_vert_nm1 = norm.cdf((stim_nm1 - 45) / 15)
        choice_nm1 = (rng.random(n_trials) < p_vert_nm1).astype(int)
        # deterministic reward: rewarded if choice matches ground truth side
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
    Negative log-likelihood for a cumulative Gaussian with lapse and guess rate.

    Model: P(vertical) = gamma + (1 - gamma - lapse) * Φ((x - mu) / sigma)

    Bernoulli likelihood per trial:  p^c * (1-p)^(1-c)
    Log-likelihood summed over trials: Σ [ c*log(p) + (1-c)*log(1-p) ]
    We return the negative because scipy.optimize.minimize minimizes.

    Parameters
    ----------
    params : [mu, sigma, gamma, lapse]
    x      : stimulus angles, shape (N,)
    c      : binary choices (1=vertical), shape (N,)
    """
    mu, sigma, gamma, lapse = params
    p = gamma + (1 - gamma - lapse) * norm.cdf((x - mu) / sigma)
    p = np.clip(p, 1e-10, 1 - 1e-10)   # guard against log(0)
    return -np.sum(c * np.log(p) + (1 - c) * np.log(1 - p))


def neg_log_likelihood_minimal(params, x, c):
    """2-parameter version with gamma and lapse fixed at 0.05."""
    mu, sigma = params
    return neg_log_likelihood([mu, sigma, 0.05, 0.05], x, c)


def fit_psychometric(df, min_trials=MIN_TRIALS_PER_ANGLE, minimal_fit=MINIMAL_FIT):
    """
    Fit cumulative Gaussian with lapses to binary choice data using MLE.

    Parameters
    ----------
    df          : DataFrame with columns [COL_STIM, COL_CHOICE]
    min_trials  : minimum trials per angle required (data quality filter)
    minimal_fit : if True, fix gamma=lapse=0.05 and fit only mu, sigma

    Returns
    -------
    dict with keys {mu, sigma, gamma, lapse, success} or None if insufficient data
    """
    # data quality filter: drop angles with too few trials
    counts       = df.groupby(COL_STIM)[COL_CHOICE].count()
    valid_angles = counts[counts >= min_trials].index
    df           = df[df[COL_STIM].isin(valid_angles)]

    if df[COL_STIM].nunique() < 3:
        return None  # not enough angles to constrain the curve shape

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
# SECTION 3 — Trial preparation (fake 45° + reward balancing)
# =============================================================================

def prepare_trials(df):
    """
    Build balanced fake 45° pool from 40° and 50° n-1 trials.

    Step 1: sample equal N without replacement from 40° and 50° pools
            → approximates n-1 stimulus = 45° symmetrically
            → without replacement because we are selecting real trials,
              not estimating a distribution (no duplicates in the pool)

    Step 2: balance rewarded vs unrewarded with replacement
            → with replacement because this is the bootstrap sampling step:
              we want to capture variability in the reward distribution

    Input:  DataFrame already filtered by n-1 choice,
            containing only rows where COL_STIM_NM1 ∈ {40, 50}
    Output: (df_rewarded, df_unrewarded) — balanced, ready for fitting
            or (None, None) if one reward condition is empty
    """
    pool_40 = df[df[COL_STIM_NM1] == 40]
    pool_50 = df[df[COL_STIM_NM1] == 50]

    # step 1: equal sampling without replacement → fake 45°
    n1      = min(len(pool_40), len(pool_50))
    samp_40 = pool_40.sample(n=n1, replace=False)
    samp_50 = pool_50.sample(n=n1, replace=False)
    df_45   = pd.concat([samp_40, samp_50], ignore_index=True)

    # step 2: balance reward conditions with replacement (bootstrap)
    rew   = df_45[df_45[COL_REWARD] == 1]
    unrew = df_45[df_45[COL_REWARD] == 0]
    n2    = min(len(rew), len(unrew))

    if n2 == 0:
        return None, None  # degenerate: one condition has no trials at all

    df_rewarded   = rew.sample(n=n2,  replace=True)
    df_unrewarded = unrew.sample(n=n2, replace=True)

    return df_rewarded, df_unrewarded

# =============================================================================
# SECTION 4 — Bootstrap
# =============================================================================

def bootstrap_psychometric(df_fake45, n_bootstrap=N_BOOTSTRAP,
                            minimal_fit=MINIMAL_FIT):
    """
    Run bootstrap to estimate parameter distributions for both reward conditions.

    Each iteration:
        - resamples the fake 45° pool and balances reward conditions
        - fits a psychometric curve to rewarded and unrewarded trials
        - stores fitted parameters (without 'success' flag)

    Δμ is computed by pairing rewarded and unrewarded estimates
    iteration-by-iteration: both come from the same sampled dataset,
    so pairing correctly captures their covariance.

    Input:  df_fake45  — filtered by n-1 choice, COL_STIM_NM1 ∈ {40, 50}
    Output: dict with keys 'rewarded' and 'unrewarded',
            each a DataFrame of shape (n_valid_iters, 4): [mu, sigma, gamma, lapse]
    """
    params_rew   = []
    params_unrew = []
    n_skipped    = 0

    for i in range(n_bootstrap):
        df_rew, df_unrew = prepare_trials(df_fake45)

        # handle degenerate prepare_trials output
        if df_rew is None or df_unrew is None:
            n_skipped += 1
            continue

        fit_rew   = fit_psychometric(df_rew,   minimal_fit=minimal_fit)
        fit_unrew = fit_psychometric(df_unrew, minimal_fit=minimal_fit)

        # skip if either fit returned None (insufficient data after sampling)
        if not fit_rew or not fit_unrew:
            n_skipped += 1
            continue

        # warn if optimizer did not converge (but keep the iteration)
        if not fit_rew['success'] or not fit_unrew['success']:
            print(f"  ⚠ Bootstrap iter {i}: optimizer did not converge")

        # remove 'success' flag before storing — we only need the 4 parameters
        fit_rew.pop('success')
        fit_unrew.pop('success')

        params_rew.append(fit_rew)
        params_unrew.append(fit_unrew)

    if n_skipped > 0:
        print(f"  ℹ {n_skipped}/{n_bootstrap} bootstrap iterations skipped "
              f"(insufficient data after sampling)")
    if len(params_rew) < 20:
        print(f"  ⚠ Only {len(params_rew)} valid iterations — CIs may be unreliable")

    return {
        'rewarded':   pd.DataFrame(params_rew),
        'unrewarded': pd.DataFrame(params_unrew)
    }

# =============================================================================
# SECTION 5 — Plotting
# =============================================================================

def plot_psychometric(boot_results, df_45_full, rat_id, x_plot=X_PLOT):
    """
    Plot psychometric curves with bootstrap CI and Δμ.

    Panel 1: rewarded vs unrewarded curves with pointwise 95% CI
             + dotted reference curve fitted on the full unbalanced df_45_full
             (reference reflects real trial distribution without resampling)
    Panel 2: Δμ = mu_rewarded - mu_unrewarded with 95% bootstrap CI

    Parameters
    ----------
    boot_results : output of bootstrap_psychometric
    df_45_full   : full fake 45° dataset (unbalanced, for reference curve only)
    rat_id       : used in plot title and filename
    x_plot       : angles at which to evaluate curves for plotting
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = {'rewarded': '#2196F3', 'unrewarded': '#F44336'}
    labels = {'rewarded': 'Rewarded (n-1)', 'unrewarded': 'Unrewarded (n-1)'}

    # --- panel 1: psychometric curves ---
    for condition in ['rewarded', 'unrewarded']:
        params = boot_results[condition]   # DataFrame: (n_bootstrap, 4)
        color  = colors[condition]

        # evaluate curve at x_plot for every bootstrap iteration
        # shape: (n_bootstrap, len(x_plot))
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

        # median curve (robust to outlier fits) + pointwise 95% CI
        median_curve = np.median(curves, axis=0)
        ci_low       = np.percentile(curves,  2.5, axis=0)
        ci_high      = np.percentile(curves, 97.5, axis=0)

        axes[0].plot(x_plot, median_curve, color=color,
                     label=labels[condition], lw=2)
        axes[0].fill_between(x_plot, ci_low, ci_high,
                             alpha=0.25, color=color)

    # reference curves: fitted on full unbalanced df_45_full (dotted)
    # one per reward condition, showing the fit without any resampling
    for cond_val, color in [(1, colors['rewarded']),
                             (0, colors['unrewarded'])]:
        ref_df  = df_45_full[df_45_full[COL_REWARD] == cond_val]
        ref_fit = fit_psychometric(ref_df)
        if ref_fit is not None:
            ref_curve = eval_psychometric(
                x_plot, ref_fit['mu'], ref_fit['sigma'],
                ref_fit['gamma'], ref_fit['lapse']
            )
            axes[0].plot(x_plot, ref_curve, color=color,
                         lw=1.5, linestyle='--', alpha=0.6,
                         label=f"{'Rew' if cond_val else 'Unrew'} reference")

    axes[0].axvline(45,  color='gray', linestyle=':', alpha=0.5, label='45°')
    axes[0].axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    axes[0].set_xlabel('Stimulus angle (°)', fontsize=12)
    axes[0].set_ylabel('P(vertical)', fontsize=12)
    axes[0].set_title(f'Rat {rat_id} — psychometric curves\n'
                      f'(dashed = reference fit on full pool)', fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(-0.05, 1.05)

    # --- panel 2: Δμ with CI ---
    # pair iteration-by-iteration: same sampled data → captures covariance
    delta_mu    = (boot_results['rewarded']['mu'].values -
                   boot_results['unrewarded']['mu'].values)
    med_delta   = np.median(delta_mu)
    ci_low_d    = np.percentile(delta_mu,  2.5)
    ci_high_d   = np.percentile(delta_mu, 97.5)

    axes[1].errorbar(
        x=0, y=med_delta,
        yerr=[[med_delta - ci_low_d], [ci_high_d - med_delta]],
        fmt='o', color='black', capsize=8, markersize=10, lw=2
    )
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.6)
    axes[1].set_xlim(-1, 1)
    axes[1].set_xticks([])
    axes[1].set_ylabel('Δμ  (rewarded − unrewarded) [°]', fontsize=12)
    axes[1].set_title(f'Rat {rat_id} — shift in perceptual boundary\n'
                      f'95% bootstrap CI', fontsize=11)
    axes[1].annotate(
        f'Δμ = {med_delta:.2f}°\n95% CI [{ci_low_d:.2f}, {ci_high_d:.2f}]',
        xy=(0.05, 0.92), xycoords='axes fraction', fontsize=10,
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
    )

    prev_label = 'vertical' if PREV_CHOICE_CONDITION == 1 else 'horizontal'
    plt.suptitle(
        f'Rat {rat_id}  |  n-1 choice = {prev_label}  |  '
        f'n-1 stimulus ≈ 45°  |  {len(delta_mu)} bootstrap iterations',
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    fname = f'rat_{rat_id}_prev_choice_{PREV_CHOICE_CONDITION}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Figure saved: {fname}")

# =============================================================================
# SECTION 6 — Main loop
# =============================================================================

def run_analysis(df):
    """
    Main entry point. Iterates over rats and runs the full pipeline for each.

    Parameters
    ----------
    df : your full DataFrame with all columns defined at the top of this script
    """
    rats = df[COL_RAT].unique()
    print(f"\n{'='*60}")
    print(f"Psychometric curve analysis")
    print(f"  n-1 choice condition : {'vertical' if PREV_CHOICE_CONDITION == 1 else 'horizontal'}")
    print(f"  Bootstrap iterations : {N_BOOTSTRAP}")
    print(f"  Rats found           : {len(rats)}")
    print(f"{'='*60}\n")

    # step 1: filter by n-1 choice — tunable one-liner
    df_choice = df[df[COL_CHOICE_NM1] == PREV_CHOICE_CONDITION].copy()
    print(f"After n-1 choice filter ({PREV_CHOICE_CONDITION}): "
          f"{len(df_choice)} / {len(df)} trials retained\n")

    # step 2: keep only 40° and 50° n-1 trials (ingredients for fake 45°)
    df_pool = df_choice[df_choice[COL_STIM_NM1].isin([40, 50])].copy()
    print(f"After fake-45° angle filter (40° + 50°): "
          f"{len(df_pool)} trials retained\n")

    results = {}

    for rat in sorted(rats):
        print(f"--- Rat {rat} ---")
        df_rat = df_pool[df_pool[COL_RAT] == rat].copy()

        # essential data checks before running anything
        n40  = (df_rat[COL_STIM_NM1] == 40).sum()
        n50  = (df_rat[COL_STIM_NM1] == 50).sum()
        nrew = (df_rat[COL_REWARD] == 1).sum()
        nunr = (df_rat[COL_REWARD] == 0).sum()
        print(f"  n-1 angle  : 40°={n40}, 50°={n50}")
        print(f"  reward     : rewarded={nrew}, unrewarded={nunr}")

        if min(n40, n50) < 10:
            print(f"  ⚠ Skipping: too few trials in 40° or 50° pool\n")
            continue
        if min(nrew, nunr) < 10:
            print(f"  ⚠ Skipping: too few trials in one reward condition\n")
            continue

        # bootstrap
        print(f"  Running {N_BOOTSTRAP} bootstrap iterations...")
        boot_results = bootstrap_psychometric(df_rat)
        n_valid = len(boot_results['rewarded'])
        print(f"  ✓ {n_valid} valid iterations completed")

        # plot
        plot_psychometric(boot_results, df_rat, rat_id=rat)

        # store per-rat summary
        delta_mu = (boot_results['rewarded']['mu'].values -
                    boot_results['unrewarded']['mu'].values)
        results[rat] = {
            'delta_mu_median':   np.median(delta_mu),
            'delta_mu_ci_low':   np.percentile(delta_mu,  2.5),
            'delta_mu_ci_high':  np.percentile(delta_mu, 97.5),
            'n_bootstrap_valid': n_valid
        }
        print(f"  Δμ = {results[rat]['delta_mu_median']:.2f}° "
              f"[{results[rat]['delta_mu_ci_low']:.2f}, "
              f"{results[rat]['delta_mu_ci_high']:.2f}]\n")

    # summary table across all rats
    if results:
        print(f"\n{'='*60}")
        print("Summary across rats:")
        summary = pd.DataFrame(results).T
        print(summary.to_string(float_format='{:.2f}'.format))
        print(f"{'='*60}\n")

    return results

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    # replace with your real data:
    # df = pd.read_csv('your_data.csv')
    print(f"dataset: {len(df)} trials, "
          f"{df[COL_RAT].nunique()} rats\n")

    results = run_analysis(df)