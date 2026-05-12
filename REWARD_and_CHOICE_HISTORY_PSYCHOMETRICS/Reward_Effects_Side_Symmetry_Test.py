"""
Psychometric Curve Analysis — Reward Effects + Side Symmetry Test
==================================================================
Analyses the effect of n-1 reward on trial n choice,
conditioned on n-1 stimulus ≈ 45° (fake 45° from flanking angles).

For each rat:
    1. Run reward analysis for n-1 choice = vertical
    2. Run reward analysis for n-1 choice = horizontal
    3. Test side symmetry: S = Δμ_vertical + Δμ_horizontal
       If side symmetry holds (S ≈ 0), reward effects are equal and opposite
       across the two choice sides, and the average of all four groups
       provides a valid neutral reference curve.

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
# TUNABLE PARAMETERS
# =============================================================================

HISTORY_DEPTH = 1
N_BOOTSTRAP          = 1000    # bootstrap iterations (use 1000 for final figures)
N_SYMMETRY_SAMPLES   = 10000  # samples for independent resampling of S
X_PLOT               = np.linspace(0, 90, 200)
MIN_TRIALS_PER_ANGLE = 5
MINIMAL_FIT          = False

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
    angles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
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

def prepare_trials(df, pse_flanks=(40, 50)):
    """
    Build fake 45° pool and balance reward conditions.

    Step 1: sample equal N without replacement from flanking angle pools
    Step 2: balance rewarded vs unrewarded with replacement (bootstrap)

    Returns (df_rewarded, df_unrewarded) or (None, None) if a condition is empty.
    """
    pool_low  = df[df[COL_STIM_NM1] == pse_flanks[0]]
    pool_high = df[df[COL_STIM_NM1] == pse_flanks[1]]
    n1        = min(len(pool_low), len(pool_high))

    if n1 == 0:
        return None, None

    df_45 = pd.concat([
        pool_low.sample(n=n1,  replace=False),
        pool_high.sample(n=n1, replace=False)
    ], ignore_index=True)

    rew   = df_45[df_45[COL_REWARD] == 1]
    unrew = df_45[df_45[COL_REWARD] == 0]
    n2    = min(len(rew), len(unrew))

    if n2 == 0:
        return None, None

    return rew.sample(n=n2, replace=True), unrew.sample(n=n2, replace=True)

# =============================================================================
# SECTION 4 — Bootstrap
# =============================================================================

def bootstrap_psychometric(df_rat, pse_flanks=(40, 50),
                            n_bootstrap=N_BOOTSTRAP,
                            minimal_fit=MINIMAL_FIT):
    """
    Run bootstrap to estimate parameter distributions for rewarded vs unrewarded.

    Returns dict with keys 'rewarded' and 'unrewarded',
    each a DataFrame of shape (n_valid_iters, 4): [mu, sigma, gamma, lapse]
    """
    params_rew   = []
    params_unrew = []
    n_skipped    = 0

    for i in range(n_bootstrap):
        df_rew, df_unrew = prepare_trials(df_rat, pse_flanks=pse_flanks)

        if df_rew is None or df_unrew is None:
            n_skipped += 1
            continue

        fit_rew   = fit_psychometric(df_rew,   minimal_fit=minimal_fit)
        fit_unrew = fit_psychometric(df_unrew, minimal_fit=minimal_fit)

        if not fit_rew or not fit_unrew:
            n_skipped += 1
            continue

        if not fit_rew['success'] or not fit_unrew['success']:
            print(f"  ⚠ Bootstrap iter {i}: optimizer did not converge")

        fit_rew.pop('success')
        fit_unrew.pop('success')

        params_rew.append(fit_rew)
        params_unrew.append(fit_unrew)

    if n_skipped > 0:
        print(f"  ℹ {n_skipped}/{n_bootstrap} iterations skipped")
    if len(params_rew) < 20:
        print(f"  ⚠ Only {len(params_rew)} valid iterations — CIs may be unreliable")

    return {
        'rewarded':   pd.DataFrame(params_rew),
        'unrewarded': pd.DataFrame(params_unrew)
    }

# =============================================================================
# SECTION 5 — Plotting
# =============================================================================

def plot_psychometric(boot_results, df_rat, rat_id,
                      choice_label, choice_condition, pse_flanks=(40, 50),
                      x_plot=X_PLOT):
    """
    Plot psychometric curves (rewarded vs unrewarded) with bootstrap CI and Δμ.

    Panel 1: two curves with pointwise 95% CI + dotted reference curve
    Panel 2: Δμ = mu_rewarded - mu_unrewarded with 95% bootstrap CI
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = {'rewarded': "#21F333", 'unrewarded': "#F436E4"}
    labels = {'rewarded': 'Rewarded (n-1)', 'unrewarded': 'Unrewarded (n-1)'}

    # --- panel 1: psychometric curves ---
    for condition in ['rewarded', 'unrewarded']:
        params = boot_results[condition]
        color  = colors[condition]

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
                     label=labels[condition], lw=2)
        axes[0].fill_between(x_plot, ci_low, ci_high,
                             alpha=0.25, color=color)

    # reference curves on full unbalanced dataset (dotted)
    #df_45_full = df_rat[df_rat[COL_STIM_NM1].isin(pse_flanks)].copy()
    df_45_full = df_rat[
        (df_rat[COL_CHOICE_NM1] == choice_condition) &
        (df_rat[COL_STIM_NM1].isin(pse_flanks))
    ].copy()

    for cond_val, color in [(1, colors['rewarded']),
                             (0, colors['unrewarded'])]:
        ref_df  = df_45_full[df_45_full[COL_REWARD] == cond_val]
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
    axes[0].set_title(f'Rat {rat_id} — reward effect\n'
                      f'n-1 choice = {choice_label} | dashed = reference',
                      fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(-0.05, 1.05)

    # --- panel 2: Δμ ---
    delta_mu  = (boot_results['rewarded']['mu'].values -
                 boot_results['unrewarded']['mu'].values)
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
    axes[1].set_ylabel('Δμ  (rewarded − unrewarded) [°]', fontsize=12)
    axes[1].set_title(f'Rat {rat_id} — shift in perceptual boundary\n'
                      f'95% bootstrap CI', fontsize=11)
    axes[1].annotate(
        f'Δμ = {med_delta:.2f}°\n95% CI [{ci_low_d:.2f}, {ci_high_d:.2f}]',
        xy=(0.05, 0.92), xycoords='axes fraction', fontsize=10,
        va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
    )

    plt.suptitle(
        f'Rat {rat_id}  |  n-1 choice = {choice_label}  |  '
        f'n-1 stimulus ≈ 45°  |  {len(delta_mu)} bootstrap iterations',
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    fname = f'rat_{rat_id}_reward_effect_{choice_label}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Figure saved: {fname}")

    return delta_mu  # return for symmetry test


def plot_symmetry_test(s_distribution, delta_mus, rat_id,
                       n_bootstrap_vert, n_bootstrap_horiz):
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
    for i, (side, color) in enumerate([('vertical',   "#F32121"),
                                        ('horizontal', "#364FF4")]):
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
    axes[1].set_xticklabels(['n-1 vertical', 'n-1 horizontal'], fontsize=11)
    axes[1].set_ylabel('Δμ  (rewarded − unrewarded) [°]', fontsize=12)
    axes[1].set_title(f'Rat {rat_id} — Δμ per choice side\n'
                      f'(for reference)', fontsize=11)

    plt.suptitle(
        f'Rat {rat_id}  |  Side symmetry test  |  '
        f'vertical: {n_bootstrap_vert} iters, '
        f'horizontal: {n_bootstrap_horiz} iters  |  '
        f'{N_SYMMETRY_SAMPLES} resamples for S',
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    fname = f'rat_{rat_id}_symmetry_test.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  ✓ Figure saved: {fname}")

# =============================================================================
# SECTION 6 — Main loop
# =============================================================================

def run_analysis(df):
    """
    Main entry point.

    For each rat:
        1. Run reward analysis for n-1 choice = vertical
        2. Run reward analysis for n-1 choice = horizontal
        3. Test side symmetry: S = Δμ_vertical + Δμ_horizontal
    """
    rats = df[COL_RAT].unique()
    print(f"\n{'='*60}")
    print(f"Reward effect + side symmetry analysis")
    print(f"  Bootstrap iterations : {N_BOOTSTRAP}")
    print(f"  Symmetry resamples   : {N_SYMMETRY_SAMPLES}")
    print(f"  Rats found           : {len(rats)}")
    print(f"{'='*60}\n")

    # filter to fake 45° pool across all rats
    all_flanks  = set()
    for flanks in PSE_FLANKS.values():
        all_flanks.update(flanks)
    df_pool = df[df[COL_STIM_NM1].isin(all_flanks)].copy()
    print(f"After fake-45° angle filter: "
          f"{len(df_pool)} / {len(df)} trials retained\n")

    all_results = {}

    for rat in sorted(rats):
        pse_flanks = PSE_FLANKS.get(rat, (40, 50))
        print(f"{'='*60}")
        print(f"Rat {rat}  |  PSE flanks: {pse_flanks}")
        print(f"{'='*60}")

        df_rat   = df[df[COL_RAT] == rat].copy()
        delta_mus = {}
        n_valid   = {}

        for choice_condition, choice_label in [(1, 'vertical'),
                                               (0, 'horizontal')]:
            print(f"\n  --- n-1 choice = {choice_label} ---")

            # filter by n-1 choice and flanking angles
            df_cond = df_rat[
                (df_rat[COL_CHOICE_NM1] == choice_condition) &
                (df_rat[COL_STIM_NM1].isin(pse_flanks))
            ].copy()

            # data checks
            n_low  = (df_cond[COL_STIM_NM1] == pse_flanks[0]).sum()
            n_high = (df_cond[COL_STIM_NM1] == pse_flanks[1]).sum()
            n_rew  = (df_cond[COL_REWARD] == 1).sum()
            n_unr  = (df_cond[COL_REWARD] == 0).sum()
            print(f"  n-1 angle  : {pse_flanks[0]}°={n_low}, {pse_flanks[1]}°={n_high}")
            print(f"  reward     : rewarded={n_rew}, unrewarded={n_unr}")

            if min(n_low, n_high) < 10 or min(n_rew, n_unr) < 10:
                print(f"  ⚠ Skipping: too few trials\n")
                continue

            # bootstrap
            print(f"  Running {N_BOOTSTRAP} bootstrap iterations...")
            boot_results = bootstrap_psychometric(df_cond, pse_flanks=pse_flanks)
            n_valid[choice_label] = len(boot_results['rewarded'])
            print(f"  ✓ {n_valid[choice_label]} valid iterations")

            # plot reward effect for this choice side
            delta_mu = plot_psychometric(
                boot_results, df_rat, rat_id=rat,
                choice_label=choice_label, choice_condition=choice_condition, pse_flanks=pse_flanks
            )

            # store Δμ distribution for symmetry test
            delta_mus[choice_label] = delta_mu

        # --- symmetry test (only if both sides were successfully fitted) ---
        if 'vertical' in delta_mus and 'horizontal' in delta_mus:
            print(f"\n  --- Side symmetry test ---")

            # independent resampling: no pairing justified across separate runs
            s_distribution = (
                np.random.choice(delta_mus['vertical'],
                                 size=N_SYMMETRY_SAMPLES, replace=True) +
                np.random.choice(delta_mus['horizontal'],
                                 size=N_SYMMETRY_SAMPLES, replace=True)
            )

            plot_symmetry_test(
                s_distribution, delta_mus, rat_id=rat,
                n_bootstrap_vert=n_valid.get('vertical',   0),
                n_bootstrap_horiz=n_valid.get('horizontal', 0)
            )

            med_s    = np.median(s_distribution)
            ci_low_s = np.percentile(s_distribution,  2.5)
            ci_high_s= np.percentile(s_distribution, 97.5)
            symmetric = ci_low_s <= 0 <= ci_high_s

            all_results[rat] = {
                'delta_mu_vertical_median':   np.median(delta_mus['vertical']),
                'delta_mu_horizontal_median': np.median(delta_mus['horizontal']),
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
    if all_results:
        print(f"\n{'='*60}")
        print("Summary across rats:")
        summary = pd.DataFrame(all_results).T
        print(summary.to_string(float_format='{:.2f}'.format))
        print(f"{'='*60}\n")

    return all_results

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    # replace with your real data:
    # df = pd.read_csv('your_data.csv')

    #print("Generating simulated data...")
    #df = simulate_data(n_subjects=4, n_trials=1000)
    print(f"Simulated dataset: {len(df)} trials, "
          f"{df[COL_RAT].nunique()} rats\n")

    results = run_analysis(df)