def filter_training_trials(
    df,
    method='criterion',
    n_training=5,
    criterion=0.80,
    min_consec=4,
    outcome_col='hitmiss',
    angle_col='angle',
    trial_id_col='trialID',
    warmup_angles=(0, 90),
    trim_first_n=150,          # NEW — drop first N trials AFTER warmup block
    min_session_length=150,   # NEW — don't trim short sessions to a stub
    verbose=True
):
    """
    Filters a trial-level dataframe by:
      1. Removing warm-up trials at the start of each session (initial 0/90 block)
      2. Removing early training sessions (fixed cutoff or performance criterion)

    Parameters
    ----------
    df          : must already have columns: rat, session_idx, trialID, angle, 
                  angle_n-1, hitmiss_n-1, and whatever outcome_col is
    method      : 'fixed' drops first n_training sessions per rat
                  'criterion' drops sessions before rat reaches stable accuracy
    verbose     : prints trial counts at each filtering step

    Returns
    -------
    Filtered dataframe with columns added: 
    'first_real_trial' 
    'date_real'
    'session_idx'
    'first_real_trial'
    
    To call it: 
    df_fixed = filter_training_trials(df, method='fixed', n_training=5)
    df_crit  = filter_training_trials(df, method='criterion', criterion=0.80, min_consec=4)
    """
    df = df.copy()
    n_start = len(df)

    # ── Step 1: Warm-up removal ───────────────────────────────────────────────
    warmup_set = set(warmup_angles)

    def get_first_real_trial(session_df):
        session_sorted = session_df.sort_values(trial_id_col)
        for _, row in session_sorted.iterrows():
            if pd.notna(row[angle_col]) and row[angle_col] not in warmup_set:
                return row[trial_id_col]
        return None

    #def matlab_datenum_to_datetime(datenum):
        #return pd.Timestamp('0000-01-01') + pd.to_timedelta(datenum - 1, unit='D')
    def matlab_datenum_to_datetime(datenum):
        return pd.to_datetime(datenum - 719529, unit='D')

    df['date_real'] = df['date'].apply(matlab_datenum_to_datetime)
    df = df.sort_values(['rat', 'date_real']).reset_index(drop=True)

    # Add session index per rat (0-indexed)
    df['session_idx'] = df.groupby('rat')['date_real'].transform(
        lambda x: pd.factorize(x)[0]
    )
    
    first_real = (
        df.groupby(['rat', 'session_idx'])
        .apply(get_task_start_trial)
        .rename('first_real_trial')
        .reset_index()
    )

    df = df.merge(first_real, on=['rat', 'session_idx'])
    df = df[
        df['first_real_trial'].notna() &
        (df[trial_id_col] >= df['first_real_trial'])
    ].copy()

    # Nullify lag columns on the first real trial of each session
    boundary = df[trial_id_col] == df['first_real_trial']
    df.loc[boundary, [f'{angle_col}_n-1', f'{outcome_col}_n-1']] = np.nan

    
    n_after_warmup = len(df)

    # NEW: per-session trim of transition zone
    if trim_first_n > 0:
        n_before_trim = len(df)
        df = df.sort_values(['rat', 'session_idx', trial_id_col])
        df['post_warmup_pos'] = df.groupby(['rat', 'session_idx']).cumcount()
        df['sess_len']        = df.groupby(['rat', 'session_idx'])[trial_id_col].transform('count')

        keep = (df['post_warmup_pos'] >= trim_first_n) | (df['sess_len'] < min_session_length)
        df = df[keep].copy().drop(columns=['post_warmup_pos', 'sess_len'])

        # Nullify n-1 lag columns at the NEW first trial of each session,
        # so the perceptual-bias analysis doesn't use lags that point into
        # the trimmed region.
        new_first = df.groupby(['rat', 'session_idx'])[trial_id_col].transform('min')
        boundary  = df[trial_id_col] == new_first
        df.loc[boundary, [f'{angle_col}_n-1', f'{outcome_col}_n-1']] = np.nan

        if verbose:
            print(f"After per-session trim (first {trim_first_n}, "
                  f"skip if session < {min_session_length}): "
                  f"{len(df):>8}  (-{n_before_trim - len(df)})")

    n_after_warmup = len(df)

    # ── Step 2: Session filtering ─────────────────────────────────────────────
    if method == 'fixed':
        df_out = df[df['session_idx'] >= n_training].copy()

    elif method == 'criterion':
        def get_criterion_session(rat_df):
            #easy = rat_df[rat_df[angle_col].isin(warmup_angles)]
            easy = rat_df[
                (rat_df[angle_col] <= 15) |
                (rat_df[angle_col] >= 75)
            ]
            # check the numebr of trials at easy angles per session
            easy_counts = easy.groupby('session_idx').size()
            # use only sessions with at least 10 easy trials to compute accuracy
            # but I should be oke in any case because I don't have any session with less than 10 easy trials
            rat_id = rat_df['rat'].iloc[0]
            low_count_sessions = easy_counts[easy_counts < 50]

            if low_count_sessions.empty:
                print(
                    f"Rat {rat_id}: all sessions have at least "
                    f"50 easy trials."
                )
            else:
                print(
                    f"Rat {rat_id}: sessions with fewer than "
                    f"50 easy trials:\n{low_count_sessions}"
                )

            acc = easy.groupby('session_idx')[outcome_col].mean()
            consec = 0
            for sess, val in acc.items():
                if val >= criterion:
                    consec += 1
                    if consec >= min_consec:
                        return sess - (min_consec - 1)  # first session of run
                else:
                    consec = 0
            return None

        criterion_sessions = (
            df.groupby('rat')
            .apply(get_criterion_session)
            .rename('criterion_session')
            .reset_index()
        )
        df = df.merge(criterion_sessions, on='rat')
        df_out = df[
            df['criterion_session'].notna() &
            (df['session_idx'] >= df['criterion_session'])
        ].copy()

    else:
        raise ValueError(f"method must be 'fixed' or 'criterion', got '{method}'")

    n_after_sessions = len(df_out)

    # ── Verbose report ────────────────────────────────────────────────────────
    if verbose:
        print(f"\n--- filter_training_trials (method='{method}') ---")
        print(f"Trials at start:              {n_start:>8}")
        print(f"After warm-up removal:        {n_after_warmup:>8}  (-{n_start - n_after_warmup})")
        print(f"After session filtering:      {n_after_sessions:>8}  (-{n_after_warmup - n_after_sessions})")
        print(f"Total removed:                {n_start - n_after_sessions:>8}")
        print(f"\nTrials remaining per rat:")
        print(df_out.groupby('rat').size().to_string())
        if method == 'criterion':
            print(f"\nCriterion sessions per rat:")
            print(criterion_sessions.to_string(index=False))

    return df_out
    
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import sys
sys.path.append(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy")
from functions_Ale import get_task_start_trial
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
TARGET_RATS           = [3]      # None = all rats; else list like [9, 11, 13]
MIN_TRIALS_PER_BIN    = 20        # per (rat, mod, bin)
MIN_SESSIONS          = 3         # min sessions for reliable session-bootstrap
N_BOOT                = 10       # session-bootstrap iterations
N_INDEPENDENT_SAMPLES = 10_000    # paired resamples for Δmu / Δsigma CIs
N_DIFFICULTY_BINS     = 3
X_GRID                = np.linspace(-45, 45, 200)
BUBBLE_SCALE          = 1         # scatter marker area = n_trials * this
N_EMP_POINTS          = 5         # empirical dots per bin curve
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "figures_perceptual_bias")


# =============================================================================
# LOAD + PREPROCESS  (identical to your previous script)
# =============================================================================
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data()

df = an.create_lagged_features()
df = filter_training_trials(df, method='criterion', criterion=0.80, min_consec=2)

# derive P(repeat) and signed evidence for repetition
df[REP_COL]      = (df['action_n-1'] == df[RESP_COL]).astype(int)
df[EVIDENCE_COL] = (df[ANGLE_COL] - BOUNDARY) * (2 * df['action_n-1'] - 1)

# key filters: n-1 correct only; exclude ambiguous n-1 (difficulty code 0)
df = df[df[STATE_COL] == 1].copy()
df = df[df[DIFFICULTY_NM1_COL] != 0].copy()

MODALITIES = df[MODALITY_COL].dropna().unique()


# =============================================================================
# MLE psychometric fit (Bernoulli negative log-likelihood)
# =============================================================================

def _nll(params, x, c):
    """4-param cum. Gaussian NLL: p = γ + (1 − γ − λ)·Φ((x − μ)/σ)."""
    mu, sigma, lapse, gamma = params
    p = gamma + (1 - gamma - lapse) * norm.cdf(x, loc=mu, scale=sigma)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -np.sum(c * np.log(p) + (1 - c) * np.log(1 - p))


def fit_psychometric(x, y):
    """MLE fit. Returns [mu, sigma, lapse, gamma] or None on failure."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 10 or len(np.unique(x)) < 3:
        return None

    x0     = [0.0, 15.0, 0.05, 0.05]
    bounds = [(-45, 45), (1e-3, 90), (0, 0.2), (0, 0.2)]

    try:
        result = minimize(_nll, x0, args=(x, y),
                          bounds=bounds, method='L-BFGS-B')
        return result.x if result.success else None
    except (RuntimeError, ValueError, FloatingPointError):
        return None


def psychometric(x, mu, sigma, lapse, gamma):
    return gamma + (1 - gamma - lapse) * norm.cdf(x, loc=mu, scale=sigma)


# =============================================================================
# Balanced difficulty binning
# =============================================================================

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
# Session bootstrap — stores fit parameters
# =============================================================================

def bootstrap_psychometric(df_bin, n_boot=N_BOOT, seed=0):
    """
    Resample sessions with replacement; refit each time; store params.

    Returns
    -------
    dict with keys:
        popt        : best-fit params on full data (or None)
        param_boot  : (n_valid, 4) array of bootstrapped params, or None
                      if fewer than MIN_SESSIONS sessions available or
                      too few bootstrap fits succeeded
        ci_valid    : bool
        n_valid     : int (successful bootstrap fits)
    """
    x = df_bin[EVIDENCE_COL].values
    y = df_bin[REP_COL].values

    popt = fit_psychometric(x, y)
    if popt is None:
        return {'popt': None, 'param_boot': None,
                'ci_valid': False, 'n_valid': 0}

    sessions = df_bin[SESSION_COL].dropna().unique()
    if len(sessions) < MIN_SESSIONS:
        return {'popt': popt, 'param_boot': None,
                'ci_valid': False, 'n_valid': 0}

    rng         = np.random.default_rng(seed)
    boot_params = []
    for _ in range(n_boot):
        sampled = rng.choice(sessions, size=len(sessions), replace=True)
        boot_df = pd.concat([df_bin[df_bin[SESSION_COL] == s] for s in sampled],
                            ignore_index=True)
        p = fit_psychometric(boot_df[EVIDENCE_COL].values,
                             boot_df[REP_COL].values)
        if p is not None:
            boot_params.append(p)

    if len(boot_params) < 10:
        return {'popt': popt, 'param_boot': None,
                'ci_valid': False, 'n_valid': len(boot_params)}

    return {'popt': popt, 'param_boot': np.array(boot_params),
            'ci_valid': True, 'n_valid': len(boot_params)}


# =============================================================================
# ΔPSE / Δslope: independent resampling across bins
# =============================================================================

def compute_bin_differences(param_by_bin,
                             n_samples=N_INDEPENDENT_SAMPLES, seed=0):
    """
    For every pair of bins (a, b) compute Δmu = μ_b − μ_a and
    Δsigma = σ_b − σ_a via independent resampling of the two bootstrap
    parameter distributions. Independence is the right null because the
    two bootstraps use disjoint trials and independent iterations.
    """
    rng     = np.random.default_rng(seed)
    bins    = sorted(param_by_bin.keys())
    results = {}
    for i, bin_a in enumerate(bins):
        for bin_b in bins[i + 1:]:
            p_a = param_by_bin[bin_a]
            p_b = param_by_bin[bin_b]

            mu_a    = rng.choice(p_a[:, 0], size=n_samples, replace=True)
            mu_b    = rng.choice(p_b[:, 0], size=n_samples, replace=True)
            sigma_a = rng.choice(p_a[:, 1], size=n_samples, replace=True)
            sigma_b = rng.choice(p_b[:, 1], size=n_samples, replace=True)

            d_mu    = mu_b - mu_a
            d_sigma = sigma_b - sigma_a

            mu_lo, mu_hi = np.percentile(d_mu,    [2.5, 97.5])
            si_lo, si_hi = np.percentile(d_sigma, [2.5, 97.5])

            results[(bin_a, bin_b)] = {
                'delta_mu_median':    float(np.median(d_mu)),
                'delta_mu_ci':        (float(mu_lo), float(mu_hi)),
                'delta_mu_excl_0':    not (mu_lo <= 0 <= mu_hi),
                'delta_sigma_median': float(np.median(d_sigma)),
                'delta_sigma_ci':     (float(si_lo), float(si_hi)),
                'delta_sigma_excl_0': not (si_lo <= 0 <= si_hi),
            }
    return results


def classify_bias(differences, primary_pair):
    """
    Classification uses the PRIMARY pair only (avoids multiple-comparison
    inflation from testing every pair). Other pairs are still reported.

    bias_detected : primary ΔPSE CI excludes 0.
    null          : primary ΔPSE CI includes 0.
                    Bound on the effect = CI half-width.
    """
    if primary_pair not in differences:
        primary_pair = next(iter(differences))  # fallback

    primary  = differences[primary_pair]
    half_w   = 0.5 * (primary['delta_mu_ci'][1] - primary['delta_mu_ci'][0])
    max_hw   = max(0.5 * (d['delta_mu_ci'][1] - d['delta_mu_ci'][0])
                   for d in differences.values())
    return {
        'primary_pair':          primary_pair,
        'primary_delta_mu':      primary['delta_mu_median'],
        'primary_ci':            primary['delta_mu_ci'],
        'primary_ci_halfwidth':  float(half_w),
        'max_ci_halfwidth':      float(max_hw),
        'bias_detected':         primary['delta_mu_excl_0'],
        'classification':        'bias_detected' if primary['delta_mu_excl_0']
                                                 else 'null',
        'any_pair_excl_0':       any(d['delta_mu_excl_0']
                                     for d in differences.values()),
    }


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
# Plotting — 2-panel per (rat, modality)
# =============================================================================

def plot_bin_curves_with_delta(bin_data, differences, classification,
                                rat_id, modality, savefig=False):
    """
    Left panel  : psychometric curves per difficulty bin
                  (fit + 95% bootstrap band + empirical dots + PSE ticks)
    Right panel : ΔPSE with 95% CI for every bin pair; primary pair in red.
    """
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

        if d['param_boot'] is not None:
            curves = np.array([psychometric(X_GRID, *p) for p in d['param_boot']])
            y_lo   = np.percentile(curves, 2.5,  axis=0)
            y_hi   = np.percentile(curves, 97.5, axis=0)
            ax.fill_between(X_GRID, y_lo, y_hi, alpha=0.15, color=c, linewidth=0)
            ci_tag = ""
        else:
            ci_tag = " (no CI)"

        h, = ax.plot(X_GRID, psychometric(X_GRID, *popt),
                     color=c, lw=2,
                     label=f"bin {lv}{ci_tag}  (n={d['n_trials']})")
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

    leg_curves = ax.legend(handles=curve_handles, title="difficulty n-1 bin",
                           fontsize=9, title_fontsize=9, loc='upper left')
    ax.add_artist(leg_curves)

    bubble_handles = [
        plt.scatter([], [], s=n * BUBBLE_SCALE, color='grey', alpha=0.6,
                    edgecolors='white', linewidths=0.5, label=f'n={n}')
        for n in (20, 50, 100)
    ]
    ax.legend(handles=bubble_handles, title="trials / level",
              fontsize=8, title_fontsize=8, loc='lower right',
              framealpha=0.7)

    # ---------- Panel 2 : ΔPSE per pair ----------
    ax     = axes[1]
    pairs  = list(differences.keys())
    xpos   = np.arange(len(pairs))
    primary_pair = classification['primary_pair']

    for i, pair in enumerate(pairs):
        d      = differences[pair]
        med    = d['delta_mu_median']
        lo, hi = d['delta_mu_ci']
        is_primary = (pair == primary_pair)

        color = 'crimson' if is_primary else 'steelblue'
        lw    = 3 if is_primary else 2
        ax.errorbar(i, med, yerr=[[med - lo], [hi - med]],
                    fmt='o', color=color, capsize=8, markersize=10, lw=lw)
        mark = '*' if d['delta_mu_excl_0'] else ''
        y_annot = hi + 0.03 * (abs(hi) + abs(lo) + 1)
        ax.text(i, y_annot,
                f"{med:+.2f}°{mark}\n[{lo:+.2f}, {hi:+.2f}]",
                ha='center', va='bottom', fontsize=9)

    ax.axhline(0, color='gray', ls='--', alpha=0.6)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"bin {a} → {b}"
                        + (' (primary)' if (a, b) == primary_pair else '')
                        for (a, b) in pairs], fontsize=10)
    ax.set_ylabel("ΔPSE  (μ_bin_b − μ_bin_a) [°]")
    label_txt = classification['classification'].upper()
    color_txt = 'crimson' if classification['bias_detected'] else 'seagreen'
    ax.set_title(
        f"ΔPSE bootstrap 95% CIs  →  {label_txt}\n"
        f"primary CI half-width = {classification['primary_ci_halfwidth']:.2f}°  "
        f"|  max across pairs = {classification['max_ci_halfwidth']:.2f}°",
        color=color_txt,
    )

    if savefig:
        os.makedirs(FIGURE_DIR, exist_ok=True)
        stem = f"rat_{rat_id}_{modality}_perceptual_bias"
        for ext in ('png', 'svg'):
            path = os.path.join(FIGURE_DIR, f"{stem}.{ext}")
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"      ✓ Figure saved: {path}")
    plt.show()


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
                'classification':     c['classification'],
                'any_pair_excl_0':    c['any_pair_excl_0'],
            })
    if not rows:
        print("  (no results to summarise)")
        return None

    df_out = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("Per-(rat × modality) perceptual bias classification")
    print("=" * 90)
    print(df_out.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nGroup counts of 'bias_detected' by modality:")
    for mod in sorted(df_out['modality'].unique()):
        sub    = df_out[df_out['modality'] == mod]
        n_bias = int((sub['classification'] == 'bias_detected').sum())
        n_tot  = len(sub)
        print(f"  {mod:<15}: {n_bias:>2}/{n_tot} rats  "
              f"({100 * n_bias / n_tot:.0f}%)  "
              f"— median CI half-width across nulls: "
              f"{sub[sub['classification'] == 'null']['CI_halfwidth'].median():.2f}°"
              if (sub['classification'] == 'null').any() else
              f"  {mod:<15}: {n_bias:>2}/{n_tot} rats  "
              f"({100 * n_bias / n_tot:.0f}%)")
    print("=" * 90)
    return df_out


# =============================================================================
# MAIN LOOP — all rats × all modalities
# =============================================================================

def run_analysis(df, modalities, target_rats=TARGET_RATS):
    all_results = {}

    rats = sorted(df[RAT_COL].dropna().unique())
    if target_rats is not None:
        rats = [r for r in rats if r in target_rats]

    print(f"\n{'=' * 70}")
    print(f"Perceptual bias test — ΔPSE / Δslope via psychometric bootstrap")
    print(f"  Rats             : {len(rats)}  ({rats})")
    print(f"  Modalities       : {list(modalities)}")
    print(f"  Bootstrap iters  : {N_BOOT}")
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
            continue
        rat_df.loc[:, 'difficulty_n-1_bin'] = rat_df[DIFFICULTY_NM1_COL].map(bin_map)
        print(f"  Difficulty bin map: {bin_map}")

        all_results[rat] = {}

        for mod in modalities:
            print(f"\n  Modality: {mod}")
            sub_md = rat_df[rat_df[MODALITY_COL] == mod]
            if len(sub_md) < MIN_TRIALS_PER_BIN * N_DIFFICULTY_BINS:
                print(f"    ⚠ Skipping: too few trials ({len(sub_md)})")
                continue

            bin_data = {}
            for lv in sorted(sub_md['difficulty_n-1_bin'].dropna().unique()):
                sub = sub_md[sub_md['difficulty_n-1_bin'] == lv]
                n   = len(sub)
                if n < MIN_TRIALS_PER_BIN:
                    print(f"    bin {lv}: skipped ({n} trials)")
                    continue

                seed = int(1000 * rat + 10 * lv)
                res  = bootstrap_psychometric(sub, seed=seed)
                if res['popt'] is None:
                    print(f"    bin {lv}: fit failed")
                    continue

                bin_data[lv] = {**res, 'n_trials': n, 'df': sub}
                mu, sigma, lam, gamma = res['popt']
                ci_tag = (f", {res['n_valid']} valid boot iters"
                          if res['ci_valid'] else " (no CI)")
                print(f"    bin {lv}: n={n}, μ={mu:+.2f}°, σ={sigma:.2f}°{ci_tag}")

            param_by_bin = {lv: d['param_boot'] for lv, d in bin_data.items()
                            if d['param_boot'] is not None}
            if len(param_by_bin) < 2:
                print(f"    ⚠ Fewer than 2 bins with valid bootstrap — "
                      f"no ΔPSE test possible")
                continue

            differences    = compute_bin_differences(param_by_bin)
            bins_sorted    = sorted(param_by_bin.keys())
            primary_pair   = (bins_sorted[0], bins_sorted[-1])   # extreme pair
            classification = classify_bias(differences, primary_pair=primary_pair)

            for pair, d in differences.items():
                mark = ' *' if d['delta_mu_excl_0'] else ''
                tag  = ' (primary)' if pair == primary_pair else ''
                print(f"    ΔPSE bin {pair[0]}→{pair[1]}: "
                      f"{d['delta_mu_median']:+.2f}° "
                      f"[{d['delta_mu_ci'][0]:+.2f}, "
                      f"{d['delta_mu_ci'][1]:+.2f}]{mark}{tag}")
            print(f"    → {classification['classification']}  "
                  f"(primary CI half-width "
                  f"{classification['primary_ci_halfwidth']:.2f}°)")

            plot_bin_curves_with_delta(bin_data, differences, classification,
                                        rat_id=rat, modality=mod)

            all_results[rat][mod] = {
                'bin_data':       bin_data,
                'differences':    differences,
                'classification': classification,
            }

    summary_df = summarise_bias(all_results)
    return all_results, summary_df


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    results, summary = run_analysis(df, MODALITIES)