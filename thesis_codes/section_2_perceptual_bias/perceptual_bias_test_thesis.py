"""
perceptual_bias_test_thesis.py
========================
Tests whether the difficulty of the n-1 stimulus biases the current
perceptual boundary (PSE shift) — i.e. whether history-dependent
PERCEPTUAL bias is present, and, as a secondary output, whether it
changes sensitivity (slope).

For every (rat, modality):
    - fit a 4-parameter cumulative Gaussian P(repeat) vs
      evidence-for-repetition, per difficulty_n-1 bin, via MLE
    - session-level bootstrap to obtain distributions of μ (PSE) and σ (slope)
    - compute ΔPSE and Δslope between every pair of bins with 95% CIs
    - classify per (rat, modality) on the "primary" (extreme-bin) pair:
          bias_detected  → primary ΔPSE CI excludes 0
          null           → primary ΔPSE CI includes 0
                            (report the CI half-width as the effective bound)

Then a group-level table aggregates classifications across all rats
and modalities.

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

Notes and caveats
-----------------
    ▸ With N_DIFFICULTY_BINS = 3 there are 3 pair-wise ΔPSE CIs.
      Classification uses only the PRIMARY (extreme) pair — this
      avoids the multiple-comparison inflation of "any pair excludes 0".
      The other pairs are shown in the figure and summary for context.
    ▸ Independent (not paired) resampling for Δmu / Δsigma: the bootstrap
      draws are independent across bins (disjoint trials, independent
      bootstrap iterations), so pairing would impose a correlation
      that isn't there.
"""

import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
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
N_BOOT                = 1000       # session-bootstrap iterations
N_INDEPENDENT_SAMPLES = 10_000    # paired resamples for Δmu / Δsigma CIs
N_DIFFICULTY_BINS     = 3
X_GRID                = np.linspace(-45, 45, 200)
BUBBLE_SCALE          = 0.5         # scatter marker area = n_trials * this
N_EMP_POINTS          = 5         # empirical dots per bin curve
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "figures_perceptual_bias")


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
df = df[df[STATE_COL] == 1].copy()
df = df[df[DIFFICULTY_NM1_COL] != 0].copy()

MODALITIES = df[MODALITY_COL].dropna().unique()

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
    """MLE fit. Returns [mu, sigma, lapse, gamma] or None on failure."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 10 or len(np.unique(x)) < 3:
        return None

    x0     = [0.0, 15.0, 0.05, 0.05]
    bounds = [(-45, 45), (1, 90), (0, 0.4), (0, 0.4)]

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
        med    = d['delta_mu_median']
        lo, hi = d['delta_mu_ci']
        is_primary = (pair == primary_pair)

        color = 'crimson' if is_primary else 'steelblue'
        lw    = 3 if is_primary else 2
        ax.errorbar(i, med, yerr=[[med - lo], [hi - med]],
                    fmt='o', color=color, capsize=8, markersize=10, lw=lw)

        mark    = ' *' if d['delta_mu_excl_0'] else ''
        y_annot = hi + 0.04 * data_range
        ax.text(i, y_annot,
                f"{med:+.2f}°{mark}\n[{lo:+.2f}, {hi:+.2f}]",
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
# Table export — LaTeX (booktabs) + CSV
# =============================================================================
TABLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "tables_perceptual_bias")


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


def export_tables(all_results):
    """
    Export two tables per run:
      1) SUMMARY  — one row per (rat, modality), primary-pair result
      2) DETAILS  — one row per (rat, modality, bin-pair) with ΔPSE + ΔSlope

    Each is written both as .tex (booktabs) and .csv (raw values, easy to edit).
    """
    os.makedirs(TABLE_DIR, exist_ok=True)

    # ---- SUMMARY ---------------------------------------------------------
    summary_rows = []
    for rat, mods in all_results.items():
        for mod, r in mods.items():
            c        = r['classification']
            lo, hi   = c['primary_ci']
            pp       = c['primary_pair']
            bl       = r.get('bin_labels', {})
            pp_label = f"{bl.get(pp[0], pp[0])} → {bl.get(pp[1], pp[1])}"
            summary_rows.append({
                'rat':            int(rat),
                'modality':       str(mod),
                'primary_pair':   pp_label,
                'delta_pse':      c['primary_delta_mu'],
                'ci_lo':          lo,
                'ci_hi':          hi,
                'ci_halfwidth':   c['primary_ci_halfwidth'],
                'classification': c['classification'],
                'any_pair_ne_0':  bool(c['any_pair_excl_0']),
            })
    summary_raw = pd.DataFrame(summary_rows)
    summary_raw.to_csv(os.path.join(TABLE_DIR, 'summary_perceptual_bias.csv'),
                        index=False, float_format='%.4f')

    # LaTeX-formatted view
    summary_tex = pd.DataFrame({
        'Rat':               summary_raw['rat'],
        'Modality':          summary_raw['modality'],
        'Primary pair':      summary_raw['primary_pair'],
        r'$\Delta$PSE (°)':  summary_raw['delta_pse'].map(_fmt_pm),
        r'95\% CI':          [f"[{_fmt_pm(lo)}, {_fmt_pm(hi)}]"
                              for lo, hi in zip(summary_raw['ci_lo'],
                                                 summary_raw['ci_hi'])],
        'Half-width (°)':    summary_raw['ci_halfwidth'].map(lambda v: f"{v:.2f}"),
        'Result':            summary_raw['classification'].map(
                                lambda s: s.replace('_', r'\_')),
    })
    tex = _df_to_booktabs(
        summary_tex,
        column_format='rllrlrl',
        header_labels=list(summary_tex.columns),
        caption=(r"Per-rat $\times$ modality perceptual-bias classification. "
                 r"$\Delta$PSE and 95\% CI are for the primary "
                 r"(extreme-bin) pair; the half-width bounds the effect "
                 r"under the null."),
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
                    'delta_pse':          d['delta_mu_median'],
                    'delta_pse_ci_lo':    d['delta_mu_ci'][0],
                    'delta_pse_ci_hi':    d['delta_mu_ci'][1],
                    'delta_pse_excl_0':   bool(d['delta_mu_excl_0']),
                    'delta_sigma':        d['delta_sigma_median'],
                    'delta_sigma_ci_lo':  d['delta_sigma_ci'][0],
                    'delta_sigma_ci_hi':  d['delta_sigma_ci'][1],
                    'delta_sigma_excl_0': bool(d['delta_sigma_excl_0']),
                })
    detail_raw = pd.DataFrame(detail_rows)
    detail_raw.to_csv(os.path.join(TABLE_DIR, 'details_perceptual_bias.csv'),
                       index=False, float_format='%.4f')

    detail_tex = pd.DataFrame({
        'Rat':                    detail_raw['rat'],
        'Modality':               detail_raw['modality'],
        'Pair':                   detail_raw['pair'] + detail_raw['is_primary'].map(
                                    lambda p: r' $\star$' if p else ''),
        r'$\Delta$PSE (°)':       detail_raw['delta_pse'].map(_fmt_pm),
        r'95\% CI $\Delta$PSE':   [f"[{_fmt_pm(lo)}, {_fmt_pm(hi)}]"
                                    for lo, hi in zip(detail_raw['delta_pse_ci_lo'],
                                                       detail_raw['delta_pse_ci_hi'])],
        r'$\Delta\sigma$ (°)':    detail_raw['delta_sigma'].map(_fmt_pm),
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

    print(f"\n✓ Tables written to {TABLE_DIR}")
    for name in ('summary_perceptual_bias', 'details_perceptual_bias'):
        for ext in ('csv', 'tex'):
            print(f"    - {name}.{ext}")


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
            bin_labels_short = make_bin_labels_short(bin_map)
            bin_labels = make_bin_labels(bin_map)
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
                                        rat_id=rat, modality=mod,
                                        bin_labels=bin_labels, bin_labels_short=bin_labels_short)

            all_results[rat][mod] = {
                'bin_data':       bin_data,
                'differences':    differences,
                'classification': classification,
                'bin_labels':     bin_labels,
                'bin_labels_short': bin_labels_short,
            }

    summary_df = summarise_bias(all_results)
    export_tables(all_results)  
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
    results, summary = run_analysis(df, MODALITIES)