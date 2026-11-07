"""
Section_3_group_figures.py
Group-level sketch figures for the choice-effects psychometric analysis
=======================================================================

Reads the two CSVs written by ``Section_3_pyscho_choice_reward_thesis.py``:

    <results_root>/<subset>/tables/delta_mu_per_analysis.csv
    <results_root>/<subset>/tables/orthogonal_contrasts.csv

and produces five group-level figures plus an augmented contrasts table:

    A. Group SD plane (rat medians on the (Δµ_rew, Δµ_unrew) plane)
    B. Forest plot of S, D, P per rat
    C. Axis-pattern classification bar chart
    D. Real 45° vs fake 45° validation scatter
    E. Group-level tests on S, D, P (beeswarm + Wilcoxon)

Default input/output layout matches the analysis script's ``OUTPUT_ROOT``:

    choice_effects_psychometric_results/
        pooled/
            tables/     <-- input (from analysis)
            figures/    <-- per-rat figures (from analysis)
            group_figures/  <-- written by this script

Override with --results-root, --subset, --out-dir on the command line if
your paths differ. Classification thresholds are also exposed as CLI args.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# =============================================================================
# Defaults — match Section_3_pyscho_choice_reward_thesis.py output layout
# =============================================================================
BASE_DIR             = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = BASE_DIR / "choice_effects_psychometric_results"
DEFAULT_SUBSET       = "pooled"
DEFAULT_R_THRESHOLD  = 2.0     # rats with r below this are null
DEFAULT_WEDGE_HALF   = 22.5    # ± this many degrees around each axis

PATTERN_COLOURS = {
    'null'           : '#95a5a6',
    'rew-only'       : '#2ecc71',
    'unrew-only'     : '#d946ef',
    'S (choice)'     : '#E68722',
    'D (interaction)': '#2922E6',
}
PATTERN_ORDER = ['null', 'rew-only', 'unrew-only', 'S (choice)', 'D (interaction)']


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
                   help=f"Root of the analysis output (default: {DEFAULT_RESULTS_ROOT})")
    p.add_argument("--subset", default=DEFAULT_SUBSET,
                   help=f"Subset directory name (default: {DEFAULT_SUBSET!r})")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory for figures "
                        "(default: <results-root>/<subset>/group_figures/)")
    p.add_argument("--r-threshold", type=float, default=DEFAULT_R_THRESHOLD,
                   help=f"Magnitude threshold for the 'null' axis pattern "
                        f"(default: {DEFAULT_R_THRESHOLD})")
    p.add_argument("--wedge-half", type=float, default=DEFAULT_WEDGE_HALF,
                   help=f"Half-width of the axis wedges, degrees "
                        f"(default: {DEFAULT_WEDGE_HALF})")
    p.add_argument("--forest-order", choices=["effect", "rat"], default="effect",
                   help="Sort forest-plot rows by effect size ('effect', "
                        "most negative at the bottom) or by rat ID ascending "
                        "('rat'). Default: 'effect'.")
    return p.parse_args()


# =============================================================================
# I/O
# =============================================================================
def load_inputs(results_root: Path, subset: str):
    """Load the two CSVs the analysis script wrote for the given subset."""
    tables_dir = results_root / subset / "tables"
    per_path   = tables_dir / "delta_mu_per_analysis.csv" # Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\REWARD_and_CHOICE_HISTORY_PSYCHOMETRICS\tables_choice_effects\delta_mu_per_analysis.csv")  #
    contr_path = tables_dir / "orthogonal_contrasts.csv"  #Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\REWARD_and_CHOICE_HISTORY_PSYCHOMETRICS\tables_choice_effects\orthogonal_contrasts.csv")  #

    missing = [p for p in (per_path, contr_path) if not p.exists()]
    if missing:
        msg = ("Could not find the following expected input file(s):\n"
               + "\n".join(f"  - {p}" for p in missing)
               + f"\n\nRun Section_3_pyscho_choice_reward_thesis.py first, "
                 f"or pass --results-root / --subset to point at existing tables.")
        raise FileNotFoundError(msg)

    df_pa = pd.read_csv(per_path)
    # keep_default_na=False stops pandas from turning the literal string "null"
    # (used in the 'profile' column) into NaN. Empty cells remain NA.
    df_c  = pd.read_csv(contr_path, keep_default_na=False, na_values=[''])
    return df_pa, df_c, tables_dir


# =============================================================================
# Axis-pattern classification
# =============================================================================
def _sort_forest(sub, med_col, order):
    """Sort a forest-plot subset either by effect size or by rat ID."""
    if order == 'effect':
        return sub.sort_values(med_col).reset_index(drop=True)
    if order == 'rat':
        return sub.sort_values('rat').reset_index(drop=True)
    raise ValueError(f"Unknown order: {order!r}. Use 'effect' or 'rat'.")


def classify_axis_pattern(dr, du, r_threshold):
    """
    Nearest-axis classifier on the (Δµ_rew, Δµ_unrew) plane.

    Each of the four axes (±x = rew-only, ±y = unrew-only, y=x = S, y=-x = D)
    owns a 45°-wide wedge; they tile the plane exactly at wedge_half=22.5°.
    Setting wedge_half smaller would leave 'unclassified' gaps between
    axes, which we currently ignore (assigned to nearest anyway).
    """
    r = float(np.hypot(dr, du))
    if r < r_threshold:
        return 'null'
    theta = float(np.degrees(np.arctan2(du, dr)))
    d_x = min(abs(theta - 0), abs(theta - 180), abs(theta - (-180)))
    d_y = min(abs(theta - 90), abs(theta - (-90)))
    d_S = min(abs(theta - 45), abs(theta - (-135)))
    d_D = min(abs(theta - 135), abs(theta - (-45)))
    dists = {'rew-only': d_x, 'unrew-only': d_y,
             'S (choice)': d_S, 'D (interaction)': d_D}
    return min(dists, key=dists.get)


def add_axis_pattern_columns(df_c, df_pa, r_threshold):
    """Attach r, theta_deg, axis_pattern, and marginal CIs from analyses 3 and 4."""
    df_c = df_c.copy()
    df_c['r']         = np.hypot(df_c['delta_mu_rew'], df_c['delta_mu_unrew'])
    df_c['theta_deg'] = np.degrees(np.arctan2(df_c['delta_mu_unrew'],
                                              df_c['delta_mu_rew']))
    df_c['axis_pattern'] = df_c.apply(
        lambda row: classify_axis_pattern(row['delta_mu_rew'],
                                          row['delta_mu_unrew'],
                                          r_threshold=r_threshold),
        axis=1,
    )

    # Pull marginal 95% CIs on Δµ_rew and Δµ_unrew from the per-analysis file
    def get_ci(rat, analysis):
        sub = df_pa[(df_pa['rat'] == rat) & (df_pa['analysis'] == analysis)]
        if sub.empty:
            return np.nan, np.nan
        row = sub.iloc[0]
        return float(row['ci_lo']), float(row['ci_hi'])

    for rat in df_c['rat']:
        lo_r, hi_r = get_ci(rat, 'analysis_3_fake45_reward_controlled')
        lo_u, hi_u = get_ci(rat, 'analysis_4_fake45_reward_controlled_opposite')
        mask = df_c['rat'] == rat
        df_c.loc[mask, 'ci_lo_rew']   = lo_r
        df_c.loc[mask, 'ci_hi_rew']   = hi_r
        df_c.loc[mask, 'ci_lo_unrew'] = lo_u
        df_c.loc[mask, 'ci_hi_unrew'] = hi_u

    # Per-rat CI-exclusion flags for the raw Δµ (bootstrap CIs from analyses 3, 4)
    df_c['dmu_rew_excludes_0'] = ~(
        (df_c['ci_lo_rew']   <= 0) & (df_c['ci_hi_rew']   >= 0)
    )
    df_c['dmu_unrew_excludes_0'] = ~(
        (df_c['ci_lo_unrew'] <= 0) & (df_c['ci_hi_unrew'] >= 0)
    )

    return df_c


# =============================================================================
# Figure A: Group SD plane
# =============================================================================
def figure_A_sd_plane(df_c, out_dir):
    fig, ax = plt.subplots(figsize=(9, 9))
    lim = max(22, float(np.ceil(df_c[['delta_mu_rew', 'delta_mu_unrew']]
                                .abs().values.max())) + 2)

    ax.axhline(0, color='k', lw=0.6, zorder=1)
    ax.axvline(0, color='k', lw=0.6, zorder=1)
    ax.plot([-lim, lim], [-lim, lim], ':', color='#E68722', lw=1.6,
            label='y = x   (pure choice — S axis)', zorder=2)
    ax.plot([-lim, lim], [lim, -lim], ':', color='#2922E6', lw=1.6,
            label='y = −x  (reward × choice — D axis)', zorder=2)

    box = dict(boxstyle='round,pad=0.3', facecolor='white',
               edgecolor='none', alpha=0.75)
    ax.text(0.97, 0.97, 'both switch\n(win-switch\n+ lose-switch)',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8.5, color='#7d4a13', bbox=box, linespacing=1.15)
    ax.text(0.03, 0.03, 'both stay\n(win-stay\n+ lose-stay)',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=8.5, color='#7d4a13', bbox=box, linespacing=1.15)
    ax.text(0.03, 0.97, 'win-stay\n+ lose-switch',
            transform=ax.transAxes, ha='left', va='top',
            fontsize=8.5, color='#151785', bbox=box, linespacing=1.15)
    ax.text(0.97, 0.03, 'win-switch\n+ lose-stay',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8.5, color='#151785', bbox=box, linespacing=1.15)

    for _, row in df_c.iterrows():
        colour = PATTERN_COLOURS[row['axis_pattern']]
        x, y   = row['delta_mu_rew'], row['delta_mu_unrew']

        if not np.isnan(row.get('ci_lo_rew', np.nan)):
            ax.plot([row['ci_lo_rew'], row['ci_hi_rew']], [y, y],
                    color=colour, lw=1.0, alpha=0.35, zorder=3)
            ax.plot([x, x], [row['ci_lo_unrew'], row['ci_hi_unrew']],
                    color=colour, lw=1.0, alpha=0.35, zorder=3)

        # Ring the rat if it's significant in ANY of the tests shown in the
        # forest plots (S, D, P, Δµ_rew, Δµ_unrew). Uses the boolean columns
        # rather than a string comparison on 'profile' — that was fragile
        # against pandas' default NA-string handling.
        is_sig = bool(row['S_excludes_0'] or row['D_excludes_0']
                      or row['P_excludes_0']
                      or row['dmu_rew_excludes_0']
                      or row['dmu_unrew_excludes_0'])
        ax.scatter(x, y, s=140, c=colour,
                   edgecolors='black' if is_sig else 'none',
                   linewidths=2.0 if is_sig else 0,
                   zorder=5)
        ax.annotate(f"{int(row['rat'])}", (x, y),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold' if is_sig else 'normal',
                    zorder=6)

    handles = [plt.Line2D([0], [0], marker='o', linestyle='',
                          markerfacecolor=PATTERN_COLOURS[p],
                          markeredgecolor='none', markersize=10,
                          label=f"{p} (n={int((df_c['axis_pattern']==p).sum())})")
               for p in PATTERN_ORDER]
    handles.append(plt.Line2D([0], [0], marker='o', linestyle='',
                              markerfacecolor='white', markeredgecolor='black',
                              markersize=11, markeredgewidth=2.0,
                              label='dark ring: sig. in ≥1 of S, D, P, Δµ$_{rew}$, Δµ$_{unrew}$'))
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0),
              fontsize=9, frameon=False, title='Axis pattern')

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.set_xlabel(r'$\Delta\mu_{\mathrm{rew}}$  [°]  '
                  r'(prev-V − prev-H | prev rewarded)', fontsize=11)
    ax.set_ylabel(r'$\Delta\mu_{\mathrm{unrew}}$  [°]  '
                  r'(prev-V − prev-H | prev unrewarded)', fontsize=11)
    ax.set_title('Group SD plane — pooled scope\n'
                 r'Median ($\Delta\mu_{rew}$, $\Delta\mu_{unrew}$) '
                 'per rat with marginal 95% CIs', fontsize=12)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(out_dir / f'fig_A_sd_plane.{ext}',
                    dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Figure B: Forest plot of S, D, P
# =============================================================================
def figure_B_forest(df_c, out_dir, order='effect'):
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))

    panels = [
        ('S (main effect of previous choice)',    'S', 'S_ci_lo', 'S_ci_hi', 'S_excludes_0', '#E68722'),
        ('D (previous-reward × previous-choice)', 'D', 'D_ci_lo', 'D_ci_hi', 'D_excludes_0', '#2922E6'),
        ('P (main effect of previous outcome)',   'P', 'P_ci_lo', 'P_ci_hi', 'P_excludes_0', '#2ecc71'),
    ]

    for ax, (name, med_col, lo_col, hi_col, ex_col, colour) in zip(axes, panels):
        sub = df_c[['rat', med_col, lo_col, hi_col, ex_col]].copy()
        sub = _sort_forest(sub, med_col, order)

        for i, row in sub.iterrows():
            excl = bool(row[ex_col])
            face = colour if excl else 'white'
            ax.errorbar(row[med_col], i,
                        xerr=[[row[med_col] - row[lo_col]],
                              [row[hi_col] - row[med_col]]],
                        fmt='o', ecolor=colour, elinewidth=1.4,
                        markerfacecolor=face, markeredgecolor=colour,
                        markeredgewidth=1.8, markersize=9, capsize=3)

        ax.axvline(0, color='k', lw=0.8, linestyle='--', alpha=0.7)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([f"rat {int(r)}" for r in sub['rat']], fontsize=9)
        ax.set_xlabel('Contrast value [°]', fontsize=10)
        n_sig = int(sub[ex_col].sum())
        ax.set_title(f'{name}\n{n_sig}/{len(sub)} rats: 95% CI excludes 0',
                     fontsize=10)
        ax.grid(True, axis='x', alpha=0.2)

    fig.suptitle('Forest plot — orthogonal contrasts across rats (pooled scope)',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(out_dir / f'fig_B_forest.{ext}',
                    dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Figure C: Axis-pattern bar chart
# =============================================================================
def figure_C_axis_bar(df_c, r_threshold, wedge_half, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    counts = df_c['axis_pattern'].value_counts().reindex(PATTERN_ORDER,
                                                        fill_value=0)
    colours = [PATTERN_COLOURS[p] for p in counts.index]
    bars = ax.barh(counts.index, counts.values, color=colours,
                   edgecolor='black', linewidth=0.9)

    for bar, val, patt in zip(bars, counts.values, counts.index):
        rats_here = df_c[df_c['axis_pattern'] == patt]['rat'].tolist()
        ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val}   " + ('rats: ' + ', '.join(map(str, rats_here))
                              if val else ''),
                va='center', fontsize=9)

    ax.set_xlim(0, max(counts.values) + 2)
    ax.set_xlabel('Number of rats', fontsize=11)
    ax.set_title(f'Axis-pattern classification '
                 f'(r threshold = {r_threshold}°, wedge half-width = {wedge_half}°)',
                 fontsize=11)
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.2)
    plt.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(out_dir / f'fig_C_axis_bar.{ext}',
                    dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Figure D: Real 45° vs fake 45° validation
# =============================================================================
def figure_D_real_vs_fake(df_pa, out_dir):
    a1 = df_pa[df_pa['analysis'] == 'analysis_1_real45'].set_index('rat')
    a2 = df_pa[df_pa['analysis'] == 'analysis_2_fake45'].set_index('rat')
    common = sorted(set(a1.index) & set(a2.index))
    x  = a1.loc[common, 'delta_mu'].values
    y  = a2.loc[common, 'delta_mu'].values
    xl = a1.loc[common, 'ci_lo'].values
    xh = a1.loc[common, 'ci_hi'].values
    yl = a2.loc[common, 'ci_lo'].values
    yh = a2.loc[common, 'ci_hi'].values

    fig, ax = plt.subplots(figsize=(7, 7))
    lim = max(np.max(np.abs([x, y, xl, xh, yl, yh])), 10) + 1
    ax.plot([-lim, lim], [-lim, lim], ':', color='gray', lw=1.4, label='y = x')
    ax.plot([-lim, lim], [0, 0], color='k', lw=0.6)
    ax.plot([0, 0], [-lim, lim], color='k', lw=0.6)
    ax.errorbar(x, y, xerr=[x - xl, xh - x], yerr=[y - yl, yh - y],
                fmt='o', color='#34495e', ecolor='#34495e',
                elinewidth=1.0, alpha=0.55, markersize=7, capsize=2)
    for xi, yi, rat in zip(x, y, common):
        ax.annotate(f"{int(rat)}", (xi, yi),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    r_pears = float(np.corrcoef(x, y)[0, 1])
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.set_xlabel(r'$\Delta\mu$ from Analysis 1 (real 45°) [°]', fontsize=11)
    ax.set_ylabel(r'$\Delta\mu$ from Analysis 2 (fake 45°) [°]', fontsize=11)
    ax.set_title(f'Real vs fake 45° validation of the previous-choice effect\n'
                 f'Pearson r = {r_pears:.2f} (n = {len(common)} rats)',
                 fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(out_dir / f'fig_D_real_vs_fake.{ext}',
                    dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Figure E: Group-level tests on S, D, P
# =============================================================================
def figure_E_group_tests(df_c, out_dir):
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    rng = np.random.default_rng(0)

    tests = [
        ('S (choice)',      df_c['S'].values,              '#E68722'),
        ('D (interaction)', df_c['D'].values,              '#2922E6'),
        ('P (reward)',      df_c['P'].values,              '#2ecc71'),
        (r'$\Delta\mu_{\mathrm{rew}}$',   df_c['delta_mu_rew'].values,   '#0FA023'),
        (r'$\Delta\mu_{\mathrm{unrew}}$', df_c['delta_mu_unrew'].values, '#B0289F'),
    ]

    for ax, (name, vals, colour) in zip(axes, tests):
        jitter = rng.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(jitter, vals, s=90, c=colour, edgecolors='black',
                   linewidths=0.9, alpha=0.9, zorder=3)
        med = float(np.median(vals))
        ax.hlines(med, -0.2, 0.2, color='black', lw=2.5, zorder=4)
        ax.axhline(0, color='gray', lw=0.8, linestyle='--', alpha=0.7)

        try:
            w = wilcoxon(vals, alternative='two-sided', zero_method='wilcox')
            p_str = f"Wilcoxon signed-rank\nW = {w.statistic:.1f}, p = {w.pvalue:.3f}"
        except ValueError as e:
            p_str = f"Wilcoxon failed: {e}"

        n_pos = int(np.sum(vals > 0)); n_neg = int(np.sum(vals < 0))
        sign_str = f"{n_pos}+ / {n_neg}−  (median = {med:+.2f}°)"

        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_ylabel(f'{name} [°]', fontsize=11)
        ax.set_title(sign_str + "\n" + p_str, fontsize=10)
        ax.grid(True, axis='y', alpha=0.2)

    fig.suptitle('Group-level tests — per-rat medians (n = '
                 f'{len(df_c)}): S, D, P and the raw Δµ by outcome',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(out_dir / f'fig_E_group_tests.{ext}',
                    dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Figure F: Forest plot of Δµ_rew and Δµ_unrew (per-rat)
# =============================================================================
def figure_F_forest_delta_mu(df_c, out_dir, order='effect'):
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))

    panels = [
        (r'$\Delta\mu_{\mathrm{rew}}$ (prev-V − prev-H | prev rewarded)',
         'delta_mu_rew',   'ci_lo_rew',   'ci_hi_rew',
         'dmu_rew_excludes_0',   '#0FA023'),
        (r'$\Delta\mu_{\mathrm{unrew}}$ (prev-V − prev-H | prev unrewarded)',
         'delta_mu_unrew', 'ci_lo_unrew', 'ci_hi_unrew',
         'dmu_unrew_excludes_0', '#B0289F'),
    ]

    for ax, (name, med_col, lo_col, hi_col, ex_col, colour) in zip(axes, panels):
        sub = df_c[['rat', med_col, lo_col, hi_col, ex_col]].copy()
        sub = _sort_forest(sub, med_col, order)

        for i, row in sub.iterrows():
            excl = bool(row[ex_col])
            face = colour if excl else 'white'
            ax.errorbar(row[med_col], i,
                        xerr=[[row[med_col] - row[lo_col]],
                              [row[hi_col] - row[med_col]]],
                        fmt='o', ecolor=colour, elinewidth=1.4,
                        markerfacecolor=face, markeredgecolor=colour,
                        markeredgewidth=1.8, markersize=9, capsize=3)

        ax.axvline(0, color='k', lw=0.8, linestyle='--', alpha=0.7)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels([f"rat {int(r)}" for r in sub['rat']], fontsize=9)
        ax.set_xlabel('Δµ [°]', fontsize=10)

        # Group Wilcoxon on the per-rat medians of this contrast
        vals = df_c[med_col].values
        try:
            w = wilcoxon(vals, alternative='two-sided', zero_method='wilcox')
            group_str = f"group Wilcoxon: W = {w.statistic:.1f}, p = {w.pvalue:.3f}"
        except ValueError as e:
            group_str = f"group Wilcoxon: n/a ({e})"

        n_sig = int(sub[ex_col].sum())
        ax.set_title(f'{name}\n{n_sig}/{len(sub)} rats: 95% CI excludes 0\n{group_str}',
                     fontsize=9.5)
        ax.grid(True, axis='x', alpha=0.2)

    fig.suptitle('Forest plot — previous-choice-driven boundary shifts by outcome '
                 '(pooled scope)', fontsize=12, y=1.02)
    plt.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(out_dir / f'fig_F_forest_delta_mu.{ext}',
                    dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()

    out_dir = args.out_dir or (args.results_root / args.subset / "group_figures")

    print(f"Reading from: {args.results_root / args.subset / 'tables'}")
    print(f"Writing to  : {out_dir}")
    print(f"Params      : r_threshold={args.r_threshold}, "
          f"wedge_half={args.wedge_half}")

    df_pa, df_c, _ = load_inputs(args.results_root, args.subset)
    out_dir.mkdir(parents=True, exist_ok=True)  # created only after inputs are validated
    df_c = add_axis_pattern_columns(df_c, df_pa, r_threshold=args.r_threshold)

    print("\nAxis-pattern classification:")
    print(df_c[['rat', 'delta_mu_rew', 'delta_mu_unrew', 'r', 'theta_deg',
                'axis_pattern', 'profile']].to_string(index=False))
    print("\nCounts by pattern:")
    print(df_c['axis_pattern'].value_counts()
          .reindex(PATTERN_ORDER, fill_value=0))

    figure_A_sd_plane(df_c, out_dir)
    figure_B_forest(df_c, out_dir, order=args.forest_order)
    figure_C_axis_bar(df_c, args.r_threshold, args.wedge_half, out_dir)
    figure_D_real_vs_fake(df_pa, out_dir)
    figure_E_group_tests(df_c, out_dir)
    figure_F_forest_delta_mu(df_c, out_dir, order=args.forest_order)

    df_c.to_csv(out_dir / 'contrasts_with_axis_pattern.csv',
                index=False, float_format='%.3f')

    print(f"\n✓ All figures written to {out_dir}")


if __name__ == '__main__':
    main()