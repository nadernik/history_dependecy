"""
figure_3_per_modality_distributions.py
--------------------------------------
Three-panel strip plot of the fit-derived summary parameters across rats,
per modality. Uses the (unfiltered) task_performance_summary.csv only —
no pickles required.

Panels (left → right):
    1) Accuracy on unambiguous trials (%)
    2) σ 97.5-percentile (°)
    3) γ + λ  (total lapse rate)

Layout:
    - x-axis   : modality (Mod 1, Mod 2, Mod 3)
    - y-axis   : parameter value
    - each dot : one rat's cell for that modality
    - colour   : overall tier of that cell (clean / usable / poor)
    - open marker + × : cell EXCLUDED from downstream (degenerate in both filter states)
    - dashed reference lines : tier thresholds
    - thick horizontal bar   : median across rats per modality

Usage:
    python figure_3_per_modality_distributions.py
        [--csv path]
        [--out path]
        [--filter-state unfiltered|filtered]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SCRIPT_DIR  = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR /'task_performance_summary.csv'
DEFAULT_OUT = SCRIPT_DIR /'figures' / 'figure_3_per_modality.svg'

TIER_COLORS = {
    'clean':  '#2E8B57',
    'usable': '#7CB342',
    'poor':   '#DAA520',
}

CONDITIONS_ORDER = ['1', '2', '3']
COND_LABELS      = ['Mod 1', 'Mod 2', 'Mod 3']


def find_excluded_cells(df):
    """Return set of (rat, condition) that are degenerate in BOTH filter states."""
    pv = df.pivot_table(index=['rat', 'condition'], columns='filter_state',
                        values='fit_degenerate', aggfunc='first').fillna(False)
    both = pv[pv.all(axis=1)]
    return set((int(r), str(c)) for (r, c), _ in both.iterrows())


def plot_panel(ax, df_fs, param_col, title, ylabel, thresholds,
               ylim, excluded, direction='lower_is_better',
               show_ylabel=True):
    """
    Draw one panel.
        thresholds : (clean_bound, usable_bound) — dashed reference lines
        direction  : 'higher_is_better' or 'lower_is_better'
    """
    x_positions = np.arange(len(CONDITIONS_ORDER))

    # jitter within each modality column
    rng = np.random.default_rng(0)

    for xi, cond in enumerate(CONDITIONS_ORDER):
        sub = df_fs[df_fs['condition'].astype(str) == cond]
        if len(sub) == 0:
            continue

        # jitter for visibility; consistent per rat so lines/order don't jump
        rat_ids = sorted(sub['rat'].unique())
        jitter_map = {r: rng.uniform(-0.18, 0.18) for r in rat_ids}

        vals   = []
        for _, r in sub.iterrows():
            rat_id = int(r['rat'])
            key    = (rat_id, cond)
            tier   = r['classification']
            colour = TIER_COLORS.get(tier, '0.5')
            is_excluded = key in excluded

            x = xi + jitter_map[rat_id]
            y = r[param_col]
            if np.isnan(y):
                continue
            vals.append(y)

            if is_excluded:
                ax.scatter(x, y, s=90, facecolors='none', edgecolors=colour,
                           linewidths=1.8, zorder=3)
                ax.scatter(x, y, s=55, marker='x', color=colour,
                           linewidths=1.6, zorder=4)
            else:
                ax.scatter(x, y, s=70, color=colour, edgecolors='white',
                           linewidths=0.7, zorder=3)

        # median bar
        if vals:
            med = float(np.median(vals))
            ax.hlines(med, xi - 0.32, xi + 0.32, colors='0.2', linewidth=2.2,
                      zorder=5)

    # tier threshold reference lines
    for t in thresholds:
        ax.axhline(t, color='0.45', linestyle='--', linewidth=0.9, zorder=1, alpha=0.7)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(COND_LABELS, fontsize=11)
    ax.set_xlim(-0.5, len(CONDITIONS_ORDER) - 0.5)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(axis='both', labelsize=10)
    ax.grid(axis='y', color='0.9', linewidth=0.6, zorder=0)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)


def plot_figure(csv_path, out_path, filter_state='unfiltered'):
    df = pd.read_csv(csv_path)
    df['condition'] = df['condition'].astype(str)

    excluded = find_excluded_cells(df)
    print(f"Excluded cells (degenerate in both filter states): {sorted(excluded)}")

    df_fs = df[df['filter_state'] == filter_state]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5))

    # --- Panel 1: accuracy
    plot_panel(
        axes[0], df_fs, 'accuracy',
        title='Accuracy on unambiguous trials',
        ylabel='Accuracy (%)',
        thresholds=[0.65, 0.80],
        ylim=(0.55, 1.02),
        excluded=excluded,
    )
    # y-axis as %
    axes[0].set_yticks(np.arange(0.6, 1.01, 0.1))
    axes[0].set_yticklabels([f'{int(t*100)}' for t in np.arange(0.6, 1.01, 0.1)])
    # annotate threshold lines
    axes[0].text(2.45, 0.80, ' 80', va='center', ha='left', fontsize=8, color='0.45')
    axes[0].text(2.45, 0.65, ' 65', va='center', ha='left', fontsize=8, color='0.45')

    # --- Panel 2: sigma 97.5
    plot_panel(
        axes[1], df_fs, 'sigma_97_5',
        title=r'$\sigma_{97.5}$   (bootstrap slope)',
        ylabel=r'$\sigma_{97.5}$ (°)',
        thresholds=[20.0, 35.0],
        ylim=(0, 95),
        excluded=excluded,
    )
    axes[1].text(2.45, 20, ' 20°', va='center', ha='left', fontsize=8, color='0.45')
    axes[1].text(2.45, 35, ' 35°', va='center', ha='left', fontsize=8, color='0.45')
    axes[1].axhline(60, color='crimson', linestyle=':', linewidth=0.9, alpha=0.6, zorder=1)
    axes[1].text(2.45, 60, ' 60° (degenerate)', va='center', ha='left', fontsize=8, color='crimson')

    # --- Panel 3: gamma + lambda
    plot_panel(
        axes[2], df_fs, 'lapse_sum',
        title=r'Total lapse rate  $\gamma + \lambda$',
        ylabel=r'$\gamma + \lambda$',
        thresholds=[0.20, 0.40],
        ylim=(0, 0.65),
        excluded=excluded,
    )
    axes[2].text(2.45, 0.20, ' 0.20', va='center', ha='left', fontsize=8, color='0.45')
    axes[2].text(2.45, 0.40, ' 0.40', va='center', ha='left', fontsize=8, color='0.45')

    # ---------- shared legend ----------
    legend_handles = [
        Line2D([0], [0], marker='o', linestyle='',
               markerfacecolor=TIER_COLORS['clean'], markeredgecolor='white',
               markersize=8, label='clean'),
        Line2D([0], [0], marker='o', linestyle='',
               markerfacecolor=TIER_COLORS['usable'], markeredgecolor='white',
               markersize=8, label='usable'),
        Line2D([0], [0], marker='o', linestyle='',
               markerfacecolor=TIER_COLORS['poor'], markeredgecolor='white',
               markersize=8, label='poor'),
        Line2D([0], [0], marker='x', linestyle='',
               color='0.35', markersize=10, markeredgewidth=1.8,
               label='excluded (degenerate in both filter states)'),
        Line2D([0], [0], color='0.2', linewidth=2.2, label='median across rats'),
        Line2D([0], [0], color='0.45', linestyle='--', linewidth=0.9,
               label='tier boundary'),
    ]
    fig.legend(
        handles=legend_handles, loc='lower center', ncol=3,
        frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.05),
    )

    fig.suptitle(
        f'Fit-derived summary parameters across rats, per modality '
        f'({filter_state} state)',
        fontsize=13, y=1.02,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format='svg', bbox_inches='tight')
    plt.savefig(out_path.with_suffix('.png'), format='png', dpi=200,
                bbox_inches='tight')
    print(f'✓ SVG: {out_path}')
    print(f'✓ PNG: {out_path.with_suffix(".png")}')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=DEFAULT_CSV)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--filter-state', choices=['unfiltered', 'filtered'],
                        default='unfiltered')
    args = parser.parse_args()

    plot_figure(args.csv, args.out, filter_state=args.filter_state)