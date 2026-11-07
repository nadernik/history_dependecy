"""
task_performance_heatmap.py
---------------------------
Summary heatmap of the tier classification from task_performance_summary.csv.

Layout:
    rows    = rats
    columns = conditions (Mod 1, Mod 2, Mod 3, Pooled)
    panels  = filter states (unfiltered | filtered)

Each cell:
    - background colour = tier (clean / usable / poor)
    - annotation        = accuracy on unambiguous trials, in %
    - trailing '*'      = fit_degenerate cell (σ 97.5-pctile > 60°)

Classification rules used to produce the CSV (see reclassify_from_pickles.py):
    metric tiers  : accuracy (higher-better), σ 97.5-pctile (lower-better),
                    γ+λ (lower-better)
    aggregation   : majority vote across the three metric tiers, with the
                    three-way tie (one of each) resolved to the middle tier
    lapse bounds  : clean ≤ 0.20, usable ≤ 0.40, poor > 0.40

Usage:
    python task_performance_heatmap.py
        [--csv path/to/task_performance_summary.csv]
        [--out path/to/output.svg]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from pathlib import Path


# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
DEFAULT_CSV = SCRIPT_DIR / 'task_performance_summary.csv'
DEFAULT_OUT = SCRIPT_DIR / 'figures' / 'task_performance_heatmap.svg'

# Softer palette (matches the second heatmap you produced).
# 'poor' is now a real exclusion signal, so goldenrod reads as
# "flag, don't use" without being alarming.
TIER_COLORS = {
    'clean':  '#2E8B57',   # sea green
    'usable': '#7CB342',   # light green
    'poor':   '#DAA520',   # goldenrod
}
TIER_ORDER  = ['clean', 'usable', 'poor']
TIER_TO_INT = {t: i for i, t in enumerate(TIER_ORDER)}

# Columns in display order; pooled last so modalities read left-to-right.
COND_ORDER  = ['1', '2', '3', 'pooled']
COND_LABELS = ['Mod 1', 'Mod 2', 'Mod 3', 'Pooled']

FILTER_ORDER = ['unfiltered', 'filtered']


# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------
def _build_matrix(df, filter_state, rats, conditions):
    """Return (tier_int, accuracy, degenerate) each shaped (n_rats, n_cond)."""
    n_r, n_c = len(rats), len(conditions)
    tier  = np.full((n_r, n_c), np.nan)
    acc   = np.full((n_r, n_c), np.nan)
    degen = np.zeros((n_r, n_c), dtype=bool)

    sub = df[df['filter_state'] == filter_state]
    for i, rat in enumerate(rats):
        for j, cond in enumerate(conditions):
            row = sub[(sub['rat'] == rat) & (sub['condition'] == cond)]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            label = row['classification']
            if label in TIER_TO_INT:
                tier[i, j] = TIER_TO_INT[label]
            acc[i, j]   = row['accuracy']
            degen[i, j] = bool(row['fit_degenerate'])
    return tier, acc, degen


# --------------------------------------------------------------------------
# PLOT
# --------------------------------------------------------------------------
def plot_heatmap(csv_path=DEFAULT_CSV, out_path=DEFAULT_OUT, show=False):
    df = pd.read_csv(csv_path)
    df['condition'] = df['condition'].astype(str)

    rats = sorted(df['rat'].unique())
    cmap = ListedColormap([TIER_COLORS[t] for t in TIER_ORDER])

    fig, axes = plt.subplots(
        1, len(FILTER_ORDER),
        figsize=(9, max(4.5, 0.42 * len(rats) + 1.5)),
        sharey=True,
    )
    if len(FILTER_ORDER) == 1:
        axes = [axes]

    for ax, fs in zip(axes, FILTER_ORDER):
        tier, acc, degen = _build_matrix(df, fs, rats, COND_ORDER)

        ax.imshow(
            tier, cmap=cmap,
            vmin=0, vmax=len(TIER_ORDER) - 1,
            aspect='auto', interpolation='none',
        )

        # cell annotations
        for i in range(len(rats)):
            for j in range(len(COND_ORDER)):
                if np.isnan(acc[i, j]):
                    ax.text(j, i, '—', ha='center', va='center',
                            color='gray', fontsize=10)
                    continue
                acc_txt = f'{acc[i, j] * 100:.0f}'
                if degen[i, j]:
                    # accuracy in white, red '*' appended
                    ax.text(j, i, acc_txt + ' ',
                            ha='center', va='center',
                            color='white', fontsize=10, fontweight='bold')
                    # place the asterisk just to the right of the number
                    ax.annotate('*', xy=(j, i), xytext=(12, 0),
                                textcoords='offset points',
                                ha='left', va='center',
                                color='#D32F2F', fontsize=14, fontweight='bold')
                else:
                    ax.text(j, i, acc_txt, ha='center', va='center',
                            color='white', fontsize=10, fontweight='bold')

        # white grid between cells for a clean tiled look
        ax.set_xticks(np.arange(len(COND_ORDER)) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(rats)) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linewidth=1.5)
        ax.tick_params(which='minor', length=0)

        ax.set_xticks(range(len(COND_ORDER)))
        ax.set_xticklabels(COND_LABELS, fontsize=11)
        ax.set_title(fs.capitalize(), fontsize=13, fontweight='bold', pad=8)
        ax.set_xlabel('Condition', fontsize=11)

    axes[0].set_yticks(range(len(rats)))
    axes[0].set_yticklabels([f'Rat {r}' for r in rats], fontsize=10)
    axes[0].set_ylabel('Animal', fontsize=11)

    # shared legend
    handles = [
        Patch(facecolor=TIER_COLORS[t], edgecolor='white', label=t.capitalize())
        for t in TIER_ORDER
    ]
    fig.legend(
        handles=handles, loc='lower center', ncol=len(TIER_ORDER),
        frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02),
        title='Overall classification', title_fontsize=10,
    )
    fig.text(
        0.5, -0.06,
        'Numbers = accuracy on unambiguous trials (%).  '
        'Red * marks cells with a degenerate psychometric fit '
        '(σ 97.5-percentile > 60°, fit essentially flat).',
        ha='center', fontsize=9, color='0.35',
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format='svg', bbox_inches='tight')
    plt.savefig(out_path.with_suffix('.png'), format='png', dpi=200,
                bbox_inches='tight')
    print(f'✓ SVG saved: {out_path}')
    print(f'✓ PNG saved: {out_path.with_suffix(".png")}')

    if show:
        plt.show()
    else:
        plt.close(fig)


# --------------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=DEFAULT_CSV,
                        help='Path to task_performance_summary.csv')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT,
                        help='Output SVG path (a .png sibling is also written)')
    parser.add_argument('--show', action='store_true',
                        help='Open the figure window after saving')
    args = parser.parse_args()

    plot_heatmap(csv_path=args.csv, out_path=args.out, show=args.show)