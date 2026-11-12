"""
figure_2_representative_fits_se.py
-----------------------------------
Compact 2x2 grid of exemplar psychometric fits spanning the tier ranking.

Panels:
    (a) CLEAN                          — rat 6, Mod 3, filtered
    (b) USABLE                         — rat 3, Mod 2, filtered
    (c) POOR (non-degenerate)          — rat 9, Mod 1, filtered
    (d) POOR (degenerate fit)          — rat 13, Mod 1, filtered

Each panel:
    - binned empirical points (fixed marker size)
    - binomial standard-error bars for the empirical proportions
    - fitted 4-parameter cumulative-Gaussian curve in black
    - panel border coloured by modality
    - (mu, sigma, gamma, lambda) values in-panel

Requires:
    - The pickled fits at <OUTPUT_DIR>/raw/all_cells.pkl
    - The raw trial-level data, obtained by importing load_datasets() from
      your main script (task_performance_check_thesis.py).

No bootstrap confidence intervals are read or plotted.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DIR = SCRIPT_DIR / 'results' / 'task_check_output_no_boot'
DEFAULT_OUT = DEFAULT_DIR / 'figures' / 'figure_2_exemplars.svg'

sys.path.insert(0, str(SCRIPT_DIR))

EXEMPLARS = [
    dict(rat=6, condition=3, filter_state='filtered',
         tier_label='(a) Clean', subplot=(0, 0)),
    dict(rat=3, condition=2, filter_state='filtered',
         tier_label='(b) Usable', subplot=(0, 1)),
    dict(rat=9, condition=1, filter_state='filtered',
         tier_label='(c) Poor', subplot=(1, 0)),
    dict(rat=13, condition=1, filter_state='filtered',
         tier_label='(d) Poor (degenerate)', subplot=(1, 1)),
]

MODALITY_COLORS = {
    1: "#0B0C0B",       # tactile
    2: "#080808",       # visual
    3: "#060606",       # visuotactile
    '1': "#0D0E0E",
    '2': "#0B0B0C",
    '3': "#040303",
    'pooled': '0.35',
}

ANGLE_COL = 'angle'
RESP_COL = 'action'
RAT_COL = 'rat'
MODALITY_COL = 'mod'
X_GRID = np.linspace(0, 90, 200)

MODALITY_NAMES = {
    1: 'Tactile',
    2: 'Visual',
    3: 'Visuotactile',
    '1': 'Tactile',
    '2': 'Visual',
    '3': 'Visuotactile',
    'pooled': 'Pooled',
}


def psychometric(x, mu, sigma, gamma, lapse):
    return gamma + (1 - gamma - lapse) * norm.cdf(
        x, loc=mu, scale=sigma
    )


def _match_cond(mod_col, cond):
    """Match condition robustly to integer/string inconsistencies."""
    if cond == 'pooled':
        return np.ones(len(mod_col), dtype=bool)
    try:
        return (mod_col == cond) | (mod_col.astype(str) == str(cond))
    except Exception:
        return mod_col.astype(str) == str(cond)


def _find_cell(all_cells, rat, condition, filter_state):
    """Find a cell robustly to integer/string condition keys."""
    candidates = [
        (rat, condition, filter_state),
        (rat, str(condition), filter_state),
        (rat,
         int(condition) if str(condition).isdigit() else condition,
         filter_state),
    ]
    for key in candidates:
        if key in all_cells:
            return all_cells[key]

    for key, cell in all_cells.items():
        if (int(key[0]) == int(rat)
                and str(key[1]) == str(condition)
                and str(key[2]) == str(filter_state)):
            return cell

    raise KeyError(
        f'Cell (rat={rat}, condition={condition}, '
        f'filter_state={filter_state}) not found. '
        f'Available keys sample: {list(all_cells)[:5]}'
    )


def _binned_empirical(x, y, n_bins=20):
    """Return centers, proportions, counts, and binomial SEs."""
    edges = np.linspace(0, 90, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)

    xs, ps, ns, ses = [], [], [], []
    for b, center in enumerate(centers):
        mask = idx == b
        n = int(mask.sum())
        if n < 3:
            continue

        p = float(y[mask].mean())
        se = np.sqrt(p * (1 - p) / n)
        xs.append(center)
        ps.append(p)
        ns.append(n)
        ses.append(se)

    return (
        np.asarray(xs),
        np.asarray(ps),
        np.asarray(ns),
        np.asarray(ses),
    )


def plot_panel(ax, df_cond, cell, tier_label):
    condition = cell.get('condition', '?')
    modality_colour = MODALITY_COLORS.get(
        condition,
        MODALITY_COLORS.get(str(condition), '0.35'),
    )

    # Use the panel border, rather than the fitted curve, to encode modality.
    for spine in ax.spines.values():
        spine.set_color(modality_colour)
        spine.set_linewidth(2.0)

    x = df_cond[ANGLE_COL].to_numpy()
    y = df_cond[RESP_COL].to_numpy()
    obs_x, obs_p, obs_n, obs_se = _binned_empirical(x, y)

    # Empirical proportions with fixed-size markers and binomial SE bars.
    ax.errorbar(
        obs_x,
        obs_p,
        yerr=obs_se,
        fmt='o',
        markersize=5.5,
        markerfacecolor='steelblue',
        markeredgecolor='white',
        markeredgewidth=0.7,
        color='steelblue',
        ecolor='steelblue',
        elinewidth=1.2,
        capsize=3,
        capthick=1.0,
        alpha=0.85,
        zorder=5,
    )

    # Full-data fitted curve only: no bootstrap CI band.
    popt = cell.get('popt')
    if popt is not None:
        y_fit = psychometric(X_GRID, *popt)
        ax.plot(X_GRID, y_fit, color='black', linewidth=2.4, zorder=3)

        mu, sigma, gamma, lapse = popt
        ax.text(
            0.02,
            0.98,
            (f"μ = {mu:.1f}°\n"
             f"σ = {sigma:.1f}°\n"
             f"γ = {gamma:.3f}\n"
             f"λ = {lapse:.3f}"),
            transform=ax.transAxes,
            ha='left',
            va='top',
            fontsize=9,
            family='monospace',
            bbox=dict(
                boxstyle='round,pad=0.35',
                facecolor='white',
                edgecolor='0.75',
                linewidth=0.6,
                alpha=0.92,
            ),
        )

    ax.axhline(0.5, color='0.6', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.axvline(45, color='0.6', linestyle='--', linewidth=0.7, alpha=0.7)

    ax.set_xlim(0, 90)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(np.arange(0, 91, 15))
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(alpha=0.15, linewidth=0.5)

    n_trials = len(df_cond)
    n_sessions = (
        int(df_cond['date'].nunique())
        if 'date' in df_cond.columns else None
    )
    modality = MODALITY_NAMES.get(
        condition,
        MODALITY_NAMES.get(str(condition), str(condition)),
    )
    id_line = (
        f"rat {cell.get('rat', '?')}   "
        f"{modality}   "
        f"{cell.get('filter_state', '?')}   "
        f"n = {n_trials:,}"
        + (f", {n_sessions} sessions" if n_sessions else "")
    )
    ax.set_title(
        f"{tier_label}\n{id_line}",
        fontsize=10,
        loc='left',
        pad=6,
        color='0.15',
    )


def build_figure(out_dir, out_path):
    master_pkl = out_dir / 'raw' / 'all_cells.pkl'
    if not master_pkl.exists():
        raise FileNotFoundError(
            f'Cannot find {master_pkl}. Run the main script first, or point '
            '--dir at the correct output directory.'
        )

    with open(master_pkl, 'rb') as file:
        all_cells = pickle.load(file)
    print(f'Loaded {len(all_cells)} cells from {master_pkl}')

    print('Importing load_datasets from task_performance_check_thesis ...')
    from task_performance_check_thesis import load_datasets

    datasets = load_datasets()
    print(f"Loaded raw datasets: {list(datasets.keys())}")

    fig, axes = plt.subplots(
        2, 2, figsize=(11, 8), sharex=True, sharey=True
    )

    for ex in EXEMPLARS:
        ax = axes[ex['subplot']]
        cell = _find_cell(
            all_cells,
            ex['rat'],
            ex['condition'],
            ex['filter_state'],
        )
        df = datasets[ex['filter_state']]
        df_r = df[df[RAT_COL] == ex['rat']]
        mask = _match_cond(df_r[MODALITY_COL], ex['condition'])
        df_cond = df_r[mask]

        if len(df_cond) == 0:
            print(f"  WARNING: No raw data for {ex}; panel will show fit only.")
        plot_panel(ax, df_cond, cell, ex['tier_label'])

    for ax in axes[-1]:
        ax.set_xlabel('Stimulus orientation (°)', fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel('P(action = vertical)', fontsize=11)

    fig.suptitle(
        'Representative psychometric fits, one per performance tier',
        fontsize=13,
        fontweight='bold',
        y=1.00,
    )
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format='svg', bbox_inches='tight')
    plt.savefig(
        out_path.with_suffix('.png'),
        format='png',
        dpi=200,
        bbox_inches='tight',
    )
    print(f'Wrote SVG: {out_path}')
    print(f'Wrote PNG: {out_path.with_suffix(".png")}')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dir',
        type=Path,
        default=DEFAULT_DIR,
        help='Output directory containing raw/all_cells.pkl',
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=DEFAULT_OUT,
        help='Output figure path (SVG; .png also written)',
    )
    args = parser.parse_args()

    build_figure(args.dir, args.out)
