"""
figure_psychometric_by_modality.py
----------------------------------
1x3 SVG panel of psychometric curves, one per current modality
(Tactile | Visual | Visuotactile), for a single rat or for all rats pooled.

Usage:
    python figure_psychometric_by_modality.py --rat 6
    python figure_psychometric_by_modality.py --rat all

Single rat : uses the fit stored in <DIR>/raw/all_cells.pkl (filtered cell);
             falls back to a refit if the cell is missing (or with --refit).
All rats   : trials from every rat are pooled and a 4-parameter cumulative
             Gaussian is fitted here by maximum likelihood.

Only the filtered dataset is used.
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DIR = Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\task_check_output_no_boot")

sys.path.insert(0, str(SCRIPT_DIR))

FILTER_STATE = 'filtered'

# Panel order and colours: Tactile, Visual, Visuotactile
MODALITIES = [1, 2, 3]
MODALITY_COLORS = {1: 'green', 2: 'blue', 3: 'red'}

ANGLE_COL = 'angle'
RESP_COL = 'action'
RAT_COL = 'rat'
MODALITY_COL = 'mod'
X_GRID = np.linspace(0, 90, 200)


# ----------------------------------------------------------------------------
# Model and fitting
# ----------------------------------------------------------------------------
def psychometric(x, mu, sigma, gamma, lapse):
    return gamma + (1 - gamma - lapse) * norm.cdf(x, loc=mu, scale=sigma)


def fit_psychometric(x, y):
    """Bernoulli MLE of (mu, sigma, gamma, lambda) with bounded parameters."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    eps = 1e-9

    def nll(p):
        pr = np.clip(psychometric(x, *p), eps, 1 - eps)
        return -np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr))

    bounds = [(0, 90), (1, 50), (0, 0.5), (0, 0.5)]
    best = None
    for mu0 in (30, 45, 60):
        for s0 in (5, 15, 30):
            res = minimize(nll, x0=[mu0, s0, 0.05, 0.05],
                           bounds=bounds, method='L-BFGS-B')
            if best is None or res.fun < best.fun:
                best = res
    return best.x


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _match_cond(mod_col, cond):
    return (mod_col == cond) | (mod_col.astype(str) == str(cond))


def _find_cell(all_cells, rat, condition, filter_state):
    """Return the stored cell or None, robust to int/str key types."""
    for key, cell in all_cells.items():
        try:
            if (str(key[0]) == str(rat)
                    and str(key[1]) == str(condition)
                    and str(key[2]) == str(filter_state)):
                return cell
        except (TypeError, IndexError):
            continue
    return None

def _binned_empirical(x, y, min_n=3, decimals=1):
    """One point per presented stimulus level (no binning)."""
    x = np.round(np.asarray(x, float), decimals)   # absorbs float noise, e.g. 44.9999 -> 45.0
    y = np.asarray(y, float)

    xs, ps, ses = [], [], []
    for a in np.unique(x):
        m = x == a
        n = int(m.sum())
        if n < min_n:
            continue
        p = float(y[m].mean())
        xs.append(a)
        ps.append(p)
        ses.append(np.sqrt(p * (1 - p) / n))
    return np.asarray(xs), np.asarray(ps), np.asarray(ses)


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_panel(ax, df_cond, popt, colour):
    x = df_cond[ANGLE_COL].to_numpy(float)
    y = df_cond[RESP_COL].to_numpy(float)
    obs_x, obs_p, obs_se = _binned_empirical(x, y)

    ax.errorbar(
        obs_x, obs_p, yerr=obs_se, fmt='o',
        markersize=5.5, markerfacecolor=colour, markeredgecolor='white',
        markeredgewidth=0.7, ecolor=colour, elinewidth=1.2,
        capsize=3, capthick=1.0, alpha=0.6, zorder=5,
    )

    if popt is not None:
        ax.plot(X_GRID, psychometric(X_GRID, *popt),
                color=colour, linewidth=2.4, zorder=4)

        mu, sigma, gamma, lapse = popt
        ax.text(
            0.02, 0.98,
            (f"μ = {mu:.1f}°\n"
             f"σ = {sigma:.1f}°\n"
             f"γ = {gamma:.3f}\n"
             f"λ = {lapse:.3f}"),
            transform=ax.transAxes, ha='left', va='top',
            fontsize=9, family='monospace',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      edgecolor='0.75', linewidth=0.6, alpha=0.92),
        )

    ax.axhline(0.5, color='0.6', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.axvline(45, color='0.6', linestyle='--', linewidth=0.7, alpha=0.7)

    ax.set_xlim(0, 90)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks(np.arange(0, 91, 15))
    ax.tick_params(axis='both', labelsize=9)
    ax.spines[['top', 'right']].set_visible(False)


def build_figure(out_dir, out_path, rat, refit=False):
    from task_performance_check_thesis import load_datasets

    datasets = load_datasets()
    df = datasets[FILTER_STATE]

    pooled = str(rat).lower() == 'all'
    if pooled:
        df_sel = df
        label = 'All rats'
        all_cells = {}
    else:
        rat = int(rat)
        df_sel = df[df[RAT_COL] == rat]
        if len(df_sel) == 0:
            raise ValueError(f'No filtered trials for rat {rat}.')
        label = f'Rat {rat}'
        all_cells = {}
        if not refit:
            master_pkl = out_dir / 'raw' / 'all_cells.pkl'
            if master_pkl.exists():
                with open(master_pkl, 'rb') as f:
                    all_cells = pickle.load(f)
            else:
                print(f'{master_pkl} not found; refitting from trials.')

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8),
                             sharex=True, sharey=True)

    for ax, mod in zip(axes, MODALITIES):
        df_cond = df_sel[_match_cond(df_sel[MODALITY_COL], mod)]
        if len(df_cond) == 0:
            print(f'  WARNING: no trials for {label}, modality {mod}.')
            plot_panel(ax, df_cond, None, MODALITY_COLORS[mod])
            continue

        popt = None
        if not pooled and not refit:
            cell = _find_cell(all_cells, rat, mod, FILTER_STATE)
            if cell is not None and cell.get('popt') is not None:
                popt = np.asarray(cell['popt'], float)
        if popt is None:
            popt = fit_psychometric(df_cond[ANGLE_COL], df_cond[RESP_COL])

        plot_panel(ax, df_cond, popt, MODALITY_COLORS[mod])

    for ax in axes:
        ax.set_xlabel('Stimulus orientation (°)', fontsize=11)
    axes[0].set_ylabel('P(action = vertical)', fontsize=11)

    # Single label for the whole panel
    fig.text(0.01, 0.99, label, ha='left', va='top',
             fontsize=13, fontweight='bold')

    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format='svg', bbox_inches='tight')
    print(f'Wrote SVG: {out_path}')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rat', default='2',
                        help="Rat ID (e.g. 6) or 'all' to pool all rats")
    parser.add_argument('--dir', type=Path, default=DEFAULT_DIR,
                        help='Output directory containing raw/all_cells.pkl')
    parser.add_argument('--out', type=Path, default=None,
                        help='Output SVG path (default: <dir>/figures/psychometric_<rat>.svg)')
    parser.add_argument('--refit', action='store_true',
                        help='Refit single-rat curves instead of using the pickle')
    args = parser.parse_args()

    out = args.out or args.dir / 'figures' / f'psychometric_rat_{args.rat}.svg'
    build_figure(args.dir, out, args.rat, refit=args.refit)