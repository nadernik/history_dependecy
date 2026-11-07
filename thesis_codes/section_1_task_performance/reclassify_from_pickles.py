"""
reclassify_from_pickles.py
--------------------------
One-off script: re-runs the tier classification on the pickled bootstrap
data from task_performance_check_thesis.py, WITHOUT refitting anything.

Reads:
    <OUTPUT_DIR>/raw/all_cells.pkl

Writes (overwrites):
    <OUTPUT_DIR>/task_performance_summary.csv
    <OUTPUT_DIR>/task_performance_summary.tex
    <OUTPUT_DIR>/raw/all_cells.pkl                (with new 'classification')
    <OUTPUT_DIR>/raw/<filter>/rat_<rat>_<cond>.pkl (per-cell)
    <OUTPUT_DIR>/reclassify_log.json              (before/after transition matrix)

What changed vs. the original run:
    tier names       : good/medium/poor  →  clean/usable/poor
    lapse thresholds : clean ≤ 0.20  (was 0.15)
                       usable ≤ 0.40 (was 0.30)
    aggregation rule : worst-tier    →  majority vote, with the three-way
                                        tie (one of each tier) resolved to
                                        the middle tier (usable)

Justification (cite in methods):
    - International Brain Laboratory (2021, eLife 63711): trained head-fixed
      mice show γ+λ ≈ 0.10 at easy contrasts → sets the "clean" boundary.
    - Pisupati, Chartarifsky-Lynn, Khanal & Churchland (2021, eLife 55490):
      rat 4-parameter psychometric fits routinely give γ+λ up to ~0.40 on
      hard/uncertain modalities; lapses reflect uncertainty-driven exploration,
      not disengagement → 0.40 is a well-supported "usable/poor" boundary.

Usage:
    python reclassify_from_pickles.py
        [--dir task_check_output_final]
        [--dry-run]        # don't overwrite pickles, print transition only
"""

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# NEW CLASSIFICATION CONFIG
# =============================================================================
# Metric tier boundaries. Unchanged for accuracy and sigma; relaxed for lapse.
ACC_TIERS   = {'clean': 0.80, 'usable': 0.65}   # higher is better
SIGMA_TIERS = {'clean': 20.0, 'usable': 35.0}   # lower is better
LAPSE_TIERS = {'clean': 0.20, 'usable': 0.40}   # lower is better  (RELAXED)

FIT_DEGENERATE_SIGMA = 60.0

_TIER_ORDER = ('clean', 'usable', 'poor')
_TIER_RANK  = {t: i for i, t in enumerate(_TIER_ORDER)}

# Old → new label map (for logging transitions from prior outputs).
_OLD_TO_NEW = {'good': 'clean', 'medium': 'usable', 'poor': 'poor'}
SCRIPT_DIR = Path(__file__).resolve().parent

# =============================================================================
# TIER LOGIC
# =============================================================================
def _tier_higher_better(value, tiers):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'poor'
    if value >= tiers['clean']:
        return 'clean'
    if value >= tiers['usable']:
        return 'usable'
    return 'poor'


def _tier_lower_better(value, tiers):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'poor'
    if value <= tiers['clean']:
        return 'clean'
    if value <= tiers['usable']:
        return 'usable'
    return 'poor'


def _majority_tier(tiers):
    """
    Majority vote across the three metric tiers.
    Exactly three inputs → possible tier count distributions are:
        (3,0,0)  → clear majority
        (2,1,0)  → unique majority (the 2)
        (1,1,1)  → three-way tie → resolve to the middle tier by rank
    """
    counts = Counter(tiers)
    top = max(counts.values())
    if top >= 2:
        # unique majority (either 3 same, or 2 same + 1 different)
        return next(t for t, c in counts.items() if c == top)
    # (1,1,1) → middle by rank
    return sorted(tiers, key=lambda t: _TIER_RANK[t])[1]


def classify_task_performance(perf, boot):
    """
    New classifier: majority vote across {accuracy, σ 97.5-percentile, γ+λ}
    with the loosened lapse thresholds. Returns the same dict shape as the
    original classifier so downstream summarisation is unchanged.
    """
    out = {}

    # metric 1: accuracy at unambiguous trials (higher is better)
    out['accuracy'] = perf.get('accuracy', np.nan)
    out['acc_tier'] = _tier_higher_better(out['accuracy'], ACC_TIERS)

    # metric 2: σ 97.5-percentile (lower is better)
    param_boot = boot.get('param_boot')
    popt       = boot.get('popt')
    if param_boot is not None:
        sigma_hi = float(np.percentile(param_boot[:, 1], 97.5))
    elif popt is not None:
        sigma_hi = float(popt[1])
    else:
        sigma_hi = np.nan
    out['sigma_97_5']     = sigma_hi
    out['sigma_tier']     = _tier_lower_better(sigma_hi, SIGMA_TIERS)
    out['fit_degenerate'] = bool(
        (not np.isnan(sigma_hi)) and (sigma_hi > FIT_DEGENERATE_SIGMA)
    )

    # metric 3: total lapses γ + λ  (lower is better)
    if popt is not None:
        _, _, gamma, lapse = popt
        out['lapse_sum'] = float(gamma + lapse)
    else:
        out['lapse_sum'] = np.nan
    out['lapse_tier'] = _tier_lower_better(out['lapse_sum'], LAPSE_TIERS)

    # overall
    tiers = (out['acc_tier'], out['sigma_tier'], out['lapse_tier'])
    out['classification'] = _majority_tier(tiers)
    out['tier_breakdown'] = tiers
    return out


# =============================================================================
# CSV + LaTeX regeneration  (copied from the main script, updated tier names)
# =============================================================================
def _rows_from_cells(all_cells):
    rows = []
    for (rat, cond, fs), cell in all_cells.items():
        p = cell['perf']
        c = cell['classification']
        ci = p.get('ci_two_sided', (np.nan, np.nan))
        rows.append({
            'rat':                 int(rat),
            'condition':           cond,
            'filter_state':        fs,
            'n_trials':            cell.get('n_trials', np.nan),
            'n_sessions':          cell.get('n_sessions', np.nan),
            'n_extreme':           p.get('n_extreme', np.nan),
            'n_correct':           p.get('n_correct', np.nan),
            'accuracy':            p.get('accuracy', np.nan),
            'acc_ci_low':          ci[0],
            'acc_ci_high':         ci[1],
            'acc_ci_lower_1sided': p.get('ci_one_sided_lower', np.nan),
            'binom_p':             p.get('binom_p', np.nan),
            'sigma_97_5':          c['sigma_97_5'],
            'lapse_sum':           c['lapse_sum'],
            'fit_degenerate':      c['fit_degenerate'],
            'acc_tier':            c['acc_tier'],
            'sigma_tier':          c['sigma_tier'],
            'lapse_tier':          c['lapse_tier'],
            'classification':      c['classification'],
        })
    return rows


def _write_csv(all_cells, csv_path, filter_order=('unfiltered', 'filtered')):
    df = pd.DataFrame(_rows_from_cells(all_cells))
    if df.empty:
        print("  (no cells to write)")
        return df
    fs_order = {fs: i for i, fs in enumerate(filter_order)}
    df['_fs'] = df['filter_state'].map(fs_order).fillna(len(fs_order))
    df = df.sort_values(['_fs', 'rat', 'condition']).drop(columns='_fs')
    df.to_csv(csv_path, index=False, float_format='%.4g')
    print(f'  ✓ CSV:   {csv_path}')
    return df


def _write_latex(df, latex_path):
    """
    Booktabs LaTeX table with the new tier names. Trailing * on σ marks
    fit_degenerate cells. Requires \\usepackage{booktabs, longtable} (and
    textcomp for \\textdegree).
    """
    tex = df.copy()
    tex['accuracy_pct'] = tex.apply(
        lambda r: (f"{r['accuracy']*100:.1f} "
                   f"[{r['acc_ci_low']*100:.1f}, {r['acc_ci_high']*100:.1f}]"),
        axis=1)
    tex['sigma_fmt'] = tex.apply(
        lambda r: (f"{r['sigma_97_5']:.1f}$^\\ast$"
                   if r['fit_degenerate']
                   else f"{r['sigma_97_5']:.1f}"),
        axis=1)
    tex['lapse_fmt'] = tex['lapse_sum'].map(lambda v: f"{v:.3f}")
    tex['binom_fmt'] = tex['binom_p'].map(lambda v: f"{v:.1e}")

    cols_out = {
        'rat':            'Rat',
        'condition':      'Cond.',
        'filter_state':   'Filter',
        'n_trials':       '$n$',
        'accuracy_pct':   r'Acc.\ (\%) [95\% CI]',
        'binom_fmt':      r'$p_{\text{binom}}$',
        'sigma_fmt':      r'$\sigma_{97.5}$ (\textdegree)',
        'lapse_fmt':      r'$\gamma+\lambda$',
        'acc_tier':       'Acc.\\ tier',
        'sigma_tier':     r'$\sigma$ tier',
        'lapse_tier':     'Lapse tier',
        'classification': 'Overall',
    }
    tex = tex[list(cols_out.keys())].rename(columns=cols_out)

    latex_str = tex.to_latex(
        index=False, escape=False, longtable=True,
        column_format='lll' + 'r' * 5 + 'l' * 4,
        caption=(
            'Task-performance classification per rat, condition, and filter '
            'state. Each metric is scored clean / usable / poor; the overall '
            'label is the majority tier across the three, with ties resolved '
            'to the middle tier (usable). Two-sided Wilson 95\\% CI is '
            'reported for accuracy at unambiguous angles '
            '($|\\text{angle} - 45| \\geq 15\\degree$). An asterisk '
            '($^\\ast$) on $\\sigma_{97.5}$ marks cells with a degenerate '
            'fit (essentially flat psychometric).'
        ),
        label='tab:task_performance',
    )
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write("% Requires \\usepackage{booktabs, longtable} "
                "(and textcomp for \\textdegree).\n")
        f.write(latex_str)
    print(f'  ✓ LaTeX: {latex_path}')


# =============================================================================
# MAIN
# =============================================================================
def reclassify(output_dir, dry_run=False):
    output_dir = Path(output_dir)
    raw_dir    = output_dir / 'raw'
    master_pkl = raw_dir / 'all_cells.pkl'

    if not master_pkl.exists():
        raise FileNotFoundError(f'Cannot find master pickle at {master_pkl}')

    with open(master_pkl, 'rb') as f:
        all_cells = pickle.load(f)
    print(f'Loaded {len(all_cells)} cells from {master_pkl}')

    # ---- transition tracking
    transitions = []
    for key, cell in all_cells.items():
        old = cell['classification'].get('classification', None)
        # apply new classifier
        new_class = classify_task_performance(cell['perf'], {
            'popt':       cell.get('popt'),
            'param_boot': cell.get('param_boot'),
        })
        # keep the new dict + note config used
        cell['classification'] = new_class
        cell.setdefault('config', {})
        cell['config'].update({
            'acc_tiers':            ACC_TIERS,
            'sigma_tiers':          SIGMA_TIERS,
            'lapse_tiers':          LAPSE_TIERS,
            'fit_degenerate_sigma': FIT_DEGENERATE_SIGMA,
            'aggregation':          'majority_vote_middle_tiebreak',
            'reclassified':         True,
        })
        transitions.append({
            'rat':          int(cell.get('rat', key[0])),
            'condition':    str(cell.get('condition', key[1])),
            'filter_state': str(cell.get('filter_state', key[2])),
            'old':          _OLD_TO_NEW.get(old, old),
            'new':          new_class['classification'],
        })

    # ---- transition summary
    tdf = pd.DataFrame(transitions)
    print('\nTransition matrix (old label → new label):')
    print(pd.crosstab(tdf['old'], tdf['new'], margins=True))
    print('\nNew counts by (filter × condition):')
    ct = pd.crosstab([tdf['filter_state'], tdf['condition']], tdf['new'])
    for t in _TIER_ORDER:
        if t not in ct.columns:
            ct[t] = 0
    print(ct[list(_TIER_ORDER)])

    if dry_run:
        print('\n[dry-run] no files written.')
        return

    # ---- overwrite per-cell pickles + master pickle
    print('\nWriting outputs to', output_dir)
    for key, cell in all_cells.items():
        rat, cond, fs = key
        sub = raw_dir / fs
        sub.mkdir(parents=True, exist_ok=True)
        with open(sub / f'rat_{rat}_{cond}.pkl', 'wb') as f:
            pickle.dump(cell, f)
    with open(master_pkl, 'wb') as f:
        pickle.dump(all_cells, f)
    print(f'  ✓ Pickles rewritten under {raw_dir}')

    # ---- rewrite CSV and LaTeX
    csv_path   = output_dir / 'task_performance_summary.csv'
    latex_path = output_dir / 'task_performance_summary.tex'
    df = _write_csv(all_cells, csv_path)
    _write_latex(df, latex_path)

    # ---- log
    log_path = output_dir / 'reclassify_log.json'
    with open(log_path, 'w') as f:
        json.dump({
            'transitions':          transitions,
            'acc_tiers':            ACC_TIERS,
            'sigma_tiers':          SIGMA_TIERS,
            'lapse_tiers':          LAPSE_TIERS,
            'fit_degenerate_sigma': FIT_DEGENERATE_SIGMA,
            'aggregation':          'majority_vote_middle_tiebreak',
        }, f, indent=2)
    print(f'  ✓ Log:   {log_path}')

    n_changed = int((tdf['old'] != tdf['new']).sum())
    print(f'\nDone. {n_changed} of {len(tdf)} cells changed tier.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', type=Path,
                    default=SCRIPT_DIR / 'results' / 'task_check_output_final',
                    help='Path to OUTPUT_DIR (containing raw/all_cells.pkl)')
    parser.add_argument('--dry-run', action='store_true',
                        help="Don't overwrite anything; print transition only")
    args = parser.parse_args()
    reclassify(args.dir, dry_run=args.dry_run)