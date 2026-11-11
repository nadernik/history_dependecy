"""
validation_against_per_cell_2.py

Adapted for Formula 2 with full random structure:
- formula2_uncorrelated_try_more_random
- Includes acuity × history fixed effects and random slopes

Compares four estimators of per-rat-per-cell history-effect slopes:
  1. raw_slope_unc   : empirical slope on raw P(right), uncentered
  2. raw_slope_cent  : empirical slope, after within-session centering
  3. per_cell_glm    : beta from independent per-cell logistic regression
                       (in probability units via local sigmoid scaling)
  4. model_prob      : slope of the mixed-model fitted_p (includes BLUPs)

Plus diagnostic Wilcoxon tests on per-rat shifts (down-same, up-same).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import wilcoxon
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[4]   # history_dependecy/ (script is in GLMM_ANALYSIS/scripts/04_acuity_transitions/<step>/)
OUT04 = PROJECT_ROOT / "GLMM_ANALYSIS" / "outputs" / "04_acuity_transitions"


# ─── CONFIG ───────────────────────────────────────────────────────────────────

GLM_DIR = str(PROJECT_ROOT / "per_cell_glm_results")
FIT_DIR = str(OUT04 / "b_rank_transition_fits" / "formula2_uncorrelated_try_more_random")
OUT_DIR = str(OUT04 / "c_validation_vs_per_cell" / "formula2")

TYPE_B = {11, 13, 16}
MIN_TRIALS_PER_CELL = 50
MIN_PER_BIN         = 20


# ─── LOAD ─────────────────────────────────────────────────────────────────────

def load_all():
    print("Loading data and model results...")

    data = pd.read_csv(os.path.join(FIT_DIR, 'data.csv'))
    preds = pd.read_csv(os.path.join(FIT_DIR, 'predictions.csv'))
    assert len(data) == len(preds), "Row count mismatch — files not aligned!"

    df = data.copy()
    df['fitted_p'] = preds['fitted_p'].values
    df['transition'] = df['transition_cell'].str.replace('_to_', '->', regex=False)
    df['rat'] = df['rat'].astype(int)

    fixed = pd.read_csv(os.path.join(FIT_DIR, 'fixed_effects.csv')).set_index('term')

    blups = pd.read_csv(os.path.join(FIT_DIR, 'blups_rat.csv'))
    blups = blups.rename(columns={'(Intercept)': 'intercept',
                                  'X.Intercept.': 'intercept'})
    blups['rat'] = blups['id'].astype(int)

    glm = pd.read_csv(os.path.join(GLM_DIR, 'per_cell_betas.csv'))
    glm['rat'] = glm['rat'].astype(int)

    mapping = pd.read_csv(os.path.join(GLM_DIR, 'transition_dir_mapping.csv'))
    mapping['rat'] = mapping['rat'].astype(int)

    print(f"  Trials: {len(df):,} across {df['rat'].nunique()} rats")
    print(f"  Fixed effects: {len(fixed)} terms")
    print(f"  BLUPs: {len(blups)} rats × {len([c for c in blups.columns if c not in ['id','rat']])} components")
    print(f"  Available BLUP columns: {[c for c in blups.columns if c not in ['id','rat']]}")

    return df, fixed, blups, glm, mapping


# ─── EMPIRICAL SLOPES (unchanged) ────────────────────────────────────────────

def slope_uncentered(cell_df, hist_col, response='action'):
    plus  = cell_df.loc[cell_df[hist_col] == +1, response]
    minus = cell_df.loc[cell_df[hist_col] == -1, response]
    if len(plus) < MIN_PER_BIN or len(minus) < MIN_PER_BIN:
        return np.nan
    return (plus.mean() - minus.mean()) / 2


def slope_session_centered(cell_df, hist_col, response='action'):
    if len(cell_df) < MIN_TRIALS_PER_CELL:
        return np.nan
    g = cell_df.copy()
    g['h_c'] = g.groupby('date')[hist_col].transform(lambda x: x - x.mean())
    g['a_c'] = g.groupby('date')[response].transform(lambda x: x - x.mean())
    denom = (g['h_c'] ** 2).sum()
    if denom < 1e-8:
        return np.nan
    return (g['h_c'] * g['a_c']).sum() / denom


# ─── PER-CELL COMPARISON (UPDATED for Formula 2) ─────────────────────────────

HIST_CONFIG = {
    'failure_n1': {
        'glm_col':     'failure_n-1',
        'fe_main':     'failure_n1_c',
        'fe_x_down':   'failure_n1_c:td_down',
        'fe_x_up':     'failure_n1_c:td_up',
        'fe_x_lo':     'failure_n1_c:ac_low',     # NEW
        'fe_x_hi':     'failure_n1_c:ac_high',    # NEW
        'blup_main':   'failure_n1_c',
        'blup_x_down': 'failure_n1_c:td_down',
        'blup_x_up':   'failure_n1_c:td_up',
        'blup_x_lo':   'failure_n1_c:ac_low',     # NEW
        'blup_x_hi':   'failure_n1_c:ac_high',    # NEW
    },
    'success_n1': {
        'glm_col':     'success_n-1',
        'fe_main':     'success_n1_c',
        'fe_x_down':   'success_n1_c:td_down',
        'fe_x_up':     'success_n1_c:td_up',
        'fe_x_lo':     'success_n1_c:ac_low',     # NEW
        'fe_x_hi':     'success_n1_c:ac_high',    # NEW
        'blup_main':   'success_n1_c',
        'blup_x_down': 'success_n1_c:td_down',
        'blup_x_up':   'success_n1_c:td_up',
        'blup_x_lo':   'success_n1_c:ac_low',     # NEW
        'blup_x_hi':   'success_n1_c:ac_high',    # NEW
    },
}


def safe_blup(b, key):
    """Return BLUP value or 0 if column doesn't exist (e.g. singular component)."""
    return b[key] if key in b.index else 0.0


def build_per_cell_table(df, fixed, blups, glm, mapping, hist_col):
    """One row per (rat, transition) cell; estimates on probability scale.
    Includes contributions from both transition_dir AND ac_low/ac_high.
    """
    cfg     = HIST_CONFIG[hist_col]
    fe_main = fixed.loc[cfg['fe_main'],   'Estimate']
    fe_x_d  = fixed.loc[cfg['fe_x_down'], 'Estimate']
    fe_x_u  = fixed.loc[cfg['fe_x_up'],   'Estimate']
    fe_x_lo = fixed.loc[cfg['fe_x_lo'],   'Estimate']   # NEW
    fe_x_hi = fixed.loc[cfg['fe_x_hi'],   'Estimate']   # NEW

    rows = []
    for rat in sorted(df['rat'].unique()):
        rat_df = df[df['rat'] == rat]
        b      = blups.loc[blups['rat'] == rat]
        if len(b) == 0:
            continue
        b = b.iloc[0]

        for trans in sorted(rat_df['transition'].unique()):
            cell = rat_df[rat_df['transition'] == trans]
            n    = len(cell)
            if n < MIN_TRIALS_PER_CELL:
                continue

            # Empirical and model-prediction slopes (probability scale)
            raw_unc  = slope_uncentered(cell, hist_col)
            raw_cent = slope_session_centered(cell, hist_col)
            model_p  = slope_uncentered(cell, hist_col, response='fitted_p')

            # Cell baseline P(right) for logit→prob conversion
            p_bar = cell['action'].mean()
            scale = p_bar * (1 - p_bar)

            # Per-cell GLM beta (logit) → probability
            mask = ((glm['rat'] == rat) &
                    (glm['transition'] == trans) &
                    (glm['beta_name'] == cfg['glm_col']))
            beta_logit = glm.loc[mask, 'beta'].values
            beta_logit = beta_logit[0] if len(beta_logit) else np.nan
            beta_prob  = beta_logit * scale if not np.isnan(beta_logit) else np.nan

            # Determine cell's transition_dir AND ac level
            map_row = mapping[(mapping['rat'] == rat) & (mapping['transition'] == trans)]
            td      = map_row['transition_dir'].iloc[0] if len(map_row) else None

            # Determine ac level from the data (constant within rat × cell)
            ac_low_val  = cell['ac_low'].iloc[0]   if 'ac_low'  in cell.columns else 0
            ac_high_val = cell['ac_high'].iloc[0]  if 'ac_high' in cell.columns else 0
            if   ac_low_val  == 1: ac_level = 'low'
            elif ac_high_val == 1: ac_level = 'high'
            else:                  ac_level = 'med'

            # ── Mixed-model per-rat logit slope ──
            # base: fixed main effect + rat-level main effect BLUP
            m_logit = fe_main + safe_blup(b, cfg['blup_main'])

            # add transition_dir contribution
            if td == 'down':
                m_logit += fe_x_d + safe_blup(b, cfg['blup_x_down'])
            elif td == 'up':
                m_logit += fe_x_u + safe_blup(b, cfg['blup_x_up'])
            # else 'same': no contribution

            # add acuity-level contribution (NEW for Formula 2)
            if ac_level == 'low':
                m_logit += fe_x_lo + safe_blup(b, cfg['blup_x_lo'])
            elif ac_level == 'high':
                m_logit += fe_x_hi + safe_blup(b, cfg['blup_x_hi'])
            # else 'med': no contribution

            m_prob = m_logit * scale

            rows.append({
                'rat': rat, 'transition': trans, 'transition_dir': td,
                'ac_level': ac_level,
                'n_trials': n, 'p_baseline': p_bar,
                'raw_slope_unc':       raw_unc,
                'raw_slope_cent':      raw_cent,
                'per_cell_glm_prob':   beta_prob,
                'per_cell_glm_logit':  beta_logit,
                'model_logit_per_rat': m_logit,
                'model_logit_to_prob': m_prob,
                'model_prob':          model_p,
            })
    return pd.DataFrame(rows)


# ─── AGGREGATE TO 3-LEVEL DIRECTION (unchanged) ──────────────────────────────

def aggregate_to_direction(df, hist_col):
    rows = []
    for rat in sorted(df['rat'].unique()):
        rat_df = df[df['rat'] == rat]
        for td in ['same', 'down', 'up']:
            cell = rat_df[rat_df['transition_dir'] == td]
            rows.append({
                'rat': rat, 'transition_dir': td, 'n_trials': len(cell),
                'raw_slope_unc':  slope_uncentered(cell, hist_col),
                'raw_slope_cent': slope_session_centered(cell, hist_col),
                'model_prob':     slope_uncentered(cell, hist_col, response='fitted_p'),
            })
    return pd.DataFrame(rows)


def compute_shifts(direction_table):
    pivot = direction_table.pivot(
        index='rat', columns='transition_dir',
        values=['raw_slope_unc', 'raw_slope_cent', 'model_prob']
    )
    rows = []
    for rat in pivot.index:
        for shift_name in ['down', 'up']:
            row = {'rat': rat, 'shift': f'{shift_name}-same'}
            for est in ['raw_slope_unc', 'raw_slope_cent', 'model_prob']:
                row[est] = pivot.loc[rat, (est, shift_name)] - pivot.loc[rat, (est, 'same')]
            rows.append(row)
    return pd.DataFrame(rows)

def aggregate_to_acuity(df, hist_col):
    """Per-rat × ac_level slopes, pooling cells within acuity level."""
    rows = []
    for rat in sorted(df['rat'].unique()):
        rat_df = df[df['rat'] == rat]
        for ac in ['med', 'low', 'high']:
            if ac == 'low':
                cell = rat_df[rat_df['ac_low'] == 1]
            elif ac == 'high':
                cell = rat_df[rat_df['ac_high'] == 1]
            else:  # med = both dummies are 0
                cell = rat_df[(rat_df['ac_low'] == 0) & (rat_df['ac_high'] == 0)]
            rows.append({
                'rat': rat, 'ac_level': ac, 'n_trials': len(cell),
                'raw_slope_unc':  slope_uncentered(cell, hist_col),
                'raw_slope_cent': slope_session_centered(cell, hist_col),
                'model_prob':     slope_uncentered(cell, hist_col, response='fitted_p'),
            })
    return pd.DataFrame(rows)

def aggregate_to_acuity_within_same(df, hist_col):
    """Failure/success slope in same-modality cells, by acuity level.
    Isolates the pure acuity effect by holding transition_dir = 'same'."""
    rows = []
    for rat in sorted(df['rat'].unique()):
        rat_df = df[(df['rat'] == rat) & (df['transition_dir'] == 'same')]
        for ac in ['med', 'low', 'high']:
            if ac == 'low':
                cell = rat_df[rat_df['ac_low'] == 1]
            elif ac == 'high':
                cell = rat_df[rat_df['ac_high'] == 1]
            else:
                cell = rat_df[(rat_df['ac_low'] == 0) & (rat_df['ac_high'] == 0)]
            rows.append({
                'rat': rat, 'ac_level': ac, 'n_trials': len(cell),
                'raw_slope_unc':  slope_uncentered(cell, hist_col),
                'raw_slope_cent': slope_session_centered(cell, hist_col),
                'model_prob':     slope_uncentered(cell, hist_col, response='fitted_p'),
            })
    return pd.DataFrame(rows)


def compute_acuity_shifts(acuity_table):
    """Per-rat high−med and low−med shifts for each estimator."""
    pivot = acuity_table.pivot(
        index='rat', columns='ac_level',
        values=['raw_slope_unc', 'raw_slope_cent', 'model_prob']
    )
    rows = []
    for rat in pivot.index:
        for shift_name in ['high', 'low']:
            row = {'rat': rat, 'shift': f'{shift_name}-med'}
            for est in ['raw_slope_unc', 'raw_slope_cent', 'model_prob']:
                row[est] = pivot.loc[rat, (est, shift_name)] - pivot.loc[rat, (est, 'med')]
            rows.append(row)
    return pd.DataFrame(rows)


# ─── WILCOXON TESTS (unchanged) ──────────────────────────────────────────────

def run_wilcoxon(shifts, hist_col):
    print(f"\n{'='*68}")
    print(f"  Wilcoxon signed-rank tests — {hist_col}")
    print(f"  H0: per-rat shift = 0   (probability-scale slope)")
    print('='*68)

    for shift_type in ['down-same', 'up-same']:
        sub = shifts[shifts['shift'] == shift_type]
        print(f"\n  Shift: {shift_type}    (n = {len(sub)} rats)")
        for est in ['raw_slope_unc', 'raw_slope_cent', 'model_prob']:
            vals = sub[est].dropna()
            if len(vals) < 6:
                print(f"    {est:<18}: insufficient data ({len(vals)})")
                continue
            stat, p = wilcoxon(vals)
            n_neg = (vals < 0).sum()
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            print(f"    {est:<18}: median={vals.median():+.4f}, "
                  f"n_neg={n_neg}/{len(vals)}, W={stat:5.1f}, p={p:.4f} {sig}")
            

def run_wilcoxon_acuity(shifts, hist_col):
    print(f"\n{'='*68}")
    print(f"  Wilcoxon signed-rank tests — {hist_col} (acuity level)")
    print(f"  H0: per-rat shift = 0   (probability-scale slope)")
    print('='*68)

    for shift_type in ['high-med', 'low-med']:
        sub = shifts[shifts['shift'] == shift_type]
        print(f"\n  Shift: {shift_type}    (n = {len(sub)} rats)")
        for est in ['raw_slope_unc', 'raw_slope_cent', 'model_prob']:
            vals = sub[est].dropna()
            if len(vals) < 6:
                print(f"    {est:<18}: insufficient data ({len(vals)})")
                continue
            stat, p = wilcoxon(vals)
            n_neg = (vals < 0).sum()
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            print(f"    {est:<18}: median={vals.median():+.4f}, "
                  f"n_neg={n_neg}/{len(vals)}, W={stat:5.1f}, p={p:.4f} {sig}")


# ─── PLOTS (unchanged, plus one new) ─────────────────────────────────────────

def color_for_rat(rat):
    return 'crimson' if int(rat) in TYPE_B else 'steelblue'


def plot_per_rat_shifts(shifts_fail, shifts_succ, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
    np.random.seed(42)

    estimators = [('raw_slope_unc',  'Raw (uncentered)', 'o'),
                  ('raw_slope_cent', 'Session-centered', 's'),
                  ('model_prob',     'Model fitted_p',   '^')]

    for ax, shifts, label in zip(axes, [shifts_fail, shifts_succ],
                                  ['Failure', 'Success']):
        for shift_idx, shift_name in enumerate(['down-same', 'up-same']):
            sub = shifts[shifts['shift'] == shift_name]
            for est_idx, (col, _, marker) in enumerate(estimators):
                xs = shift_idx + 0.25 * (est_idx - 1) + np.random.uniform(-0.04, 0.04, len(sub))
                colors = [color_for_rat(r) for r in sub['rat']]
                ax.scatter(xs, sub[col], c=colors, marker=marker, s=80,
                           alpha=0.85, edgecolor='black', linewidth=0.4)
                med = sub[col].median()
                ax.hlines(med, xmin=shift_idx + 0.25*(est_idx-1) - 0.08,
                          xmax=shift_idx + 0.25*(est_idx-1) + 0.08,
                          colors='black', linewidth=2.5, zorder=10)

        ax.axhline(0, color='gray', lw=0.5, linestyle='--')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['down − same', 'up − same'], fontsize=11)
        ax.set_ylabel(f'Shift in {label.lower()}-history slope (P units)', fontsize=10)
        ax.set_title(f'{label}-history shifts (per rat)', fontsize=11)

    legend_items = (
        [plt.Line2D([0], [0], marker=m, color='gray', linestyle='', markersize=9, label=lab)
         for _, lab, m in estimators] +
        [mpatches.Patch(color='steelblue', label='Type A'),
         mpatches.Patch(color='crimson',   label='Type B'),
         plt.Line2D([0], [0], color='black', linewidth=2.5, label='Median (across rats)')]
    )
    fig.legend(handles=legend_items, loc='lower center', ncol=6,
               fontsize=9, bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.suptitle('Per-rat shifts in history-effect slopes (Formula 2)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  → {os.path.basename(save_path)}")
    plt.close()

def plot_per_rat_acuity_shifts(shifts_fail, shifts_succ, save_path):
    """Population view: each rat's high-med and low-med shifts, per estimator."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
    np.random.seed(42)

    estimators = [('raw_slope_unc',  'Raw (uncentered)', 'o'),
                  ('raw_slope_cent', 'Session-centered', 's'),
                  ('model_prob',     'Model fitted_p',   '^')]

    for ax, shifts, label in zip(axes, [shifts_fail, shifts_succ],
                                  ['Failure', 'Success']):
        for shift_idx, shift_name in enumerate(['high-med', 'low-med']):
            sub = shifts[shifts['shift'] == shift_name]
            for est_idx, (col, _, marker) in enumerate(estimators):
                xs = shift_idx + 0.25 * (est_idx - 1) + np.random.uniform(-0.04, 0.04, len(sub))
                colors = [color_for_rat(r) for r in sub['rat']]
                ax.scatter(xs, sub[col], c=colors, marker=marker, s=80,
                           alpha=0.85, edgecolor='black', linewidth=0.4)
                med = sub[col].median()
                ax.hlines(med, xmin=shift_idx + 0.25*(est_idx-1) - 0.08,
                          xmax=shift_idx + 0.25*(est_idx-1) + 0.08,
                          colors='black', linewidth=2.5, zorder=10)

        ax.axhline(0, color='gray', lw=0.5, linestyle='--')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['high − med', 'low − med'], fontsize=11)
        ax.set_ylabel(f'Shift in {label.lower()}-history slope (P units)', fontsize=10)
        ax.set_title(f'{label}-history acuity shifts (per rat)', fontsize=11)

    legend_items = (
        [plt.Line2D([0], [0], marker=m, color='gray', linestyle='', markersize=9, label=lab)
         for _, lab, m in estimators] +
        [mpatches.Patch(color='steelblue', label='Type A'),
         mpatches.Patch(color='crimson',   label='Type B'),
         plt.Line2D([0], [0], color='black', linewidth=2.5, label='Median (across rats)')]
    )
    fig.legend(handles=legend_items, loc='lower center', ncol=6,
               fontsize=9, bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.suptitle('Per-rat shifts in history-effect slopes across acuity levels (Formula 2)',
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  → {os.path.basename(save_path)}")
    plt.close()


def plot_per_cell_grid(per_cell_table, save_path, hist_col_label='Failure'):
    rats   = sorted(per_cell_table['rat'].unique())
    n_rats = len(rats)
    ncols  = 3
    nrows  = int(np.ceil(n_rats / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 3.5*nrows),
                              constrained_layout=True)
    axes = axes.flatten() if n_rats > 1 else [axes]

    for ax, rat in zip(axes, rats):
        rat_df = per_cell_table[per_cell_table['rat'] == rat].copy()
        rat_df['td_order'] = rat_df['transition_dir'].map({'down':0, 'same':1, 'up':2})
        rat_df = rat_df.sort_values(['td_order', 'transition'])
        x = np.arange(len(rat_df))

        for xi, td in enumerate(rat_df['transition_dir']):
            color = {'down':'#ffe0e0', 'same':'#e8e8e8', 'up':'#dde6f0'}[td]
            ax.axvspan(xi-0.5, xi+0.5, alpha=0.5, color=color, zorder=0)

        ax.axhline(0, color='black', lw=0.5, linestyle='--', alpha=0.6)
        ax.plot(x, rat_df['raw_slope_unc'],     'o-', color='steelblue', label='Raw',          markersize=7)
        ax.plot(x, rat_df['per_cell_glm_prob'], 's--', color='darkorange', label='Per-cell GLM', markersize=6)
        ax.plot(x, rat_df['model_prob'],        '^:', color='darkgreen',  label='Model',        markersize=6)

        ax.set_xticks(x)
        # Add ac_level annotation under each xtick label
        xtick_labels = [f"{row['transition']}\n[{row['ac_level']}]"
                        for _, row in rat_df.iterrows()]
        ax.set_xticklabels(xtick_labels, rotation=45, ha='right', fontsize=7)
        strat = 'B' if int(rat) in TYPE_B else 'A'
        ax.set_title(f'Rat {rat} (Type {strat})', fontsize=10)
        ax.set_ylabel(f'{hist_col_label}-history slope (P units)', fontsize=8)
        if ax is axes[0]:
            ax.legend(fontsize=7, loc='best')

    for ax in axes[n_rats:]:
        ax.set_visible(False)

    fig.suptitle(f'Per-rat per-cell {hist_col_label.lower()}-history slopes (Formula 2) — '
                 'three estimators compared\n[ac level] annotated under each cell',
                 fontsize=11, y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  → {os.path.basename(save_path)}")
    plt.close()


def plot_correlation_scatter(per_cell_table, save_path, hist_col_label='Failure'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    pairs = [
        ('raw_slope_unc',     'per_cell_glm_prob',  'Raw vs Per-cell GLM'),
        ('raw_slope_unc',     'model_prob',         'Raw vs Model'),
        ('per_cell_glm_prob', 'model_prob',         'Per-cell GLM vs Model'),
    ]
    for ax, (x_col, y_col, title) in zip(axes, pairs):
        sub = per_cell_table[[x_col, y_col, 'rat']].dropna()
        colors = [color_for_rat(r) for r in sub['rat']]
        ax.scatter(sub[x_col], sub[y_col], c=colors, alpha=0.7,
                   s=60, edgecolor='black', linewidth=0.4)

        lo = min(sub[x_col].min(), sub[y_col].min())
        hi = max(sub[x_col].max(), sub[y_col].max())
        ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, label='Identity')
        ax.axhline(0, color='gray', lw=0.3); ax.axvline(0, color='gray', lw=0.3)

        if len(sub) > 3:
            r = sub[[x_col, y_col]].corr().iloc[0, 1]
            ax.text(0.05, 0.95, f'r = {r:.3f}\n n = {len(sub)}',
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel(x_col); ax.set_ylabel(y_col); ax.set_title(title, fontsize=10)

    legend_items = [mpatches.Patch(color='steelblue', label='Type A'),
                    mpatches.Patch(color='crimson',   label='Type B')]
    fig.legend(handles=legend_items, loc='lower center', ncol=2,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f'Per-cell scatter (Formula 2) — {hist_col_label}-history (each dot = rat × cell)',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  → {os.path.basename(save_path)}")
    plt.close()


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df, fixed, blups, glm, mapping = load_all()

    results = {}
    for hist_col in ['failure_n1', 'success_n1']:
        print(f"\n{'='*68}")
        print(f"  Processing: {hist_col}")
        print('='*68)

        per_cell = build_per_cell_table(df, fixed, blups, glm, mapping, hist_col)
        per_cell.to_csv(os.path.join(OUT_DIR, f'per_cell_{hist_col}.csv'), index=False)
        print(f"  Per-cell table: {len(per_cell)} rows → per_cell_{hist_col}.csv")

        direction = aggregate_to_direction(df, hist_col)
        direction.to_csv(os.path.join(OUT_DIR, f'per_direction_{hist_col}.csv'), index=False)

        shifts = compute_shifts(direction)
        shifts.to_csv(os.path.join(OUT_DIR, f'shifts_{hist_col}.csv'), index=False)

        run_wilcoxon(shifts, hist_col)

        results[hist_col] = {'per_cell': per_cell, 'direction': direction, 'shifts': shifts}

    # Acuity-level aggregation
        acuity = aggregate_to_acuity_within_same(df, hist_col)
        acuity.to_csv(os.path.join(OUT_DIR, f'per_acuity_{hist_col}.csv'), index=False)

        acuity_shifts = compute_acuity_shifts(acuity)
        acuity_shifts.to_csv(os.path.join(OUT_DIR, f'acuity_shifts_{hist_col}.csv'), index=False)

        run_wilcoxon_acuity(acuity_shifts, hist_col)

        results[hist_col]['acuity'] = acuity
        results[hist_col]['acuity_shifts'] = acuity_shifts

    print(f"\n{'='*68}")
    print(f"  Plots saved to {OUT_DIR}")
    print('='*68)
    plot_per_rat_shifts(results['failure_n1']['shifts'],
                        results['success_n1']['shifts'],
                        os.path.join(OUT_DIR, 'fig1_per_rat_shifts.png'))
    plot_per_rat_acuity_shifts(results['failure_n1']['acuity_shifts'],
                                results['success_n1']['acuity_shifts'],
                                os.path.join(OUT_DIR, 'fig1b_per_rat_acuity_shifts.png'))
    plot_per_cell_grid(results['failure_n1']['per_cell'],
                       os.path.join(OUT_DIR, 'fig2_per_cell_grid_failure.png'),
                       hist_col_label='Failure')
    plot_per_cell_grid(results['success_n1']['per_cell'],
                       os.path.join(OUT_DIR, 'fig2_per_cell_grid_success.png'),
                       hist_col_label='Success')
    plot_correlation_scatter(results['failure_n1']['per_cell'],
                             os.path.join(OUT_DIR, 'fig3_scatter_failure.png'),
                             hist_col_label='Failure')
    plot_correlation_scatter(results['success_n1']['per_cell'],
                             os.path.join(OUT_DIR, 'fig3_scatter_success.png'),
                             hist_col_label='Success')

    print("\nDone.")


if __name__ == '__main__':
    main()