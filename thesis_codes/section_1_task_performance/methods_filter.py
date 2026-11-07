"""
Reproduces every figure and table used in the "filter diagnostics" methods
section: the two-panel typical-session figure, the rat-2-vs-9-vs-12
diagnostic figure, and the per-rat removal summary table (including the
non-0/90-among-warmup-removed column).

Depends on `filter_diagnostics.py` (delivered earlier) sitting in the same
directory as this script, or on your Python path.

Usage:
    python make_methods_figures.py

Edit the CONFIG block below to point at your own data file and to change
which rats are used in the example-session / diagnosis figures.
"""

import os
import sys
from zipfile import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from filter_diagnostics import filter_training_trials_verbose  # noqa: E402

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

# Path to the trial-level dataset (.csv or .parquet), already the output of
# create_lagged_features() / load_and_preprocess_data() -- i.e. it must
# already have the angle/action/hitmiss/mod (+ '_n-1' lag) columns.
INPUT_PATH = 'C:\\dev\\projects\\Thesis_SISSA\\Rats\\history_dependecy\\trimmed_for_diagnostics.parquet'

# Where figures/tables get written.
OUTPUT_DIR = Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\ale_thesis_fig_tab\tables_thesis\section_1")

# Columns load_base_data() drops NaNs/out-of-bounds rows on, before ever
# calling the filter -- mirrors Section 3's preprocessing exactly.
REQUIRED_COLS = ['angle', 'angle_n-1', 'action', 'action_n-1', 'hitmiss_n-1',
                  'rat', 'date', 'mod']

# filter_training_trials kwargs, matching FILTER_KWARGS in
# Section_3_choice_reward_joint_bootstrap.py.
FILTER_KWARGS = dict(method='criterion', criterion=0.80, min_consec=2)

# Rats used in the rat-level diagnosis figure (session-by-session accuracy +
# 100%-warmup-session markers). First rat should be a "clean" low-exclusion
# example; the rest are whichever rats you want to contrast it against.
DIAGNOSIS_RATS = [2.0, 9.0, 12.0]

BOUNDARY_ANGLE = 45
EASY_ANGLE_MARGIN = 30  # easy trial: angle <= boundary-margin or >= boundary+margin

STATUS_COLORS = {'kept': '#2b6cb0', 'warmup': '#e53e3e', 'learning_phase': '#dd6b20'}
STATUS_LABELS = {'kept': 'kept', 'warmup': 'removed: warm-up',
                  'learning_phase': 'removed: learning phase'}


# --------------------------------------------------------------------------
# Data loading + filtering
# --------------------------------------------------------------------------

def load_and_filter():
    """Loads the raw trial data, applies the load_base_data()-equivalent
    dropna/bounds step, runs the (bug-fixed) filter, and returns a single
    combined dataframe with a 'status' column in
    {'kept', 'warmup', 'learning_phase'} plus 'session_idx'."""
    if INPUT_PATH.endswith('.parquet'):
        df = pd.read_parquet(INPUT_PATH)
    else:
        df = pd.read_csv(INPUT_PATH, comment='#')
    print("Loaded:", df.shape)

    data = df.dropna(subset=REQUIRED_COLS).copy()
    data = data[data['angle'].between(0, 90)]
    data = data[data['angle_n-1'].between(0, 90)]
    data = data[data['action'].isin([0, 1])]
    data = data[data['action_n-1'].isin([0, 1])]
    data = data[data['hitmiss_n-1'].isin([0, 1])]
    print("After load_base_data()-equivalent dropna/bounds:", data.shape)

    df_kept, df_removed = filter_training_trials_verbose(data, verbose=False, **FILTER_KWARGS)
    print("Kept:", df_kept.shape, "Removed:", df_removed.shape)

    df_kept = df_kept.copy()
    df_kept['status'] = 'kept'
    df_removed = df_removed.copy()
    df_removed['status'] = df_removed['removed_reason']

    combined = pd.concat([df_kept, df_removed], ignore_index=True)
    combined = combined.sort_values(['rat', 'session_idx', 'trialID']).reset_index(drop=True)
    return combined


# --------------------------------------------------------------------------
# Figure 1: typical retained session vs. typical fully-discarded session
# --------------------------------------------------------------------------

def find_example_sessions(combined):
    """Auto-selects a representative 'clean retained' session (from the
    rat with the lowest overall warm-up removal, a session with a modest
    warm-up block followed by plenty of kept trials) and a representative
    'fully-discarded, 100%-warmup' session (from the rat with the highest
    overall warm-up removal). Returns ((rat, session_idx), (rat, session_idx))."""
    per_rat_pct_warmup = (
        combined.groupby('rat')['status']
        .apply(lambda s: (s == 'warmup').mean())
        .sort_values()
    )
    low_rat = per_rat_pct_warmup.index[0]
    high_rat = per_rat_pct_warmup.index[-1]

    # -- typical retained session (from the cleanest rat) --
    sub = combined[combined['rat'] == low_rat]
    per_sess = sub.groupby('session_idx')['status'].value_counts().unstack(fill_value=0)
    cand = per_sess[(per_sess.get('warmup', 0).between(5, 40)) & (per_sess.get('kept', 0) >= 80)]
    retained_session_idx = cand.index[0] if len(cand) else per_sess.index[0]

    # -- typical fully-discarded session (from the messiest rat) --
    sub2 = combined[combined['rat'] == high_rat]
    per_sess2 = sub2.groupby('session_idx')['status'].value_counts().unstack(fill_value=0)
    is_pure_dropped = (per_sess2.get('kept', 0) == 0) & (per_sess2.get('learning_phase', 0) == 0)
    cand2 = per_sess2[is_pure_dropped & (per_sess2.get('warmup', 0).between(40, 90))]
    if len(cand2):
        discarded_session_idx = cand2.index[0]
    else:
        # fall back to whatever fully-dropped session exists, any size
        cand2b = per_sess2[is_pure_dropped]
        discarded_session_idx = cand2b.index[0] if len(cand2b) else per_sess2.index[0]

    return (low_rat, retained_session_idx), (high_rat, discarded_session_idx)


def plot_typical_kept_vs_discarded(combined, retained, discarded, out_path):
    (rat_a, sess_a), (rat_b, sess_b) = retained, discarded

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)

    sess_df_a = combined[(combined['rat'] == rat_a) & (combined['session_idx'] == sess_a)] \
        .sort_values('trialID').reset_index(drop=True)
    axA = axes[0]
    for status, sub in sess_df_a.groupby('status'):
        axA.scatter(sub.index, sub['angle'], s=14, color=STATUS_COLORS[status],
                    label=STATUS_LABELS[status], zorder=3)
    axA.axhline(BOUNDARY_ANGLE, color='gray', ls=':', lw=1, zorder=1)
    n_warmup_a = (sess_df_a['status'] == 'warmup').sum()
    axA.set_title(f"Typical retained session\n(rat {rat_a:g}, session {sess_a}: "
                  f"{len(sess_df_a)} trials, {n_warmup_a} warm-up)")
    axA.set_xlabel('trial number within session')
    axA.set_ylabel('stimulus angle (deg)')

    sess_df_b = combined[(combined['rat'] == rat_b) & (combined['session_idx'] == sess_b)] \
        .sort_values('trialID').reset_index(drop=True)
    axB = axes[1]
    for status, sub in sess_df_b.groupby('status'):
        axB.scatter(sub.index, sub['angle'], s=14, color=STATUS_COLORS[status],
                    label=STATUS_LABELS[status], zorder=3)
    axB.axhline(BOUNDARY_ANGLE, color='gray', ls=':', lw=1, zorder=1)
    axB.set_title(f"Typical fully-discarded session\n(rat {rat_b:g}, session {sess_b}: "
                  f"{len(sess_df_b)} trials, all warm-up)")
    axB.set_xlabel('trial number within session')

    handles, labels_ = axA.get_legend_handles_labels()
    fig.legend(handles, labels_, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle('Example sessions illustrating the two filter outcomes', y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {out_path}")


# --------------------------------------------------------------------------
# Figure 2: per-rat session-by-session accuracy + 100%-warmup markers
# --------------------------------------------------------------------------

def plot_rat_diagnosis(combined, rat_ids, out_path):
    """For each rat: plots easy-trial accuracy per session (only sessions
    that have at least one non-warmup trial, i.e. passed stage 1), marks
    sessions that are 100% warm-up-angle trials (no real-task trials at
    all) as red ticks along the bottom, and draws the criterion_session
    cutoff as a vertical dashed line when one exists."""
    fig, axes = plt.subplots(len(rat_ids), 1, figsize=(11, 3.6 * len(rat_ids)), sharex=False)
    if len(rat_ids) == 1:
        axes = [axes]

    for ax, rat in zip(axes, rat_ids):
        sub = combined[combined['rat'] == rat]
        per_sess_status = sub.groupby('session_idx')['status'].value_counts().unstack(fill_value=0)

        # a session is "100% warm-up-angle" (no-start, wholesale dropped)
        # if it has zero kept and zero learning_phase trials
        pure_warmup_sessions = per_sess_status[
            (per_sess_status.get('kept', 0) == 0) & (per_sess_status.get('learning_phase', 0) == 0)
        ].index

        has_real = sub[sub['status'] != 'warmup']  # passed stage 1
        easy = has_real[(has_real['angle'] <= BOUNDARY_ANGLE - EASY_ANGLE_MARGIN) |
                         (has_real['angle'] >= BOUNDARY_ANGLE + EASY_ANGLE_MARGIN)]
        acc = easy.groupby('session_idx')['hitmiss'].mean()

        ax.scatter(acc.index, acc.values, s=18, color=STATUS_COLORS['kept'],
                   label='has real-task trials (easy-trial acc)', zorder=3)
        if len(pure_warmup_sessions):
            ax.scatter(pure_warmup_sessions, [-0.05] * len(pure_warmup_sessions),
                      marker='|', color=STATUS_COLORS['warmup'], s=40,
                      label='100%-warmup session (dropped)')
        ax.axhline(FILTER_KWARGS.get('criterion', 0.8), color='gray', ls=':', lw=1)

        crit_vals = has_real[has_real['status'] == 'kept']['criterion_session']
        if len(crit_vals) and crit_vals.notna().any():
            crit = crit_vals.iloc[0]
            ax.axvline(crit, color='k', ls='--', lw=1)
            ax.text(crit + 1, 0.02, f'criterion_session={crit:g}', fontsize=9)

        n_pure = len(pure_warmup_sessions)
        n_total_sess = sub['session_idx'].nunique()
        ax.set_title(f"Rat {rat:g}: {n_pure}/{n_total_sess} sessions are 100% warm-up-angle trials")
        ax.set_ylabel('easy-trial accuracy')
        ax.set_ylim(-0.15, 1.05)

    axes[0].legend(loc='lower right', fontsize=9)
    axes[-1].set_xlabel('session_idx (chronological)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# --------------------------------------------------------------------------
# Table: per-rat removal summary incl. non-0/90 among warm-up-removed
# --------------------------------------------------------------------------

def build_removal_table(combined, out_path, warmup_angles=(0, 90)):
    rows = []
    for rat, g in combined.groupby('rat'):
        n_total = len(g)
        n_kept = (g['status'] == 'kept').sum()
        wu = g[g['status'] == 'warmup']
        lp = g[g['status'] == 'learning_phase']
        n_wu = len(wu)
        n_wu_nonextreme = (~wu['angle'].isin(warmup_angles)).sum()
        n_lp = len(lp)
        rows.append(dict(
            rat=rat,
            n_input_trials=n_total,
            n_kept=n_kept,
            pct_kept=round(100 * n_kept / n_total, 1),
            n_removed_warmup=n_wu,
            pct_removed_warmup=round(100 * n_wu / n_total, 1),
            n_warmup_nonextreme=int(n_wu_nonextreme),
            pct_warmup_nonextreme=round(100 * n_wu_nonextreme / n_wu, 2) if n_wu else 0.0,
            n_removed_learning_phase=n_lp,
            pct_removed_learning_phase=round(100 * n_lp / n_total, 1),
        ))
    out = pd.DataFrame(rows).sort_values('pct_kept', ascending=False)
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print(out.to_string(index=False))
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    combined = load_and_filter()

    retained, discarded = find_example_sessions(combined)
    plot_typical_kept_vs_discarded(
        combined, retained, discarded,
        os.path.join(OUTPUT_DIR, 'typical_kept_vs_discarded_session.png'),
    )

    plot_rat_diagnosis(
        combined, DIAGNOSIS_RATS,
        os.path.join(OUTPUT_DIR, 'rat_diagnosis.png'),
    )

    build_removal_table(
        combined,
        os.path.join(OUTPUT_DIR, 'table_removal_summary_with_nonextreme.csv'),
    )

    print("\nDONE")