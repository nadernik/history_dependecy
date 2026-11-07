"""
perceptual_bias_test.py
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
#df = filter_training_trials(df, method='criterion', criterion=0.80, min_consec=2)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def diagnose_session_warmup(df, rat, n_sessions=6,
                             warmup_angles=(0, 90),
                             window_size=20,
                             angle_col='angle', trial_id_col='trialID',
                             outcome_col='hitmiss'):
    """
    For a given rat, plot the first n_sessions to reveal warmup structure.
    Two rows per session:
      row 1: running fraction of warmup angles (0/90) in a sliding window,
             with the session baseline (last 50%) as a dashed line
      row 2: running accuracy on warmup-angle trials only, in the same
             window, with the session late-baseline as a dashed line
    """
    rat_df = df[df['rat'] == rat].copy()
    sessions = sorted(rat_df['date'].dropna().unique())[:n_sessions]

    fig, axes = plt.subplots(2, len(sessions),
                              figsize=(3.2 * len(sessions), 6),
                              sharey='row')
    if len(sessions) == 1:
        axes = axes.reshape(2, 1)

    for col, sess in enumerate(sessions):
        s = (rat_df[rat_df['date'] == sess]
             .sort_values(trial_id_col).reset_index(drop=True))
        angles   = s[angle_col].values
        outcomes = s[outcome_col].values
        valid    = ~pd.isna(angles)
        is_wu    = np.isin(angles, list(warmup_angles)) & valid
        n = len(angles)
        if n < window_size:
            continue

        # running density and running accuracy-on-warmup
        density, acc_wu, pos = [], [], []
        for i in range(n - window_size + 1):
            wv = valid[i:i+window_size]
            wu = is_wu[i:i+window_size]
            if wv.sum() == 0:
                continue
            density.append(wu.sum() / wv.sum())
            wu_trials = wu & ~pd.isna(outcomes[i:i+window_size])
            if wu_trials.sum() >= 3:
                acc_wu.append(np.nanmean(outcomes[i:i+window_size][wu_trials]))
            else:
                acc_wu.append(np.nan)
            pos.append(i + window_size // 2)

        # baselines from the last half of the session
        half = n // 2
        base_density = is_wu[half:].sum() / max(1, valid[half:].sum())
        late_wu = is_wu[half:] & ~pd.isna(outcomes[half:])
        base_acc = (np.nanmean(outcomes[half:][late_wu])
                    if late_wu.sum() >= 5 else np.nan)

        axes[0, col].plot(pos, density, color='steelblue')
        axes[0, col].axhline(base_density, ls='--', color='k', alpha=0.6,
                              label=f'baseline={base_density:.2f}')
        axes[0, col].set_title(f'session {sess}\n(n={n})', fontsize=9)
        axes[0, col].set_ylim(0, 1)
        axes[0, col].legend(fontsize=7)

        axes[1, col].plot(pos, acc_wu, color='crimson')
        if not np.isnan(base_acc):
            axes[1, col].axhline(base_acc, ls='--', color='k', alpha=0.6,
                                  label=f'baseline={base_acc:.2f}')
            axes[1, col].legend(fontsize=7)
        axes[1, col].set_ylim(0, 1.05)
        axes[1, col].set_xlabel('trial position')

    axes[0, 0].set_ylabel('running frac(0° or 90°)')
    axes[1, 0].set_ylabel('running accuracy\non 0°/90° only')
    fig.suptitle(f'Rat {rat} — warmup diagnostic (window={window_size})',
                 fontsize=11)
    plt.tight_layout()
    plt.show()



def diagnose_warmup_leakage(df, warmup_angles=(0, 90), n_probe=30,
                             angle_col='angle', trial_id_col='trialID'):
    """
    For each (rat, session), compare the current filter's cut point
    against how many warmup-angle trials appear in the *next* n_probe
    trials after the cut. A high 'leak_fraction' means warmup trials
    are surviving the filter.
    """
    warmup_set = set(warmup_angles)
    rows = []
    for (rat, sess), s in df.groupby(['rat', 'date']):
        s = s.sort_values(trial_id_col).reset_index(drop=True)
        angles = s[angle_col].values

        # replicate current filter's cut
        first_real_idx = next(
            (i for i, a in enumerate(angles)
             if pd.notna(a) and a not in warmup_set),
            None
        )
        if first_real_idx is None:
            continue

        probe  = angles[first_real_idx : first_real_idx + n_probe]
        n_leak = int(np.sum(np.isin(probe, list(warmup_angles))))
        rows.append({
            'rat': rat, 'date': sess,
            'cut_at_trial_index': first_real_idx,
            f'n_warmup_in_next_{n_probe}': n_leak,
            'leak_fraction': n_leak / max(1, len(probe)),
        })
    out = pd.DataFrame(rows)
    print("Per-rat median leak fraction (after current filter's cut):")
    print(out.groupby('rat')['leak_fraction'].median().to_string())
    print(f"\nSessions with >20% leak in next {n_probe} trials:")
    print(out[out['leak_fraction'] > 0.20]
          .groupby('rat').size().to_string())
    return out

def diagnose_session_warmup_v2(df, rat, session_range=None,
                                modality=None, n_sessions=6,
                                warmup_angles=(0, 90),
                                window_size=20,
                                proposed_ratio_threshold=2.0,
                                angle_col='angle', trial_id_col='trialID',
                                outcome_col='hitmiss', modality_col='mod'):
    """
    Post-criterion diagnostic. Baseline is the median density / accuracy
    in the middle third of the session (robust to end-of-session blocks).

    proposed_ratio_threshold : the cut visualization marks the first
        position at which running density drops below
        (mid-baseline * proposed_ratio_threshold). E.g. threshold=2.0
        means "warmup ends when the running window has < 2x the
        session baseline density of 0/90 trials".
    """
    rat_df = df[df['rat'] == rat].copy()

    # robust modality matching (int vs str vs float)
    if modality is not None:
        available = rat_df[modality_col].dropna().unique()
        # try exact, then string-cast
        if modality in available:
            rat_df = rat_df[rat_df[modality_col] == modality]
        else:
            matches = [m for m in available if str(m) == str(modality)]
            if not matches:
                print(f"Rat {rat}: modality {modality!r} not found. "
                      f"Available: {list(available)}")
                return
            rat_df = rat_df[rat_df[modality_col] == matches[0]]

    sess_all = sorted(rat_df['date'].dropna().unique())
    if session_range is not None:
        lo, hi = session_range
        sess_all = [s for s in sess_all if lo <= s <= hi]

    if not sess_all:
        print(f"Rat {rat}: no sessions in range {session_range} "
              f"for modality {modality}.")
        return

    if len(sess_all) > n_sessions:
        idx = np.linspace(0, len(sess_all) - 1, n_sessions).astype(int)
        sessions = [sess_all[i] for i in idx]
    else:
        sessions = sess_all

    fig, axes = plt.subplots(2, len(sessions),
                              figsize=(3.2 * len(sessions), 6),
                              sharey='row', squeeze=False)

    for col, sess in enumerate(sessions):
        s = (rat_df[rat_df['date'] == sess]
             .sort_values(trial_id_col).reset_index(drop=True))
        angles   = s[angle_col].values
        outcomes = s[outcome_col].values
        valid    = ~pd.isna(angles)
        is_wu    = np.isin(angles, list(warmup_angles)) & valid
        n = len(angles)
        if n < window_size:
            continue

        density, acc_wu, pos = [], [], []
        for i in range(n - window_size + 1):
            wv = valid[i:i+window_size]
            wu = is_wu[i:i+window_size]
            if wv.sum() == 0:
                continue
            density.append(wu.sum() / wv.sum())
            wu_hit = wu & ~pd.isna(outcomes[i:i+window_size])
            acc_wu.append(np.nanmean(outcomes[i:i+window_size][wu_hit])
                          if wu_hit.sum() >= 3 else np.nan)
            pos.append(i + window_size // 2)

        density = np.array(density)
        pos     = np.array(pos)

        # robust mid-third baseline
        lo_b, hi_b = n // 3, 2 * n // 3
        mid_v  = valid[lo_b:hi_b]
        mid_wu = is_wu[lo_b:hi_b]
        base_density = mid_wu.sum() / max(1, mid_v.sum())
        mid_wu_hit = mid_wu & ~pd.isna(outcomes[lo_b:hi_b])
        base_acc = (np.nanmean(outcomes[lo_b:hi_b][mid_wu_hit])
                    if mid_wu_hit.sum() >= 5 else np.nan)

        # proposed cut: first window position where density
        # falls below threshold * baseline
        cut_thresh = max(0.15, proposed_ratio_threshold * base_density)
        below = np.where(density <= cut_thresh)[0]
        cut_pos = pos[below[0]] if len(below) else None

        axes[0, col].plot(pos, density, color='steelblue')
        axes[0, col].axhline(base_density, ls='--', color='k', alpha=0.6,
                              label=f'mid={base_density:.2f}')
        axes[0, col].axhline(cut_thresh, ls=':', color='orange', alpha=0.7,
                              label=f'thresh={cut_thresh:.2f}')
        if cut_pos is not None:
            axes[0, col].axvline(cut_pos, color='crimson', lw=2, alpha=0.7)
            axes[1, col].axvline(cut_pos, color='crimson', lw=2, alpha=0.7)
        axes[0, col].set_title(f'sess {sess} (n={n})\ncut@{cut_pos}', fontsize=9)
        axes[0, col].set_ylim(0, 1)
        axes[0, col].legend(fontsize=7)

        axes[1, col].plot(pos, acc_wu, color='crimson')
        if not np.isnan(base_acc):
            axes[1, col].axhline(base_acc, ls='--', color='k', alpha=0.6,
                                  label=f'mid={base_acc:.2f}')
            axes[1, col].legend(fontsize=7)
        axes[1, col].set_ylim(0, 1.05)
        axes[1, col].set_xlabel('trial position')

    axes[0, 0].set_ylabel('running frac(0° or 90°)')
    axes[1, 0].set_ylabel('running accuracy\non 0°/90° only')
    fig.suptitle(f'Rat {rat} — modality={modality} — '
                 f'sessions {sessions[0]}–{sessions[-1]} '
                 f'(window={window_size}, cut = {proposed_ratio_threshold}× baseline)',
                 fontsize=11)
    plt.tight_layout()
    plt.show()


    # what modalities does rat 12 have, and at which session_idx?
#r12 = df[df['rat'] == 12]
#print(r12.groupby('mod')['date'].agg(['min', 'max', 'nunique']))

def diagnose_session_warmup_v3(df, rat, modality=None,
                                session_quantile=(0.5, 1.0),
                                n_sessions=6,
                                warmup_angles=(0, 90),
                                window_size=20,
                                proposed_ratio_threshold=2.0,
                                min_baseline_floor=0.15,
                                angle_col='angle',
                                trial_id_col='trialID',
                                outcome_col='hitmiss',
                                modality_col='mod',
                                session_col='date'):
    """
    Rank-based session selection: picks n_sessions evenly from within
    the given quantile range of the rat's (optionally modality-filtered)
    sessions. Robust to any session_idx encoding.

    session_quantile=(0.5, 1.0) → sample from the later half.
    session_quantile=(0.0, 0.3) → sample from the early sessions.
    """
    rat_df = df[df['rat'] == rat].copy()
    if modality is not None:
        available = rat_df[modality_col].dropna().unique()
        if modality not in available:
            matches = [m for m in available if str(m) == str(modality)]
            if not matches:
                print(f"Rat {rat}: modality {modality!r} not found. "
                      f"Available: {sorted(available.tolist())}")
                return
            modality = matches[0]
        rat_df = rat_df[rat_df[modality_col] == modality]

    sess_all = sorted(rat_df[session_col].dropna().unique())
    if not sess_all:
        print(f"Rat {rat}: no sessions after filtering.")
        return

    lo_q, hi_q = session_quantile
    i_lo = int(np.floor(lo_q * (len(sess_all) - 1)))
    i_hi = int(np.ceil (hi_q * (len(sess_all) - 1)))
    pool = sess_all[i_lo : i_hi + 1]
    if len(pool) > n_sessions:
        idx = np.linspace(0, len(pool) - 1, n_sessions).astype(int)
        sessions = [pool[i] for i in idx]
    else:
        sessions = pool

    print(f"Rat {rat} | mod={modality}: {len(sess_all)} sessions total, "
          f"sampling {len(sessions)} from quantile {session_quantile}")

    fig, axes = plt.subplots(2, len(sessions),
                              figsize=(3.2 * len(sessions), 6),
                              sharey='row', squeeze=False)

    for col, sess in enumerate(sessions):
        s = (rat_df[rat_df[session_col] == sess]
             .sort_values(trial_id_col).reset_index(drop=True))
        angles   = s[angle_col].values
        outcomes = s[outcome_col].values
        valid    = ~pd.isna(angles)
        is_wu    = np.isin(angles, list(warmup_angles)) & valid
        n = len(angles)
        if n < window_size:
            axes[0, col].set_title(f'sess {sess}\n(n={n}, skip)', fontsize=8)
            continue

        density, acc_wu, pos = [], [], []
        for i in range(n - window_size + 1):
            wv = valid[i:i+window_size]
            wu = is_wu[i:i+window_size]
            if wv.sum() == 0:
                continue
            density.append(wu.sum() / wv.sum())
            wu_hit = wu & ~pd.isna(outcomes[i:i+window_size])
            acc_wu.append(np.nanmean(outcomes[i:i+window_size][wu_hit])
                          if wu_hit.sum() >= 3 else np.nan)
            pos.append(i + window_size // 2)

        density = np.array(density); pos = np.array(pos)
        lo_b, hi_b = n // 3, 2 * n // 3
        mid_v, mid_wu = valid[lo_b:hi_b], is_wu[lo_b:hi_b]
        base_density = mid_wu.sum() / max(1, mid_v.sum())
        mid_wu_hit = mid_wu & ~pd.isna(outcomes[lo_b:hi_b])
        base_acc = (np.nanmean(outcomes[lo_b:hi_b][mid_wu_hit])
                    if mid_wu_hit.sum() >= 5 else np.nan)

        cut_thresh = max(min_baseline_floor,
                         proposed_ratio_threshold * base_density)
        below = np.where(density <= cut_thresh)[0]
        cut_pos = int(pos[below[0]]) if len(below) else None

        axes[0, col].plot(pos, density, color='steelblue')
        axes[0, col].axhline(base_density, ls='--', color='k', alpha=0.6,
                              label=f'mid={base_density:.2f}')
        axes[0, col].axhline(cut_thresh, ls=':', color='orange', alpha=0.8,
                              label=f'cut≤{cut_thresh:.2f}')
        if cut_pos is not None:
            axes[0, col].axvline(cut_pos, color='crimson', lw=2, alpha=0.7)
            axes[1, col].axvline(cut_pos, color='crimson', lw=2, alpha=0.7)
        axes[0, col].set_title(f'sess {sess}\n(n={n}, cut@{cut_pos})',
                                fontsize=8)
        axes[0, col].set_ylim(0, 1); axes[0, col].legend(fontsize=7)

        axes[1, col].plot(pos, acc_wu, color='crimson')
        if not np.isnan(base_acc):
            axes[1, col].axhline(base_acc, ls='--', color='k', alpha=0.6,
                                  label=f'mid={base_acc:.2f}')
            axes[1, col].legend(fontsize=7)
        axes[1, col].set_ylim(0, 1.05)
        axes[1, col].set_xlabel('trial position')

    axes[0, 0].set_ylabel('running frac(0° or 90°)')
    axes[1, 0].set_ylabel('running accuracy\non 0°/90° only')
    fig.suptitle(f'Rat {rat} — mod={modality} — '
                 f'quantile {session_quantile} of {len(sess_all)} sessions '
                 f'(cut = {proposed_ratio_threshold}× baseline, floor '
                 f'{min_baseline_floor})', fontsize=10)
    plt.tight_layout()
    plt.show()


# Late VT sessions (post-training) for the problem rat
diagnose_session_warmup_v3(df_filtered, rat=12, modality=3,
                            session_quantile=(0.4, 1.0), n_sessions=6)

# Compare against a "good" rat's late VT sessions
diagnose_session_warmup_v3(df_filtered, rat=2, modality=3,
                            session_quantile=(0.4, 1.0), n_sessions=6)