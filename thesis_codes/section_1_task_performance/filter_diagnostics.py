"""
filter_diagnostics.py

Diagnostics for `filter_training_trials`: what does the filter actually
remove, and does it look like it's doing its job?

It gives you:

  1. `filter_training_trials_verbose()` -- runs the exact same two-stage
     filter (warm-up removal, then fixed/criterion session cutoff) as your
     original `filter_training_trials`, but instead of throwing the removed
     trials away it returns them too, tagged with *why* they were removed:
     'warmup' or 'learning_phase'.

  2. `summarize_removed_trials()` / `print_removed_summary()` -- for each
     removal reason: how many trials, which stimulus (angle) they came
     from, reward rate overall / per-angle, and the sensory-modality
     breakdown.

  3. `plot_empirical_psychometric()` -- a *non-parametric* psychometric
     curve (raw proportions + Wilson confidence intervals, no sigmoid
     fitting) comparing kept trials vs. warmup-removed vs.
     learning-phase-removed trials. If the filter is doing its job, the
     removed groups should look flat / low-accuracy relative to the kept
     data. Can be split by modality.

  4. `plot_modality_breakdown()` -- stacked bar of modality composition
     per removal reason.

--------------------------------------------------------------------------
BEFORE RUNNING: edit the CONFIG block below to match your dataframe's
actual column names. Everything else is parameterised, so you shouldn't
need to touch the function bodies.
--------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 0. CONFIG -- edit these to match your dataframe
# ---------------------------------------------------------------------------

# Column holding sensory modality. Confirmed from processed_behavior_data.csv:
# 'mod', coded 1/2/3. MODALITY_CODE_MAP below turns those into readable labels
# for the summaries/plots (mapping taken from Section_3_choice_reward_joint_bootstrap.py's
# MODALITY_CODES). Set MODALITY_COL to None to skip modality breakdowns.
MODALITY_COL = 'mod'
MODALITY_CODE_MAP = {1: 'tactile', 2: 'visual', 3: 'visuotactile'}

# Confirmed: processed_behavior_data.csv already has a raw choice column,
# 'action', coded 0/1 -- so the psychometric uses it directly instead of the
# angle+hitmiss inference fallback (CHOICE_COL = None would trigger that).
CHOICE_COL = 'action'
CHOICE_MAP = None          # e.g. {'L': 0, 'R': 1} if CHOICE_COL is categorical instead
BOUNDARY_ANGLE = 45        # category boundary in degrees, used only if CHOICE_COL is None

# If reward isn't simply "hit == rewarded", set this to the real reward
# column (e.g. a probabilistic-reward flag). Leave None to use outcome_col
# (hitmiss) as the reward indicator.
REWARD_COL = None

# Minimum trials an angle bin needs before it's shown on the empirical
# psychometric plot. Real multi-rat datasets often have a handful of
# idiosyncratic one-off angle values (e.g. a rat-specific calibration probe
# hit once or twice) alongside the shared stimulus grid; those near-empty
# bins are exactly 0 or 1 and make the plot look wrongly noisy if not
# dropped. Matches the spirit of MIN_TRIALS_PER_ANGLE in the fitting script.
MIN_TRIALS_PER_ANGLE_BIN = 5


# ---------------------------------------------------------------------------
# 1. Task-start detection -- unchanged from your original code
# ---------------------------------------------------------------------------

def get_task_start_trial(session_df, warmup_angles=(0, 90),
                          window_size=10, max_warmup_fraction=0.30,
                          angle_col='angle', trial_id_col='trialID'):
    """
    Return the trialID at which the real task phase begins.

    Slides a forward-looking window of `window_size` trials over the
    session; the task phase begins at the first position where the
    fraction of warmup-angle trials in the window drops to
    <= max_warmup_fraction. Returns the trialID of the first
    non-warmup trial at/after that position (so the kept trial itself
    is a real task trial).
    """
    s = session_df.sort_values(trial_id_col).reset_index(drop=True)
    angles = s[angle_col].values
    trial_ids = s[trial_id_col].values

    valid = ~pd.isna(angles)
    is_warmup = np.isin(angles, list(warmup_angles)) & valid

    n = len(angles)
    if n == 0:
        return None
    w = min(window_size, n)

    for i in range(n - w + 1):
        wv = valid[i:i + w]
        if wv.sum() == 0:
            continue
        frac_warmup = is_warmup[i:i + w].sum() / wv.sum()
        if frac_warmup <= max_warmup_fraction:
            for j in range(i, n):
                if valid[j] and not is_warmup[j]:
                    return trial_ids[j]
            return None
    return None


def _matlab_datenum_to_datetime(datenum):
    return pd.to_datetime(datenum - 719529, unit='D')


# ---------------------------------------------------------------------------
# 2. The filter, instrumented to keep what it removes
# ---------------------------------------------------------------------------

def filter_training_trials_verbose(
    df,
    method='criterion',        # 'criterion' or 'fixed'
    n_training=5,               # used if method='fixed'
    criterion=0.80,              # accuracy threshold
    min_consec=4,               # consecutive sessions above threshold
    outcome_col='hitmiss',
    angle_col='angle',
    trial_id_col='trialID',
    warmup_angles=(0, 90),
    verbose=True
):
    """
    Same two-stage logic as filter_training_trials (warm-up removal, then
    fixed/criterion session cutoff), but returns BOTH the kept trials and
    the removed trials, with removed trials tagged by removal reason.

    Returns
    -------
    df_kept    : dataframe of trials that survive both filtering stages
                 (same as filter_training_trials()'s return value).
    df_removed : dataframe of every trial dropped at either stage, with an
                 added 'removed_reason' column in {'warmup', 'learning_phase'}.
    """
    df = df.copy()
    n_start = len(df)

    df['date_real'] = df['date'].apply(_matlab_datenum_to_datetime)
    df = df.sort_values(['rat', 'date_real']).reset_index(drop=True)
    df['session_idx'] = df.groupby('rat')['date_real'].transform(lambda x: pd.factorize(x)[0])

    # ---- stage 1: warm-up removal -----------------------------------------
    first_real = (
        df.groupby(['rat', 'session_idx'])
        .apply(get_task_start_trial, warmup_angles=warmup_angles, window_size=10,
               max_warmup_fraction=0.30, angle_col=angle_col, trial_id_col=trial_id_col)
        .rename('first_real_trial')
        .reset_index()
    )
    df = df.merge(first_real, on=['rat', 'session_idx'])

    is_kept_stage1 = df['first_real_trial'].notna() & (df[trial_id_col] >= df['first_real_trial'])
    df_after_warmup = df[is_kept_stage1].copy()
    df_removed_warmup = df[~is_kept_stage1].copy()
    df_removed_warmup['removed_reason'] = 'warmup'
    df_removed_warmup['removed_stage'] = 1

    # nullify lag columns at the session boundary, same as the original
    boundary = df_after_warmup[trial_id_col] == df_after_warmup['first_real_trial']
    lag_cols = [c for c in (f'{angle_col}_n-1', f'{outcome_col}_n-1') if c in df_after_warmup.columns]
    if lag_cols:
        df_after_warmup.loc[boundary, lag_cols] = np.nan

    n_after_warmup = len(df_after_warmup)

    # ---- stage 2: session-level filtering ---------------------------------
    criterion_sessions = None

    if method == 'fixed':
        is_kept_stage2 = df_after_warmup['session_idx'] >= n_training
        df_out = df_after_warmup[is_kept_stage2].copy()
        df_removed_learning = df_after_warmup[~is_kept_stage2].copy()

    elif method == 'criterion':
        # NOTE: pandas >= 2.2 (and 3.x) drops the grouping column from the
        # sub-dataframe inside groupby(...).apply(...) by default. Your
        # original get_criterion_session() reads rat_df['rat'], which
        # KeyErrors under those pandas versions -- so it's looped manually
        # here instead, with rat_id passed in explicitly.
        def get_criterion_session(rat_df, rat_id):
            easy = rat_df[(rat_df[angle_col] <= 15) | (rat_df[angle_col] >= 75)]
            easy_counts = easy.groupby('session_idx').size()
            low_count_sessions = easy_counts[easy_counts < 50]
            if verbose:
                if low_count_sessions.empty:
                    print(f"Rat {rat_id}: all sessions have at least 50 easy trials.")
                else:
                    print(f"Rat {rat_id}: sessions with fewer than 50 easy trials:\n{low_count_sessions}")
            acc = easy.groupby('session_idx')[outcome_col].mean()
            consec = 0
            for sess, val in acc.items():
                if val >= criterion:
                    consec += 1
                    if consec >= min_consec:
                        return sess - (min_consec - 1)
                else:
                    consec = 0
            return None

        criterion_sessions = pd.DataFrame([
            {'rat': rat_id, 'criterion_session': get_criterion_session(rat_df, rat_id)}
            for rat_id, rat_df in df_after_warmup.groupby('rat')
        ])
        df_after_warmup = df_after_warmup.merge(criterion_sessions, on='rat')
        is_kept_stage2 = df_after_warmup['criterion_session'].notna() & \
                         (df_after_warmup['session_idx'] >= df_after_warmup['criterion_session'])
        df_out = df_after_warmup[is_kept_stage2].copy()
        df_removed_learning = df_after_warmup[~is_kept_stage2].copy()

    else:
        raise ValueError(f"method must be 'fixed' or 'criterion', got '{method}'")

    df_removed_learning = df_removed_learning.copy()
    df_removed_learning['removed_reason'] = 'learning_phase'
    df_removed_learning['removed_stage'] = 2

    common_cols = df_removed_warmup.columns.union(df_removed_learning.columns)
    df_removed = pd.concat(
        [df_removed_warmup.reindex(columns=common_cols),
         df_removed_learning.reindex(columns=common_cols)],
        ignore_index=True
    )

    if verbose:
        print(f"\n--- filter_training_trials_verbose (method='{method}') ---")
        print(f"Trials at start:              {n_start:>8}")
        print(f"Removed as warmup:            {len(df_removed_warmup):>8}")
        print(f"After warm-up removal:        {n_after_warmup:>8}")
        print(f"Removed as learning phase:    {len(df_removed_learning):>8}")
        print(f"After session filtering:      {len(df_out):>8}")
        print(f"Total removed:                {n_start - len(df_out):>8}  (check: {len(df_removed)})")
        if criterion_sessions is not None:
            print(f"\nCriterion sessions per rat:\n{criterion_sessions.to_string(index=False)}")

    return df_out, df_removed


# ---------------------------------------------------------------------------
# 3. Summary statistics on the removed trials
# ---------------------------------------------------------------------------

def _modality_labels(series, code_map=MODALITY_CODE_MAP):
    """Map numeric modality codes (e.g. 1/2/3) to readable labels, if a map is given."""
    if code_map:
        return series.map(code_map).fillna(series.astype(str))
    return series


def summarize_removed_trials(df_removed, angle_col='angle', outcome_col='hitmiss',
                              modality_col=MODALITY_COL, reward_col=REWARD_COL,
                              modality_code_map=MODALITY_CODE_MAP):
    """
    Per removal reason ('warmup', 'learning_phase'): trial/rat/session
    counts, stimulus (angle) distribution, reward rate overall and per
    angle, and (if modality_col is available) modality counts/fractions
    and reward rate per modality.
    """
    reward_col = reward_col or outcome_col
    summary = {}
    for reason, sub in df_removed.groupby('removed_reason'):
        stats = {
            'n_trials': len(sub),
            'n_rats': sub['rat'].nunique(),
            'n_sessions': sub.groupby('rat')['session_idx'].nunique().sum(),
            'angle_counts': sub[angle_col].value_counts().sort_index(),
            'reward_rate_overall': sub[reward_col].mean(),
            'reward_rate_by_angle': sub.groupby(angle_col)[reward_col].mean(),
        }
        if modality_col and modality_col in sub.columns:
            labels = _modality_labels(sub[modality_col], modality_code_map)
            stats['modality_counts'] = labels.value_counts()
            stats['modality_fraction'] = labels.value_counts(normalize=True)
            stats['reward_rate_by_modality'] = sub.groupby(labels)[reward_col].mean()
        else:
            stats['modality_counts'] = None
        summary[reason] = stats
    return summary


def print_removed_summary(summary):
    for reason, stats in summary.items():
        print(f"\n=== Removed as: {reason} ===")
        print(f"  n_trials              : {stats['n_trials']}")
        print(f"  n_rats                : {stats['n_rats']}")
        print(f"  n_rat-sessions        : {stats['n_sessions']}")
        print(f"  reward rate (overall) : {stats['reward_rate_overall']:.3f}")
        print(f"  trials by stimulus angle:\n{stats['angle_counts'].to_string()}")
        print(f"  reward rate by angle:\n{stats['reward_rate_by_angle'].to_string()}")
        if stats['modality_counts'] is not None:
            print(f"  trials by modality:\n{stats['modality_counts'].to_string()}")
            print(f"  modality fraction:\n{stats['modality_fraction'].to_string()}")
            print(f"  reward rate by modality:\n{stats['reward_rate_by_modality'].to_string()}")


# ---------------------------------------------------------------------------
# 4. Empirical (non-parametric) psychometric curve
# ---------------------------------------------------------------------------

def _wilson_ci(k, n, z=1.96):
    """Wilson score interval for a binomial proportion (no fitting)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    half = (z * np.sqrt((p * (1 - p) + z ** 2 / (4 * n)) / n)) / denom
    return center - half, center + half


def infer_choice(df, angle_col='angle', outcome_col='hitmiss',
                  boundary=BOUNDARY_ANGLE, choice_col=CHOICE_COL, choice_map=CHOICE_MAP):
    """
    Binary choice per trial (1 = chose the 'high' category, angle > boundary;
    0 = chose the 'low' category).

    If choice_col is given and present in df, that column is used directly
    (mapped through choice_map if it isn't already 0/1).

    Otherwise choice is INFERRED from angle + hit/miss:
        correct_side = 1 if angle > boundary else 0
        choice       = correct_side        if outcome == 1 (hit)
                        1 - correct_side    if outcome == 0 (miss)

    This assumes: (a) outcome_col is coded hit=1 / miss=0, (b) every trial
    has one unambiguous correct side defined by `boundary`, and (c) trials
    exactly at the boundary (if any) aren't meaningful -- check these hold
    for your task before trusting the resulting curve.
    """
    if choice_col and choice_col in df.columns:
        vals = df[choice_col]
        if choice_map:
            vals = vals.map(choice_map)
        return vals.astype(float)

    correct_side = (df[angle_col] > boundary).astype(float)
    choice = np.where(df[outcome_col] == 1, correct_side, 1 - correct_side)
    return pd.Series(choice, index=df.index)


def empirical_psychometric(df, angle_col='angle', outcome_col='hitmiss',
                            boundary=BOUNDARY_ANGLE, choice_col=CHOICE_COL,
                            min_trials_per_bin=MIN_TRIALS_PER_ANGLE_BIN):
    """
    Raw (unfit) psychometric: for each unique stimulus angle, the empirical
    P(choice = high category), trial count, and a Wilson 95% CI. Angle bins
    with fewer than min_trials_per_bin trials are dropped (a real n=1 bin is
    always exactly 0 or 1 and just adds noise, not signal).
    """
    d = df.copy()
    d['_choice'] = infer_choice(d, angle_col, outcome_col, boundary, choice_col)
    grp = d.groupby(angle_col)['_choice'].agg(['mean', 'count']).rename(
        columns={'mean': 'p_choice_high', 'count': 'n'})
    grp = grp[grp['n'] >= min_trials_per_bin]
    ci = grp.apply(lambda r: _wilson_ci(r['p_choice_high'] * r['n'], r['n']), axis=1)
    grp['ci_lo'] = ci.apply(lambda x: x[0])
    grp['ci_hi'] = ci.apply(lambda x: x[1])
    return grp.reset_index()


def _plot_psych_panel(ax, groups, angle_col, outcome_col, boundary, choice_col,
                       min_trials_per_bin=MIN_TRIALS_PER_ANGLE_BIN):
    colors = {'kept': 'tab:green', 'removed: warmup': 'tab:orange',
              'removed: learning phase': 'tab:red'}
    for label, sub in groups.items():
        if len(sub) == 0:
            continue
        curve = empirical_psychometric(sub, angle_col, outcome_col, boundary, choice_col,
                                        min_trials_per_bin)
        curve = curve[curve['n'] > 0]
        yerr = np.vstack([curve['p_choice_high'] - curve['ci_lo'],
                           curve['ci_hi'] - curve['p_choice_high']])
        yerr = np.clip(yerr, 0, None)  # guard against floating-point noise at n small / p near 0 or 1
        ax.errorbar(curve[angle_col], curve['p_choice_high'], yerr=yerr,
                     marker='o', linestyle='-', capsize=3,
                     label=f"{label} (n={len(sub)})", color=colors.get(label))
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1)
    ax.axvline(boundary, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('stimulus angle')
    ax.set_ylabel('P(choice = high category)')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)


def plot_empirical_psychometric(df_kept, df_removed, angle_col='angle', outcome_col='hitmiss',
                                 boundary=BOUNDARY_ANGLE, choice_col=CHOICE_COL,
                                 modality_col=MODALITY_COL, split_by_modality=False):
    """
    Empirical (unfit) psychometric curves comparing kept trials vs.
    warmup-removed vs. learning-phase-removed trials -- raw proportions +
    Wilson CIs, no sigmoid fitting. If split_by_modality is True, draws one
    panel per sensory modality.
    """
    groups = {
        'kept': df_kept,
        'removed: warmup': df_removed[df_removed['removed_reason'] == 'warmup'],
        'removed: learning phase': df_removed[df_removed['removed_reason'] == 'learning_phase'],
    }

    if not split_by_modality or not modality_col or modality_col not in df_kept.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        _plot_psych_panel(ax, groups, angle_col, outcome_col, boundary, choice_col)
        ax.set_title('Empirical psychometric: kept vs. removed trials')
        fig.tight_layout()
        return fig

    kept_labels = _modality_labels(df_kept[modality_col])
    removed_labels = _modality_labels(df_removed[modality_col])
    modalities = sorted(set(kept_labels.dropna().unique()) | set(removed_labels.dropna().unique()))
    fig, axes = plt.subplots(1, len(modalities), figsize=(5 * len(modalities), 5), sharey=True)
    if len(modalities) == 1:
        axes = [axes]
    for ax, mod in zip(axes, modalities):
        sub_groups = {
            'kept': df_kept[kept_labels == mod],
            'removed: warmup': df_removed[(removed_labels == mod) & (df_removed['removed_reason'] == 'warmup')],
            'removed: learning phase': df_removed[(removed_labels == mod) & (df_removed['removed_reason'] == 'learning_phase')],
        }
        _plot_psych_panel(ax, sub_groups, angle_col, outcome_col, boundary, choice_col)
        ax.set_title(str(mod))
    fig.suptitle('Empirical psychometric by modality: kept vs. removed trials')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Modality composition of removed trials
# ---------------------------------------------------------------------------

def plot_modality_breakdown(df_removed, modality_col=MODALITY_COL):
    """Stacked bar: fraction of removed trials from each modality, per removal reason."""
    if not modality_col or modality_col not in df_removed.columns:
        raise ValueError(f"modality_col '{modality_col}' not found in df_removed; "
                          f"check the MODALITY_COL setting at the top of this file.")
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = _modality_labels(df_removed[modality_col])
    ct = pd.crosstab(df_removed['removed_reason'], labels)
    ct_frac = ct.div(ct.sum(axis=1), axis=0)
    ct_frac.plot(kind='bar', stacked=True, ax=ax)
    ax.set_ylabel('fraction of removed trials')
    ax.set_xlabel('removal reason')
    ax.set_title('Sensory modality composition of removed trials')
    ax.legend(title='modality', bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    return fig, ct


# ---------------------------------------------------------------------------
# 6. Example usage
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # df = pd.read_pickle('your_data.pkl')  # must have: rat, date, trialID,
    #                                        # angle, hitmiss (+ modality if used)
    #
    # df_kept, df_removed = filter_training_trials_verbose(
    #     df, method='criterion', criterion=0.80, min_consec=4
    # )
    #
    # summary = summarize_removed_trials(df_removed)
    # print_removed_summary(summary)
    #
    # fig1 = plot_empirical_psychometric(df_kept, df_removed, split_by_modality=True)
    # fig2, modality_table = plot_modality_breakdown(df_removed)
    # plt.show()
    pass