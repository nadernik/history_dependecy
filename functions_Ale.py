'functions _ Ale'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bin_by_unique_angles(df, name_new_col, n_groups, angle_col, exclude_angle= None, min_count=5):
    """
    Bin by *unique angle values* so each bin has (nearly) the same count of distinct angles.
    Rows with angle==exclude are removed first.

    label options:
      - 'range' : labels like '0–16°'
      - 'center': numeric labels = group midpoint (mean of angles in the group)
      - 'index' : integer bin index 0..n_groups-1
    """
    decision_boundary = 45
  # === COUNT OCCURRENCES OF EACH ANGLE ===
    counts = df[angle_col].value_counts()

    # === KEEP ONLY ANGLES THAT APPEAR AT LEAST `min_count` TIMES ===
    valid_angles = counts[counts >= min_count].index
    # === EXCLUDE 45° AND GET UNIQUE ANGLES ===
    unique_angles = np.sort(df.loc[(df[angle_col].isin(valid_angles)) & (df[angle_col] != decision_boundary), angle_col].unique())


    # === SPLIT UNIQUE ANGLES INTO ~EQUAL-SIZED GROUPS ===
    groups = np.array_split(unique_angles, n_groups)

    # === ASSIGN LABELS (RANGE LABELS, e.g., '0–16°') ===
    labels = [f"{g[0]:.0f}–{g[-1]:.0f}°" for g in groups]

    # === MAP EACH ANGLE VALUE TO ITS BIN LABEL ===
    angle_to_label = {a: labels[i] for i, g in enumerate(groups) for a in g}
    if exclude_angle is None:
        angle_to_label[decision_boundary] = '45°'

    # === CREATE NEW COLUMN WITH BIN LABEL ===
    df[name_new_col] = df[angle_col].map(angle_to_label)

    #print(df[[angle_col, name_new_col]].sample(15, random_state=42))
    
    return df, angle_to_label

def midpoint(interval):
    if hasattr(interval, 'mid'):
        return interval.mid  # for pd.Interval bins
    try:
        # for string labels like "0–15°"
        parts = [float(x.replace('°','')) for x in str(interval).replace('–','-').split('-')]
        return np.mean(parts)
    except:
        return np.nan


def color_bin(bin_means): 
        
        bins_sorted = sorted(bin_means, key=bin_means.get)
        blue_bins = [b for b in bins_sorted if bin_means[b] < 45]
        red_bins  = [b for b in bins_sorted if bin_means[b] > 45]
        green_bin = [b for b in bins_sorted if bin_means[b] == 45]

        blues = plt.cm.Blues(np.linspace(0.85, 0.25, len(blue_bins)))
        reds  = plt.cm.Reds(np.linspace(0.25, 0.85, len(red_bins)))

        color_by_bin = {b: c for b, c in zip(blue_bins, blues)}
        color_by_bin.update({b: c for b, c in zip(red_bins, reds)})
        green_color = None
        if len(green_bin) == 1:
            green_color = np.array([0.2, 0.8, 0.2, 1.0])

        return color_by_bin, green_color

def aggregate_data(dfc, BIN_COL, ANGLE_COL, RESP_COL, RAT_COL=None, MOD_TRANS_COL=None):
    grup_col = [BIN_COL, ANGLE_COL]
    if RAT_COL is not None:
        grup_col.insert(0, RAT_COL)
    if MOD_TRANS_COL is not None:
        grup_col.insert(0, MOD_TRANS_COL)

    agg = (dfc.groupby(grup_col)[RESP_COL]
            .agg(mean='mean', n='size')
            .reset_index())
    agg = agg[agg['n'] >= 5]  # or 10, depending on your data
    agg['se'] = np.sqrt(agg['mean'] * (1 - agg['mean']) / agg['n'])
    return agg

def flatten_model_results(model_results_dict):
    # this is a helper function to flatten the nested dict structure of model results into a single dataframe, 
    # with one row per rat and columns for each feature coefficient and Cross Validation scores
    rows=[]
    for rat_id,d in model_results_dict.items():
        fn=d["feature_names"]
        b=np.asarray(d["coefficients"]).ravel()
        print(f"Rat {rat_id} – features: {fn}, check function")
        cv=d.get("cross_validation_scores", {})
        row={"rat_id": rat_id, "score_mean": cv.get("score_mean", np.nan), "score_std": cv.get("score_std", np.nan)}
        row.update({f"beta_{n}": v for n,v in zip(fn,b)})
        rows.append(row)
    return pd.DataFrame(rows).set_index("rat_id").sort_index()

def plot_angle_diagnostics(df, title="Angle distribution", n_cols=4, save_csv=None):
    """
    Show unique angles per rat:
    - dot plot: x=angle, y=trial count, colored by in/out of [0,90]
    - console: per-rat summary + per-angle trial counts + balance + modality coverage
    - save_csv: if a filepath string is provided, saves per-angle stats to CSV
    """
    rats = sorted(df['rat'].unique())
    n_rows = int(np.ceil(len(rats) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 3),
                             facecolor="#1A1A2E")
    axes = axes.flatten()

    print(f"\n{'='*70}")
    print(f"  {title.upper()}")
    print(f"{'='*70}")

    # --- Global summary first ---
    total_trials   = len(df)
    total_in_range = np.sum((df['angle'] >= 0) & (df['angle'] <= 90))
    total_outside  = total_trials - total_in_range
    print(f"  Total trials   : {total_trials:>8}")
    print(f"  In [0, 90]     : {total_in_range:>8}  ({100*total_in_range/total_trials:.1f}%)")
    print(f"  Outside [0,90] : {total_outside:>8}  ({100*total_outside/total_trials:.1f}%)")
    if 'mod' in df.columns:
        print(f"  Modalities     : {sorted(df['mod'].dropna().unique())}")
    print(f"{'='*70}")

    # --- Accumulate rows for CSV ---
    csv_rows = []
    mod_vals = sorted(df['mod'].dropna().unique()) if 'mod' in df.columns else []

    for idx, rat in enumerate(rats):
        ax         = axes[idx]
        rat_df     = df[df['rat'] == rat]
        rat_angles = rat_df['angle'].dropna()
        unique_angles = np.sort(rat_angles.unique())

        counts     = np.array([np.sum(rat_angles == a) for a in unique_angles])
        n_unique   = len(unique_angles)
        n_in_range = int(np.sum((unique_angles >= 0) & (unique_angles <= 90)))
        n_outside  = int(np.sum((unique_angles < 0)  | (unique_angles > 90)))
        total_rat  = len(rat_angles)
        colors     = ['#5DCAA5' if 0 <= a <= 90 else '#E07B6A' for a in unique_angles]

        # --- Per-rat header ---
        print(f"\n  RAT {int(rat)}  |  {total_rat} trials  |  {n_unique} unique angles  "
              f"|  {n_in_range} in [0,90]  |  {n_outside} outside")
        print(f"  {'-'*66}")

        # --- Per-angle counts table ---
        row_w = 8
        in_range_angles = [(a, c) for a, c in zip(unique_angles, counts) if 0 <= a <= 90]
        outside_angles  = [(a, c) for a, c in zip(unique_angles, counts) if not (0 <= a <= 90)]

        print(f"  Angles in [0, 90]:")
        for i in range(0, len(in_range_angles), row_w):
            chunk = in_range_angles[i:i+row_w]
            angle_row = "  ".join(f"{a:5.1f}°" for a, _ in chunk)
            count_row = "  ".join(f"{c:6}"     for _, c in chunk)
            print(f"    angle: {angle_row}")
            print(f"    count: {count_row}")

        if outside_angles:
            print(f"  ⚠ Angles OUTSIDE [0, 90]:")
            for i in range(0, len(outside_angles), row_w):
                chunk = outside_angles[i:i+row_w]
                angle_row = "  ".join(f"{a:5.1f}°" for a, _ in chunk)
                count_row = "  ".join(f"{c:6}"     for _, c in chunk)
                print(f"    angle: {angle_row}")
                print(f"    count: {count_row}")

        # --- Balance check ---
        in_range_counts = np.array([c for a, c in zip(unique_angles, counts) if 0 <= a <= 90])
        cv = 0.0
        if len(in_range_counts) > 1:
            cv = in_range_counts.std() / in_range_counts.mean()
            balance_flag = "⚠ UNBALANCED" if cv > 0.5 else "OK"
            print(f"  Balance (CV of trial counts): {cv:.2f}  {balance_flag}")
            print(f"  Min trials at any angle: {in_range_counts.min()}  "
                  f"Max: {in_range_counts.max()}  "
                  f"Mean: {in_range_counts.mean():.0f}")

        # --- Modality coverage per angle ---
        if 'mod' in rat_df.columns:
            print(f"  Modality coverage per angle (angles in [0,90]):")
            mod_header = "    " + f"{'angle':>7}" + "".join(f"  mod{int(m):>1}" for m in mod_vals)
            print(mod_header)
            for a in [ang for ang in unique_angles if 0 <= ang <= 90]:
                angle_df   = rat_df[rat_df['angle'] == a]
                mod_counts = "".join(
                    f"  {int(np.sum(angle_df['mod'] == m)):>5}"
                    for m in mod_vals
                )
                print(f"    {a:>7.1f}°{mod_counts}")

        # --- Date coverage ---
        if 'date' in rat_df.columns:
            n_dates_total = rat_df['date'].nunique()
            angle_date_coverage = {
                a: rat_df[rat_df['angle'] == a]['date'].nunique()
                for a in unique_angles if 0 <= a <= 90
            }
            min_cov = min(angle_date_coverage.values())
            max_cov = max(angle_date_coverage.values())
            sparse_angles = [a for a, d in angle_date_coverage.items()
                             if d < n_dates_total * 0.2]
            print(f"  Date coverage: {n_dates_total} total sessions | "
                  f"angles appear in {min_cov}–{max_cov} sessions")
            if sparse_angles:
                print(f"  ⚠ Angles appearing in <20% of sessions: "
                      f"{[f'{a:.1f}' for a in sparse_angles]}")

        # --- Accumulate CSV rows (one row per unique angle per rat) ---
        for a, c in zip(unique_angles, counts):
            in_range = bool(0 <= a <= 90)
            row = {
                'rat'           : int(rat),
                'angle'         : round(float(a), 2),
                'trial_count'   : int(c),
                'in_range'      : in_range,
                'pct_of_rat'    : round(100 * c / total_rat, 3),
                'rat_total'     : total_rat,
                'rat_n_unique'  : n_unique,
                'rat_cv'        : round(cv, 4),
            }
            # per-modality counts for this angle
            for m in mod_vals:
                angle_df = rat_df[rat_df['angle'] == a]
                row[f'mod{int(m)}_count'] = int(np.sum(angle_df['mod'] == m))
            # date coverage for this angle
            if 'date' in rat_df.columns:
                row['n_sessions_with_angle'] = rat_df[rat_df['angle'] == a]['date'].nunique()
                row['n_sessions_total']      = rat_df['date'].nunique()
            csv_rows.append(row)

        # --- Plot ---
        ax.scatter(unique_angles, counts, c=colors, s=18, alpha=0.85, zorder=3)
        ax.axvline(90,  color='#E07B6A', linewidth=1, linestyle='--', alpha=0.7)
        ax.axvline(180, color='#F5B97F', linewidth=1, linestyle='--', alpha=0.7)
        ax.set_facecolor("#12122A")
        ax.set_title(f"Rat {int(rat)}  ({n_unique} unique, CV={cv:.2f})",
                     color="white", fontsize=9)
        ax.tick_params(colors="#AAAACC", labelsize=7)
        ax.set_xlabel("angle", color="#AAAACC", fontsize=7)
        ax.set_ylabel("n trials", color="#AAAACC", fontsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

        for a, c, col in zip(unique_angles, counts, colors):
            if col == '#E07B6A':
                ax.annotate(f"{a:.0f}", (a, c),
                            textcoords="offset points", xytext=(0, 5),
                            color='#E07B6A', fontsize=6, ha='center')

    for ax in axes[len(rats):]:
        ax.set_visible(False)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#5DCAA5',
               markersize=6, label='angle ∈ [0, 90]'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E07B6A',
               markersize=6, label='angle outside [0, 90]'),
    ]
    fig.legend(handles=legend_elements, loc='lower right',
               facecolor="#2A2A4A", labelcolor="white", fontsize=8,
               edgecolor="#555577")

    fig.suptitle(title, color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # --- Save CSV ---
    if save_csv is not None:
        csv_df = pd.DataFrame(csv_rows)
        csv_df = csv_df.sort_values(['rat', 'angle']).reset_index(drop=True)
        csv_df.to_csv(save_csv, index=False)
        print(f"  Angle diagnostics saved to: {save_csv}  ({len(csv_df)} rows)")

    print(f"\n{'='*70}\n")


def filter_training_trials(
    df,
    method='criterion',        # 'criterion' or 'fixed'
    n_training=5,              # used if method='fixed'
    criterion=0.80,            # accuracy threshold
    min_consec=4,              # consecutive sessions above threshold
    outcome_col='hitmiss',     # current trial outcome column
    angle_col='angle',
    trial_id_col='trialID',
    warmup_angles=(0, 90),
    verbose=True
):
    """
    Filters a trial-level dataframe by:
      1. Removing warm-up trials at the start of each session (initial 0/90 block)
      2. Removing early training sessions (fixed cutoff or performance criterion)

    Parameters
    ----------
    df          : must already have columns: rat, session_idx, trialID, angle, 
                  angle_n-1, hitmiss_n-1, and whatever outcome_col is
    method      : 'fixed' drops first n_training sessions per rat
                  'criterion' drops sessions before rat reaches stable accuracy
    verbose     : prints trial counts at each filtering step

    Returns
    -------
    Filtered dataframe with a 'first_real_trial' column added.

    To call it: 
    df_fixed = filter_training_trials(df, method='fixed', n_training=5)
    df_crit  = filter_training_trials(df, method='criterion', criterion=0.80, min_consec=4)
    """
    df = df.copy()
    n_start = len(df)

    # ── Step 1: Warm-up removal ───────────────────────────────────────────────
    warmup_set = set(warmup_angles)

    def get_first_real_trial(session_df):
        session_sorted = session_df.sort_values(trial_id_col)
        for _, row in session_sorted.iterrows():
            if row[angle_col] not in warmup_set:
                return row[trial_id_col]
        return None

    def matlab_datenum_to_datetime(datenum):
        return pd.Timestamp('0000-01-01') + pd.to_timedelta(datenum - 1, unit='D')

    df['date_real'] = df['date'].apply(matlab_datenum_to_datetime)
    df = df.sort_values(['rat', 'date_real']).reset_index(drop=True)

    # Add session index per rat (0-indexed)
    df['session_idx'] = df.groupby('rat')['date_real'].transform(
        lambda x: pd.factorize(x)[0]
    )
    
    first_real = (
        df.groupby(['rat', 'session_idx'])
        .apply(get_first_real_trial)
        .rename('first_real_trial')
        .reset_index()
    )

    df = df.merge(first_real, on=['rat', 'session_idx'])
    df = df[
        df['first_real_trial'].notna() &
        (df[trial_id_col] >= df['first_real_trial'])
    ].copy()

    # Nullify lag columns on the first real trial of each session
    boundary = df[trial_id_col] == df['first_real_trial']
    df.loc[boundary, [f'{angle_col}_n-1', f'{outcome_col}_n-1']] = np.nan

    n_after_warmup = len(df)

    # ── Step 2: Session filtering ─────────────────────────────────────────────
    if method == 'fixed':
        df_out = df[df['session_idx'] >= n_training].copy()

    elif method == 'criterion':
        def get_criterion_session(rat_df):
            trained = rat_df[rat_df[angle_col].isin(warmup_angles)]
            acc = trained.groupby('session_idx')[outcome_col].mean()
            consec = 0
            for sess, val in acc.items():
                if val >= criterion:
                    consec += 1
                    if consec >= min_consec:
                        return sess - (min_consec - 1)  # first session of run
                else:
                    consec = 0
            return None

        criterion_sessions = (
            df.groupby('rat')
            .apply(get_criterion_session)
            .rename('criterion_session')
            .reset_index()
        )
        df = df.merge(criterion_sessions, on='rat')
        df_out = df[
            df['criterion_session'].notna() &
            (df['session_idx'] >= df['criterion_session'])
        ].copy()

    else:
        raise ValueError(f"method must be 'fixed' or 'criterion', got '{method}'")

    n_after_sessions = len(df_out)

    # ── Verbose report ────────────────────────────────────────────────────────
    if verbose:
        print(f"\n--- filter_training_trials (method='{method}') ---")
        print(f"Trials at start:              {n_start:>8}")
        print(f"After warm-up removal:        {n_after_warmup:>8}  (-{n_start - n_after_warmup})")
        print(f"After session filtering:      {n_after_sessions:>8}  (-{n_after_warmup - n_after_sessions})")
        print(f"Total removed:                {n_start - n_after_sessions:>8}")
        print(f"\nTrials remaining per rat:")
        print(df_out.groupby('rat').size().to_string())
        if method == 'criterion':
            print(f"\nCriterion sessions per rat:")
            print(criterion_sessions.to_string(index=False))

    return df_out

'''
def plot_angle_diagnostics(df, title="Angle distribution", n_cols=4):
    """
    Show unique angles per rat as:
    - a dot plot (intuitive: each dot = one unique angle value)
    - a count table printed to console
    """
    rats = sorted(df['rat'].unique())
    n_rows = int(np.ceil(len(rats) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, 
                              figsize=(n_cols * 4, n_rows * 3), 
                              facecolor="#1A1A2E")
    axes = axes.flatten()
    
    print(f"\n{'='*60}")
    print(f"{title.upper()}")
    print(f"{'='*60}")
    print(f"{'Rat':<6} {'N unique angles':<18} {'Min':>6} {'Max':>6} {'In [0,90]':>10} {'Outside':>10}")
    print(f"{'-'*60}")
    
    for idx, rat in enumerate(rats):
        ax = axes[idx]
        rat_angles = df[df['rat'] == rat]['angle'].dropna()
        unique_angles = np.sort(rat_angles.unique())
        
        # Count stats
        n_unique = len(unique_angles)
        n_in_range = np.sum((unique_angles >= 0) & (unique_angles <= 90))
        n_outside  = np.sum((unique_angles < 0)  | (unique_angles > 90))
        
        print(f"{int(rat):<6} {n_unique:<18} {unique_angles.min():>6.1f} "
              f"{unique_angles.max():>6.1f} {n_in_range:>10} {n_outside:>10}")
        
        # Color each dot by whether it's inside [0,90] or not
        colors = ['#5DCAA5' if 0 <= a <= 90 else '#E07B6A' for a in unique_angles]
        
        # Dot plot: x = angle value, y = trial count at that angle
        counts = [np.sum(rat_angles == a) for a in unique_angles]
        
        ax.scatter(unique_angles, counts, c=colors, s=18, alpha=0.85, zorder=3)
        ax.axvline(90,  color='#E07B6A', linewidth=1, linestyle='--', alpha=0.7)
        ax.axvline(180, color='#F5B97F', linewidth=1, linestyle='--', alpha=0.7)
        
        ax.set_facecolor("#12122A")
        ax.set_title(f"Rat {int(rat)}  ({n_unique} unique)", 
                     color="white", fontsize=9)
        ax.tick_params(colors="#AAAACC", labelsize=7)
        ax.set_xlabel("angle", color="#AAAACC", fontsize=7)
        ax.set_ylabel("n trials", color="#AAAACC", fontsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")
        
        # Annotate outside-range dots with their value
        for a, c, col in zip(unique_angles, counts, colors):
            if col == '#E07B6A':  # outside [0,90]
                ax.annotate(f"{a:.0f}", (a, c), 
                            textcoords="offset points", xytext=(0, 5),
                            color='#E07B6A', fontsize=6, ha='center')
    
    # Hide unused axes
    for ax in axes[len(rats):]:
        ax.set_visible(False)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#5DCAA5', 
               markersize=6, label='angle ∈ [0, 90]'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E07B6A', 
               markersize=6, label='angle outside [0, 90]'),
    ]
    fig.legend(handles=legend_elements, loc='lower right', 
               facecolor="#2A2A4A", labelcolor="white", fontsize=8,
               edgecolor="#555577")
    
    fig.suptitle(title, color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
    print(f"{'='*60}\n")
'''