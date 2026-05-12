# this script plots psychometric curves of P(repeat) as a function of evidence for repetition, split by difficulty of the previous trial (difficulty_n-1).
# evidence for repetition is defined as the signed distance of the current angle from the boundary, 
# signed according to the previous choice (positive = evidence for repeating previous action, negative = evidence for switching).
# No distance among the curves implies no perceptual history effect, while a shift of the curves implies a bias (PSE shift) and a change in slope implies a change in sensitivity.
# this is the psychometric sibling of empirical_repeat_difficulty.py
# the analysis are also split by current sensory modality, to see if there are differences in the history effects between modalities.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from functions_Ale import filter_training_trials
from serial_dependence_analysis import SerialDependenceAnalyzer

# -----------------------------
# COLUMN NAMES
# -----------------------------
ANGLE_COL          = 'angle'
RAT_COL            = 'rat'
SESSION_COL        = 'date'
EVIDENCE_COL       = 'evidence_4_repetition'
REP_COL            = 'repeated_action'
STATE_COL          = 'hitmiss_n-1'      # 0/1
DIFFICULTY_NM1_COL = 'difficulty_n-1'
BOUNDARY           = 45
RESP_COL           = 'action'

# -----------------------------
# TUNABLE PARAMETERS
# -----------------------------
TARGET_RAT     = 12      # <-- change to run on a different rat
MIN_TRIALS     = 20     # minimum trials per difficulty level
MIN_SESSIONS   = 3      # minimum sessions for reliable session-bootstrap CIs
N_BOOT         = 500    # number of bootstrap resamples
X_GRID         = np.linspace(-45, 45, 200)   # smooth x-axis for plotting fitted curves

# Empirical dot scaling: marker area = BUBBLE_SCALE * n_trials_at_level
# Adjust BUBBLE_SCALE up/down to taste (default gives ~40–200 pt² range)
BUBBLE_SCALE   = 1

# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data()

df_processed = an.create_lagged_features()
df_processed  = filter_training_trials(df_processed, method='criterion', criterion=0.80, min_consec=2)
modalities    = df_processed["mod"].dropna().unique()

df_processed[REP_COL]      = (df_processed["action_n-1"] == df_processed[RESP_COL]).astype(int)
df_processed[EVIDENCE_COL] = (df_processed[ANGLE_COL] - BOUNDARY) * (
    2 * df_processed["action_n-1"] - 1
)

# -----------------------------
# KEY FILTERS
# -----------------------------
df_processed = df_processed[df_processed[STATE_COL] == 1].copy()
df_processed = df_processed[df_processed[DIFFICULTY_NM1_COL] != 0].copy()


# ============================================================
#  PSYCHOMETRIC FUNCTION  (4-parameter cumulative Gaussian)
#
#   F(x; mu, sigma, lambda, gamma) =
#       gamma + (1 - gamma - lambda) * Phi((x - mu) / sigma)
#
#   mu     : PSE  – point of subjective equality  (bias)
#   sigma  : slope parameter  (sensitivity; smaller = steeper)
#   lambda : lapse rate  (upper asymptote offset, 0–0.2)
#   gamma  : guess rate  (lower asymptote offset, 0–0.2)
# ============================================================

def psychometric(x, mu, sigma, lam, gamma):
    """4-parameter cumulative Gaussian psychometric function."""
    return gamma + (1.0 - gamma - lam) * norm.cdf(x, loc=mu, scale=sigma)


def fit_psychometric(x, y):
    """
    Fit the 4-parameter psychometric function to binary trial data.

    Parameters
    ----------
    x : array-like  – evidence values (one per trial)
    y : array-like  – binary responses (0/1, one per trial)

    Returns
    -------
    popt : array  [mu, sigma, lambda, gamma]  or None if fit fails
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    p0     = [0.0, 15.0, 0.05, 0.05]          # initial guess
    bounds = ([-45, 1e-3, 0.0, 0.0],           # lower bounds
              [ 45, 90.0, 0.2, 0.2])            # upper bounds

    try:
        popt, _ = curve_fit(psychometric, x, y, p0=p0, bounds=bounds,
                            maxfev=10_000)
        return popt
    except (RuntimeError, ValueError):
        return None


# -----------------------------
# EMPIRICAL POINTS PER STIMULUS LEVEL
# -----------------------------

N_EMP_POINTS = 5   # number of empirical dots per curve

def empirical_points(df, n_points=N_EMP_POINTS):
    """
    Compute P(repeat) and trial count at n_points representative evidence
    levels chosen from the actual stimulus values in df.

    Selection: compute P(repeat) and n at every unique evidence level,
    then pick n_points levels whose x-positions are evenly spaced across
    the quantiles of the observed evidence distribution — so dots are
    spread across the x-axis rather than clustered where data are dense.
    """
    grp      = df.groupby(EVIDENCE_COL)[REP_COL]
    p_rep    = grp.mean()
    n_trials = grp.count()

    all_x = p_rep.index.to_numpy()
    all_p = p_rep.to_numpy()
    all_n = n_trials.to_numpy()

    if len(all_x) <= n_points:
        return all_x, all_p, all_n

    # pick indices whose x values sit at evenly-spaced quantiles
    quantiles = np.linspace(0, 1, n_points)
    targets   = np.quantile(all_x, quantiles)
    idx       = [np.argmin(np.abs(all_x - t)) for t in targets]
    idx       = sorted(set(idx))          # deduplicate, preserve order

    return all_x[idx], all_p[idx], all_n[idx]


# -----------------------------
# SESSION BOOTSTRAP FOR CIs
# -----------------------------

def bootstrap_psychometric(df, n_boot=N_BOOT, seed=0):
    """
    Session bootstrap: resample sessions with replacement, refit psychometric
    each time.  Returns point-estimate params + per-x CI band.

    Returns
    -------
    popt        : best-fit parameters on full data  [mu, sigma, lam, gamma]
    y_low/high  : 95 % CI band evaluated on X_GRID  (None, None if < MIN_SESSIONS)
    ci_valid    : bool – True when enough sessions for reliable CIs
    """
    x = df[EVIDENCE_COL].values
    y = df[REP_COL].values

    popt = fit_psychometric(x, y)
    if popt is None:
        return None, None, None, False

    sessions  = df[SESSION_COL].dropna().unique()
    ci_valid  = len(sessions) >= MIN_SESSIONS

    if not ci_valid:
        print(f"      ⚠️  Only {len(sessions)} session(s) — plotting curve without CIs.")
        return popt, None, None, False

    rng    = np.random.default_rng(seed)
    curves = []

    for _ in range(n_boot):
        sampled = rng.choice(sessions, size=len(sessions), replace=True)
        boot    = pd.concat([df[df[SESSION_COL] == s] for s in sampled],
                            ignore_index=True)
        p = fit_psychometric(boot[EVIDENCE_COL].values, boot[REP_COL].values)
        if p is not None:
            curves.append(psychometric(X_GRID, *p))

    if len(curves) < 10:          # too few successful fits
        return popt, None, None, False

    curves = np.array(curves)
    y_low  = np.quantile(curves, 0.025, axis=0)
    y_high = np.quantile(curves, 0.975, axis=0)
    return popt, y_low, y_high, True


# -----------------------------
# DIFFICULTY BINNING  (same logic as original)
# -----------------------------

rat_df = df_processed[df_processed[RAT_COL] == TARGET_RAT].dropna(
    subset=[EVIDENCE_COL, REP_COL, SESSION_COL, DIFFICULTY_NM1_COL]
)
print(f"Rat {TARGET_RAT}: {len(rat_df)} trials (after filtering n-1 correct, excluding diff_n-1=45)")

counts = rat_df[DIFFICULTY_NM1_COL].value_counts().sort_index()
print(f"\nTrial counts per difficulty_n-1 level (rat {TARGET_RAT}):")
for level, n in counts.items():
    flag = "  <-- WARNING: too few trials, will be skipped" if n < MIN_TRIALS else ""
    print(f"  difficulty_n-1 = {level:>5}:  {n:>4} trials{flag}")

valid_levels = counts[counts >= MIN_TRIALS].index.tolist()
dropped      = counts[counts < MIN_TRIALS].index.tolist()
if dropped:
    print(f"\nDropping levels with < {MIN_TRIALS} trials: {dropped}")
rat_df = rat_df[rat_df[DIFFICULTY_NM1_COL].isin(valid_levels)]

difficulty_levels = sorted(valid_levels)
bin_size  = len(difficulty_levels) // 3
low_lev   = difficulty_levels[:bin_size]
mid_lev   = difficulty_levels[bin_size:2 * bin_size]
high_lev  = difficulty_levels[2 * bin_size:]

diff_bin_map = {lv: 3 for lv in low_lev}
diff_bin_map.update({lv: 2 for lv in mid_lev})
diff_bin_map.update({lv: 1 for lv in high_lev})

rat_df["difficulty_n-1_bin"] = rat_df[DIFFICULTY_NM1_COL].map(diff_bin_map)
difficulty_levels = sorted(rat_df["difficulty_n-1_bin"].dropna().unique())
print(f"\nDifficulty n-1 bins kept for plotting: {difficulty_levels}")

# colormap
cmap       = plt.cm.viridis
colors     = cmap(np.linspace(0.1, 0.9, len(difficulty_levels)))
diff_colors = {lv: colors[i] for i, lv in enumerate(difficulty_levels)}


# ============================================================
#  MAIN LOOP  –  one figure per modality,
#               one curve per difficulty bin
# ============================================================

all_params = []   # collect fitted parameters for summary table

for mod in modalities:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    print(f"\n{'='*55}")
    print(f"  Modality: {mod}  |  Rat {TARGET_RAT}")
    print(f"{'='*55}")

    sub_md = rat_df[rat_df['mod'] == mod]

    for diff_level in difficulty_levels:
        sub = sub_md[sub_md["difficulty_n-1_bin"] == diff_level].dropna(
            subset=[EVIDENCE_COL, REP_COL, SESSION_COL]
        )
        n_trials   = len(sub)
        n_sessions = sub[SESSION_COL].nunique()
        print(f"\n  diff_bin={diff_level}  |  {n_trials} trials, {n_sessions} sessions")

        if n_trials < MIN_TRIALS:
            print(f"      ⚠️  Skipping: fewer than {MIN_TRIALS} trials.")
            continue

        popt, y_low, y_high, ci_valid = bootstrap_psychometric(sub, seed=42 + int(diff_level))

        if popt is None:
            print(f"      ⚠️  Curve fit failed — skipping.")
            continue

        mu, sigma, lam, gamma = popt
        print(f"      PSE (mu)    = {mu:+.2f}°")
        print(f"      Slope (σ)  = {sigma:.2f}°")
        print(f"      Lapse (λ)  = {lam:.3f}")
        print(f"      Guess (γ)  = {gamma:.3f}")

        all_params.append({
            'rat': TARGET_RAT, 'modality': mod,
            'diff_bin': diff_level,
            'mu': mu, 'sigma': sigma, 'lambda': lam, 'gamma': gamma,
            'n_trials': n_trials, 'n_sessions': n_sessions,
            'ci_valid': ci_valid
        })

        c          = diff_colors[diff_level]
        y_fit      = psychometric(X_GRID, *popt)
        ci_label   = "" if ci_valid else " (no CI)"
        label      = f"diff_n-1 = {diff_level}{ci_label}"

        if ci_valid:
            ax.fill_between(X_GRID, y_low, y_high, alpha=0.15, linewidth=0, color=c)

        ax.plot(X_GRID, y_fit, linestyle='-', color=c, linewidth=2, label=label)

        # Empirical dots: one per actual stimulus level, size ∝ n trials
        x_pts, p_pts, n_pts = empirical_points(sub)
        ax.scatter(x_pts, p_pts,
                   s=n_pts * BUBBLE_SCALE,
                   color=c, alpha=0.6,
                   edgecolors='white', linewidths=0.5,
                   zorder=3)

        # PSE marker
        ax.axvline(mu, color=c, linestyle=':', linewidth=1.2, alpha=0.7)

    ax.axhline(0.5, color='k', ls='--', alpha=0.4)
    ax.axvline(0,   color='k', ls='--', alpha=0.4)
    ax.set_xlim(-45, 45)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Evidence for repetition")
    ax.set_ylabel("P(repeat)")
    ax.legend(title="difficulty n-1", fontsize=9, title_fontsize=9, loc='upper left')

    # Bubble size legend (shows what marker area corresponds to in trial counts)
    bubble_ns    = [20, 50, 100]
    bubble_handles = [
        plt.scatter([], [], s=n * BUBBLE_SCALE, color='grey', alpha=0.6,
                    edgecolors='white', linewidths=0.5, label=f'n={n}')
        for n in bubble_ns
    ]
    ax.legend(handles=bubble_handles, title="trials / level",
              fontsize=8, title_fontsize=8, loc='lower right',
              framealpha=0.7)

    # Restore main legend on top
    main_handles = [
        plt.Line2D([0], [0], color=diff_colors[lv], linewidth=2,
                   label=f"diff_n-1 = {lv}" + ("" if any(
                       p['diff_bin'] == lv and p['ci_valid']
                       for p in all_params if p['modality'] == mod
                   ) else " (no CI)"))
        for lv in difficulty_levels
        if any(p['diff_bin'] == lv and p['modality'] == mod for p in all_params)
    ]
    leg1 = ax.legend(handles=main_handles, title="difficulty n-1",
                     fontsize=9, title_fontsize=9, loc='upper left')
    ax.add_artist(leg1)
    ax.legend(handles=bubble_handles, title="trials / level",
              fontsize=8, title_fontsize=8, loc='lower right', framealpha=0.7)
    ax.set_title(
        f"Rat {TARGET_RAT}  |  Modality: {mod}\n"
        f"Psychometric fit — P(repeat) by difficulty n-1\n"
        f"(only trials where n-1 was correct; dashed verticals = PSE)"
    )
    plt.show()

# -----------------------------
# SUMMARY TABLE OF FITTED PARAMS
# -----------------------------
if all_params:
    df_params = pd.DataFrame(all_params)
    print("\n" + "="*65)
    print("  SUMMARY: Fitted psychometric parameters")
    print("="*65)
    print(df_params.to_string(index=False, float_format=lambda x: f"{x:.3f}"))




'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from functions_Ale import filter_training_trials
from serial_dependence_analysis import SerialDependenceAnalyzer

# -----------------------------
# COLUMN NAMES
# -----------------------------
ANGLE_COL          = 'angle'
RAT_COL            = 'rat'
SESSION_COL        = 'date'
EVIDENCE_COL       = 'evidence_4_repetition'
REP_COL            = 'repeated_action'
STATE_COL          = 'hitmiss_n-1'      # 0/1
DIFFICULTY_NM1_COL = 'difficulty_n-1'
BOUNDARY           = 45
RESP_COL           = 'action'

# -----------------------------
# TUNABLE PARAMETERS
# -----------------------------
TARGET_RAT     = 12      # <-- change to run on a different rat
MIN_TRIALS     = 20     # minimum trials per difficulty level
MIN_SESSIONS   = 3      # minimum sessions for reliable session-bootstrap CIs
N_BOOT         = 1    # number of bootstrap resamples
X_GRID         = np.linspace(-45, 45, 200)   # smooth x-axis for plotting fitted curves

# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data()

df_processed = an.create_lagged_features()
df_processed  = filter_training_trials(df_processed, method='criterion', criterion=0.80, min_consec=2)
modalities    = df_processed["mod"].dropna().unique()

df_processed[REP_COL]      = (df_processed["action_n-1"] == df_processed[RESP_COL]).astype(int)
df_processed[EVIDENCE_COL] = (df_processed[ANGLE_COL] - BOUNDARY) * (
    2 * df_processed["action_n-1"] - 1
)

# -----------------------------
# KEY FILTERS
# -----------------------------
df_processed = df_processed[df_processed[STATE_COL] == 1].copy()
df_processed = df_processed[df_processed[DIFFICULTY_NM1_COL] != 0].copy()


# ============================================================
#  PSYCHOMETRIC FUNCTION  (4-parameter cumulative Gaussian)
#
#   F(x; mu, sigma, lambda, gamma) =
#       gamma + (1 - gamma - lambda) * Phi((x - mu) / sigma)
#
#   mu     : PSE  – point of subjective equality  (bias)
#   sigma  : slope parameter  (sensitivity; smaller = steeper)
#   lambda : lapse rate  (upper asymptote offset, 0–0.2)
#   gamma  : guess rate  (lower asymptote offset, 0–0.2)
# ============================================================

def psychometric(x, mu, sigma, lam, gamma):
    """4-parameter cumulative Gaussian psychometric function."""
    return gamma + (1.0 - gamma - lam) * norm.cdf(x, loc=mu, scale=sigma)


def fit_psychometric(x, y):
    """
    Fit the 4-parameter psychometric function to binary trial data.

    Parameters
    ----------
    x : array-like  – evidence values (one per trial)
    y : array-like  – binary responses (0/1, one per trial)

    Returns
    -------
    popt : array  [mu, sigma, lambda, gamma]  or None if fit fails
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    p0     = [0.0, 15.0, 0.05, 0.05]          # initial guess
    bounds = ([-45, 1e-3, 0.0, 0.0],           # lower bounds
              [ 45, 90.0, 0.3, 0.3])            # upper bounds

    try:
        popt, _ = curve_fit(psychometric, x, y, p0=p0, bounds=bounds,
                            maxfev=10_000)
        return popt
    except (RuntimeError, ValueError):
        return None


# -----------------------------
# SESSION BOOTSTRAP FOR CIs
# -----------------------------

def bootstrap_psychometric(df, n_boot=N_BOOT, seed=0):
    """
    Session bootstrap: resample sessions with replacement, refit psychometric
    each time.  Returns point-estimate params + per-x CI band.

    Returns
    -------
    popt        : best-fit parameters on full data  [mu, sigma, lam, gamma]
    y_low/high  : 95 % CI band evaluated on X_GRID  (None, None if < MIN_SESSIONS)
    ci_valid    : bool – True when enough sessions for reliable CIs
    """
    x = df[EVIDENCE_COL].values
    y = df[REP_COL].values

    popt = fit_psychometric(x, y)
    if popt is None:
        return None, None, None, False

    sessions  = df[SESSION_COL].dropna().unique()
    ci_valid  = len(sessions) >= MIN_SESSIONS

    if not ci_valid:
        print(f"      ⚠️  Only {len(sessions)} session(s) — plotting curve without CIs.")
        return popt, None, None, False

    rng    = np.random.default_rng(seed)
    curves = []

    for _ in range(n_boot):
        sampled = rng.choice(sessions, size=len(sessions), replace=True)
        boot    = pd.concat([df[df[SESSION_COL] == s] for s in sampled],
                            ignore_index=True)
        p = fit_psychometric(boot[EVIDENCE_COL].values, boot[REP_COL].values)
        if p is not None:
            curves.append(psychometric(X_GRID, *p))

    if len(curves) < 10:          # too few successful fits
        return popt, None, None, False

    curves = np.array(curves)
    y_low  = np.quantile(curves, 0.025, axis=0)
    y_high = np.quantile(curves, 0.975, axis=0)
    return popt, y_low, y_high, True


# -----------------------------
# DIFFICULTY BINNING  (same logic as original)
# -----------------------------

rat_df = df_processed[df_processed[RAT_COL] == TARGET_RAT].dropna(
    subset=[EVIDENCE_COL, REP_COL, SESSION_COL, DIFFICULTY_NM1_COL]
)
print(f"Rat {TARGET_RAT}: {len(rat_df)} trials (after filtering n-1 correct, excluding diff_n-1=45)")

counts = rat_df[DIFFICULTY_NM1_COL].value_counts().sort_index()
print(f"\nTrial counts per difficulty_n-1 level (rat {TARGET_RAT}):")
for level, n in counts.items():
    flag = "  <-- WARNING: too few trials, will be skipped" if n < MIN_TRIALS else ""
    print(f"  difficulty_n-1 = {level:>5}:  {n:>4} trials{flag}")

valid_levels = counts[counts >= MIN_TRIALS].index.tolist()
dropped      = counts[counts < MIN_TRIALS].index.tolist()
if dropped:
    print(f"\nDropping levels with < {MIN_TRIALS} trials: {dropped}")
rat_df = rat_df[rat_df[DIFFICULTY_NM1_COL].isin(valid_levels)]

difficulty_levels = sorted(valid_levels)
bin_size  = len(difficulty_levels) // 3
low_lev   = difficulty_levels[:bin_size]
mid_lev   = difficulty_levels[bin_size:2 * bin_size]
high_lev  = difficulty_levels[2 * bin_size:]

diff_bin_map = {lv: 3 for lv in low_lev}
diff_bin_map.update({lv: 2 for lv in mid_lev})
diff_bin_map.update({lv: 1 for lv in high_lev})

rat_df["difficulty_n-1_bin"] = rat_df[DIFFICULTY_NM1_COL].map(diff_bin_map)
difficulty_levels = sorted(rat_df["difficulty_n-1_bin"].dropna().unique())
print(f"\nDifficulty n-1 bins kept for plotting: {difficulty_levels}")

# colormap
cmap       = plt.cm.viridis
colors     = cmap(np.linspace(0.1, 0.9, len(difficulty_levels)))
diff_colors = {lv: colors[i] for i, lv in enumerate(difficulty_levels)}


# ============================================================
#  MAIN LOOP  –  one figure per modality,
#               one curve per difficulty bin
# ============================================================

all_params = []   # collect fitted parameters for summary table

for mod in modalities:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    print(f"\n{'='*55}")
    print(f"  Modality: {mod}  |  Rat {TARGET_RAT}")
    print(f"{'='*55}")

    sub_md = rat_df[rat_df['mod'] == mod]

    for diff_level in difficulty_levels:
        sub = sub_md[sub_md["difficulty_n-1_bin"] == diff_level].dropna(
            subset=[EVIDENCE_COL, REP_COL, SESSION_COL]
        )
        n_trials   = len(sub)
        n_sessions = sub[SESSION_COL].nunique()
        print(f"\n  diff_bin={diff_level}  |  {n_trials} trials, {n_sessions} sessions")

        if n_trials < MIN_TRIALS:
            print(f"      ⚠️  Skipping: fewer than {MIN_TRIALS} trials.")
            continue

        popt, y_low, y_high, ci_valid = bootstrap_psychometric(sub, seed=42 + int(diff_level))

        if popt is None:
            print(f"      ⚠️  Curve fit failed — skipping.")
            continue

        mu, sigma, lam, gamma = popt
        print(f"      PSE (mu)    = {mu:+.2f}°")
        print(f"      Slope (σ)  = {sigma:.2f}°")
        print(f"      Lapse (λ)  = {lam:.3f}")
        print(f"      Guess (γ)  = {gamma:.3f}")

        all_params.append({
            'rat': TARGET_RAT, 'modality': mod,
            'diff_bin': diff_level,
            'mu': mu, 'sigma': sigma, 'lambda': lam, 'gamma': gamma,
            'n_trials': n_trials, 'n_sessions': n_sessions,
            'ci_valid': ci_valid
        })

        c          = diff_colors[diff_level]
        y_fit      = psychometric(X_GRID, *popt)
        ci_label   = "" if ci_valid else " (no CI)"
        label      = f"diff_n-1 = {diff_level}{ci_label}"

        if ci_valid:
            ax.fill_between(X_GRID, y_low, y_high, alpha=0.15, linewidth=0, color=c)

        ax.plot(X_GRID, y_fit, linestyle='-', color=c, linewidth=2, label=label)

        # PSE marker
        ax.axvline(mu, color=c, linestyle=':', linewidth=1.2, alpha=0.7)

    ax.axhline(0.5, color='k', ls='--', alpha=0.4)
    ax.axvline(0,   color='k', ls='--', alpha=0.4)
    ax.set_xlim(-45, 45)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Evidence for repetition")
    ax.set_ylabel("P(repeat)")
    ax.legend(title="difficulty n-1", fontsize=9, title_fontsize=9, loc='upper left')
    ax.set_title(
        f"Rat {TARGET_RAT}  |  Modality: {mod}\n"
        f"Psychometric fit — P(repeat) by difficulty n-1\n"
        f"(only trials where n-1 was correct; dashed verticals = PSE)"
    )
    plt.show()

# -----------------------------
# SUMMARY TABLE OF FITTED PARAMS
# -----------------------------
if all_params:
    df_params = pd.DataFrame(all_params)
    print("\n" + "="*65)
    print("  SUMMARY: Fitted psychometric parameters")
    print("="*65)
    print(df_params.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

'''