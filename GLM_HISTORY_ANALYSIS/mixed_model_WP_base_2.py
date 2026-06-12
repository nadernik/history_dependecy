# ============================================================
# IMPORTS AND R ENVIRONMENT
# ============================================================
import os
os.environ['R_HOME']      = r'C:\Program Files\R\R-4.4.1'
os.environ['PATH']        = r'C:\Program Files\R\R-4.4.1\bin' + ';' + os.environ['PATH']
os.environ['R_LIBS_USER'] = r'C:\Users\alech\AppData\Local\R\win-library\4.4'

import sys, re, subprocess, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf

sys.path.append(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy")
from functions_Ale import filter_training_trials
from serial_dependence_analysis import SerialDependenceAnalyzer

# ★ NEW: path to Rscript executable (adjust if you move R)
RSCRIPT_EXE = r'C:\Program Files\R\R-4.4.1\bin\Rscript.exe'
print(subprocess.run([RSCRIPT_EXE, '--version'], capture_output=True, text=True).stderr)

# ============================================================
# CONFIG — column names (verify they match your data)
# ============================================================
RAT_COL     = 'rat'
SESSION_COL = 'date'       
MOD_COL     = 'mod'
MOD_N1_COL  = 'mod_n-1'
ACTION_COL  = 'action'
ANGLE_COL   = 'angle'
ANGLE_N1_COL = 'angle_n-1'
HIT_N1_COL  = 'hit_n-1'
ACTION_N1_COL = 'action_n-1'

HISTORY_DEPTH  = 1
RANDOM_SEED    = 42
PSYCH_FRACTION = 1/3            # fraction of SESSIONS held out for psychometric fits


# ============================================================
# 1. LOAD AND PREPROCESS  
# ============================================================
an = SerialDependenceAnalyzer(history_depth=HISTORY_DEPTH, save_figures=False)
an.load_and_preprocess_data()
df = an.create_lagged_features()
df = filter_training_trials(df, method='criterion', criterion=0.80, min_consec=2) # filter out training trials based on performance criterion (adjust as needed)

print(f"Loaded {len(df)} trials, {df[RAT_COL].nunique()} rats")

# ============================================================
# 2. ★ NEW: SESSION-LEVEL SPLIT (psychometric vs model set)
# ─────────────────────────────────────────────────────────────
# Why split by WHOLE SESSIONS and not by trials:
#   Trials within a session share state (motivation, fatigue,
#   apparatus drift). A trial-level split would let session
#   state leak across the two sets, so σ and the history
#   betas would still be dependent. Whole-session assignment
#   makes them independent at the session level.
# Why stratify by rat:
#   Each rat must contribute to both sets, otherwise we lose rats.
# ============================================================
def split_sessions(df, psych_fraction=1/3, seed=RANDOM_SEED):
    df = df.copy()
    rng = np.random.default_rng(seed)
    df['split'] = ''
    for rat_id, rat_df in df.groupby(RAT_COL):
        sessions = rat_df[SESSION_COL].unique().copy()
        rng.shuffle(sessions)
        n_psych = max(1, int(round(len(sessions) * psych_fraction)))
        psych_sessions = set(sessions[:n_psych])
        in_rat = df[RAT_COL] == rat_id
        df.loc[in_rat & df[SESSION_COL].isin(psych_sessions),  'split'] = 'psych'
        df.loc[in_rat & ~df[SESSION_COL].isin(psych_sessions), 'split'] = 'model'
    print(f"  psych set: {(df['split']=='psych').sum():>7d} trials")
    print(f"  model set: {(df['split']=='model').sum():>7d} trials")
    return df

df = split_sessions(df, psych_fraction=PSYCH_FRACTION, seed=RANDOM_SEED)
# this just adds a coloumn for each trial with the value 'psych' or 'model' depending on the session it belongs to. 
# This is used later to separate the data into two sets: one for fitting psychometric functions and one for fitting the mixed model.

# ============================================================
# 3. PSYCHOMETRICS ON PSYCH SET → ACUITY RANK PER RAT
# ─────────────────────────────────────────────────────────────
# Same cumulative-Gaussian-with-lapses fit as before, restricted
# to the 1/3 of sessions reserved for it. From σ per rat per
# modality, we build an ordinal RANK (1=lowest acuity → 3=highest)
# which becomes the basis for transition_dir and acuity_n1.
# ============================================================
def fit_psychometric_per_rat(df_psych, n_bins=19):
    def fn(x, mu, sigma, gamma, lam):
        x = np.asarray(x, dtype=float)
        return gamma + (1 - gamma - lam) * (
            0.5 * (1 + erf((x - mu) / np.sqrt(2 * sigma**2)))
        )
    p0     = [45, 15, 0.01, 0.02]
    bounds = ([20, 5, 0.0, 0.0], [70, 50, 0.3, 0.3])
    edges   = np.linspace(0, 90, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    rows = []
    for rat_id in sorted(df_psych[RAT_COL].unique()):
        for mod in sorted(df_psych[MOD_COL].unique()):
            sub = df_psych[(df_psych[RAT_COL] == rat_id) & (df_psych[MOD_COL] == mod)]
            if len(sub) < 50:
                rows.append({'rat': rat_id, 'mod': mod, 'sigma': np.nan, 'fit_success': False})
                continue
            angles    = sub[ANGLE_COL].values.astype(float)
            responses = sub[ACTION_COL].values.astype(float)
            bin_idx = np.clip(np.digitize(angles, edges, right=False) - 1, 0, n_bins - 1)
            obs_x, obs_p = [], []
            for b, c in enumerate(centers):
                m = bin_idx == b
                if m.sum() < 5: continue
                obs_x.append(c); obs_p.append(responses[m].mean())
            try:
                popt, _ = curve_fit(fn, obs_x, obs_p, p0=p0, bounds=bounds, maxfev=5000)
                rows.append({'rat': rat_id, 'mod': mod, 'sigma': popt[1], 'fit_success': True})
            except Exception as e:
                print(f"  Rat {rat_id} mod {mod}: fit failed ({e})")
                rows.append({'rat': rat_id, 'mod': mod, 'sigma': np.nan, 'fit_success': False})
    return pd.DataFrame(rows)


def build_acuity_ranks(psych_df):
    """For each rat, rank modalities by 1/σ. Returns {rat: {mod: rank}}."""
    ranks = {}
    for rat_id, sub in psych_df[psych_df['fit_success']].groupby('rat'):
        sub = sub.copy()
        sub['strength'] = 1 / sub['sigma']
        sub['rank'] = sub['strength'].rank(method='dense').astype(int)
        ranks[rat_id] = dict(zip(sub['mod'], sub['rank']))
    return ranks


psych_df    = fit_psychometric_per_rat(df[df['split'] == 'psych'])
acuity_rank = build_acuity_ranks(psych_df) # --> {
#    rat_1: {mod_1: 3, mod_2: 1, mod_3: 2},
#    rat_2: {mod_1: 2, mod_2: 3, mod_3: 1}
#}, but mod_1 is just called 1 and so on for the others.
# We have something like this for each rat, where the numbers are ranks of acuity (1=lowest, 3=highest) for each modality. 
print("\nAcuity ranks per rat (3 = best acuity for that rat):")
print(pd.DataFrame(acuity_rank).T)


# ============================================================
# 4. ★ NEW: FEATURE ENGINEERING ON THE MODEL SET
# ─────────────────────────────────────────────────────────────
# Performed on the 2/3 sessions reserved for the mixed model.
# Drops the first trial of each session (its n-1 is part of the warm up). Builds:
#   success_n1 / failure_n1 — signed history (as before)
#   hit_n1                  — unsigned outcome (NEW covariate)
#   angle / angle_n1        — fixed -45 centering (as before)
#   acuity_n1               — low/mid/high rank of n−1 modality
#   transition_dir          — down/same/up rank comparison
#   transition_cell         — full 9-level cell (for Formula 3)
# ============================================================
def engineer_features(df_in, acuity_rank):
    df = df_in.copy().sort_values([RAT_COL, SESSION_COL]).reset_index(drop=True)
    df['_trial_in_sess'] = df.groupby([RAT_COL, SESSION_COL]).cumcount()
    n_first = (df['_trial_in_sess'] == 0).sum()
    df = df[df['_trial_in_sess'] > 0].drop(columns='_trial_in_sess')
    print(f"  Dropped {n_first} first-of-session trials") # if trial number 0 repeats it's a problem with .cumcount()
    # I am sure I can do this in a easier way.

    # signed action_n-1: ensure -1/+1 coding
    if df[ACTION_N1_COL].between(0, 1).all():
        df[ACTION_N1_COL] = df[ACTION_N1_COL] * 2 - 1

    # signed history (as before)
    df['success_n1'] = df[HIT_N1_COL]       * df[ACTION_N1_COL]
    df['failure_n1'] = (1 - df[HIT_N1_COL]) * df[ACTION_N1_COL]

    # ★ NEW: unsigned outcome covariate
    df['hit_n1'] = df[HIT_N1_COL].astype(float)

    # angle centering (as before)
    df['angle']    = df[ANGLE_COL]    - 45.0
    df['angle_n1'] = df[ANGLE_N1_COL] - 45.0

    # acuity_n1 from rank lookup
    rank_label = {1: 'low', 2: 'mid', 3: 'high'}
    df['acuity_n1'] = df.apply(
        lambda r: rank_label.get(acuity_rank.get(r[RAT_COL], {}).get(r[MOD_N1_COL], np.nan), np.nan),
        axis=1
    )

    # transition_dir
    def trans_dir(row):
        rmap   = acuity_rank.get(row[RAT_COL], {}) # rmap = { 1: 1,   # T = low 2: 3,   # V = high3: 2    # VT = mid}
        r_prev = rmap.get(row[MOD_N1_COL])
        r_curr = rmap.get(row[MOD_COL])
        if r_prev is None or r_curr is None: return np.nan
        if r_curr > r_prev: return 'up'
        if r_curr < r_prev: return 'down'
        return 'same'
    df['transition_dir'] = df.apply(trans_dir, axis=1)

    # transition_cell (for Formula 3) --> but don't we have this already in the data? check if they are the same thing
    mod_label = {1: 'T', 2: 'V', 3: 'VT'}
    df['transition_cell'] = (
        df[MOD_N1_COL].map(mod_label).astype(str)
        + '_to_'
        + df[MOD_COL].map(mod_label).astype(str)
    )

    print(f"\n  transition_dir counts:\n{df['transition_dir'].value_counts().to_string()}")
    print(f"\n  acuity_n1 counts:\n{df['acuity_n1'].value_counts().to_string()}")
    return df

df_model = engineer_features(df[df['split'] == 'model'], acuity_rank)


# ============================================================
# 5. ★ NEW: SESSION-MEAN CENTERING (level-1 variables)
# ─────────────────────────────────────────────────────────────
# Centering each level-1 numeric predictor by its session mean
# isolates the WITHIN-SESSION effect of each predictor. Without
# this, slopes mix within-session and between-session variation,
# which can have different signs.
# ============================================================
def session_mean_center(df, cols, group=(RAT_COL, SESSION_COL)):
    df = df.copy()
    for c in cols:
        sess_mean = df.groupby(list(group))[c].transform('mean') # transform applies the groupby mean back to the original dataframe, 
                                                                 # so each trial gets the mean of its session.
        df[f'{c}_c'] = df[c] - sess_mean
    return df

df_model = session_mean_center(
    df_model,
    cols=['success_n1', 'failure_n1', 'hit_n1', 'angle', 'angle_n1']
)

# create a way to check if the centering worked correctly by comparing the original and centered columns for a few sessions
# or a plot of where session points are before and after centering. 
# ============================================================
# 6. ★ NEW: DUMMY CODING with chosen REFERENCE CATEGORIES
# ─────────────────────────────────────────────────────────────
# transition_dir:  reference = 'same'  → coefs read as down-vs-same, up-vs-same
# acuity_n1:       reference = 'mid'   → coefs read as low-vs-mid,  high-vs-mid
# transition_cell: reference = the most-populated cell
# ============================================================
def make_dummies(df):
    df = df.copy()
    df['td_down'] = (df['transition_dir'] == 'down').astype(int)
    df['td_up']   = (df['transition_dir'] == 'up').astype(int)
    df['ac_low']  = (df['acuity_n1'] == 'low').astype(int)
    df['ac_high'] = (df['acuity_n1'] == 'high').astype(int)

    cell_counts = df['transition_cell'].value_counts()
    ref_cell    = cell_counts.idxmax() # we can set the reference category manually if we want, here it is T --> T because it is the most populated cell
    print(f"\n  transition_cell reference category: {ref_cell}")
    tc_dummies = []
    for cell in cell_counts.index:
        if cell == ref_cell: continue
        name = f'tc_{cell}'
        df[name] = (df['transition_cell'] == cell).astype(int)
        tc_dummies.append(name)
    return df, ref_cell, tc_dummies

df_model, ref_cell, transition_cell_dummies = make_dummies(df_model)


# ============================================================
# 7. ★ NEW: EMPIRICAL LOGIT DIAGNOSTIC PLOT
# ─────────────────────────────────────────────────────────────
# Check linearity-on-logit assumption for continuous predictors.
# Bin x into quantiles → mean(y) per bin → logit transform.
# Straight line = linear OK. Curvature = use basis expansion.
# References:
#   Harrell (2015) Regression Modeling Strategies, ch. 2
#   Carandini (2024) Neuron 112:2854 — lapse rates and the
#     need for nonlinear stimulus mapping in choice models.
# ============================================================
def empirical_logit_plot(df, x_col, y_col=ACTION_COL, n_bins=15, min_per_bin=30, ax=None):
    work = df[[x_col, y_col]].dropna().copy()
    work['bin'] = pd.qcut(work[x_col], n_bins, duplicates='drop')
    g = work.groupby('bin', observed=True).agg(
        x_center=(x_col, 'mean'), n=(y_col, 'size'), p=(y_col, 'mean')
    ).reset_index(drop=True)
    g = g[g['n'] >= min_per_bin].copy()
    g['p_adj']     = (g['p'] * g['n'] + 0.5) / (g['n'] + 1)
    g['emp_logit'] = np.log(g['p_adj'] / (1 - g['p_adj']))
    g['se_logit']  = np.sqrt(1 / (g['n'] * g['p_adj']) + 1 / (g['n'] * (1 - g['p_adj'])))
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(g['x_center'], g['emp_logit'], yerr=g['se_logit'], fmt='o', capsize=3)
    coef = np.polyfit(g['x_center'], g['emp_logit'], 1)
    xs = np.linspace(g['x_center'].min(), g['x_center'].max(), 100)
    ax.plot(xs, np.polyval(coef, xs), '--', color='gray',
            label=f'linear fit: slope={coef[0]:.3f}')
    ax.set_xlabel(x_col); ax.set_ylabel('empirical logit')
    ax.axhline(0, color='k', lw=0.5, alpha=0.5)
    ax.set_title(f'Empirical logit: {x_col}')
    ax.legend(); plt.tight_layout()
    return g


# ============================================================
# 8. ★ NEW: BASIS EXPANSION for angle / angle_n1 (tunable)
# ─────────────────────────────────────────────────────────────
# 'linear'    : raw, 1 term
# 'cubic'     : x + x³   (symmetric about 0, 2 terms)
# 'poly3'     : x + x² + x³  (3 terms)
# 'spline_k'  : natural cubic spline, k df  (Harrell default ~ 4)
# ============================================================
def expand_basis(df, col, representation): # I got the basic idea of what splines are doing, I still need to test if we need them
    df = df.copy()
    x = df[col].values.astype(float)
    if representation == 'linear':
        return df, [col]
    if representation == 'cubic':
        df[f'{col}_cu'] = x**3
        return df, [col, f'{col}_cu']
    if representation == 'poly3':
        df[f'{col}_sq'] = x**2
        df[f'{col}_cu'] = x**3
        return df, [col, f'{col}_sq', f'{col}_cu']
    if representation.startswith('spline_'):
        n_df = int(representation.split('_')[1])
        from patsy import dmatrix
        basis = dmatrix(f"cr(x, df={n_df}) - 1", {'x': x}, return_type='dataframe')
        new_cols = []
        for i in range(basis.shape[1]):
            name = f'{col}_s{i+1}'
            df[name] = basis.iloc[:, i].values
            new_cols.append(name)
        return df, new_cols
    raise ValueError(f"Unknown representation: {representation}")


# ============================================================
# 9. MODEL CONFIG — pick formula, basis, random effects
# ============================================================
MODEL_CONFIG = {
    # which of the three formulas to fit
    'FORMULA':                       3,
           # 1 (minimal), 2 (+acuity_n1), 3 (saturated)

    # basis representation for angle terms
    'angle_representation':          'spline_4', # 'linear' | 'cubic' | 'poly3' | 'spline_3..5'
    'angle_n1_representation':       'linear',

    # random effects (rat-level — level 3)
    'random_intercept_rat':          True,
    'random_success_n1_rat':         True,
    'random_failure_n1_rat':         True,
    'random_success_n1_c:td_down_rat': False,  
    'random_success_n1_c:td_up_rat':   False,  
    'random_failure_n1_c:td_down_rat': False,  
    'random_failure_n1_c:td_up_rat':   False,  
    'random_success_n1_c:ac_low_rat':  False,
    'random_success_n1_c:ac_high_rat': False,
    'random_failure_n1_c:ac_low_rat':  False,
    'random_failure_n1_c:ac_high_rat': False,
     
     # Success × cell interactions (8 — T→T is reference, no slope needed)
    'random_success_n1_c:tc_T_to_V_rat':               True,
    'random_success_n1_c:tc_T_to_VT_rat':              True,
    'random_success_n1_c:tc_V_to_T_rat':               True,
    'random_success_n1_c:tc_V_to_V_rat':               True,
    'random_success_n1_c:tc_V_to_VT_rat':              True,
    'random_success_n1_c:tc_VT_to_T_rat':              True,
    'random_success_n1_c:tc_VT_to_V_rat':              True,
    'random_success_n1_c:tc_VT_to_VT_rat':             True,

    # Failure × cell interactions (8)
    'random_failure_n1_c:tc_T_to_V_rat':               True,
    'random_failure_n1_c:tc_T_to_VT_rat':              True,
    'random_failure_n1_c:tc_V_to_T_rat':               True,
    'random_failure_n1_c:tc_V_to_V_rat':               True,
    'random_failure_n1_c:tc_V_to_VT_rat':              True,
    'random_failure_n1_c:tc_VT_to_T_rat':              True,
    'random_failure_n1_c:tc_VT_to_V_rat':              True,
    'random_failure_n1_c:tc_VT_to_VT_rat':             True,   

    # session-level random effects (level 2) — intercept only for now
    'random_intercept_rat_session':  True,
    'random_success_n1_rat_session': False,   # off for speed
    'random_failure_n1_rat_session': False,   # off for speed
    'random_session_slopes':         False,   # ★ master switch — keep False

    'use_random_correlations':       False,
    'nAGQ':                          1,       # use 0 if 1 is still too slow
}


# ============================================================
# ★ DIAGNOSTIC: VIF on the fixed-effects design matrix
# ─────────────────────────────────────────────────────────────
# Tests whether any fixed-effect predictor is a near-linear
# combination of the others (multicollinearity). VIF > 5 is
# concerning, > 10 is a real problem. Concern here is whether
# hit_n1_c is collinear with success_n1_c / failure_n1_c via
# their interactions.
# ============================================================
def compute_vif(df_in, columns):
    """Compute VIF for each column by regressing it on all others."""
    from sklearn.linear_model import LinearRegression
    vif_rows = []
    X = df_in[columns].dropna().values
    for i, col in enumerate(columns):
        y = X[:, i]
        X_others = np.delete(X, i, axis=1)
        lr = LinearRegression().fit(X_others, y)
        r2 = lr.score(X_others, y)
        vif = 1 / (1 - r2) if r2 < 1 else np.inf
        vif_rows.append({'term': col, 'R²_on_others': r2, 'VIF': vif})
    return pd.DataFrame(vif_rows).sort_values('VIF', ascending=False)


# Build the fixed-effects design matrix as the model sees it.
# Need to expand splines and include interaction columns.
_df_vif, _ = expand_basis(df_model[['angle_c']].copy(), 'angle_c',
                          MODEL_CONFIG['angle_representation'])
df_design = df_model.copy()
for col in _df_vif.columns:
    df_design[col] = _df_vif[col].values

# add interaction columns explicitly so VIF sees them
df_design['success_x_td_down'] = df_design['success_n1_c'] * df_design['td_down']
df_design['success_x_td_up']   = df_design['success_n1_c'] * df_design['td_up']
df_design['failure_x_td_down'] = df_design['failure_n1_c'] * df_design['td_down']
df_design['failure_x_td_up']   = df_design['failure_n1_c'] * df_design['td_up']

vif_cols = [c for c in [
    'angle_c_s1', 'angle_c_s2', 'angle_c_s3', 'angle_c_s4',  # spline (some may be dropped)
    'angle_n1_c', 'hit_n1_c',
    'success_n1_c', 'failure_n1_c',
    'td_down', 'td_up',
    'success_x_td_down', 'success_x_td_up',
    'failure_x_td_down', 'failure_x_td_up',
] if c in df_design.columns]

print("\n" + "="*60)
print("DIAGNOSTIC — VIF on fixed-effects design matrix")
print("="*60)
vif_df = compute_vif(df_design, vif_cols)
print(vif_df.round(2).to_string(index=False))
print("\nVIF > 5: suspicious; VIF > 10: clear multicollinearity.")

# ============================================================
# 10. FORMULA BUILDER
# ============================================================
def build_formula(config, df, transition_cell_dummies):
    # angle bases
    df, angle_terms    = expand_basis(df, 'angle_c',    config['angle_representation'])
    df, angle_n1_terms = expand_basis(df, 'angle_n1_c', config['angle_n1_representation'])

    fixed = []
    fixed += angle_terms + angle_n1_terms
    fixed += ['hit_n1_c']                                # ★ unsigned outcome covariate
    fixed += ['success_n1_c', 'failure_n1_c']            # signed history main effects

    if config['FORMULA'] == 1:
        fixed += ['td_down', 'td_up']
        fixed += ['success_n1_c:td_down', 'success_n1_c:td_up']
        fixed += ['failure_n1_c:td_down', 'failure_n1_c:td_up']

    elif config['FORMULA'] == 2:
        fixed += ['td_down', 'td_up']
        fixed += ['ac_low', 'ac_high']
        fixed += ['success_n1_c:td_down', 'success_n1_c:td_up']
        fixed += ['failure_n1_c:td_down', 'failure_n1_c:td_up']
        fixed += ['success_n1_c:ac_low',  'success_n1_c:ac_high']
        fixed += ['failure_n1_c:ac_low',  'failure_n1_c:ac_high']

    elif config['FORMULA'] == 3:
        fixed += transition_cell_dummies
        for d in transition_cell_dummies:
            fixed += [f'success_n1_c:{d}', f'failure_n1_c:{d}']
    else:
        raise ValueError("FORMULA must be 1, 2, or 3")
# random effects
    pipe = '||' if not config.get('use_random_correlations', True) else '|'

    rat_terms = []
    if config['random_intercept_rat']:  rat_terms.append('1')
    if config['random_success_n1_rat']: rat_terms.append('success_n1_c')
    if config['random_failure_n1_rat']: rat_terms.append('failure_n1_c')
    if config['random_success_n1_c:td_down_rat']: rat_terms.append('success_n1_c:td_down')
    if config['random_success_n1_c:td_up_rat']: rat_terms.append('success_n1_c:td_up')
    if config['random_failure_n1_c:td_down_rat']: rat_terms.append('failure_n1_c:td_down')
    if config['random_failure_n1_c:td_up_rat']: rat_terms.append('failure_n1_c:td_up') 
    if config['random_success_n1_c:ac_low_rat']: rat_terms.append('success_n1_c:ac_low')
    if config['random_success_n1_c:ac_high_rat']: rat_terms.append('success_n1_c:ac_high')
    if config['random_failure_n1_c:ac_low_rat']: rat_terms.append('failure_n1_c:ac_low')
    if config['random_failure_n1_c:ac_high_rat']: rat_terms.append('failure_n1_c:ac_high')

    sess_terms = []
    if config['random_intercept_rat_session']:
        sess_terms.append('1')
    if config.get('random_session_slopes', False):
        # only include session-level slopes when explicitly enabled (expensive!)
        if config['random_success_n1_rat_session']: sess_terms.append('success_n1_c')
        if config['random_failure_n1_rat_session']: sess_terms.append('failure_n1_c')

    # use || only when there are multiple terms to de-correlate; otherwise use |
    rat_pipe  = pipe if len(rat_terms)  > 1 else '|'
    sess_pipe = pipe if len(sess_terms) > 1 else '|'

    random_str = f"({' + '.join(rat_terms)} {rat_pipe} {RAT_COL})"
    if sess_terms:
        random_str += f" + ({' + '.join(sess_terms)} {sess_pipe} {RAT_COL}:{SESSION_COL})"
                    
    formula = f"{ACTION_COL} ~ " + " + ".join(fixed) + " + " + random_str
    return formula, df


# ============================================================
# ★ NEW: R FITTING SCRIPT (embedded, written to disk per fit)
# ─────────────────────────────────────────────────────────────
# Reads the prepared CSV, fits glmer with lme4, writes:
#   fixed_effects.csv   — estimates, SE, z, p
#   ranef_variances.csv — variance/covariance components
#   blups_<group>.csv   — per-group random-effect estimates
#   meta.txt            — AIC, logLik, n_obs, singular, converged
# Requires only base R + lme4 (no jsonlite, no rpy2).
# ============================================================
R_FIT_SCRIPT = r"""
suppressPackageStartupMessages(library(lme4))

args <- commandArgs(trailingOnly = TRUE)
data_path   <- args[1]
formula_str <- args[2]
out_dir     <- args[3]
nAGQ_val    <- as.integer(args[4])

t_start <- Sys.time()
cat("[", format(t_start), "] Started\n", sep="")

cat("[", format(Sys.time()), "] Reading data...\n", sep="")
dat <- read.csv(data_path)
cat("  rows:", nrow(dat), " cols:", ncol(dat), "\n")
cat("  read elapsed:",
    format(round(difftime(Sys.time(), t_start, units="secs"), 1)), "\n")
flush.console()

cat("[", format(Sys.time()), "] Formula:\n  ", formula_str, "\n", sep="")
cat("  nAGQ =", nAGQ_val, "\n")
flush.console()

cat("[", format(Sys.time()), "] Starting glmer fit (verbose iteration log below)...\n", sep="")
flush.console()

t_fit <- Sys.time()
model <- glmer(
  as.formula(formula_str),
  data    = dat,
  family  = binomial,
  nAGQ    = nAGQ_val,
  control = glmerControl(
    optimizer = "bobyqa",
    optCtrl   = list(maxfun = 100000, rhoend = 1e-4)
  ),
  verbose = 2
)
fit_secs <- as.numeric(difftime(Sys.time(), t_fit, units="secs"))
cat("[", format(Sys.time()), "] Fit complete in ",
    sprintf("%.1f min", fit_secs/60), "\n", sep="")
flush.console()

# fixed effects
fixed_df <- as.data.frame(summary(model)$coefficients)
fixed_df$term <- rownames(fixed_df)
rownames(fixed_df) <- NULL
write.csv(fixed_df, file.path(out_dir, "fixed_effects.csv"), row.names = FALSE)

# random effect variances/covariances
vc <- as.data.frame(VarCorr(model))
write.csv(vc, file.path(out_dir, "ranef_variances.csv"), row.names = FALSE)

# BLUPs
ranef_list <- ranef(model)
for (grp in names(ranef_list)) {
  re_df <- as.data.frame(ranef_list[[grp]])
  re_df$id <- rownames(re_df)
  rownames(re_df) <- NULL
  safe <- gsub("[^A-Za-z0-9]", "_", grp)
  write.csv(re_df, file.path(out_dir, paste0("blups_", safe, ".csv")),
            row.names = FALSE)
}

# metadata
conv_msgs <- model@optinfo$conv$lme4$messages
meta_lines <- c(
  paste0("aic=",       AIC(model)),
  paste0("logLik=",    as.numeric(logLik(model))),
  paste0("n_obs=",     nobs(model)),
  paste0("singular=",  isSingular(model)),
  paste0("converged=", is.null(conv_msgs)),
  paste0("fit_minutes=", sprintf("%.2f", fit_secs/60))
)
if (!is.null(conv_msgs)) {
  meta_lines <- c(meta_lines,
                  paste0("conv_msg=", paste(conv_msgs, collapse="; ")))
}
writeLines(meta_lines, file.path(out_dir, "meta.txt"))

# NEW: write fitted probabilities (with random effects) per trial
fitted_p <- fitted(model)
preds_df <- data.frame(
  fitted_p = fitted_p,
  action   = dat$action,
  rat      = dat$rat,
  date     = dat$date,
  transition_dir = dat$transition_dir,
  failure_n1     = dat$failure_n1,
  success_n1     = dat$success_n1
)
write.csv(preds_df, file.path(out_dir, "predictions.csv"), row.names = FALSE)


cat("[", format(Sys.time()), "] Done\n", sep="")
"""


# ============================================================
# 11. ★ CHANGED: FIT VIA SUBPROCESS, REPORT FROM DICT
# ============================================================
def _columns_in_formula(formula):
    tokens = re.split(r'[\s+~()|*:]+', formula)
    return [t for t in tokens if t and not t.replace('.', '').replace('-', '').isdigit()]



def fit_model(df_model, config, transition_cell_dummies,
              rscript_exe=RSCRIPT_EXE,
              output_dir=None):     # ★ NEW: optional permanent save location
    """
    If output_dir is given, results are saved there permanently
    (and not deleted). Otherwise a temp dir is used and deleted at the end.
    """
    formula, df_ready = build_formula(config, df_model, transition_cell_dummies)
    print(f"\n--- Formula {config['FORMULA']} ---")
    print(formula)

    cols = _columns_in_formula(formula)
    cols = [c for c in cols if c in df_ready.columns]
    df_fit = df_ready.dropna(subset=cols)
    print(f"N trials in fit: {len(df_fit)}")

    if output_dir is not None:
        tmpdir = Path(output_dir)
        tmpdir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        tmpdir = Path(tempfile.mkdtemp(prefix='glmm_fit_'))
        cleanup = True

    try:
        data_path   = tmpdir / 'data.csv'
        script_path = tmpdir / 'fit.R'
        df_fit.to_csv(data_path, index=False)
        script_path.write_text(R_FIT_SCRIPT)

        cmd = [rscript_exe, str(script_path), str(data_path),
               formula, str(tmpdir), str(config.get('nAGQ', 1))]
        print(f"Running Rscript with nAGQ={config.get('nAGQ', 1)}...")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in iter(process.stdout.readline, ''):
            print(line, end='', flush=True)
        process.stdout.close()
        if process.wait() != 0:
            raise RuntimeError(f"R fitting failed")

        fixed   = pd.read_csv(tmpdir / 'fixed_effects.csv')
        ranef_v = pd.read_csv(tmpdir / 'ranef_variances.csv')
        meta = {}
        for line in (tmpdir / 'meta.txt').read_text().splitlines():
            if '=' in line:
                k, v = line.split('=', 1)
                meta[k.strip()] = v.strip()
        blups = {p.stem[len('blups_'):]: pd.read_csv(p)
                 for p in tmpdir.glob('blups_*.csv')}

        result = {
            'fixed': fixed, 'ranef_var': ranef_v, 'blups': blups,
            'meta': meta, 'formula': formula, 'n_fit': len(df_fit),
        }
        # also save the Python-side dict for easy reload
        if output_dir is not None:
            import pickle
            with open(tmpdir / 'result.pkl', 'wb') as f:
                pickle.dump(result, f)
            print(f"Saved to {tmpdir}")
        return result
    finally:
        if cleanup:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


def report(result):
    """Pretty-print a fit result dict from fit_model()."""
    print("\n=== FIXED EFFECTS ===")
    fixed = result['fixed'].copy()
    # round numeric columns
    for c in fixed.columns:
        if pd.api.types.is_numeric_dtype(fixed[c]):
            fixed[c] = fixed[c].round(4)
    # put term first
    cols = ['term'] + [c for c in fixed.columns if c != 'term']
    print(fixed[cols].to_string(index=False))

    print("\n=== RANDOM EFFECT VARIANCES / COVARIANCES ===")
    print(result['ranef_var'].round(4).to_string(index=False))

    print("\n=== META ===")
    for k, v in result['meta'].items():
        print(f"  {k}: {v}")
    print(f"  n_fit: {result['n_fit']}")


# ============================================================
# 12. RUN
# ============================================================
# ============================================================
# Formula 1 || refit — THE LONG ONE
# ============================================================
input("\n>>> Press Enter to start the || refit (will take 1-2h)... ")

MODEL_CONFIG['FORMULA'] = 3
result_1_uncorr = fit_model(
    df_model, MODEL_CONFIG, transition_cell_dummies,
    output_dir=r'C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\fits\formula3_uncorrelated_try_more_random'
)
report(result_1_uncorr)

# ============================================================
# ★ DIAGNOSTIC — model predictions vs raw probabilities per cell
# ─────────────────────────────────────────────────────────────
# If the model is correctly specified, its fitted probabilities
# should match raw P(right) in each (rat × transition) cell.
# This is the most direct test of whether the model is fitting
# the data well, independent of how individual coefficients
# might be interpreted.
# ============================================================
preds_path = r'C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\fits\formula3_uncorrelated_try_more_random\predictions.csv'
preds = pd.read_csv(preds_path)
print(f"Loaded predictions: {len(preds)} trials")

# overall calibration
print(f"\nOverall mean action:      {preds['action'].mean():.4f}")
print(f"Overall mean fitted_p:    {preds['fitted_p'].mean():.4f}")
print(f"Overall difference:       {preds['fitted_p'].mean() - preds['action'].mean():+.4f}")

# Per-cell summary
cell_summary = preds.groupby(['rat', 'transition_dir']).agg(
    n      = ('action',  'size'),
    raw_p  = ('action',  'mean'),
    pred_p = ('fitted_p', 'mean'),
).reset_index()
cell_summary['residual']     = cell_summary['pred_p'] - cell_summary['raw_p']
cell_summary['abs_residual'] = cell_summary['residual'].abs()

print(f"\nPer-rat × transition_dir summary (worst-fitting cells first):")
print(cell_summary.sort_values('abs_residual', ascending=False).head(15).round(3).to_string(index=False))

print(f"\nMean abs residual across cells: {cell_summary['abs_residual'].mean():.4f}")
print(f"Max  abs residual across cells: {cell_summary['abs_residual'].max():.4f}")
print(f"Cells with |residual| > 0.05:   {(cell_summary['abs_residual'] > 0.05).sum()} / {len(cell_summary)}")

# Now the key check: predicted failure-slope shift per cell vs raw
# Slope = P(right|failure=+1) − P(right|failure=−1), per cell, for both raw and predicted
def slope_in_cell(cell_df, history_col, response_col):
    p_plus  = cell_df.loc[cell_df[history_col] == +1, response_col].mean() if (cell_df[history_col] == +1).sum() > 20 else np.nan
    p_minus = cell_df.loc[cell_df[history_col] == -1, response_col].mean() if (cell_df[history_col] == -1).sum() > 20 else np.nan
    return (p_plus - p_minus) / 2 if not (np.isnan(p_plus) or np.isnan(p_minus)) else np.nan

print("\n" + "="*60)
print("Per-rat failure-slope: model vs raw, by transition")
print("="*60)
slope_compare = []
for (rat_id, td), cell_df in preds.groupby(['rat', 'transition_dir']):
    raw_slope  = slope_in_cell(cell_df, 'failure_n1', 'action')
    pred_slope = slope_in_cell(cell_df, 'failure_n1', 'fitted_p')
    slope_compare.append({
        'rat': rat_id, 'transition_dir': td,
        'raw_slope': raw_slope, 'pred_slope': pred_slope,
        'n': len(cell_df),
    })
slope_df = pd.DataFrame(slope_compare)
print(slope_df.round(3).to_string(index=False))

print("\n" + "="*60)
print("Average raw vs predicted slope by transition")
print("="*60)

summary = (
    slope_df.groupby('transition_dir')
            .agg(
                raw_mean=('raw_slope', 'mean'),
                pred_mean=('pred_slope', 'mean'),
                raw_sd=('raw_slope', 'std'),
                pred_sd=('pred_slope', 'std'),
                n=('raw_slope', 'count')
            )
)

print(summary.round(4))