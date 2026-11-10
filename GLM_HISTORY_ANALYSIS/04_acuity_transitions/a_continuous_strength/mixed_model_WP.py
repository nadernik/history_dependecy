import os
os.environ['R_HOME']      = r'C:\Program Files\R\R-4.4.1'
os.environ['PATH']        = r'C:\Program Files\R\R-4.4.1\bin' + ';' + os.environ['PATH']
os.environ['R_LIBS_USER'] = r'C:\Users\alech\AppData\Local\R\win-library\4.4'
from scipy.optimize import curve_fit
from scipy.special import erf
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy")
from functions_Ale import flatten_model_results, filter_training_trials
from serial_dependence_analysis import SerialDependenceAnalyzer
from pymer4.models import Lmer   # ← add here, after R env vars are set
HISTORY_DEPTH = 1


# =============================
# CONFIG
# =============================
RAT_COL        = "rat"
ANGLE_COL      = "angle"          # trial n angle
ACTION_COL     = "action"         # trial n choice (0/1)
ANGLE_N1_COL   = "angle_n-1"      # trial n-1 angle
ACTION_N1_COL  = "action_n-1"     # trial n-1 choice (0/1)
BOUNDARY_ANGLE = 45
HITMISS_N1_COL = "hitmiss_n-1"    # trial n-1 outcome (0/1)

# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=HISTORY_DEPTH, save_figures=False)
an.load_and_preprocess_data()
df = an.create_lagged_features()
df_processed = filter_training_trials(df, method='criterion', criterion=0.80, min_consec=2)


def fit_psychometric_all_rats(df_processed, rat_col='rat', angle_col='angle', 
                               resp_col='action', mod_col='mod', n_bins=19):
    """
    Fit cumulative Gaussian psychometric curve per rat per modality.
    Returns dataframe with columns: rat, mod, mu, sigma, gamma, lapse, fit_success
    sigma is the slope proxy — smaller sigma = steeper = more precise modality for that rat.
    """

    def cumulative_gaussian_lapse(x, mu, sigma, gamma, lambda_param):
        x = np.asarray(x, dtype=float)
        return gamma + (1 - gamma - lambda_param) * (
            0.5 * (1 + erf((x - mu) / np.sqrt(2 * sigma**2)))
        )

    p0     = [45, 15, 0.01, 0.02]
    bounds = ([20,  5,  0.0, 0.0],
              [70, 50,  0.3, 0.3])

    bin_edges   = np.linspace(0, 90, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    rats      = sorted(df_processed[rat_col].unique())
    modalities = sorted(df_processed[mod_col].unique())

    rows = []

    for rat_id in rats:
        for mod in modalities:

            mask   = (df_processed[rat_col] == rat_id) & (df_processed[mod_col] == mod)
            df_sub = df_processed[mask].copy()

            if len(df_sub) < 20:   # too few trials to fit
                rows.append({'rat': rat_id, 'mod': mod, 'fit_success': False,
                             'mu': np.nan, 'sigma': np.nan, 
                             'gamma': np.nan, 'lapse': np.nan})
                continue

            angles    = df_sub[angle_col].values.astype(float)
            responses = df_sub[resp_col].values.astype(float)

            # bin into ~5 degree bins
            bin_idx = np.clip(np.digitize(angles, bin_edges, right=False) - 1, 0, n_bins - 1)

            obs_x, obs_p = [], []
            for b, center in enumerate(bin_centers):
                bin_mask = bin_idx == b
                if bin_mask.sum() < 5:
                    continue
                obs_x.append(center)
                obs_p.append(np.mean(responses[bin_mask]))

            obs_x = np.array(obs_x)
            obs_p = np.array(obs_p)

            try:
                popt, _ = curve_fit(
                    cumulative_gaussian_lapse, obs_x, obs_p,
                    p0=p0, bounds=bounds, maxfev=5000
                )
                mu, sigma, gamma, lapse = popt
                rows.append({'rat': rat_id, 'mod': mod, 'fit_success': True,
                             'mu': mu, 'sigma': sigma, 
                             'gamma': gamma, 'lapse': lapse})
                print(f"Rat {rat_id}, mod {mod}: sigma={sigma:.2f}")

            except Exception as e:
                print(f"Rat {rat_id}, mod {mod}: fit failed — {e}")
                rows.append({'rat': rat_id, 'mod': mod, 'fit_success': False,
                             'mu': np.nan, 'sigma': np.nan,
                             'gamma': np.nan, 'lapse': np.nan})

    return pd.DataFrame(rows)



psychometric_df = fit_psychometric_all_rats(df_processed)
# BETA EXTRACTION STEP
df = df_processed
# FEATURE CREATION:

# re-code action_n-1 to be -1 for left and +1 for right, instead of 0/1, so that it can be used in the model as a signed variable (e.g. for interaction with hit_n-1)
df['action_n-1'] = df['action_n-1']*2 -1
#df['hitmiss_n-1'] = df['hitmiss_n-1']*2 -1
df['success_n-1'] = df['hit_n-1']*df['action_n-1'] # 1 if previous trial was a hit and the action was right (1*1), -1 if hit and left (1*-1), 0 if miss (0*anything)
df['failure_n-1'] = (1 - df['hit_n-1'])*df['action_n-1'] # 1 if previous trial was a miss and the action was right (1*1), -1 if miss and left (1*-1), 0 if hit (0*anything)
#df['hit_n-1'] = 1 - df['hit_n-1'] # TMEPORARY 
#df['angle_hit_n-1'] = df['angle_n-1']*df['hit_n-1'] # angle of the previous trial if it was a hit, 0 if it was a miss
#df['angle_miss_n-1'] = df['angle_n-1']*(1 - df['hit_n-1']) # angle of the previous trial if it was a miss, 0 if it was a hit
df['angle'] = df['angle'] - 45
df['angle_n-1'] = df['angle_n-1'] - 45
df['tactile'] = df['mod'] == 1
# df['visual'] = df['mod'] == 2
df['visuotactile'] = df['mod'] == 3
df['angle_tactile'] = df['angle']*df['tactile']
df['angle_visuotactile'] = df['angle']*df['visuotactile']
# 4 models with increasing complexity: now it is a bit hardcoded but we can easily make it more flexible later
# FAMILY 1: only angle
features_0 = ['angle']
features_1 = ['angle', 'tactile', 'visuotactile'] 
features_2 = ['angle', 'tactile', 'visuotactile', 'angle_tactile', 'angle_visuotactile']
features_3 = ['angle', 'angle_visuotactile', 'angle_tactile'] # this is the best model according to the previous analysis, but it is not very interpretable, as we don't know if the effect of angle is driven by tactile trials, visuotactile trials, or both (it is likely that it is driven by tactile trials, as they are more numerous and have a stronger effect, but we cannot be sure without looking at the coefficients)
# FAMILY 2: only angle and history effects (e.g. hit/miss, choice, angle of the previous trial)
features_4 = ['angle', 'tactile', 'visuotactile', 'angle_tactile', 'angle_visuotactile', 'angle_n-1']
features_5 = ['angle', 'tactile', 'visuotactile', 'angle_tactile', 'angle_visuotactile', 'action_n-1']
features_6 = ['angle', 'tactile', 'visuotactile', 'angle_tactile', 'angle_visuotactile', 'angle_n-1', 'action_n-1']
features_7 = ['angle', 'tactile', 'visuotactile', 'angle_tactile', 'angle_visuotactile', 'success_n-1', 'failure_n-1'] 
features_8 = ['angle', 'tactile', 'visuotactile', 'angle_tactile', 'angle_visuotactile', 'angle_n-1', 'success_n-1', 'failure_n-1']
# ---------------------------------------------------------
features_10 = ['angle']
features_11 = ['angle', 'action_n-1']
features_12 = ['angle', 'angle_n-1'] 
features_13 = ['angle', 'angle_n-1', 'action_n-1']
features_14 = ['angle', 'action_n-1', 'success_n-1']
features_15 = ['angle', 'action_n-1', 'failure_n-1']
features_16 = ['angle', 'action_n-1', 'angle_n-1', 'success_n-1']
features_17 = ['angle','angle_n-1', 'hit_n-1', 'failure_n-1', 'success_n-1']

#FAMILY 1
#models = [features_0, features_1, features_2, features_3]
# FAMILY 2
#models = [features_4, features_5, features_6, features_7, features_8]
models = [features_17]
results = []
for model in models:
    result = an.analyze_individual_rats_ale_N(features_keys=model, df_processed=df, k=None, scaling='cont_only')
    # result is a dict with one key per rat, and each key is another dict with keys: 
    # 'data', 'model', 'scaler', 'feature_names', 'train_accuracy', 'n_trials', 'coefficients', 'cross_validation_scores'
    results.append(result)

# -----------------------------
# PLOT RESULTS
# -----------------------------
# ---- build 4 dfs in a list + one shared beta-column order (union across models) ----
dfs=[flatten_model_results(m) for m in results] # dfs is a list of n dataframes, one per model, with columns: rat_id, feature_name, beta_mean, beta_ci_lower, beta_ci_upper, score_mean, score_std
beta_cols = []
for df_n in dfs:
    beta_cols.extend([col for col in df_n.columns if col.startswith("beta_")])
beta_cols = sorted(set(beta_cols))  # unique and sorted (by alphabetical order, this is also something to change, as it is a bit arbitrary)
score_cols=["score_mean","score_std"]


for i, result in enumerate(results):
    
    ci_df = an.bootstrap_ci(result, n_iterations=1, ci=95,
                        random_state=42, n_jobs=4)
    an.plot_bootstrap_ci(dfs[i], ci_df, model_label=f"Model {i}", beta_cols=beta_cols)
    an.plot_forest_ci(ci_df, model_label="")

def extract_stage1_betas(result, features_of_interest=['success_n-1', 'failure_n-1']):
    """
    Extract beta values for specified features from analyze_individual_rats_ale_N output.
    
    Parameters
    ----------
    result : dict
        Output of analyze_individual_rats_ale_N. Keys are rat_ids.
    features_of_interest : list
        Feature names to extract betas for.
    
    Returns
    -------
    pd.DataFrame with columns: rat, beta_success_n-1, beta_failure_n-1
    """
    rows = []
    for rat_id, rat_result in result.items():
        feature_names = rat_result['feature_names']
        coefficients  = rat_result['coefficients']   # numpy array, aligned to feature_names
        
        beta_dict = dict(zip(feature_names, coefficients))
        
        row = {'rat': rat_id}
        for feat in features_of_interest:
            if feat in beta_dict:
                row[f'beta_{feat}'] = beta_dict[feat]
            else:
                row[f'beta_{feat}'] = np.nan   # feature not in this model
        rows.append(row)
    
    return pd.DataFrame(rows).set_index('rat')


# result is results[0] if you ran one model, or whichever model has success_n-1 and failure_n-1
stage1_betas = extract_stage1_betas(results[0])


# MERGE INTO A COMMON LOOK UP TABLE

def build_rat_lookup(psychometric_df, stage1_betas):
    """
    Combine psychometric slopes and stage-1 betas into a single lookup table.
    
    Converts sigma → strength as 1/sigma (larger = more precise).
    
    Returns
    -------
    lookup : pd.DataFrame
        index = rat
        columns = strength_mod1, strength_mod2, strength_mod3,
                  beta_success_n-1, beta_failure_n-1
    
    strength_prev : dict of dicts
        { rat_id: { 1: strength, 2: strength, 3: strength } }
        for fast trial-level lookup in Step 4
    """
    # pivot psychometric_df so each rat has one row with three strength columns
    psych_wide = psychometric_df[psychometric_df['fit_success']][['rat', 'mod', 'sigma']].copy()
    psych_wide['strength'] = 1 / psych_wide['sigma']
    psych_wide = psych_wide.pivot(index='rat', columns='mod', values='strength')
    psych_wide.columns = [f'strength_mod{int(c)}' for c in psych_wide.columns]

    # merge with stage1 betas (both indexed by rat)
    lookup = psych_wide.join(stage1_betas, how='inner')

    print("Lookup table:")
    print(lookup.round(4))
    print(f"\nAll rats present: {len(lookup) == 12}")

    # also build a nested dict for fast row-level access in Step 4
    strength_dict = {}
    for rat_id, row in lookup.iterrows():
        strength_dict[rat_id] = {
            1: row['strength_mod1'],
            2: row['strength_mod2'],
            3: row['strength_mod3']
        }

    return lookup, strength_dict





lookup, strength_dict = build_rat_lookup(psychometric_df, stage1_betas)


def add_strength_prev(df, strength_dict, mod_col='mod_n-1', rat_col='rat'):
    
    df = df.copy()
    
    # flatten strength_dict into a dataframe for merging
    strength_rows = []
    for rat_id, mod_strengths in strength_dict.items():
        for mod, strength in mod_strengths.items():
            strength_rows.append({rat_col: rat_id, mod_col: mod, 'strength_prev': strength})
    strength_lookup_df = pd.DataFrame(strength_rows)
    
    # reset index in case rat is the index
    if df.index.name == rat_col:
        df = df.reset_index()
    
    df = df.merge(strength_lookup_df, on=[rat_col, mod_col], how='left')
    
    # sanity checks
    n_missing = df['strength_prev'].isna().sum()
    n_total   = len(df)
    print(f"strength_prev: {n_total - n_missing}/{n_total} trials filled ({n_missing} missing)")
    print(f"\nMean strength_prev by modality:")
    print(df.groupby(mod_col)['strength_prev'].mean().round(4))
    
    return df

print(df.index.name)
print(df.columns.tolist()[:15])  # first 15 columns


df = add_strength_prev(df, strength_dict)


def add_interaction_terms(df, lookup, rat_col='rat'):
    """
    Add to the trial dataframe:
    
    Two-way interactions (fixed effects, average amplification):
        success_x_strength = success_n-1 * strength_prev
        failure_x_strength = failure_n-1 * strength_prev
    
    Cross-level interactions (fixed effects, per-rat amplification):
        cl_success = beta_success_rat * success_n-1 * strength_prev
        cl_failure = beta_failure_rat * failure_n-1 * strength_prev
    """
    df = df.copy()

    # map rat-level betas onto every trial of that rat
    df['beta_success_rat'] = df[rat_col].map(lookup['beta_success_n-1'])
    df['beta_failure_rat'] = df[rat_col].map(lookup['beta_failure_n-1'])

    # two-way interaction terms
    df['success_x_strength'] = df['success_n-1'] * df['strength_prev']
    df['failure_x_strength'] = df['failure_n-1'] * df['strength_prev']

    # cross-level interaction terms
    df['cl_success'] = df['beta_success_rat'] * df['success_n-1'] * df['strength_prev']
    df['cl_failure'] = df['beta_failure_rat'] * df['failure_n-1'] * df['strength_prev']

    # sanity check — print means by rat to confirm no NaNs and sensible scale
    print("Mean absolute value of interaction terms by rat:")
    cols = ['success_x_strength', 'failure_x_strength', 'cl_success', 'cl_failure']
    print(df.groupby(rat_col)[cols].apply(lambda x: x.abs().mean()).round(5))

    return df

df = add_interaction_terms(df, lookup)



print("aaaaaaaaaaaaaaa why I can't organize")
print(df.columns.tolist())
print(df.shape)


import os
import pandas as pd
import numpy as np
from pymer4.models import Lmer

# ─────────────────────────────────────────────
# 1. COLUMN RENAMING
# hyphens are illegal in R variable names
# ─────────────────────────────────────────────

RENAME_MAP = {
    'success_n-1':   'success_n1',
    'failure_n-1':   'failure_n1',
    'angle_n-1':     'angle_n1',
    'action_n-1':    'action_n1',
    'hit_n-1':       'hit_n1',
    'mod_n-1':       'mod_n1',
}

def prepare_model_df(df, rename_map=RENAME_MAP):
    """
    Rename hyphenated columns and return a clean copy
    with only the columns needed for the model.
    Drops rows with any NaN in model columns.
    """
    df = df.rename(columns=rename_map)
    
    # all columns that could appear in any model configuration
    keep = [
        'rat', 'action',
        'angle', 'angle_n1',
        'success_n1', 'failure_n1',
        'strength_prev',
        'failure_x_strength',
        'success_x_strength',
        'cl_success', 'cl_failure',
    ]
    
    df = df[keep].dropna()
    print(f"Model dataframe: {len(df)} trials, {df['rat'].nunique()} rats")
    print(f"Dropped {719421 - len(df)} rows with NaN")
    return df


# ─────────────────────────────────────────────
# 2. MODEL CONFIGURATION
# edit this dict to add/remove terms
# ─────────────────────────────────────────────

MODEL_CONFIG = {
    # --- control terms (always on) ---
    'angle':                True,
    'angle_n1':             True,

    # --- baseline history terms ---
    'success_n1':           True,
    'failure_n1':           True,

    # --- main effect of precision ---
    'strength_prev':        True,

    # --- population-level amplification ---
    'failure_x_strength':   True,
    'success_x_strength':   False,   # off — sign cancellation problem

    # --- cross-level interactions (core hypothesis) ---
    'cl_success':           True,
    'cl_failure':           True,

    # --- random effects ---
    'random_intercept':     True,
    'random_success_n1':    True,
    'random_failure_n1':    True,
}


# ─────────────────────────────────────────────
# 3. FORMULA BUILDER
# ─────────────────────────────────────────────

def build_formula(config):
    """
    Build lme4 formula string from MODEL_CONFIG.
    
    Returns a string like:
    'action ~ angle + angle_n1 + success_n1 + ... + (1 + success_n1 + failure_n1 | rat)'
    """
    # fixed effects
    fixed_candidates = [
        'angle', 'angle_n1',
        'success_n1', 'failure_n1',
        'strength_prev',
        'failure_x_strength', 'success_x_strength',
        'cl_success', 'cl_failure',
    ]
    fixed_terms = [t for t in fixed_candidates if config.get(t, False)]

    # random effects
    random_terms = []
    if config.get('random_intercept', True):
        random_terms.append('1')
    if config.get('random_success_n1', False):
        random_terms.append('success_n1')
    if config.get('random_failure_n1', False):
        random_terms.append('failure_n1')

    random_str = ' + '.join(random_terms) + ' | rat'

    formula = f"action ~ {' + '.join(fixed_terms)} + ({random_str})"
    print(f"Formula: {formula}")
    return formula


# ─────────────────────────────────────────────
# 4. MODEL FITTING
# ─────────────────────────────────────────────

def fit_mixed_model(model_df, config, family='binomial'):
    """
    Fit logistic mixed model using pymer4 (lme4 backend).
    
    Parameters
    ----------
    model_df : output of prepare_model_df()
    config   : MODEL_CONFIG dict
    family   : 'binomial' for logistic, 'gaussian' for linear
    
    Returns
    -------
    model : fitted Lmer object
        model.coefs   → fixed effects table (estimate, SE, z, p)
        model.ranef   → random effects per rat
        model.AIC     → model AIC
    """
    os.environ['PATH'] += r";C:\Program Files\R\R-4.5.3\bin"
    
    formula = build_formula(config)
    
    print("Fitting model... (this may take a few minutes on 700k trials)")
    model = Lmer(formula, data=model_df, family=family)
    model.fit(
        factors=None,
        permute=False,
        conf_int='Wald',       # faster than 'profile' for large data
        REML=False,            # use ML for fixed effect comparison
        verbose=False,
    )
    
    return model


# ─────────────────────────────────────────────
# 5. RESULTS EXTRACTION
# ─────────────────────────────────────────────

def print_results(model):
    """
    Print fixed effects table and random effect variances.
    """
    print("\n=== FIXED EFFECTS ===")
    print(model.coefs.round(4))
    
    print("\n=== RANDOM EFFECT VARIANCES ===")
    print(model.ranef_var.round(4))
    
    print(f"\nAIC: {model.AIC:.2f}")


# ─────────────────────────────────────────────
# 6. RUN
# ─────────────────────────────────────────────

model_df = prepare_model_df(df)
model    = fit_mixed_model(model_df, MODEL_CONFIG)
print_results(model)
