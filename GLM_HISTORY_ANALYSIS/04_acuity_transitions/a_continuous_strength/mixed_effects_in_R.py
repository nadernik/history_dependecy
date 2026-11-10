import os
os.environ['R_HOME']      = r'C:\Program Files\R\R-4.4.1'
os.environ['PATH']        = r'C:\Program Files\R\R-4.4.1\bin' + ';' + os.environ['PATH']
os.environ['R_LIBS_USER'] = r'C:\Users\alech\AppData\Local\R\win-library\4.4'

from scipy.optimize import curve_fit
from scipy.special import erf
import numpy as np
import pandas as pd
import subprocess
import tempfile
import sys
sys.path.append(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy")
from functions_Ale import flatten_model_results, filter_training_trials
from serial_dependence_analysis import SerialDependenceAnalyzer

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
HISTORY_DEPTH = 1
RSCRIPT       = r'C:\Program Files\R\R-4.4.1\bin\Rscript.exe'
R_LIB         = r'C:\Users\alech\AppData\Local\R\win-library\4.4'
BASE_DIR      = r'C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\GLM_HISTORY_ANALYSIS\04_acuity_transitions\a_continuous_strength'
CSV_PATH      = BASE_DIR + r'\model_data.csv'
RESULTS_PATH  = BASE_DIR + r'\model_results.csv'
RANEF_PATH    = BASE_DIR + r'\model_ranef.csv'

# ─────────────────────────────────────────────
# FUNCTIONS — PSYCHOMETRIC
# ─────────────────────────────────────────────

def fit_psychometric_all_rats(df_processed, rat_col='rat', angle_col='angle',
                               resp_col='action', mod_col='mod', n_bins=19):
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
    rats       = sorted(df_processed[rat_col].unique())
    modalities = sorted(df_processed[mod_col].unique())
    rows = []

    for rat_id in rats:
        for mod in modalities:
            mask   = (df_processed[rat_col] == rat_id) & (df_processed[mod_col] == mod)
            df_sub = df_processed[mask].copy()
            if len(df_sub) < 20:
                rows.append({'rat': rat_id, 'mod': mod, 'fit_success': False,
                             'mu': np.nan, 'sigma': np.nan,
                             'gamma': np.nan, 'lapse': np.nan})
                continue
            angles    = df_sub[angle_col].values.astype(float)
            responses = df_sub[resp_col].values.astype(float)
            bin_idx   = np.clip(np.digitize(angles, bin_edges, right=False) - 1, 0, n_bins - 1)
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
                popt, _ = curve_fit(cumulative_gaussian_lapse, obs_x, obs_p,
                                    p0=p0, bounds=bounds, maxfev=5000)
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


# ─────────────────────────────────────────────
# FUNCTIONS — STAGE 1 BETAS
# ─────────────────────────────────────────────

def extract_stage1_betas(result, features_of_interest=['success_n-1', 'failure_n-1']):
    rows = []
    for rat_id, rat_result in result.items():
        feature_names = rat_result['feature_names']
        coefficients  = rat_result['coefficients']
        beta_dict = dict(zip(feature_names, coefficients))
        row = {'rat': rat_id}
        for feat in features_of_interest:
            row[f'beta_{feat}'] = beta_dict.get(feat, np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index('rat')


# ─────────────────────────────────────────────
# FUNCTIONS — LOOKUP TABLE
# ─────────────────────────────────────────────

def build_rat_lookup(psychometric_df, stage1_betas):
    psych_wide = psychometric_df[psychometric_df['fit_success']][['rat', 'mod', 'sigma']].copy()
    psych_wide['strength'] = 1 / psych_wide['sigma']
    psych_wide = psych_wide.pivot(index='rat', columns='mod', values='strength')
    psych_wide.columns = [f'strength_mod{int(c)}' for c in psych_wide.columns]
    lookup = psych_wide.join(stage1_betas, how='inner')
    print("Lookup table:")
    print(lookup.round(4))
    print(f"\nAll rats present: {len(lookup) == 12}")
    strength_dict = {}
    for rat_id, row in lookup.iterrows():
        strength_dict[rat_id] = {1: row['strength_mod1'],
                                  2: row['strength_mod2'],
                                  3: row['strength_mod3']}
    return lookup, strength_dict


# ─────────────────────────────────────────────
# FUNCTIONS — FEATURE CONSTRUCTION
# ─────────────────────────────────────────────

def add_strength_prev(df, strength_dict, mod_col='mod_n-1', rat_col='rat'):
    df = df.copy()
    strength_rows = []
    for rat_id, mod_strengths in strength_dict.items():
        for mod, strength in mod_strengths.items():
            strength_rows.append({rat_col: rat_id, mod_col: mod, 'strength_prev': strength})
    strength_lookup_df = pd.DataFrame(strength_rows)
    if df.index.name == rat_col:
        df = df.reset_index()
    df = df.merge(strength_lookup_df, on=[rat_col, mod_col], how='left')
    n_missing = df['strength_prev'].isna().sum()
    print(f"strength_prev: {len(df) - n_missing}/{len(df)} trials filled ({n_missing} missing)")
    print(f"Mean strength_prev by modality:")
    print(df.groupby(mod_col)['strength_prev'].mean().round(4))
    return df


def add_interaction_terms(df, lookup, rat_col='rat'):
    df = df.copy()
    df['beta_success_rat'] = df[rat_col].map(lookup['beta_success_n-1'])
    df['beta_failure_rat'] = df[rat_col].map(lookup['beta_failure_n-1'])
    df['success_x_strength'] = df['success_n-1'] * df['strength_prev']
    df['failure_x_strength'] = df['failure_n-1'] * df['strength_prev']
    df['cl_success'] = df['beta_success_rat'] * df['success_n-1'] * df['strength_prev']
    df['cl_failure'] = df['beta_failure_rat'] * df['failure_n-1'] * df['strength_prev']
    cols = ['success_x_strength', 'failure_x_strength', 'cl_success', 'cl_failure']
    print("Mean absolute value of interaction terms by rat:")
    print(df.groupby(rat_col)[cols].apply(lambda x: x.abs().mean()).round(5))
    return df


def prepare_model_df(df):
    rename_map = {
        'success_n-1': 'success_n1',
        'failure_n-1': 'failure_n1',
        'angle_n-1':   'angle_n1',
    }
    df = df.rename(columns=rename_map)
    keep = ['rat', 'action', 'angle', 'angle_n1',
            'success_n1', 'failure_n1', 'strength_prev',
            'failure_x_strength', 'success_x_strength',
            'cl_success', 'cl_failure']
    df = df[keep].dropna()
    print(f"Model dataframe: {len(df)} trials, {df['rat'].nunique()} rats")
    return df


# ─────────────────────────────────────────────
# FUNCTIONS — MODEL CONFIG AND FORMULA
# ─────────────────────────────────────────────

MODEL_CONFIG = {
    'angle':              True,
    'angle_n1':           True,
    'success_n1':         True,
    'failure_n1':         True,
    'strength_prev':      True,
    'failure_x_strength': True,
    'success_x_strength': False,
    'cl_success':         True,
    'cl_failure':         True,
    'random_intercept':   True,
    'random_success_n1':  True,
    'random_failure_n1':  True,
}


def build_formula(config):
    fixed_candidates = ['angle', 'angle_n1', 'success_n1', 'failure_n1',
                        'strength_prev', 'failure_x_strength',
                        'success_x_strength', 'cl_success', 'cl_failure']
    fixed_terms = [t for t in fixed_candidates if config.get(t, False)]
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
# FUNCTIONS — FIT VIA R SCRIPT
# ─────────────────────────────────────────────

def fit_model_via_r(model_df, config,
                    csv_path=CSV_PATH,
                    results_path=RESULTS_PATH,
                    ranef_path=RANEF_PATH,
                    rscript=RSCRIPT,
                    r_lib=R_LIB):
    """
    Fit logistic mixed model by:
    1. Saving model_df to CSV
    2. Writing and running an R script
    3. Reading results back into Python
    """

    # step 1 — save data
    model_df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")

    # step 2 — build formula
    formula = build_formula(config)

    # step 3 — write R script
    r_script = f"""
library(lme4, lib.loc="{r_lib.replace(chr(92), '/')}")

cat("Reading data...\\n")
df <- read.csv("{csv_path.replace(chr(92), '/')}")
df$rat <- as.factor(df$rat)
cat("Rows:", nrow(df), "\\n")

cat("Fitting model...\\n")
model <- glmer(
    {formula},
    data   = df,
    family = binomial,
    control = glmerControl(optimizer="bobyqa",
                           optCtrl=list(maxfun=2e5))
)

cat("Saving fixed effects...\\n")
fe <- as.data.frame(summary(model)$coefficients)
fe$term <- rownames(fe)
write.csv(fe, "{results_path.replace(chr(92), '/')}", row.names=FALSE)

cat("Saving random effects...\\n")
re <- as.data.frame(ranef(model)$rat)
re$rat <- rownames(re)
write.csv(re, "{ranef_path.replace(chr(92), '/')}", row.names=FALSE)

cat("AIC:", AIC(model), "\\n")
cat("Done.\\n")
"""

    # step 4 — write to temp file and run
    r_script_path = BASE_DIR + r'\model_fit.R'
    with open(r_script_path, 'w') as f:
        f.write(r_script)
    print(f"R script written to {r_script_path}")
    print("Running R... (may take several minutes on 700k trials)")

    result = subprocess.run(
        [rscript, r_script_path],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print("R ERROR:")
        print(result.stderr)
        return None, None

    # step 5 — read results back
    fixed_effects = pd.read_csv(results_path)
    random_effects = pd.read_csv(ranef_path)

    return fixed_effects, random_effects


def print_results(fixed_effects, random_effects):
    print("\n=== FIXED EFFECTS ===")
    print(fixed_effects.round(4).to_string(index=False))
    print("\n=== RANDOM EFFECTS (per rat) ===")
    print(random_effects.round(4).to_string(index=False))


# ─────────────────────────────────────────────
# MAIN — DATA LOADING AND PREPROCESSING
# ─────────────────────────────────────────────

an = SerialDependenceAnalyzer(history_depth=HISTORY_DEPTH, save_figures=False)
an.load_and_preprocess_data()
df = an.create_lagged_features()
df_processed = filter_training_trials(df, method='criterion', criterion=0.80, min_consec=2)

# psychometric slopes
psychometric_df = fit_psychometric_all_rats(df_processed)

# feature engineering
df = df_processed.copy()
df['action_n-1']  = df['action_n-1'] * 2 - 1
df['success_n-1'] = df['hit_n-1'] * df['action_n-1']
df['failure_n-1'] = (1 - df['hit_n-1']) * df['action_n-1']
df['angle']       = df['angle'] - 45
df['angle_n-1']   = df['angle_n-1'] - 45
print('HERE')
print(df.groupby('action')['angle'].mean())
# stage 1 GLM
features_17 = ['angle', 'angle_n-1', 'hit_n-1', 'failure_n-1', 'success_n-1']
result = an.analyze_individual_rats_ale_N(
    features_keys=features_17, df_processed=df, k=None, scaling='cont_only'
)
stage1_betas = extract_stage1_betas(result)

# lookup table
lookup, strength_dict = build_rat_lookup(psychometric_df, stage1_betas)

# add strength and interaction terms
df = add_strength_prev(df, strength_dict)
df = add_interaction_terms(df, lookup)

# ─────────────────────────────────────────────
# MAIN — FIT MODEL
# ─────────────────────────────────────────────

model_df = prepare_model_df(df)
fixed_effects, random_effects = fit_model_via_r(model_df, MODEL_CONFIG)

if fixed_effects is not None:
    print_results(fixed_effects, random_effects)