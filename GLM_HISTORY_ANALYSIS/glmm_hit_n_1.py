# ============================================================
# TRIAL-LEVEL GLMM FOR POPULATION INFERENCE ON HISTORY EFFECTS
# ─────────────────────────────────────────────────────────────
# Model:
#   logit P(action=1) =
#       β0
#     + angle terms   (linear or spline; see ANGLE_REPRESENTATION)
#     + angle_n1 terms
#     + β_success · success_n1_c
#     + β_failure · failure_n1_c
#     + β_hit     · hit_n1_c
#     + β_V · 1[mod=V] + β_VT · 1[mod=VT]
#     + β_success:V·[V] + β_success:VT·[VT]  (× success_n1_c)
#     + β_failure:V·[V] + β_failure:VT·[VT]  (× failure_n1_c)
#     + β_hit:V·[V]     + β_hit:VT·[VT]      (× hit_n1_c)
#     + (1 + success_n1_c + failure_n1_c || rat)
#
# Modality treatment-coded, tactile (mod=1) as reference.
# Random effects uncorrelated (||).
# hit_n1 enters only as fixed at this stage; add (0+hit_n1_c|rat)
# as a manual follow-up if this converges cleanly.
# ============================================================

# ============================================================
# IMPORTS AND R ENVIRONMENT 
# ============================================================
import os
os.environ['R_HOME']      = r'C:\Program Files\R\R-4.4.1'
os.environ['PATH']        = r'C:\Program Files\R\R-4.4.1\bin' + ';' + os.environ['PATH']
os.environ['R_LIBS_USER'] = r'C:\Users\alech\AppData\Local\R\win-library\4.4'

import sys, re, subprocess, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm, shapiro

sys.path.append(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy")
from functions_Ale import filter_training_trials
from serial_dependence_analysis import SerialDependenceAnalyzer

RSCRIPT_EXE = r'C:\Program Files\R\R-4.4.1\bin\Rscript.exe'
print(subprocess.run([RSCRIPT_EXE, '--version'], capture_output=True, text=True).stderr)

# ============================================================
# CONFIG
# ============================================================
RAT_COL       = 'rat'
SESSION_COL   = 'date'
MOD_COL       = 'mod'
ACTION_COL    = 'action'
ANGLE_COL     = 'angle'
ANGLE_N1_COL  = 'angle_n-1'
HIT_N1_COL    = 'hit_n-1'
ACTION_N1_COL = 'action_n-1'
HISTORY_DEPTH = 1

# ---- flags -------------------------------------------------
# Session-mean-center the discrete history covariates?
#   True  → within-session interpretation.
#   False → raw signed history, comparable to features_18 betas.
SESSION_CENTER_HISTORY = True

# Angle basis representation. Options:
#   'linear'   : raw (1 term)
#   'cubic'    : x + x³           (symmetric about 0, 2 terms)
#   'poly3'    : x + x² + x³      (3 terms)
#   'spline_k' : natural cubic spline, k df (patsy cr())
# Set from the pre-fit linearity check output (see section 12).
ANGLE_REPRESENTATION    = 'linear'
ANGLE_N1_REPRESENTATION = 'linear'

# Run DHARMa simulated-residual diagnostics in R after the fit?
# Requires the DHARMa package installed in R. Adds ~1-5 min to
# the fit. If package not installed, warns and skips.
USE_DHARMA = False
# ------------------------------------------------------------

OUTPUT_DIR = (
    r'C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\fits\glmm_population'
    + ('_centered' if SESSION_CENTER_HISTORY else '_uncentered')
    + f'_{ANGLE_REPRESENTATION}'
)

MODEL_CONFIG = {
    'nAGQ': 1,                 # drop to 0 if speed is a problem
    'use_dharma': USE_DHARMA,  # passed to R
}


# ============================================================
# 1. LOAD AND PREPROCESS
# ============================================================
an = SerialDependenceAnalyzer(history_depth=HISTORY_DEPTH, save_figures=False)
an.load_and_preprocess_data()
df = an.create_lagged_features()
df = filter_training_trials(df, method='criterion', criterion=0.80, min_consec=2)
print(f"Loaded {len(df)} trials, {df[RAT_COL].nunique()} rats")


# ============================================================
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
# - drop first trial of each session (n-1 is warm-up)
# - signed history: success_n1, failure_n1 ∈ {-1, 0, +1}
# - unsigned outcome: hit_n1               ∈ {0, 1}
# - THEORETICAL angle centering: angle_c = angle - 45
# - modality dummies with T (mod=1) as reference
# ============================================================
def engineer_features(df_in):
    df = df_in.copy().sort_values([RAT_COL, SESSION_COL]).reset_index(drop=True)
    df['_trial_in_sess'] = df.groupby([RAT_COL, SESSION_COL]).cumcount()
    n_first = (df['_trial_in_sess'] == 0).sum()
    df = df[df['_trial_in_sess'] > 0].drop(columns='_trial_in_sess')
    print(f"  Dropped {n_first} first-of-session trials")

    if df[ACTION_N1_COL].between(0, 1).all():
        df[ACTION_N1_COL] = df[ACTION_N1_COL] * 2 - 1

    df['success_n1'] = df[HIT_N1_COL]       * df[ACTION_N1_COL]
    df['failure_n1'] = (1 - df[HIT_N1_COL]) * df[ACTION_N1_COL]
    df['hit_n1']     = df[HIT_N1_COL].astype(float)

    df['angle_c']    = df[ANGLE_COL]    - 45.0
    df['angle_n1_c'] = df[ANGLE_N1_COL] - 45.0

    df['mod_V']  = (df[MOD_COL] == 2).astype(int)
    df['mod_VT'] = (df[MOD_COL] == 3).astype(int)

    print(f"\n  Modality counts (1=T, 2=V, 3=VT):")
    print(df[MOD_COL].value_counts().sort_index().to_string())
    return df

df_feat = engineer_features(df)


# ============================================================
# 3. SESSION-MEAN CENTERING (history covariates only)
# ============================================================
def session_mean_center(df, cols, group=(RAT_COL, SESSION_COL)):
    df = df.copy()
    for c in cols:
        sess_mean = df.groupby(list(group))[c].transform('mean')
        df[f'{c}_c'] = df[c] - sess_mean
    return df

if SESSION_CENTER_HISTORY:
    df_model = session_mean_center(df_feat, cols=['success_n1', 'failure_n1', 'hit_n1'])
    print("\n  SESSION_CENTER_HISTORY = True  → history vars centered within session")
else:
    df_model = df_feat.copy()
    for c in ['success_n1', 'failure_n1', 'hit_n1']:
        df_model[f'{c}_c'] = df_model[c]
    print("\n  SESSION_CENTER_HISTORY = False → history vars NOT centered (raw signed)")


# ============================================================
# 4. BASIS EXPANSION (angle terms)
# ─────────────────────────────────────────────────────────────
# Applied here so downstream (VIF, formula, fit) all see the
# same expanded columns.
# ============================================================
def expand_basis(df, col, representation):
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

df_model, ANGLE_TERMS    = expand_basis(df_model, 'angle_c',    ANGLE_REPRESENTATION)
df_model, ANGLE_N1_TERMS = expand_basis(df_model, 'angle_n1_c', ANGLE_N1_REPRESENTATION)
print(f"\n  angle_c    → {ANGLE_TERMS}")
print(f"  angle_n1_c → {ANGLE_N1_TERMS}")


# ============================================================
# 5. VIF DIAGNOSTIC ON FIXED EFFECTS
# ============================================================
def compute_vif(df_in, columns):
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

vif_cols = ANGLE_TERMS + ANGLE_N1_TERMS + [
    'success_n1_c', 'failure_n1_c', 'hit_n1_c', 'mod_V', 'mod_VT'
]
print("\n" + "="*60)
print("VIF on fixed-effects design matrix")
print("="*60)
print(compute_vif(df_model, vif_cols).round(2).to_string(index=False))
print("VIF > 5: suspicious;  > 10: clear multicollinearity.")


# ============================================================
# 6. FORMULA BUILDER
# ============================================================
FIXED_BASE = ANGLE_TERMS + ANGLE_N1_TERMS + [
    'success_n1_c', 'failure_n1_c', 'hit_n1_c', 'mod_V', 'mod_VT'
]
INTERACTIONS = {
    'success': ['success_n1_c:mod_V', 'success_n1_c:mod_VT'],
    'failure': ['failure_n1_c:mod_V', 'failure_n1_c:mod_VT'],
    'hit':     ['hit_n1_c:mod_V',     'hit_n1_c:mod_VT'],
}
RANDOM = '(1 + success_n1_c + failure_n1_c || rat)'

def build_formula(drop_term=None):
    fixed = list(FIXED_BASE)
    for term, ix in INTERACTIONS.items():
        if term != drop_term:
            fixed += ix
    return f"{ACTION_COL} ~ " + " + ".join(fixed) + f" + {RANDOM}"


# ============================================================
# 7. R FITTING SCRIPT
# ─────────────────────────────────────────────────────────────
# Fits glmer, writes fixed effects, vcov, RE variances, BLUPs,
# per-trial predictions, meta (AIC/BIC/logLik/df/singular).
# Optionally runs DHARMa (if use_dharma flag passed).
# ============================================================
R_FIT_SCRIPT = r"""
suppressPackageStartupMessages(library(lme4))

args <- commandArgs(trailingOnly = TRUE)
data_path   <- args[1]
formula_str <- args[2]
out_dir     <- args[3]
nAGQ_val    <- as.integer(args[4])
use_dharma  <- as.logical(as.integer(args[5]))

t_start <- Sys.time()
cat("[", format(t_start), "] Started\n", sep="")

dat <- read.csv(data_path)
cat("  rows:", nrow(dat), " cols:", ncol(dat), "\n")
cat("[", format(Sys.time()), "] Formula:\n  ", formula_str, "\n", sep="")
cat("  nAGQ =", nAGQ_val, "  use_dharma =", use_dharma, "\n"); flush.console()

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
    sprintf("%.1f min", fit_secs/60), "\n", sep=""); flush.console()

# fixed effects
fixed_df <- as.data.frame(summary(model)$coefficients)
fixed_df$term <- rownames(fixed_df); rownames(fixed_df) <- NULL
write.csv(fixed_df, file.path(out_dir, "fixed_effects.csv"), row.names = FALSE)

# fixed-effects vcov (needed for VT-vs-V contrast)
vcov_mat <- as.matrix(vcov(model))
vcov_df  <- as.data.frame(vcov_mat)
vcov_df$term <- rownames(vcov_mat)
write.csv(vcov_df, file.path(out_dir, "vcov.csv"), row.names = FALSE)

# random-effect variances / covariances
vc <- as.data.frame(VarCorr(model))
write.csv(vc, file.path(out_dir, "ranef_variances.csv"), row.names = FALSE)

# BLUPs
ranef_list <- ranef(model)
for (grp in names(ranef_list)) {
  re_df <- as.data.frame(ranef_list[[grp]])
  re_df$id <- rownames(re_df); rownames(re_df) <- NULL
  safe <- gsub("[^A-Za-z0-9]", "_", grp)
  write.csv(re_df, file.path(out_dir, paste0("blups_", safe, ".csv")),
            row.names = FALSE)
}

# per-trial predictions (with random effects)
preds_df <- data.frame(
  fitted_p   = fitted(model),
  action     = dat$action,
  rat        = dat$rat,
  date       = dat$date,
  mod        = dat$mod,
  success_n1 = dat$success_n1,
  failure_n1 = dat$failure_n1,
  hit_n1     = dat$hit_n1
)
write.csv(preds_df, file.path(out_dir, "predictions.csv"), row.names = FALSE)

# metadata
conv_msgs <- model@optinfo$conv$lme4$messages
meta_lines <- c(
  paste0("aic=",         AIC(model)),
  paste0("bic=",         BIC(model)),
  paste0("logLik=",      as.numeric(logLik(model))),
  paste0("df=",          attr(logLik(model), "df")),
  paste0("n_obs=",       nobs(model)),
  paste0("singular=",    isSingular(model)),
  paste0("converged=",   is.null(conv_msgs)),
  paste0("fit_minutes=", sprintf("%.2f", fit_secs/60))
)
if (!is.null(conv_msgs)) {
  meta_lines <- c(meta_lines,
                  paste0("conv_msg=", paste(conv_msgs, collapse="; ")))
}
writeLines(meta_lines, file.path(out_dir, "meta.txt"))

# ---- optional DHARMa diagnostics ----
if (use_dharma) {
  if (requireNamespace("DHARMa", quietly = TRUE)) {
    suppressPackageStartupMessages(library(DHARMa))
    cat("[", format(Sys.time()), "] Running DHARMa (n=250 sims)...\n", sep="")
    sim_res <- simulateResiduals(model, n = 250, seed = 42)

    ks_test   <- testUniformity(sim_res, plot = FALSE)
    disp_test <- testDispersion(sim_res, plot = FALSE)
    out_test  <- testOutliers(sim_res, plot = FALSE, type = "binomial")

    dh_df <- data.frame(
      test      = c("uniformity_KS", "dispersion", "outliers"),
      statistic = c(as.numeric(ks_test$statistic),
                    as.numeric(disp_test$statistic),
                    as.numeric(out_test$statistic)),
      p_value   = c(as.numeric(ks_test$p.value),
                    as.numeric(disp_test$p.value),
                    as.numeric(out_test$p.value))
    )
    write.csv(dh_df, file.path(out_dir, "dharma_tests.csv"), row.names = FALSE)
    write.csv(data.frame(scaled_residual = sim_res$scaledResiduals),
              file.path(out_dir, "dharma_residuals.csv"), row.names = FALSE)

    png(file.path(out_dir, "dharma_plot.png"), width = 900, height = 600)
    plot(sim_res)
    dev.off()

    cat("[", format(Sys.time()), "] DHARMa done. Tests:\n", sep="")
    print(dh_df)
  } else {
    cat("[", format(Sys.time()),
        "] DHARMa requested but package not installed. Skipping.\n", sep="")
    cat("  Install with: install.packages('DHARMa')\n")
  }
}

cat("[", format(Sys.time()), "] Done\n", sep="")
"""


# ============================================================
# 8. FIT VIA SUBPROCESS
# ============================================================
def _columns_in_formula(formula):
    tokens = re.split(r'[\s+~()|*:]+', formula)
    return [t for t in tokens if t and not t.replace('.', '').replace('-', '').isdigit()]

def fit_model(df, formula, config, output_dir, rscript_exe=RSCRIPT_EXE):
    """Fit a glmer via subprocess. Saves to output_dir and returns
    a result dict with fixed, vcov, ranef_var, blups, meta."""
    print(f"\n--- Fitting ---")
    print(formula)

    cols = _columns_in_formula(formula)
    cols = [c for c in cols if c in df.columns]
    df_fit = df.dropna(subset=cols)
    print(f"N trials in fit: {len(df_fit)}")

    tmpdir = Path(output_dir)
    tmpdir.mkdir(parents=True, exist_ok=True)

    data_path   = tmpdir / 'data.csv'
    script_path = tmpdir / 'fit.R'
    df_fit.to_csv(data_path, index=False)
    script_path.write_text(R_FIT_SCRIPT)

    cmd = [rscript_exe, str(script_path), str(data_path),
           formula, str(tmpdir),
           str(config.get('nAGQ', 1)),
           str(int(config.get('use_dharma', False)))]
    print(f"Running Rscript with nAGQ={config.get('nAGQ',1)}, "
          f"use_dharma={config.get('use_dharma',False)}...")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in iter(process.stdout.readline, ''):
        print(line, end='', flush=True)
    process.stdout.close()
    if process.wait() != 0:
        raise RuntimeError("R fitting failed")

    fixed   = pd.read_csv(tmpdir / 'fixed_effects.csv')
    vcov    = pd.read_csv(tmpdir / 'vcov.csv').set_index('term')
    ranef_v = pd.read_csv(tmpdir / 'ranef_variances.csv')
    meta = {}
    for line in (tmpdir / 'meta.txt').read_text().splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            meta[k.strip()] = v.strip()
    blups = {p.stem[len('blups_'):]: pd.read_csv(p)
             for p in tmpdir.glob('blups_*.csv')}

    result = {
        'fixed': fixed, 'vcov': vcov, 'ranef_var': ranef_v,
        'blups': blups, 'meta': meta,
        'formula': formula, 'n_fit': len(df_fit),
        'output_dir': str(tmpdir),
    }
    with open(tmpdir / 'result.pkl', 'wb') as f:
        pickle.dump(result, f)
    print(f"Saved to {tmpdir}")
    return result


def report(result):
    """Pretty-print + spec-based convergence check."""
    print("\n=== FIXED EFFECTS ===")
    fixed = result['fixed'].copy()
    for c in fixed.columns:
        if pd.api.types.is_numeric_dtype(fixed[c]):
            fixed[c] = fixed[c].round(4)
    cols = ['term'] + [c for c in fixed.columns if c != 'term']
    print(fixed[cols].to_string(index=False))

    print("\n=== RANDOM EFFECT VARIANCES ===")
    print(result['ranef_var'].round(4).to_string(index=False))

    print("\n=== META ===")
    for k, v in result['meta'].items():
        print(f"  {k}: {v}")
    print(f"  n_fit: {result['n_fit']}")

    print("\n=== CONVERGENCE STATUS (spec) ===")
    singular  = result['meta'].get('singular', 'NA')  == 'TRUE'
    converged = result['meta'].get('converged', 'NA') == 'TRUE'
    vc = result['ranef_var']
    sd_rows = vc[vc['var2'].isna()][['grp', 'var1', 'sdcor']].dropna()
    min_sd = sd_rows['sdcor'].min() if len(sd_rows) else np.nan
    sd_ok = (sd_rows['sdcor'] > 0.01).all() if len(sd_rows) else False
    max_se = result['fixed']['Std. Error'].max()
    se_ok = max_se < 5
    print(f"  optimizer converged:      {converged}")
    print(f"  singular fit:             {singular}")
    print(f"  all sd > 0.01:            {sd_ok}   (min = {min_sd:.4f})")
    print(f"  max fixed-eff SE < 5:     {se_ok}   (max = {max_se:.4f})")
    clean = converged and not singular and sd_ok and se_ok
    print(f"  → CLEAN CONVERGENCE:      {clean}")
    if not clean:
        print("\n  Per spec, fallback = drop modality interactions on the")
        print("  slower-converging history term (typically the smaller-σ one)")
        print("  and refit. Handing back for you to decide.")
    return clean


# ============================================================
# 9. PAIRWISE CONTRASTS
# ============================================================
def _wald_row(term, contrast, est, se):
    z = est / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return {'term': term, 'contrast': contrast,
            'estimate': est, 'SE': se,
            'CI_low':  est - 1.96*se, 'CI_high': est + 1.96*se,
            'z': z, 'p': p}

def pairwise_contrasts(result):
    fixed = result['fixed'].set_index('term')
    vcov  = result['vcov']
    est_c, se_c = 'Estimate', 'Std. Error'
    rows = []
    for label, X in [('success', 'success_n1_c'),
                     ('failure', 'failure_n1_c'),
                     ('hit',     'hit_n1_c')]:
        base, v_ix, vt_ix = X, f'{X}:mod_V', f'{X}:mod_VT'
        rows.append(_wald_row(label, 'T (baseline)',
                              fixed.loc[base, est_c], fixed.loc[base, se_c]))
        rows.append(_wald_row(label, 'V vs T',
                              fixed.loc[v_ix, est_c], fixed.loc[v_ix, se_c]))
        rows.append(_wald_row(label, 'VT vs T',
                              fixed.loc[vt_ix, est_c], fixed.loc[vt_ix, se_c]))
        est = fixed.loc[vt_ix, est_c] - fixed.loc[v_ix, est_c]
        var = (vcov.loc[vt_ix, vt_ix] + vcov.loc[v_ix, v_ix]
               - 2 * vcov.loc[v_ix, vt_ix])
        rows.append(_wald_row(label, 'VT vs V', est, np.sqrt(var)))
    return pd.DataFrame(rows)


# ============================================================
# 10. LRT + ΔBIC — REDUCED MODEL REFITS  (call manually)
# ============================================================
def fit_lrt_reduced(df_model, primary_result, base_output_dir, config=MODEL_CONFIG):
    ll_full  = float(primary_result['meta']['logLik'])
    df_full  = int(primary_result['meta']['df'])
    bic_full = float(primary_result['meta']['bic'])

    rows = []
    for term in ['success', 'failure', 'hit']:
        print(f"\n{'='*70}\nReduced model: dropping {term} × modality interactions\n{'='*70}")
        formula_r = build_formula(drop_term=term)
        out_dir_r = Path(base_output_dir) / f'reduced_{term}'
        res_r = fit_model(df_model, formula_r, config, out_dir_r)
        ll_r  = float(res_r['meta']['logLik'])
        df_r  = int(res_r['meta']['df'])
        bic_r = float(res_r['meta']['bic'])
        chi2_val = 2 * (ll_full - ll_r)
        df_diff  = df_full - df_r
        p_val    = 1 - chi2.cdf(chi2_val, df_diff)
        rows.append({
            'term': term, 'chi2': chi2_val, 'df': df_diff, 'p': p_val,
            'logLik_full': ll_full, 'logLik_reduced': ll_r,
            'BIC_full': bic_full, 'BIC_reduced': bic_r,
            'dBIC': bic_r - bic_full,  # positive → full preferred
        })
    lrt_df = pd.DataFrame(rows)
    lrt_df.to_csv(Path(base_output_dir) / 'lrt_omnibus.csv', index=False)
    return lrt_df


# ============================================================
# 11. CALIBRATION CHECK (post-hoc)
# ============================================================
def calibration_check(result):
    preds = pd.read_csv(Path(result['output_dir']) / 'predictions.csv')
    print(f"\nOverall mean action:    {preds['action'].mean():.4f}")
    print(f"Overall mean fitted_p:  {preds['fitted_p'].mean():.4f}")
    print(f"Overall bias:           {preds['fitted_p'].mean() - preds['action'].mean():+.4f}")
    cell = preds.groupby(['rat', 'mod']).agg(
        n=('action', 'size'), raw_p=('action', 'mean'), pred_p=('fitted_p', 'mean'),
    ).reset_index()
    cell['abs_resid'] = (cell['pred_p'] - cell['raw_p']).abs()
    print(f"\nWorst-fitting rat × mod cells:")
    print(cell.sort_values('abs_resid', ascending=False).head(10).round(3).to_string(index=False))
    print(f"\nMean |resid| across cells: {cell['abs_resid'].mean():.4f}")
    print(f"Cells with |resid| > 0.05: {(cell['abs_resid']>0.05).sum()} / {len(cell)}")
    return cell


# ============================================================
# 12. ASSUMPTION CHECK — angle linearity on the logit scale
# ─────────────────────────────────────────────────────────────
# Pre-fit, data-only. Bins the predictor, takes mean(action)
# per bin, transforms to logit. If empirical logit is close to
# the linear fit → linear rep is fine. Curvature → switch
# ANGLE_REPRESENTATION to 'spline_4' (or similar) and re-run.
#
# Marginal check (doesn't control for other covariates). Rough
# but catches obvious non-linearity.
# ============================================================
def check_angle_linearity(df, x_col, y_col=ACTION_COL,
                          n_bins=15, min_per_bin=30, save_dir=None):
    work = df[[x_col, y_col]].dropna().copy()
    work['bin'] = pd.qcut(work[x_col], n_bins, duplicates='drop')
    g = work.groupby('bin', observed=True).agg(
        x_center=(x_col, 'mean'), n=(y_col, 'size'), p=(y_col, 'mean')
    ).reset_index(drop=True)
    g = g[g['n'] >= min_per_bin].copy()
    g['p_adj']     = (g['p'] * g['n'] + 0.5) / (g['n'] + 1)  # smooth 0/1
    g['emp_logit'] = np.log(g['p_adj'] / (1 - g['p_adj']))
    g['se_logit']  = np.sqrt(1/(g['n']*g['p_adj']) + 1/(g['n']*(1-g['p_adj'])))

    coef = np.polyfit(g['x_center'], g['emp_logit'], 1)
    fitted = np.polyval(coef, g['x_center'])
    resid = g['emp_logit'] - fitted
    max_dev = resid.abs().max()
    rmse    = np.sqrt((resid**2).mean())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(g['x_center'], g['emp_logit'], yerr=g['se_logit'],
                fmt='o', capsize=3, label='binned empirical logit')
    xs = np.linspace(g['x_center'].min(), g['x_center'].max(), 100)
    ax.plot(xs, np.polyval(coef, xs), '--', color='gray',
            label=f'linear fit (slope={coef[0]:.3f})')
    ax.set_xlabel(x_col); ax.set_ylabel('empirical logit')
    ax.axhline(0, color='k', lw=0.5, alpha=0.5)
    ax.set_title(f'Linearity on logit: {x_col}\nmax |resid|={max_dev:.3f}  RMSE={rmse:.3f}')
    ax.legend(); fig.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
        fpath = save_dir / f'linearity_{x_col}.png'
        fig.savefig(fpath, dpi=120)
        print(f"  Saved {fpath}")
    plt.close(fig)

    verdict = 'LINEAR OK' if max_dev < 0.3 else ('MILD CURVATURE' if max_dev < 0.6 else 'CLEAR CURVATURE → use splines')
    print(f"  {x_col}: max |resid|={max_dev:.3f}, RMSE={rmse:.3f}  →  {verdict}")
    return g, max_dev, rmse


# ============================================================
# 13. ASSUMPTION CHECK — random-effects normality (Q-Q + Shapiro)
# ─────────────────────────────────────────────────────────────
# For each RE term, plot BLUPs vs normal quantiles and run
# Shapiro-Wilk. Underpowered at n=12, but bimodal BLUPs
# (opposite-sign reward × choice groups) may show up visually.
# ============================================================
def check_ranef_normality(result, save_dir=None):
    from scipy import stats as sstats
    save_dir = Path(save_dir or result['output_dir'])
    rows = []
    for blup_name, blup_df in result['blups'].items():
        term_cols = [c for c in blup_df.columns if c != 'id']
        for term in term_cols:
            values = blup_df[term].dropna().values
            if len(values) < 3:
                continue
            w, p = shapiro(values)
            rows.append({'group': blup_name, 'term': term, 'n': len(values),
                         'shapiro_W': w, 'shapiro_p': p})
            fig, ax = plt.subplots(figsize=(5, 5))
            sstats.probplot(values, dist='norm', plot=ax)
            ax.set_title(f'Q-Q  {blup_name} / {term}\nShapiro W={w:.3f}, p={p:.3f}')
            fig.tight_layout()
            safe = f"{blup_name}_{term}".replace(':','_').replace('(','').replace(')','')
            fpath = save_dir / f'qq_{safe}.png'
            fig.savefig(fpath, dpi=120)
            plt.close(fig)
    df_out = pd.DataFrame(rows)
    print(df_out.round(4).to_string(index=False))
    print(f"\n  Q-Q plots saved to {save_dir}")
    print("  Shapiro underpowered at n=12; visual inspection dominates.")
    return df_out


# ============================================================
# 14. ASSUMPTION CHECK — influential rats
# ─────────────────────────────────────────────────────────────
# Standardize each BLUP by that term's SD. |z| > 2 flagged.
# Cheap proxy for leave-one-rat-out sensitivity.
# ============================================================
def check_influential_rats(result):
    rows = []
    for blup_name, blup_df in result['blups'].items():
        term_cols = [c for c in blup_df.columns if c != 'id']
        for term in term_cols:
            values = blup_df[term]
            sd = values.std()
            for _, row in blup_df.iterrows():
                b = row[term]
                if pd.isna(b): continue
                z = b / sd if sd > 0 else np.nan
                rows.append({
                    'group': blup_name, 'term': term, 'rat_id': row['id'],
                    'BLUP': b, 'z_score': z, 'outlier': abs(z) > 2 if pd.notna(z) else False,
                })
    df_out = pd.DataFrame(rows)
    df_out = df_out.reindex(df_out['z_score'].abs().sort_values(ascending=False).index)
    print("Top |z|:")
    print(df_out.head(15).round(3).to_string(index=False))
    n_out = df_out['outlier'].sum()
    print(f"\n  {n_out} BLUP(s) with |z| > 2 across all terms.")
    return df_out


# ============================================================
# 15. ASSUMPTION CHECK — residual serial dependence
# ─────────────────────────────────────────────────────────────
# Pearson residuals per trial, autocorrelation at lags 1-3 per
# rat×session. Mean autocorr near 0 = lag-1 history capture is
# sufficient. Consistently non-zero = higher-lag structure the
# model isn't absorbing.
#
# Pearson residuals for GLMMs are imperfect; use DHARMa for a
# more principled check (USE_DHARMA=True).
# ============================================================
def check_residual_autocorr(result, max_lag=3, min_trials=20):
    preds = pd.read_csv(Path(result['output_dir']) / 'predictions.csv')
    preds = preds.sort_values(['rat', 'date']).reset_index(drop=True)
    p = preds['fitted_p'].clip(1e-6, 1 - 1e-6)
    preds['resid'] = (preds['action'] - p) / np.sqrt(p * (1 - p))

    rows = []
    for (rat, date), grp in preds.groupby(['rat', 'date']):
        if len(grp) < min_trials: continue
        for lag in range(1, max_lag + 1):
            r = grp['resid'].autocorr(lag=lag)
            rows.append({'rat': rat, 'date': date, 'lag': lag, 'autocorr': r})
    ac = pd.DataFrame(rows)
    summary = ac.groupby('lag').agg(
        mean_ac=('autocorr','mean'),
        sd_ac=('autocorr','std'),
        mean_abs_ac=('autocorr', lambda x: x.abs().mean()),
        n_sessions=('autocorr','count'),
    ).round(4)
    print(summary.to_string())
    print("\n  Interpretation: |mean_ac| near 0 = model captures lag-1 dependence.")
    print("  |mean_ac| > 0.05 at lag > 1 suggests residual higher-order structure.")
    return summary


# ============================================================
# 16. DHARMa summary reader (post-fit, if USE_DHARMA=True)
# ============================================================
def dharma_summary(result):
    out_dir = Path(result['output_dir'])
    tests_path = out_dir / 'dharma_tests.csv'
    if not tests_path.exists():
        print(f"  No DHARMa output in {out_dir} — was USE_DHARMA=True?")
        return None
    tests = pd.read_csv(tests_path)
    print("DHARMa tests:")
    print(tests.round(4).to_string(index=False))
    print(f"\n  Plot saved at {out_dir / 'dharma_plot.png'}")
    print("  Uniformity KS p > 0.05 = residuals uniform (as they should be under a correct model).")
    print("  Dispersion p > 0.05    = no over/underdispersion.")
    print("  Outliers p > 0.05      = no unusual excess of extreme residuals.")
    return tests


# ============================================================
# 17. RUN
# ============================================================
if __name__ == '__main__':

    # ---- PRE-FIT: linearity check on angle terms ----
    print("\n" + "="*70)
    print("PRE-FIT ASSUMPTION CHECK: angle linearity on logit")
    print("="*70)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    check_angle_linearity(df_model, 'angle_c',    save_dir=OUTPUT_DIR)
    check_angle_linearity(df_model, 'angle_n1_c', save_dir=OUTPUT_DIR)
    print(f"\n  Current ANGLE_REPRESENTATION    = '{ANGLE_REPRESENTATION}'")
    print(f"  Current ANGLE_N1_REPRESENTATION = '{ANGLE_N1_REPRESENTATION}'")
    print("  If verdict is 'CLEAR CURVATURE', switch to 'spline_4' at the top of the script.")

    # ---- PRIMARY FIT ----
    formula_full = build_formula(drop_term=None)
    print("\n" + "="*70)
    print("PRIMARY MODEL")
    print("="*70)
    print(formula_full)
    result_full = fit_model(df_model, formula_full, MODEL_CONFIG, OUTPUT_DIR)
    clean = report(result_full)

    # ---- CONTRASTS ----
    print("\n" + "="*70)
    print("PAIRWISE CONTRASTS")
    print("="*70)
    contrasts_df = pairwise_contrasts(result_full)
    print(contrasts_df.round(4).to_string(index=False))
    contrasts_df.to_csv(Path(OUTPUT_DIR) / 'pairwise_contrasts.csv', index=False)

    # ---- POST-FIT ASSUMPTION CHECKS ----
    print("\n" + "="*70)
    print("POST-FIT: calibration")
    print("="*70)
    calibration_check(result_full)

    print("\n" + "="*70)
    print("POST-FIT: random-effects normality (Q-Q + Shapiro)")
    print("="*70)
    check_ranef_normality(result_full)

    print("\n" + "="*70)
    print("POST-FIT: influential rats (|BLUP z| > 2)")
    print("="*70)
    check_influential_rats(result_full)

    print("\n" + "="*70)
    print("POST-FIT: residual serial dependence")
    print("="*70)
    check_residual_autocorr(result_full)

    if USE_DHARMA:
        print("\n" + "="*70)
        print("POST-FIT: DHARMa simulated-residual tests")
        print("="*70)
        dharma_summary(result_full)

    # ---- NEXT STEPS ----
    print("\n" + "="*70)
    print("NEXT STEPS (run manually)")
    print("="*70)
    print("""
  # 1. Omnibus modality LRT + ΔBIC (fits 3 reduced models — ~3× primary fit time):
  lrt_df = fit_lrt_reduced(df_model, result_full, OUTPUT_DIR)
  print(lrt_df.round(4).to_string(index=False))

  # 2. hit-random-slope follow-up. Edit RANDOM to:
  #      RANDOM = '(1 + success_n1_c + failure_n1_c + hit_n1_c || rat)'
  #    then rerun fit_model + report + pairwise_contrasts.

  # 3. If pre-fit linearity check flagged curvature: set
  #      ANGLE_REPRESENTATION = 'spline_4'  (and/or ANGLE_N1_REPRESENTATION)
  #    and re-run — output goes to a new folder so nothing is overwritten.
""")