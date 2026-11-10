"""
GLMM I: population-level history effects, matching the S/D geometry from §§3-4.

Scientific question
-------------------
Across rats, is there a non-zero population-average signed history effect along
the same post-success/post-failure geometry identified in §§3-4, and how much
between-rat heterogeneity surrounds that population average?

Model
-----
Fixed part:
    logit P(action_n = vertical) =
        intercept
        + beta_angle * angle_c                    (per modality via interaction)
        + beta_modality * modality
        + beta_angle_x_modality * (angle_c * modality)
        + beta_angle_prev * angle_prev_c
        + beta_reward * reward_prev
        + beta_success * success_prev
        + beta_failure * failure_prev

Primary random structure:
    (1 | rat)
    + (0 + angle_c | rat)
    + (0 + angle_prev_c | rat)
    + (0 + reward_prev | rat)
    + (0 + success_prev + failure_prev | rat)     <-- the ONE correlated block
    + (1 | rat:date)

Fallback random structure (if primary fails / singular / near-perfect correlation):
    same as primary, but
    (0 + success_prev | rat)
    + (0 + failure_prev | rat)
    replaces the correlated block. Independent success/failure heterogeneity;
    Var(S_beta) = Var(D_beta) becomes a modelling assumption rather than a finding.

Coding
------
    action:       1 = vertical, 0 = horizontal
    action_prev: +1 = vertical, -1 = horizontal (signed)
    hit_prev:     1 = rewarded/correct, 0 = unrewarded/error
    reward_prev = hit_prev - 0.5                  (in {-0.5, +0.5})
    success_prev = hit_prev * action_prev         (in {-1, 0, +1})
    failure_prev = (1 - hit_prev) * action_prev   (in {-1, 0, +1})
    angle_c      = angle - 45
    angle_prev_c = angle_prev - 45
    modality:    factor with levels {tactile, visual, visuotactile},
                 reference = tactile (treatment contrasts; same as §4 and GLMM II)

Preprocessing
-------------
    - Load data via SerialDependenceAnalyzer.
    - Apply 80% training-criterion filter (matching refit of §4).
    - Drop first-of-session trials (n-1 undefined).
    - No session-mean centering (matches §4).

Outputs (written to OUT_DIR/primary_ref-<reference>/ or fallback_ref-<reference>/,
e.g. outputs/01_glmm_I/primary_ref-tactile/)
---------------------------------------------------------------------
    fixed_effects.csv        term, estimate, SE, z, p
    fixed_vcov.csv           full fixed-effect covariance matrix
    ranef_variances.csv      variances and covariance from VarCorr
    blups_rat.csv            per-rat BLUPs for every random term at rat level
    blups_rat_date.csv       per-session intercept BLUPs (large)
    predictions.csv          per-trial fitted_p_full, fitted_p_pop_only, covariates
    contrasts.csv            S_beta, D_beta point + SE + 95% CI (linear contrasts)
    heterogeneity.csv        tau_S, tau_F, tau_SF, tau_S_beta, tau_D_beta
    per_rat_coords.csv       (beta_S + b_S, beta_F + b_F) per rat + implied S, D, Dp
    modality_baselines.csv   baseline P(vertical) at angle=0 per modality
    varcomp_ci.csv           (only if COMPUTE_PROFILE_CI) profile CIs on variances
    meta.txt                 aic, logLik, n_obs, singular, converged, fit_minutes
    formula.txt              the exact formula fitted
    session_info.txt         R and lme4 versions
    result.pkl               Python-side pickle of the full result dict

Notes on profile CIs
--------------------
Profile likelihood CIs for variance components require ~30-60 refits and are
infeasible for this model size (~1 hour per fit → 30-60 hours). Turned off by
default. Wald SEs from VarCorr are unreliable but reported anyway with a caveat.
"""

# ============================================================
# CONFIG
# ============================================================
import os

# --- Paths ---
from pathlib import Path
R_HOME       = r'C:\Program Files\R\R-4.4.1'
R_BIN        = r'C:\Program Files\R\R-4.4.1\bin'
R_LIBS_USER  = r'C:\Users\alech\AppData\Local\R\win-library\4.4'
RSCRIPT_EXE  = r'C:\Program Files\R\R-4.4.1\bin\Rscript.exe'
CODE_ROOT    = str(Path(__file__).resolve().parents[3])   # project root (functions_Ale.py, serial_dependence_analysis.py)

# --- Output ---
OUT_DIR      = str(Path(__file__).resolve().parents[2] / 'outputs' / '01_glmm_I')   # GLMM_ANALYSIS/outputs/01_glmm_I

# --- Data / preprocessing ---
HISTORY_DEPTH        = 1
CRITERION_THRESHOLD  = 0.80              # 80% correct on easy angles
CRITERION_MIN_CONSEC = 2                 # consecutive sessions passing
MODALITY_REFERENCE   = 'tactile'         # treatment contrast reference

# --- Column names in the input data ---
RAT_COL       = 'rat'
SESSION_COL   = 'date'
MOD_COL       = 'mod'
ACTION_COL    = 'action'
ANGLE_COL     = 'angle'
ANGLE_N1_COL  = 'angle_n-1'
HIT_N1_COL    = 'hit_n-1'
ACTION_N1_COL = 'action_n-1'

# Modality integer -> label
MODALITY_LABELS = {1: 'tactile', 2: 'visual', 3: 'visuotactile'}

# --- Fitting knobs ---
NAGQ                = 1              # Laplace; nAGQ>1 is not helpful with this many clusters/trials
OPTIMIZER           = 'bobyqa'
MAX_FUN             = 100000
COMPUTE_PROFILE_CI  = False          # keep OFF for main run — infeasible time cost
RANDOM_SEED         = 42

# --- Fallback triggers ---
# If any of these is true after the primary fit, switch to the fallback structure.
FALLBACK_ON_NONCONVERGENCE      = True
FALLBACK_ON_SINGULAR            = True
FALLBACK_IF_ABS_CORR_SF_ABOVE   = 0.98   # near-perfect correlation = failed covariance
FALLBACK_IF_VAR_SF_BELOW        = 1e-6   # essentially-zero variance on success or failure

# Set R environment before importing anything else that might load it
os.environ['R_HOME']      = R_HOME
os.environ['PATH']        = R_BIN + ';' + os.environ.get('PATH', '')
os.environ['R_LIBS_USER'] = R_LIBS_USER


# ============================================================
# IMPORTS
# ============================================================
import sys
import re
import time
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================

def load_and_filter():
    """Load raw data, apply training-criterion filter, drop first-of-session trials.

    Returns a dataframe with the same columns as SerialDependenceAnalyzer output,
    already lagged, filtered for post-criterion sessions, and with first-of-session
    trials removed.
    """
    sys.path.append(CODE_ROOT)
    from serial_dependence_analysis import SerialDependenceAnalyzer
    from functions_Ale import filter_training_trials

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading raw data...")
    an = SerialDependenceAnalyzer(history_depth=HISTORY_DEPTH, save_figures=False)
    an.load_and_preprocess_data(fold=False)
    df = an.create_lagged_features()
    n_before = len(df)
    df = df[(df[ANGLE_COL] >= 0) & (df[ANGLE_COL] <= 90)].copy()
    n_after_curr = len(df)
    df = df[(df['angle_n-1'] >= 0) & (df['angle_n-1'] <= 90)].copy()
    n_after_prev = len(df)

    print(f"  Filter: {n_before} → {n_after_curr} → {n_after_prev} trials")
    print(f"    dropped by angle filter:     {n_before - n_after_curr}")
    print(f"    dropped by angle_n-1 filter: {n_after_curr - n_after_prev}")
    print(f"  Remaining angle range:     "
        f"[{df[ANGLE_COL].min()}, {df[ANGLE_COL].max()}]")
    print(f"  Remaining angle_n-1 range: "
        f"[{df['angle_n-1'].min()}, {df['angle_n-1'].max()}]")
    print(f"  Unique angles: {sorted(df[ANGLE_COL].dropna().unique())}")
    print(f"  Unique angle_n-1: {sorted(df['angle_n-1'].dropna().unique())}")
    df = df[(df[ANGLE_COL] >= 0) & (df[ANGLE_COL] <= 90)].copy()
    df = df[(df['angle_n-1'] >= 0) & (df['angle_n-1'] <= 90)].copy()

    print(f"  raw:                       {len(df):>8d} trials, {df[RAT_COL].nunique():>3d} rats")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Applying training criterion "
          f"({int(CRITERION_THRESHOLD*100)}% on easy angles, "
          f"{CRITERION_MIN_CONSEC} consecutive sessions)...")
    df = filter_training_trials(
        df,
        method='criterion',
        criterion=CRITERION_THRESHOLD,
        min_consec=CRITERION_MIN_CONSEC,
    )
    print(f"  after criterion filter:    {len(df):>8d} trials, {df[RAT_COL].nunique():>3d} rats")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Dropping first-of-session trials...")
    df = df.sort_values([RAT_COL, SESSION_COL]).reset_index(drop=True)
    df['_trial_in_sess'] = df.groupby([RAT_COL, SESSION_COL]).cumcount()
    n_first = int((df['_trial_in_sess'] == 0).sum())
    df = df[df['_trial_in_sess'] > 0].drop(columns='_trial_in_sess').reset_index(drop=True)
    print(f"  dropped first-of-session:  {n_first:>8d} trials")
    print(f"  final:                     {len(df):>8d} trials, {df[RAT_COL].nunique():>3d} rats")

    return df


def prepare_glmm_data(df):
    """Code predictors and produce the dataframe R will read.

    Column names are R-friendly (no hyphens).
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Coding predictors for GLMM...")
    out = pd.DataFrame(index=df.index)

    # IDs (as strings so R doesn't try to interpret them numerically)
    out['rat']  = df[RAT_COL].astype(str)
    out['date'] = df[SESSION_COL].astype(str)

    # Response
    action_vals = set(pd.Series(df[ACTION_COL].dropna().unique()).astype(int))
    if not action_vals.issubset({0, 1}):
        raise ValueError(f"action must be 0/1; got {action_vals}")
    out['action'] = df[ACTION_COL].astype(int)

    # Previous choice, signed to -1/+1
    ap_raw = df[ACTION_N1_COL].astype(float)
    ap_vals = set(pd.Series(ap_raw.dropna().unique()))
    if ap_vals.issubset({0.0, 1.0}):
        ap = ap_raw * 2.0 - 1.0
    elif ap_vals.issubset({-1.0, 1.0}):
        ap = ap_raw
    else:
        raise ValueError(f"action_n-1 must be 0/1 or -1/+1; got {ap_vals}")

    # Previous hit
    hp = df[HIT_N1_COL].astype(float)
    if not set(pd.Series(hp.dropna().unique())).issubset({0.0, 1.0}):
        raise ValueError("hit_n-1 must be 0/1")

    # Derived history predictors (exactly the §4 coding)
    out['action_prev']  = ap
    out['hit_prev']     = hp
    out['reward_prev']  = hp - 0.5
    out['success_prev'] = hp * ap
    out['failure_prev'] = (1.0 - hp) * ap

    # Angle centering (angle - 45)
    out['angle_c']      = df[ANGLE_COL].astype(float) - 45.0
    out['angle_prev_c'] = df[ANGLE_N1_COL].astype(float) - 45.0

    # Modality: integer -> label -> ordered factor with reference last so
    # relevel() in R sets our chosen reference.
    if not set(df[MOD_COL].dropna().unique()).issubset(set(MODALITY_LABELS.keys())):
        raise ValueError(f"Unexpected mod values: {set(df[MOD_COL].dropna().unique())}")
    out['modality'] = df[MOD_COL].map(MODALITY_LABELS).astype(str)

    # Drop any rows with NA in the predictors we use
    required = [
        'action', 'rat', 'date', 'modality',
        'angle_c', 'angle_prev_c',
        'reward_prev', 'success_prev', 'failure_prev',
    ]
    n_before = len(out)
    out = out.dropna(subset=required).reset_index(drop=True)
    n_dropped = n_before - len(out)
    if n_dropped:
        print(f"  dropped {n_dropped} rows with NA in required predictors")

    print(f"  final for R:               {len(out):>8d} trials")
    print(f"  modalities present:        {sorted(out['modality'].unique())}")
    print(f"  n rats:                    {out['rat'].nunique()}")
    print(f"  n sessions:                {out['date'].nunique()}")

    return out


# ============================================================
# FORMULAS
# ============================================================

FIXED_FORMULA = (
    "action ~ "
    "angle_c * modality "
    "+ angle_prev_c "
    "+ reward_prev "
    "+ success_prev "
    "+ failure_prev"
)

RANDOM_PRIMARY = (
    " + (1 | rat)"
    " + (0 + angle_c | rat)"
    " + (0 + angle_prev_c | rat)"
    " + (0 + reward_prev | rat)"
    " + (0 + success_prev + failure_prev | rat)"      # correlated block
    " + (1 | rat:date)"
)

RANDOM_FALLBACK = (
    " + (1 | rat)"
    " + (0 + angle_c | rat)"
    " + (0 + angle_prev_c | rat)"
    " + (0 + reward_prev | rat)"
    " + (0 + success_prev | rat)"                     # independent
    " + (0 + failure_prev | rat)"                     # independent
    " + (1 | rat:date)"
)

FORMULA_PRIMARY  = FIXED_FORMULA + RANDOM_PRIMARY
FORMULA_FALLBACK = FIXED_FORMULA + RANDOM_FALLBACK


# ============================================================
# R FITTING SCRIPT (embedded)
# ============================================================
R_FIT_SCRIPT = r"""
suppressPackageStartupMessages({
  library(lme4)
  library(Matrix)
})

# ---- args ----
args <- commandArgs(trailingOnly = TRUE)
data_path      <- args[1]
formula_str    <- args[2]
out_dir        <- args[3]
nAGQ_val       <- as.integer(args[4])
mod_reference  <- args[5]
compute_profci <- as.logical(args[6])
optimizer      <- args[7]
max_fun        <- as.integer(args[8])

t_start <- Sys.time()
cat("[", format(t_start), "] R fit started\n", sep = "")
cat("  formula: ", formula_str, "\n", sep = "")
cat("  nAGQ = ", nAGQ_val, "; optimizer = ", optimizer,
    "; maxfun = ", max_fun, "; profile CI = ", compute_profci, "\n", sep = "")
flush.console()

# ---- data ----
dat <- read.csv(data_path, stringsAsFactors = FALSE)
dat$modality <- factor(dat$modality, levels = c("tactile", "visual", "visuotactile"))
dat$modality <- relevel(dat$modality, ref = mod_reference)
dat$rat      <- factor(dat$rat)
dat$date     <- factor(dat$date)
cat("  n_obs = ", nrow(dat), "; n_rats = ", nlevels(dat$rat),
    "; n_sessions = ", nlevels(dat$date), "\n", sep = "")
cat("  modality reference: ", mod_reference,
    " (levels: ", paste(levels(dat$modality), collapse = ", "), ")\n", sep = "")
flush.console()

# ---- fit ----
ctrl <- glmerControl(
  optimizer = optimizer,
  optCtrl   = list(maxfun = max_fun),
  calc.derivs = TRUE
)

t_fit <- Sys.time()
cat("[", format(Sys.time()), "] Starting glmer (verbose)...\n", sep = "")
flush.console()

model <- glmer(
  as.formula(formula_str),
  data    = dat,
  family  = binomial,
  nAGQ    = nAGQ_val,
  control = ctrl,
  verbose = 2
)

fit_secs <- as.numeric(difftime(Sys.time(), t_fit, units = "secs"))
cat("[", format(Sys.time()), "] Fit complete in ", sprintf("%.1f min", fit_secs / 60), "\n", sep = "")
flush.console()

# ============================================================
# EXTRACTION
# ============================================================

# ---- fixed effects ----
fixed_df <- as.data.frame(summary(model)$coefficients)
fixed_df$term <- rownames(fixed_df); rownames(fixed_df) <- NULL
colnames(fixed_df)[colnames(fixed_df) == "Estimate"]   <- "estimate"
colnames(fixed_df)[colnames(fixed_df) == "Std. Error"] <- "se"
colnames(fixed_df)[colnames(fixed_df) == "z value"]    <- "z"
colnames(fixed_df)[colnames(fixed_df) == "Pr(>|z|)"]   <- "p"
fixed_df <- fixed_df[, c("term", "estimate", "se", "z", "p")]
write.csv(fixed_df, file.path(out_dir, "fixed_effects.csv"), row.names = FALSE)

# ---- fixed-effect covariance (for post-hoc linear contrasts) ----
vc_fixed <- as.matrix(vcov(model))
vc_df <- as.data.frame(vc_fixed)
vc_df$term <- rownames(vc_fixed)
vc_df <- vc_df[, c("term", setdiff(colnames(vc_df), "term"))]
write.csv(vc_df, file.path(out_dir, "fixed_vcov.csv"), row.names = FALSE)

# ---- random-effect variances / covariances ----
vc <- as.data.frame(VarCorr(model))
write.csv(vc, file.path(out_dir, "ranef_variances.csv"), row.names = FALSE)

# ---- BLUPs (per rat and per session) ----
ranef_list <- ranef(model, condVar = FALSE)  # condVar off: cheaper and we don't rely on it
for (grp in names(ranef_list)) {
  re_df <- as.data.frame(ranef_list[[grp]])
  re_df$id <- rownames(re_df); rownames(re_df) <- NULL
  safe <- gsub("[^A-Za-z0-9]", "_", grp)
  write.csv(re_df, file.path(out_dir, paste0("blups_", safe, ".csv")), row.names = FALSE)
}

# ---- predictions per trial ----
fitted_p_full     <- fitted(model)                                       # includes random effects
fitted_p_pop_only <- predict(model, re.form = NA, type = "response")     # population-level
preds_df <- data.frame(
  rat            = dat$rat,
  date           = dat$date,
  modality       = dat$modality,
  angle_c        = dat$angle_c,
  angle_prev_c   = dat$angle_prev_c,
  hit_prev       = dat$hit_prev,
  action_prev    = dat$action_prev,
  reward_prev    = dat$reward_prev,
  success_prev   = dat$success_prev,
  failure_prev   = dat$failure_prev,
  action         = dat$action,
  fitted_p_full  = fitted_p_full,
  fitted_p_pop_only = fitted_p_pop_only
)
write.csv(preds_df, file.path(out_dir, "predictions.csv"), row.names = FALSE)

# ============================================================
# DERIVED QUANTITIES
# ============================================================

# ---- S_beta and D_beta contrasts (linear combinations of fixed effects) ----
fe <- fixef(model)
if (!("success_prev" %in% names(fe)) || !("failure_prev" %in% names(fe))) {
  stop("success_prev and failure_prev must be in the fixed-effect names")
}
b_S <- fe["success_prev"]
b_F <- fe["failure_prev"]
V   <- vc_fixed
v_SS <- V["success_prev", "success_prev"]
v_FF <- V["failure_prev", "failure_prev"]
v_SF <- V["success_prev", "failure_prev"]

S_beta      <- as.numeric(b_S + b_F)
D_beta      <- as.numeric(b_S - b_F)
se_S_beta   <- sqrt(v_SS + v_FF + 2 * v_SF)
se_D_beta   <- sqrt(v_SS + v_FF - 2 * v_SF)
z_S <- S_beta / se_S_beta; p_S <- 2 * (1 - pnorm(abs(z_S)))
z_D <- D_beta / se_D_beta; p_D <- 2 * (1 - pnorm(abs(z_D)))
crit <- qnorm(0.975)

contrasts_df <- data.frame(
  contrast = c("S_beta", "D_beta"),
  estimate = c(S_beta, D_beta),
  se       = c(se_S_beta, se_D_beta),
  ci_low   = c(S_beta - crit * se_S_beta, D_beta - crit * se_D_beta),
  ci_high  = c(S_beta + crit * se_S_beta, D_beta + crit * se_D_beta),
  z        = c(z_S, z_D),
  p        = c(p_S, p_D)
)

# Also probability-scale summaries at boundary (Δp = 0.5 * β at logit 0)
contrasts_df$estimate_dp <- 0.5 * contrasts_df$estimate
contrasts_df$se_dp       <- 0.5 * contrasts_df$se
contrasts_df$ci_low_dp   <- 0.5 * contrasts_df$ci_low
contrasts_df$ci_high_dp  <- 0.5 * contrasts_df$ci_high
write.csv(contrasts_df, file.path(out_dir, "contrasts.csv"), row.names = FALSE)

# ---- heterogeneity decomposition ----
# Extract variances / covariance for success_prev and failure_prev at the rat level.
# When success_prev and failure_prev are in the SAME correlated block, VarCorr provides Cov.
# When they are in SEPARATE blocks (fallback), the covariance is 0 by construction.
vc_all <- VarCorr(model)
rat_terms <- vc_all$rat  # matrix
tau_S <- NA; tau_F <- NA; tau_SF <- NA

if (!is.null(rat_terms)) {
  rn <- rownames(rat_terms)
  cn <- colnames(rat_terms)
  if ("success_prev" %in% rn) tau_S  <- sqrt(rat_terms["success_prev", "success_prev"])
  if ("failure_prev" %in% rn) tau_F  <- sqrt(rat_terms["failure_prev", "failure_prev"])
  if ("success_prev" %in% rn && "failure_prev" %in% cn) {
    tau_SF <- rat_terms["success_prev", "failure_prev"]  # covariance, not correlation
  }
}
# If SF are in separate blocks, look up variance components individually
if (is.na(tau_S) || is.na(tau_F)) {
  # Search all VarCorr elements
  for (nm in names(vc_all)) {
    m <- vc_all[[nm]]
    if ("success_prev" %in% rownames(m) && is.na(tau_S)) tau_S <- sqrt(m["success_prev", "success_prev"])
    if ("failure_prev" %in% rownames(m) && is.na(tau_F)) tau_F <- sqrt(m["failure_prev", "failure_prev"])
  }
  if (is.na(tau_SF)) tau_SF <- 0  # independent blocks
}
tau_S_beta_sq <- (tau_S ^ 2) + (tau_F ^ 2) + 2 * tau_SF
tau_D_beta_sq <- (tau_S ^ 2) + (tau_F ^ 2) - 2 * tau_SF
tau_S_beta <- if (!is.na(tau_S_beta_sq) && tau_S_beta_sq >= 0) sqrt(tau_S_beta_sq) else NA
tau_D_beta <- if (!is.na(tau_D_beta_sq) && tau_D_beta_sq >= 0) sqrt(tau_D_beta_sq) else NA
cor_SF <- if (!is.na(tau_S) && !is.na(tau_F) && tau_S > 0 && tau_F > 0) tau_SF / (tau_S * tau_F) else NA

het_df <- data.frame(
  quantity = c("tau_S", "tau_F", "tau_SF_cov", "cor_SF", "tau_S_beta", "tau_D_beta"),
  value    = c(tau_S, tau_F, tau_SF, cor_SF, tau_S_beta, tau_D_beta)
)
write.csv(het_df, file.path(out_dir, "heterogeneity.csv"), row.names = FALSE)

# ---- per-rat coordinates in the (post-success, post-failure) plane ----
# beta_S + b_S, beta_F + b_F for each rat, plus derived S, D and Δp equivalents.
# BLUPs come from ranef(model)$rat. In lme4, all rat-level random terms are
# concatenated into ranef(model)$rat columns.
rat_blups <- as.data.frame(ranef(model)$rat)
rat_blups$rat_id <- rownames(rat_blups); rownames(rat_blups) <- NULL

b_S_rat <- if ("success_prev" %in% colnames(rat_blups)) rat_blups$success_prev else rep(0, nrow(rat_blups))
b_F_rat <- if ("failure_prev" %in% colnames(rat_blups)) rat_blups$failure_prev else rep(0, nrow(rat_blups))

per_rat_df <- data.frame(
  rat_id           = rat_blups$rat_id,
  beta_S_plus_b    = as.numeric(b_S) + b_S_rat,
  beta_F_plus_b    = as.numeric(b_F) + b_F_rat,
  b_S              = b_S_rat,
  b_F              = b_F_rat
)
per_rat_df$S = per_rat_df$beta_S_plus_b + per_rat_df$beta_F_plus_b
per_rat_df$D = per_rat_df$beta_S_plus_b - per_rat_df$beta_F_plus_b
# Probability-scale at boundary
per_rat_df$dp_success = 0.5 * per_rat_df$beta_S_plus_b
per_rat_df$dp_failure = 0.5 * per_rat_df$beta_F_plus_b
per_rat_df$dp_S       = 0.5 * per_rat_df$S
per_rat_df$dp_D       = 0.5 * per_rat_df$D
write.csv(per_rat_df, file.path(out_dir, "per_rat_coords.csv"), row.names = FALSE)

# ---- modality-specific baseline P(vertical) at angle_c = 0 ----
# For each modality m, baseline logit = intercept + modality[m] main effect
# (angle_c contribution is 0 at boundary; history variables set to 0 here)
mod_levels <- levels(dat$modality)
mod_baseline_logit <- setNames(rep(as.numeric(fe["(Intercept)"]), length(mod_levels)), mod_levels)
for (m in mod_levels) {
  if (m == mod_reference) next
  term <- paste0("modality", m)
  if (term %in% names(fe)) {
    mod_baseline_logit[m] <- mod_baseline_logit[m] + as.numeric(fe[term])
  }
}
mod_baseline_df <- data.frame(
  modality       = names(mod_baseline_logit),
  baseline_logit = as.numeric(mod_baseline_logit),
  baseline_prob  = plogis(as.numeric(mod_baseline_logit))
)
write.csv(mod_baseline_df, file.path(out_dir, "modality_baselines.csv"), row.names = FALSE)

# ---- optional: profile CIs on variance components ----
if (compute_profci) {
  cat("[", format(Sys.time()), "] Computing profile CIs (this may take a long time)...\n", sep = "")
  flush.console()
  t_prof <- Sys.time()
  prof_ok <- TRUE
  varcomp_ci_df <- tryCatch({
    ci <- confint(model, method = "profile", parm = "theta_", oldNames = FALSE, quiet = TRUE)
    data.frame(parameter = rownames(ci), ci_low = ci[, 1], ci_high = ci[, 2])
  }, error = function(e) {
    cat("  Profile CI failed: ", conditionMessage(e), "\n", sep = "")
    prof_ok <<- FALSE
    data.frame(parameter = character(0), ci_low = numeric(0), ci_high = numeric(0))
  })
  write.csv(varcomp_ci_df, file.path(out_dir, "varcomp_ci.csv"), row.names = FALSE)
  cat("[", format(Sys.time()), "] Profile CIs finished in ",
      sprintf("%.1f min", as.numeric(difftime(Sys.time(), t_prof, units = "secs")) / 60), "\n", sep = "")
}

# ---- meta ----
conv_msgs <- model@optinfo$conv$lme4$messages
opt_warns <- model@optinfo$warnings
meta_lines <- c(
  paste0("aic=",         AIC(model)),
  paste0("bic=",         BIC(model)),
  paste0("logLik=",      as.numeric(logLik(model))),
  paste0("df_residual=", df.residual(model)),
  paste0("n_obs=",       nobs(model)),
  paste0("n_rats=",      nlevels(dat$rat)),
  paste0("n_sessions=",  nlevels(dat$date)),
  paste0("singular=",    isSingular(model, tol = 1e-4)),
  paste0("converged=",   is.null(conv_msgs)),
  paste0("fit_minutes=", sprintf("%.2f", fit_secs / 60)),
  paste0("optimizer=",   optimizer),
  paste0("nAGQ=",        nAGQ_val)
)
if (!is.null(conv_msgs)) {
  meta_lines <- c(meta_lines,
                  paste0("conv_msg=", paste(conv_msgs, collapse = " | ")))
}
if (length(opt_warns) > 0) {
  meta_lines <- c(meta_lines,
                  paste0("opt_warning=", paste(unlist(opt_warns), collapse = " | ")))
}
writeLines(meta_lines, file.path(out_dir, "meta.txt"))

# ---- record formula and R env ----
writeLines(formula_str, file.path(out_dir, "formula.txt"))
sink(file.path(out_dir, "session_info.txt"))
print(sessionInfo())
sink()

cat("[", format(Sys.time()), "] R fit done\n", sep = "")
"""


# ============================================================
# PYTHON HARNESS
# ============================================================

def fit_one(df_r, formula_str, out_subdir, label):
    """Run one R fit. Returns True on subprocess success (not on model quality)."""
    out_subdir = Path(out_subdir)
    out_subdir.mkdir(parents=True, exist_ok=True)

    data_path   = out_subdir / 'data.csv'
    script_path = out_subdir / 'fit.R'

    df_r.to_csv(data_path, index=False)
    script_path.write_text(R_FIT_SCRIPT, encoding='utf-8')

    cmd = [
        RSCRIPT_EXE, str(script_path),
        str(data_path),
        formula_str,
        str(out_subdir),
        str(NAGQ),
        MODALITY_REFERENCE,
        "TRUE" if COMPUTE_PROFILE_CI else "FALSE",
        OPTIMIZER,
        str(MAX_FUN),
    ]
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Launching R fit ({label})...")
    print(f"  formula: {formula_str}")
    print(f"  out:     {out_subdir}")

    log_path = out_subdir / 'r_console.log'
    with open(log_path, 'w', encoding='utf-8') as log:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in iter(process.stdout.readline, ''):
            print(line, end='', flush=True)
            log.write(line)
        process.stdout.close()
        rc = process.wait()

    if rc != 0:
        print(f"\n  R fit exited with code {rc} (see {log_path})")
        return False
    return True


def read_meta(out_subdir):
    """Parse meta.txt into a dict."""
    meta = {}
    meta_path = Path(out_subdir) / 'meta.txt'
    if not meta_path.exists():
        return meta
    for line in meta_path.read_text().splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            meta[k.strip()] = v.strip()
    return meta


def assess_fit(out_subdir):
    """Return (ok: bool, reasons: list[str]) describing whether primary fit passed our quality bar."""
    meta = read_meta(out_subdir)
    reasons = []

    if not meta:
        return False, ["meta.txt missing (R fit did not complete)"]

    converged = meta.get('converged', 'FALSE').upper() == 'TRUE'
    singular  = meta.get('singular',  'FALSE').upper() == 'TRUE'
    if FALLBACK_ON_NONCONVERGENCE and not converged:
        reasons.append(f"non-convergence: {meta.get('conv_msg', '<no message>')}")
    if FALLBACK_ON_SINGULAR and singular:
        reasons.append("model is singular (some variance component at boundary)")

    # Inspect the SF covariance directly
    het_path = Path(out_subdir) / 'heterogeneity.csv'
    if het_path.exists():
        het = pd.read_csv(het_path).set_index('quantity')['value']
        cor_SF = het.get('cor_SF', np.nan)
        tau_S  = het.get('tau_S',  np.nan)
        tau_F  = het.get('tau_F',  np.nan)
        if np.isfinite(cor_SF) and abs(cor_SF) > FALLBACK_IF_ABS_CORR_SF_ABOVE:
            reasons.append(f"|cor(b_S, b_F)| = {abs(cor_SF):.3f} exceeds {FALLBACK_IF_ABS_CORR_SF_ABOVE}")
        if np.isfinite(tau_S) and tau_S ** 2 < FALLBACK_IF_VAR_SF_BELOW:
            reasons.append(f"Var(b_S) = {tau_S**2:.2e} below {FALLBACK_IF_VAR_SF_BELOW}")
        if np.isfinite(tau_F) and tau_F ** 2 < FALLBACK_IF_VAR_SF_BELOW:
            reasons.append(f"Var(b_F) = {tau_F**2:.2e} below {FALLBACK_IF_VAR_SF_BELOW}")

    return (len(reasons) == 0), reasons


def load_result(out_subdir, label):
    """Read every CSV/txt from an R fit directory into a Python dict."""
    out_subdir = Path(out_subdir)
    result = {'label': label, 'output_dir': str(out_subdir)}

    def _safe_read_csv(name):
        p = out_subdir / name
        return pd.read_csv(p) if p.exists() else None

    result['fixed_effects']        = _safe_read_csv('fixed_effects.csv')
    result['fixed_vcov']           = _safe_read_csv('fixed_vcov.csv')
    result['ranef_variances']      = _safe_read_csv('ranef_variances.csv')
    result['contrasts']            = _safe_read_csv('contrasts.csv')
    result['heterogeneity']        = _safe_read_csv('heterogeneity.csv')
    result['per_rat_coords']       = _safe_read_csv('per_rat_coords.csv')
    result['modality_baselines']   = _safe_read_csv('modality_baselines.csv')
    result['predictions']          = _safe_read_csv('predictions.csv')
    result['varcomp_ci']           = _safe_read_csv('varcomp_ci.csv')

    # All BLUPs files
    result['blups'] = {}
    for p in sorted(out_subdir.glob('blups_*.csv')):
        key = p.stem[len('blups_'):]
        result['blups'][key] = pd.read_csv(p)

    result['meta']    = read_meta(out_subdir)
    formula_path = out_subdir / 'formula.txt'
    result['formula'] = formula_path.read_text().strip() if formula_path.exists() else None

    return result


def report_result(result):
    """Human-readable summary of a fit result."""
    label = result['label']
    print(f"\n{'=' * 72}")
    print(f"RESULT: {label}")
    print(f"{'=' * 72}")

    print("\n-- Meta --")
    for k, v in result['meta'].items():
        print(f"  {k}: {v}")

    if result['fixed_effects'] is not None:
        print("\n-- Fixed effects --")
        fe = result['fixed_effects'].copy()
        for c in ['estimate', 'se', 'z', 'p']:
            if c in fe.columns:
                fe[c] = fe[c].round(4)
        print(fe.to_string(index=False))

    if result['contrasts'] is not None:
        print("\n-- S_beta / D_beta contrasts --")
        c = result['contrasts'].copy()
        for col in c.columns:
            if pd.api.types.is_numeric_dtype(c[col]):
                c[col] = c[col].round(4)
        print(c.to_string(index=False))

    if result['heterogeneity'] is not None:
        print("\n-- Heterogeneity --")
        print(result['heterogeneity'].round(4).to_string(index=False))

    if result['ranef_variances'] is not None:
        print("\n-- Random-effect variances --")
        print(result['ranef_variances'].round(4).to_string(index=False))

    if result['modality_baselines'] is not None:
        print("\n-- Modality baselines (at angle_c = 0) --")
        print(result['modality_baselines'].round(4).to_string(index=False))


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 72)
    print("GLMM I — population history effects (S/D geometry)")
    print("=" * 72)
    print(f"Output directory: {OUT_DIR}")
    print(f"Modality reference: {MODALITY_REFERENCE}")
    print(f"Compute profile CIs: {COMPUTE_PROFILE_CI}")
    print(f"Fallback triggers: non-convergence={FALLBACK_ON_NONCONVERGENCE}, "
          f"singular={FALLBACK_ON_SINGULAR}, |cor_SF|>{FALLBACK_IF_ABS_CORR_SF_ABOVE}, "
          f"var<{FALLBACK_IF_VAR_SF_BELOW}")

    t0 = time.time()

    # ---- Load and prepare ----
    df_raw = load_and_filter()
    df_r   = prepare_glmm_data(df_raw)

    # Save the exact input used, for reproducibility
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    df_r.to_csv(Path(OUT_DIR) / 'input_used.csv', index=False)

    # ---- Primary fit ----
    primary_dir = Path(OUT_DIR) / f'primary_ref-{MODALITY_REFERENCE}'
    ok_launch_primary = fit_one(df_r, FORMULA_PRIMARY, primary_dir, label='primary')

    primary_result = load_result(primary_dir, label='primary')
    primary_ok, primary_reasons = assess_fit(primary_dir) if ok_launch_primary else (False, ['R subprocess failed'])

    if primary_ok:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Primary fit accepted (no fallback triggers fired).")
        report_result(primary_result)
        final_result = primary_result
        used_fallback = False
        fallback_result = None
    else:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Primary fit FAILED quality checks:")
        for r in primary_reasons:
            print(f"  - {r}")
        report_result(primary_result)  # print what we got

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Launching FALLBACK fit "
              f"(independent success/failure random slopes)...")
        fallback_dir = Path(OUT_DIR) / f'fallback_ref-{MODALITY_REFERENCE}'
        ok_launch_fallback = fit_one(df_r, FORMULA_FALLBACK, fallback_dir, label='fallback')
        fallback_result = load_result(fallback_dir, label='fallback')
        fallback_ok, fallback_reasons = assess_fit(fallback_dir) if ok_launch_fallback else (False, ['R subprocess failed'])
        if not fallback_ok:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Fallback also flagged:")
            for r in fallback_reasons:
                print(f"  - {r}")
            print("Reporting fallback anyway (nothing else to fall back to).")
        report_result(fallback_result)
        final_result = fallback_result
        used_fallback = True

    # ---- Pickle everything for painless reload ----
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'formula_primary':  FORMULA_PRIMARY,
            'formula_fallback': FORMULA_FALLBACK,
            'modality_reference': MODALITY_REFERENCE,
            'nAGQ': NAGQ,
            'optimizer': OPTIMIZER,
            'max_fun': MAX_FUN,
            'compute_profile_ci': COMPUTE_PROFILE_CI,
            'criterion_threshold': CRITERION_THRESHOLD,
            'criterion_min_consec': CRITERION_MIN_CONSEC,
            'fallback_on_nonconvergence': FALLBACK_ON_NONCONVERGENCE,
            'fallback_on_singular': FALLBACK_ON_SINGULAR,
            'fallback_if_abs_corr_sf_above': FALLBACK_IF_ABS_CORR_SF_ABOVE,
            'fallback_if_var_sf_below': FALLBACK_IF_VAR_SF_BELOW,
        },
        'primary_ok': primary_ok,
        'primary_reasons': primary_reasons,
        'used_fallback': used_fallback,
        'primary_result': primary_result,
        'fallback_result': fallback_result,
        'final_result': final_result,
        'total_minutes': (time.time() - t0) / 60,
    }

    pkl_path = Path(OUT_DIR) / 'glmm_I_full.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved full result dict to {pkl_path}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
          f"All done in {(time.time() - t0)/60:.1f} minutes total. "
          f"{'Used FALLBACK.' if used_fallback else 'Used PRIMARY.'}")