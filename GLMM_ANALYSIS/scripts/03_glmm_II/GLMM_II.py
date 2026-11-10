"""
GLMM II: modality modulation of population-level history effects.

Scientific question
-------------------
Does current-trial modality systematically shift the population-average
history effects? I.e., does the population center in the (beta_S, beta_F)
plane move between tactile, visual, and visuotactile?

This is the population-scale version of the within-rat modality contrasts
from the §4 post-hoc (section_4_GLM_interaction_pooled_modality.py):
  - §4 fits one GLM per rat and asks, for each rat separately, whether that
    rat's history effects shift with modality (12 separate answers).
  - GLMM II asks whether there is a SHARED direction of shift across the
    cohort (ONE answer, with rats as a random sample from the population).
The two questions are related but not interchangeable.

Design decisions (settled)
--------------------------
- Add all four history x modality interaction blocks to the fixed part.
- Drop the two near-zero random slopes (angle_c, angle_prev_c) that GLMM I
  showed had SDs ~ 0.01 or below.
- Keep the correlated (success_prev + failure_prev | rat) block - this is
  the whole S/D geometric machinery.
- No random slopes on the history x modality interactions (12 rats cannot
  support 6 additional random slopes).
- Modality reference: tactile (matches the §4 modality-interaction GLMs, so
  the coefficients and contrasts can be compared directly with §4).
  NOTE: GLMM I (01_glmm_I/GLMM_I.py) now also uses tactile as reference (its saved
  fit was made with visuotactile, equivalent because GLMM I pools history). The fit saved in
  outputs/03_glmm_II/legacy_ref-visuotactile/ was run with visuotactile as reference (terms named
  modalitytactile, modalityvisual); it must be refitted (about 6 h) to get
  tactile-reference output, and glmm_II_post_hoc.py must only be run on a fit
  made with the same reference.
- Omnibus tests: 2-df Wald from the fixed-effect covariance matrix. LRTs
  would require 4 additional ~10h refits; with N=588k, Wald and LRT agree
  to within Monte Carlo error.

Model
-----
Fixed part:
    action ~ angle_c        * modality
           + angle_prev_c   * modality
           + reward_prev    * modality
           + success_prev   * modality
           + failure_prev   * modality

Primary random structure:
    (1 | rat)
    + (0 + reward_prev | rat)
    + (0 + success_prev + failure_prev | rat)   # correlated block
    + (1 | rat:date)

Fallback random structure (if primary fails / singular / near-perfect corr):
    (1 | rat)
    + (0 + reward_prev | rat)
    + (0 + success_prev | rat)                  # independent
    + (0 + failure_prev | rat)                  # independent
    + (1 | rat:date)

Coding (identical to GLMM I)
----------------------------
    action:       1 = vertical, 0 = horizontal
    action_prev: +1 = vertical, -1 = horizontal (signed)
    hit_prev:     1 = rewarded/correct, 0 = unrewarded/error
    reward_prev  = hit_prev - 0.5                 (in {-0.5, +0.5})
    success_prev = hit_prev * action_prev         (in {-1, 0, +1})
    failure_prev = (1 - hit_prev) * action_prev   (in {-1, 0, +1})
    angle_c      = angle - 45
    angle_prev_c = angle_prev - 45
    modality: factor {tactile, visual, visuotactile}, reference=tactile

Preprocessing (identical to GLMM I)
-----------------------------------
    - SerialDependenceAnalyzer (fold=False).
    - Keep only trials with angle and angle_n-1 in [0, 90].
    - 80% training-criterion filter.
    - Drop first-of-session trials.
    - No session-mean centering.

Outputs (written to OUT_DIR/primary_ref-<reference>/ or fallback_ref-<reference>/,
e.g. outputs/03_glmm_II/primary_ref-tactile/)
-------------------------------------------------------------------
Standard (parallel to GLMM I):
    fixed_effects.csv                 term, estimate, SE, z, p (18 terms)
    fixed_vcov.csv                    full fixed-effect covariance matrix
    ranef_variances.csv               variances + covariance from VarCorr
    blups_rat.csv, blups_rat_date.csv per-rat / per-session BLUPs
    predictions.csv                   per-trial fitted probabilities
    contrasts.csv                     S_beta, D_beta AT THE REFERENCE MODALITY
                                      (not pooled across modalities; rows are
                                      named S_beta_at_<reference>, D_beta_at_<reference>)
    heterogeneity.csv                 tau_S, tau_F, tau_SF, tau_S_beta, tau_D_beta
    per_rat_coords.csv                per-rat coords at the REFERENCE MODALITY
                                      (same columns as GLMM I, but not pooled)
    modality_baselines.csv            baseline P(vertical) at angle=0 per modality
    meta.txt, formula.txt, session_info.txt

New in GLMM II:
    omnibus_wald_tests.csv            2-df Wald tests per history x modality block
    modality_specific_effects.csv     beta_S, beta_F, S, D, beta_R, beta_A1 per
                                      modality with SE + CI + z + p
    modality_pairwise_contrasts.csv   pairwise modality differences on the same
                                      quantities with SE + CI + z + p
    per_rat_per_modality_coords.csv   rat x modality coords in (beta_S, beta_F)
    model.rds                         saved fit object for future post-hoc

Python-side:
    glmm_II_full.pkl                  full result dict for painless reload
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
OUT_DIR      = str(Path(__file__).resolve().parents[2] / 'outputs' / '03_glmm_II')   # GLMM_ANALYSIS/outputs/03_glmm_II

# --- Data / preprocessing (identical to GLMM I) ---
HISTORY_DEPTH        = 1
CRITERION_THRESHOLD  = 0.80
CRITERION_MIN_CONSEC = 2
MODALITY_REFERENCE   = 'tactile'   # must match glmm_II_post_hoc.py and the fit that is post-processed

# --- Column names in the input data ---
RAT_COL       = 'rat'
SESSION_COL   = 'date'
MOD_COL       = 'mod'
ACTION_COL    = 'action'
ANGLE_COL     = 'angle'
ANGLE_N1_COL  = 'angle_n-1'
HIT_N1_COL    = 'hit_n-1'
ACTION_N1_COL = 'action_n-1'

MODALITY_LABELS = {1: 'tactile', 2: 'visual', 3: 'visuotactile'}

# --- Fitting knobs ---
NAGQ                = 1
OPTIMIZER           = 'bobyqa'
MAX_FUN             = 100000
RANDOM_SEED         = 42

# --- Fallback triggers ---
FALLBACK_ON_NONCONVERGENCE      = True
FALLBACK_ON_SINGULAR            = True
FALLBACK_IF_ABS_CORR_SF_ABOVE   = 0.98
FALLBACK_IF_VAR_SF_BELOW        = 1e-6

os.environ['R_HOME']      = R_HOME
os.environ['PATH']        = R_BIN + ';' + os.environ.get('PATH', '')
os.environ['R_LIBS_USER'] = R_LIBS_USER


# ============================================================
# IMPORTS
# ============================================================
import sys
import time
import pickle
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


# ============================================================
# DATA LOADING AND PREPROCESSING (identical to GLMM I)
# ============================================================

def load_and_filter():
    sys.path.append(CODE_ROOT)
    from serial_dependence_analysis import SerialDependenceAnalyzer
    from functions_Ale import filter_training_trials

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading raw data...")
    an = SerialDependenceAnalyzer(history_depth=HISTORY_DEPTH, save_figures=False)
    an.load_and_preprocess_data(fold=False)
    df = an.create_lagged_features()

    n_before = len(df)
    df = df[(df[ANGLE_COL] >= 0) & (df[ANGLE_COL] <= 90)].copy()
    df = df[(df['angle_n-1'] >= 0) & (df['angle_n-1'] <= 90)].copy()
    print(f"  Filter: {n_before} -> {len(df)} trials")
    print(f"  raw:   {len(df):>8d} trials, {df[RAT_COL].nunique():>3d} rats")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Applying training criterion...")
    df = filter_training_trials(
        df, method='criterion',
        criterion=CRITERION_THRESHOLD, min_consec=CRITERION_MIN_CONSEC,
    )
    print(f"  after criterion: {len(df):>8d} trials, {df[RAT_COL].nunique():>3d} rats")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Dropping first-of-session trials...")
    df = df.sort_values([RAT_COL, SESSION_COL]).reset_index(drop=True)
    df['_trial_in_sess'] = df.groupby([RAT_COL, SESSION_COL]).cumcount()
    n_first = int((df['_trial_in_sess'] == 0).sum())
    df = df[df['_trial_in_sess'] > 0].drop(columns='_trial_in_sess').reset_index(drop=True)
    print(f"  dropped first-of-session: {n_first:>8d}")
    print(f"  final: {len(df):>8d} trials, {df[RAT_COL].nunique():>3d} rats")

    return df


def prepare_glmm_data(df):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Coding predictors for GLMM...")
    out = pd.DataFrame(index=df.index)

    out['rat']  = df[RAT_COL].astype(str)
    out['date'] = df[SESSION_COL].astype(str)
    out['action'] = df[ACTION_COL].astype(int)

    ap_raw = df[ACTION_N1_COL].astype(float)
    ap_vals = set(pd.Series(ap_raw.dropna().unique()))
    if ap_vals.issubset({0.0, 1.0}):
        ap = ap_raw * 2.0 - 1.0
    elif ap_vals.issubset({-1.0, 1.0}):
        ap = ap_raw
    else:
        raise ValueError(f"action_n-1 must be 0/1 or -1/+1; got {ap_vals}")

    hp = df[HIT_N1_COL].astype(float)
    if not set(pd.Series(hp.dropna().unique())).issubset({0.0, 1.0}):
        raise ValueError("hit_n-1 must be 0/1")

    out['action_prev']  = ap
    out['hit_prev']     = hp
    out['reward_prev']  = hp - 0.5
    out['success_prev'] = hp * ap
    out['failure_prev'] = (1.0 - hp) * ap

    out['angle_c']      = df[ANGLE_COL].astype(float) - 45.0
    out['angle_prev_c'] = df[ANGLE_N1_COL].astype(float) - 45.0

    if not set(df[MOD_COL].dropna().unique()).issubset(set(MODALITY_LABELS.keys())):
        raise ValueError(f"Unexpected mod values: {set(df[MOD_COL].dropna().unique())}")
    out['modality'] = df[MOD_COL].map(MODALITY_LABELS).astype(str)

    required = ['action', 'rat', 'date', 'modality',
                'angle_c', 'angle_prev_c',
                'reward_prev', 'success_prev', 'failure_prev']
    n_before = len(out)
    out = out.dropna(subset=required).reset_index(drop=True)
    n_dropped = n_before - len(out)
    if n_dropped:
        print(f"  dropped {n_dropped} rows with NA")

    print(f"  final for R: {len(out):>8d} trials, "
          f"{out['rat'].nunique()} rats, {out['date'].nunique()} sessions")
    return out


# ============================================================
# FORMULAS
# ============================================================

FIXED_FORMULA = (
    "action ~ "
    "angle_c * modality "
    "+ angle_prev_c * modality "
    "+ reward_prev * modality "
    "+ success_prev * modality "
    "+ failure_prev * modality"
)

# Leaner than GLMM I: dropped (0 + angle_c | rat) and (0 + angle_prev_c | rat)
# which had near-zero SD in GLMM I. Kept the correlated success+failure block.
RANDOM_PRIMARY = (
    " + (1 | rat)"
    " + (0 + reward_prev | rat)"
    " + (0 + success_prev + failure_prev | rat)"      # correlated block
    " + (1 | rat:date)"
)

RANDOM_FALLBACK = (
    " + (1 | rat)"
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
optimizer      <- args[6]
max_fun        <- as.integer(args[7])

t_start <- Sys.time()
cat("[", format(t_start), "] R fit started\n", sep = "")
cat("  formula: ", formula_str, "\n", sep = "")
cat("  nAGQ = ", nAGQ_val, "; optimizer = ", optimizer,
    "; maxfun = ", max_fun, "\n", sep = "")
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
cat("[", format(Sys.time()), "] Fit complete in ",
    sprintf("%.1f min", fit_secs / 60), "\n", sep = "")
flush.console()

# ---- SAVE MODEL for future post-hoc without refit ----
saveRDS(model, file.path(out_dir, "model.rds"))
cat("  Saved model.rds\n")

# ============================================================
# HELPERS FOR LINEAR CONTRASTS AND WALD TESTS
# ============================================================

# Linear contrast: given weights (named numeric vector matching fe names),
# return estimate, SE, z, p, 95% CI for w' beta.
linear_contrast <- function(fe, V, weights) {
  missing_terms <- setdiff(names(weights), names(fe))
  if (length(missing_terms) > 0) {
    return(list(estimate = NA_real_, se = NA_real_, z = NA_real_,
                p = NA_real_, ci_low = NA_real_, ci_high = NA_real_,
                error = paste("Missing:", paste(missing_terms, collapse = ","))))
  }
  b <- fe[names(weights)]
  S <- V[names(weights), names(weights), drop = FALSE]
  w <- as.numeric(weights)
  est <- sum(b * w)
  var_val <- as.numeric(t(w) %*% S %*% w)
  se <- if (var_val >= 0) sqrt(var_val) else NA_real_
  z <- est / se
  p <- 2 * (1 - pnorm(abs(z)))
  crit <- qnorm(0.975)
  list(estimate = est, se = se, z = z, p = p,
       ci_low = est - crit * se, ci_high = est + crit * se, error = NA)
}

# Multi-df Wald: H0: all coefficients in `terms` = 0
wald_test <- function(fe, V, terms) {
  missing_terms <- setdiff(terms, names(fe))
  if (length(missing_terms) > 0) {
    return(list(chisq = NA_real_, df = length(terms), p = NA_real_,
                error = paste("Missing:", paste(missing_terms, collapse = ","))))
  }
  b <- fe[terms]
  S <- V[terms, terms, drop = FALSE]
  W <- as.numeric(t(b) %*% solve(S) %*% b)
  p <- 1 - pchisq(W, df = length(terms))
  list(chisq = W, df = length(terms), p = p, error = NA)
}

# Resolve the name of the history x modality interaction coefficient.
# R's formula parser orders the two parts of an interaction differently
# depending on the term (e.g. "angle_c:modalitytactile" but
# "modalitytactile:success_prev"), so both orderings are tried.
# If neither exists, the default name is returned (linear_contrast then
# reports it as missing instead of silently using a wrong term).
int_name <- function(fe_names, base_term, m) {
  cands <- c(paste0(base_term, ":modality", m),
             paste0("modality", m, ":", base_term))
  found <- intersect(cands, fe_names)
  if (length(found) > 0) found[1] else cands[1]
}

# ============================================================
# STANDARD EXTRACTION (parallel to GLMM I)
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

# ---- fixed-effect covariance ----
vc_fixed <- as.matrix(vcov(model))
vc_df <- as.data.frame(vc_fixed)
vc_df$term <- rownames(vc_fixed)
vc_df <- vc_df[, c("term", setdiff(colnames(vc_df), "term"))]
write.csv(vc_df, file.path(out_dir, "fixed_vcov.csv"), row.names = FALSE)

# ---- random-effect variances ----
vc <- as.data.frame(VarCorr(model))
write.csv(vc, file.path(out_dir, "ranef_variances.csv"), row.names = FALSE)

# ---- BLUPs ----
ranef_list <- ranef(model, condVar = FALSE)
for (grp in names(ranef_list)) {
  re_df <- as.data.frame(ranef_list[[grp]])
  re_df$id <- rownames(re_df); rownames(re_df) <- NULL
  safe <- gsub("[^A-Za-z0-9]", "_", grp)
  write.csv(re_df, file.path(out_dir, paste0("blups_", safe, ".csv")),
            row.names = FALSE)
}

# ---- predictions per trial ----
fitted_p_full     <- fitted(model)
fitted_p_pop_only <- predict(model, re.form = NA, type = "response")
preds_df <- data.frame(
  rat = dat$rat, date = dat$date, modality = dat$modality,
  angle_c = dat$angle_c, angle_prev_c = dat$angle_prev_c,
  hit_prev = dat$hit_prev, action_prev = dat$action_prev,
  reward_prev = dat$reward_prev,
  success_prev = dat$success_prev, failure_prev = dat$failure_prev,
  action = dat$action,
  fitted_p_full = fitted_p_full,
  fitted_p_pop_only = fitted_p_pop_only
)
write.csv(preds_df, file.path(out_dir, "predictions.csv"), row.names = FALSE)

# ---- pooled S_beta / D_beta contrasts (baseline at the reference modality) ----
# NOTE: in GLMM II these are the effects AT THE REFERENCE MODALITY (mod_reference),
# because the main effects success_prev and failure_prev in the fixed part now
# represent the reference modality only, not pooled across modalities. To get pooled-across-modality
# estimates comparable to GLMM I, see modality_specific_effects.csv averaged.
fe <- fixef(model)
b_S <- fe["success_prev"]
b_F <- fe["failure_prev"]

S_beta_VT_res <- linear_contrast(fe, vc_fixed,
                                  setNames(c(1, 1), c("success_prev", "failure_prev")))
D_beta_VT_res <- linear_contrast(fe, vc_fixed,
                                  setNames(c(1, -1), c("success_prev", "failure_prev")))

contrasts_df <- data.frame(
  contrast   = paste0(c("S_beta_at_", "D_beta_at_"), mod_reference),
  estimate   = c(S_beta_VT_res$estimate, D_beta_VT_res$estimate),
  se         = c(S_beta_VT_res$se,       D_beta_VT_res$se),
  ci_low     = c(S_beta_VT_res$ci_low,   D_beta_VT_res$ci_low),
  ci_high    = c(S_beta_VT_res$ci_high,  D_beta_VT_res$ci_high),
  z          = c(S_beta_VT_res$z,        D_beta_VT_res$z),
  p          = c(S_beta_VT_res$p,        D_beta_VT_res$p)
)
contrasts_df$estimate_dp <- 0.5 * contrasts_df$estimate
contrasts_df$se_dp       <- 0.5 * contrasts_df$se
contrasts_df$ci_low_dp   <- 0.5 * contrasts_df$ci_low
contrasts_df$ci_high_dp  <- 0.5 * contrasts_df$ci_high
write.csv(contrasts_df, file.path(out_dir, "contrasts.csv"), row.names = FALSE)

# ---- heterogeneity (same as GLMM I) ----
vc_all <- VarCorr(model)
rat_terms <- vc_all$rat
tau_S <- NA; tau_F <- NA; tau_SF <- NA
if (!is.null(rat_terms)) {
  rn <- rownames(rat_terms); cn <- colnames(rat_terms)
  if ("success_prev" %in% rn) tau_S <- sqrt(rat_terms["success_prev", "success_prev"])
  if ("failure_prev" %in% rn) tau_F <- sqrt(rat_terms["failure_prev", "failure_prev"])
  if ("success_prev" %in% rn && "failure_prev" %in% cn) {
    tau_SF <- rat_terms["success_prev", "failure_prev"]
  }
}
if (is.na(tau_S) || is.na(tau_F)) {
  for (nm in names(vc_all)) {
    m <- vc_all[[nm]]
    if ("success_prev" %in% rownames(m) && is.na(tau_S)) tau_S <- sqrt(m["success_prev", "success_prev"])
    if ("failure_prev" %in% rownames(m) && is.na(tau_F)) tau_F <- sqrt(m["failure_prev", "failure_prev"])
  }
  if (is.na(tau_SF)) tau_SF <- 0
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

# ---- modality baselines ----
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

# ============================================================
# NEW: OMNIBUS WALD TESTS (per history x modality block)
# ============================================================

cat("[", format(Sys.time()), "] Computing omnibus Wald tests...\n", sep = "")
history_vars <- c("success_prev", "failure_prev", "reward_prev", "angle_prev_c")
omnibus_rows <- list()
for (h in history_vars) {
  # Find all interaction terms h:modalityXXX in the fixed effects
  # either ordering: h:modalityXXX or modalityXXX:h
  interact_terms <- names(fe)[grepl(paste0("^", h, ":modality"), names(fe)) |
                              grepl(paste0("^modality[^:]+:", h, "$"), names(fe))]
  if (length(interact_terms) == 0) next
  res <- wald_test(fe, vc_fixed, interact_terms)
  omnibus_rows[[length(omnibus_rows) + 1]] <- data.frame(
    history_var    = h,
    n_interactions = length(interact_terms),
    terms_tested   = paste(interact_terms, collapse = " + "),
    chisq          = res$chisq,
    df             = res$df,
    p              = res$p
  )
}
omnibus_df <- do.call(rbind, omnibus_rows)
write.csv(omnibus_df, file.path(out_dir, "omnibus_wald_tests.csv"), row.names = FALSE)

# ============================================================
# NEW: MODALITY-SPECIFIC EFFECTS
# ============================================================

cat("[", format(Sys.time()), "] Computing modality-specific effects...\n", sep = "")

# Detect all modality levels present in the model
pat <- "^modality(.+)$"
non_ref_mods <- unique(gsub(pat, "\\1",
                             names(fe)[grepl(pat, names(fe)) & !grepl(":", names(fe))]))
mod_levels_all <- c(mod_reference, non_ref_mods)

# Base terms and their labels
history_map <- list(
  beta_S  = "success_prev",
  beta_F  = "failure_prev",
  beta_R  = "reward_prev",
  beta_A1 = "angle_prev_c"
)

# Helper: weights for one history quantity at modality m
weights_for <- function(base_term, m, mod_reference) {
  if (m == mod_reference) {
    w <- 1
    names(w) <- base_term
  } else {
    w <- c(1, 1)
    names(w) <- c(base_term, int_name(names(fe), base_term, m))
  }
  w
}

mod_effect_rows <- list()
for (m in mod_levels_all) {
  # Basic history quantities
  for (nm in names(history_map)) {
    base_term <- history_map[[nm]]
    w <- weights_for(base_term, m, mod_reference)
    res <- linear_contrast(fe, vc_fixed, w)
    mod_effect_rows[[length(mod_effect_rows) + 1]] <- data.frame(
      modality = m, quantity = nm,
      estimate = res$estimate, se = res$se,
      ci_low = res$ci_low, ci_high = res$ci_high,
      z = res$z, p = res$p
    )
  }

  # Derived: S_m = beta_S_m + beta_F_m
  if (m == mod_reference) {
    w_S <- setNames(c(1, 1), c("success_prev", "failure_prev"))
    w_D <- setNames(c(1, -1), c("success_prev", "failure_prev"))
  } else {
    w_S <- setNames(c(1, 1, 1, 1),
                    c("success_prev", "failure_prev",
                      int_name(names(fe), "success_prev", m),
                      int_name(names(fe), "failure_prev", m)))
    w_D <- setNames(c(1, -1, 1, -1),
                    c("success_prev", "failure_prev",
                      int_name(names(fe), "success_prev", m),
                      int_name(names(fe), "failure_prev", m)))
  }
  res_S <- linear_contrast(fe, vc_fixed, w_S)
  res_D <- linear_contrast(fe, vc_fixed, w_D)
  mod_effect_rows[[length(mod_effect_rows) + 1]] <- data.frame(
    modality = m, quantity = "S",
    estimate = res_S$estimate, se = res_S$se,
    ci_low = res_S$ci_low, ci_high = res_S$ci_high,
    z = res_S$z, p = res_S$p
  )
  mod_effect_rows[[length(mod_effect_rows) + 1]] <- data.frame(
    modality = m, quantity = "D",
    estimate = res_D$estimate, se = res_D$se,
    ci_low = res_D$ci_low, ci_high = res_D$ci_high,
    z = res_D$z, p = res_D$p
  )
}
mod_effects_df <- do.call(rbind, mod_effect_rows)
write.csv(mod_effects_df, file.path(out_dir, "modality_specific_effects.csv"),
          row.names = FALSE)

# ============================================================
# NEW: PAIRWISE MODALITY CONTRASTS
# ============================================================

cat("[", format(Sys.time()), "] Computing pairwise modality contrasts...\n", sep = "")

# Contrast: (effect at m1) - (effect at m2). For each history quantity and S/D.
pair_rows <- list()
pairs <- combn(mod_levels_all, 2, simplify = FALSE)

# Helper: build weight vector for (effect at m1) - (effect at m2) on `base_term`
diff_weights_single <- function(base_term, m1, m2, mod_reference) {
  weights <- c()
  if (m1 != mod_reference) {
    k <- int_name(names(fe), base_term, m1)
    weights[k] <- 1
  }
  if (m2 != mod_reference) {
    k <- int_name(names(fe), base_term, m2)
    weights[k] <- -1
  }
  weights
}

for (pair in pairs) {
  m1 <- pair[1]; m2 <- pair[2]
  pair_label <- paste(m1, "-", m2)

  # Basic history quantities
  for (nm in names(history_map)) {
    base_term <- history_map[[nm]]
    w <- diff_weights_single(base_term, m1, m2, mod_reference)
    if (length(w) == 0) next  # both are reference - impossible with distinct m1,m2
    res <- linear_contrast(fe, vc_fixed, w)
    pair_rows[[length(pair_rows) + 1]] <- data.frame(
      pair = pair_label, m1 = m1, m2 = m2, quantity = nm,
      estimate = res$estimate, se = res$se,
      ci_low = res$ci_low, ci_high = res$ci_high,
      z = res$z, p = res$p
    )
  }

  # Derived: Delta_S = Delta_beta_S + Delta_beta_F,  Delta_D = Delta_beta_S - Delta_beta_F
  for (deriv_name in c("S", "D")) {
    succ_sign <- 1
    fail_sign <- if (deriv_name == "S") 1 else -1
    weights <- c()
    if (m1 != mod_reference) {
      weights[int_name(names(fe), "success_prev", m1)] <- succ_sign
      weights[int_name(names(fe), "failure_prev", m1)] <- fail_sign
    }
    if (m2 != mod_reference) {
      weights[int_name(names(fe), "success_prev", m2)] <- -succ_sign
      weights[int_name(names(fe), "failure_prev", m2)] <- -fail_sign
    }
    if (length(weights) == 0) next
    res <- linear_contrast(fe, vc_fixed, weights)
    pair_rows[[length(pair_rows) + 1]] <- data.frame(
      pair = pair_label, m1 = m1, m2 = m2, quantity = deriv_name,
      estimate = res$estimate, se = res$se,
      ci_low = res$ci_low, ci_high = res$ci_high,
      z = res$z, p = res$p
    )
  }
}
pair_df <- do.call(rbind, pair_rows)
write.csv(pair_df, file.path(out_dir, "modality_pairwise_contrasts.csv"),
          row.names = FALSE)

# ============================================================
# NEW: PER-RAT PER-MODALITY COORDS
# ============================================================

# beta_S at rat r, modality m = beta_S_pop + beta_S:m (0 if m is the reference) + b_{S,r}
# same for beta_F
rat_blups <- as.data.frame(ranef(model)$rat)
rat_blups$rat_id <- rownames(rat_blups); rownames(rat_blups) <- NULL

b_S_rat <- if ("success_prev" %in% colnames(rat_blups)) rat_blups$success_prev else rep(0, nrow(rat_blups))
b_F_rat <- if ("failure_prev" %in% colnames(rat_blups)) rat_blups$failure_prev else rep(0, nrow(rat_blups))
beta_S_pop <- as.numeric(fe["success_prev"])
beta_F_pop <- as.numeric(fe["failure_prev"])

per_rat_mod_rows <- list()
for (m in mod_levels_all) {
  if (m == mod_reference) {
    dS_mod <- 0; dF_mod <- 0
  } else {
    s_term <- int_name(names(fe), "success_prev", m)
    f_term <- int_name(names(fe), "failure_prev", m)
    dS_mod <- if (s_term %in% names(fe)) as.numeric(fe[s_term]) else 0
    dF_mod <- if (f_term %in% names(fe)) as.numeric(fe[f_term]) else 0
  }
  for (i in seq_along(rat_blups$rat_id)) {
    bS <- beta_S_pop + dS_mod + b_S_rat[i]
    bF <- beta_F_pop + dF_mod + b_F_rat[i]
    per_rat_mod_rows[[length(per_rat_mod_rows) + 1]] <- data.frame(
      rat_id = rat_blups$rat_id[i], modality = m,
      beta_S = bS, beta_F = bF,
      S = bS + bF, D = bS - bF,
      b_S = b_S_rat[i], b_F = b_F_rat[i]
    )
  }
}
per_rat_mod_df <- do.call(rbind, per_rat_mod_rows)
write.csv(per_rat_mod_df, file.path(out_dir, "per_rat_per_modality_coords.csv"),
          row.names = FALSE)

# ---- pooled per-rat coords (same schema as GLMM I) ----
# For continuity with GLMM I's per_rat_coords.csv, we report the coords
# evaluated at the REFERENCE MODALITY (mod_reference) - which is where the
# main effects live. To get GLMM I's "pooled" numbers, average across the
# three modality-specific coords in per_rat_per_modality_coords.csv.
per_rat_df <- data.frame(
  rat_id = rat_blups$rat_id,
  beta_S_plus_b = beta_S_pop + b_S_rat,
  beta_F_plus_b = beta_F_pop + b_F_rat,
  b_S = b_S_rat, b_F = b_F_rat
)
per_rat_df$S = per_rat_df$beta_S_plus_b + per_rat_df$beta_F_plus_b
per_rat_df$D = per_rat_df$beta_S_plus_b - per_rat_df$beta_F_plus_b
per_rat_df$dp_success = 0.5 * per_rat_df$beta_S_plus_b
per_rat_df$dp_failure = 0.5 * per_rat_df$beta_F_plus_b
per_rat_df$dp_S       = 0.5 * per_rat_df$S
per_rat_df$dp_D       = 0.5 * per_rat_df$D
write.csv(per_rat_df, file.path(out_dir, "per_rat_coords.csv"), row.names = FALSE)

# ============================================================
# META / SESSION INFO
# ============================================================

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
  paste0("n_fixed=",     length(fe)),
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
    meta = read_meta(out_subdir)
    reasons = []
    if not meta:
        return False, ["meta.txt missing (R fit did not complete)"]

    converged = meta.get('converged', 'FALSE').upper() == 'TRUE'
    singular  = meta.get('singular',  'FALSE').upper() == 'TRUE'
    if FALLBACK_ON_NONCONVERGENCE and not converged:
        reasons.append(f"non-convergence: {meta.get('conv_msg', '<no msg>')}")
    if FALLBACK_ON_SINGULAR and singular:
        reasons.append("model is singular (variance component at boundary)")

    het_path = Path(out_subdir) / 'heterogeneity.csv'
    if het_path.exists():
        het = pd.read_csv(het_path).set_index('quantity')['value']
        cor_SF = het.get('cor_SF', np.nan)
        tau_S  = het.get('tau_S',  np.nan)
        tau_F  = het.get('tau_F',  np.nan)
        if np.isfinite(cor_SF) and abs(cor_SF) > FALLBACK_IF_ABS_CORR_SF_ABOVE:
            reasons.append(f"|cor(b_S, b_F)| = {abs(cor_SF):.3f} "
                           f"exceeds {FALLBACK_IF_ABS_CORR_SF_ABOVE}")
        if np.isfinite(tau_S) and tau_S ** 2 < FALLBACK_IF_VAR_SF_BELOW:
            reasons.append(f"Var(b_S) = {tau_S**2:.2e} below "
                           f"{FALLBACK_IF_VAR_SF_BELOW}")
        if np.isfinite(tau_F) and tau_F ** 2 < FALLBACK_IF_VAR_SF_BELOW:
            reasons.append(f"Var(b_F) = {tau_F**2:.2e} below "
                           f"{FALLBACK_IF_VAR_SF_BELOW}")

    return (len(reasons) == 0), reasons


def load_result(out_subdir, label):
    out_subdir = Path(out_subdir)
    result = {'label': label, 'output_dir': str(out_subdir)}

    def _read(name):
        p = out_subdir / name
        return pd.read_csv(p) if p.exists() else None

    result['fixed_effects']            = _read('fixed_effects.csv')
    result['fixed_vcov']               = _read('fixed_vcov.csv')
    result['ranef_variances']          = _read('ranef_variances.csv')
    result['contrasts']                = _read('contrasts.csv')
    result['heterogeneity']            = _read('heterogeneity.csv')
    result['per_rat_coords']           = _read('per_rat_coords.csv')
    result['modality_baselines']       = _read('modality_baselines.csv')
    result['predictions']              = _read('predictions.csv')

    # NEW in GLMM II
    result['omnibus_wald_tests']       = _read('omnibus_wald_tests.csv')
    result['modality_specific_effects'] = _read('modality_specific_effects.csv')
    result['modality_pairwise_contrasts'] = _read('modality_pairwise_contrasts.csv')
    result['per_rat_per_modality_coords'] = _read('per_rat_per_modality_coords.csv')

    result['blups'] = {}
    for p in sorted(out_subdir.glob('blups_*.csv')):
        key = p.stem[len('blups_'):]
        result['blups'][key] = pd.read_csv(p)

    result['meta'] = read_meta(out_subdir)
    formula_path = out_subdir / 'formula.txt'
    result['formula'] = formula_path.read_text().strip() if formula_path.exists() else None
    result['model_rds_path'] = str(out_subdir / 'model.rds')

    return result


def report_result(result):
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

    if result['omnibus_wald_tests'] is not None:
        print("\n-- Omnibus 2-df Wald tests (history x modality blocks) --")
        omn = result['omnibus_wald_tests'].copy()
        for c in ['chisq', 'p']:
            if c in omn.columns:
                omn[c] = omn[c].round(4)
        # Drop the verbose "terms_tested" column for console display
        cols_display = [c for c in omn.columns if c != 'terms_tested']
        print(omn[cols_display].to_string(index=False))

    if result['modality_specific_effects'] is not None:
        print("\n-- Modality-specific effects (log-odds) --")
        me = result['modality_specific_effects'].copy()
        for c in ['estimate', 'se', 'ci_low', 'ci_high', 'z', 'p']:
            if c in me.columns:
                me[c] = me[c].round(4)
        print(me.to_string(index=False))

    if result['modality_pairwise_contrasts'] is not None:
        print("\n-- Pairwise modality contrasts (log-odds) --")
        pc = result['modality_pairwise_contrasts'].copy()
        for c in ['estimate', 'se', 'ci_low', 'ci_high', 'z', 'p']:
            if c in pc.columns:
                pc[c] = pc[c].round(4)
        cols_display = [c for c in pc.columns if c not in ('m1', 'm2')]
        print(pc[cols_display].to_string(index=False))

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
    print("GLMM II - modality modulation of population history effects")
    print("=" * 72)
    print(f"Output directory: {OUT_DIR}")
    print(f"Modality reference: {MODALITY_REFERENCE}")
    print(f"Fallback triggers: nonconv={FALLBACK_ON_NONCONVERGENCE}, "
          f"singular={FALLBACK_ON_SINGULAR}, "
          f"|cor_SF|>{FALLBACK_IF_ABS_CORR_SF_ABOVE}, "
          f"var<{FALLBACK_IF_VAR_SF_BELOW}")

    t0 = time.time()

    df_raw = load_and_filter()
    df_r   = prepare_glmm_data(df_raw)

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    df_r.to_csv(Path(OUT_DIR) / 'input_used.csv', index=False)

    # ---- Primary fit ----
    primary_dir = Path(OUT_DIR) / f'primary_ref-{MODALITY_REFERENCE}'
    ok_launch_primary = fit_one(df_r, FORMULA_PRIMARY, primary_dir, label='primary')

    primary_result = load_result(primary_dir, label='primary')
    primary_ok, primary_reasons = (
        assess_fit(primary_dir) if ok_launch_primary
        else (False, ['R subprocess failed'])
    )

    if primary_ok:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
              f"Primary fit accepted.")
        report_result(primary_result)
        final_result = primary_result
        used_fallback = False
        fallback_result = None
    else:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
              f"Primary fit FAILED quality checks:")
        for r in primary_reasons:
            print(f"  - {r}")
        report_result(primary_result)

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
              f"Launching FALLBACK fit (independent success/failure slopes)...")
        fallback_dir = Path(OUT_DIR) / f'fallback_ref-{MODALITY_REFERENCE}'
        ok_launch_fallback = fit_one(df_r, FORMULA_FALLBACK, fallback_dir,
                                     label='fallback')
        fallback_result = load_result(fallback_dir, label='fallback')
        fallback_ok, fallback_reasons = (
            assess_fit(fallback_dir) if ok_launch_fallback
            else (False, ['R subprocess failed'])
        )
        if not fallback_ok:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Fallback also flagged:")
            for r in fallback_reasons:
                print(f"  - {r}")
            print("Reporting fallback anyway (nothing else to fall back to).")
        report_result(fallback_result)
        final_result = fallback_result
        used_fallback = True

    # ---- Pickle everything ----
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'formula_primary':  FORMULA_PRIMARY,
            'formula_fallback': FORMULA_FALLBACK,
            'modality_reference': MODALITY_REFERENCE,
            'nAGQ': NAGQ,
            'optimizer': OPTIMIZER,
            'max_fun': MAX_FUN,
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

    pkl_path = Path(OUT_DIR) / 'glmm_II_full.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nSaved full result dict to {pkl_path}")

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
          f"All done in {(time.time() - t0)/60:.1f} minutes. "
          f"{'Used FALLBACK.' if used_fallback else 'Used PRIMARY.'}")
