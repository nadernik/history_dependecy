
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
