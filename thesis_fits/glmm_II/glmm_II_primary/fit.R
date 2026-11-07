
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

# ---- pooled S_beta / D_beta contrasts (baseline VT-only) ----
# NOTE: in GLMM II these are the effects AT THE REFERENCE MODALITY (visuotactile),
# because the main effects success_prev and failure_prev in the fixed part now
# represent VT-only, not pooled across modalities. To get pooled-across-modality
# estimates comparable to GLMM I, see modality_specific_effects.csv averaged.
fe <- fixef(model)
b_S <- fe["success_prev"]
b_F <- fe["failure_prev"]

S_beta_VT_res <- linear_contrast(fe, vc_fixed,
                                  setNames(c(1, 1), c("success_prev", "failure_prev")))
D_beta_VT_res <- linear_contrast(fe, vc_fixed,
                                  setNames(c(1, -1), c("success_prev", "failure_prev")))

contrasts_df <- data.frame(
  contrast   = c("S_beta_at_VT", "D_beta_at_VT"),
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
  pattern <- paste0("^", h, ":modality")
  interact_terms <- names(fe)[grepl(pattern, names(fe))]
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
    names(w) <- c(base_term, paste0(base_term, ":modality", m))
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
                      paste0("success_prev:modality", m),
                      paste0("failure_prev:modality", m)))
    w_D <- setNames(c(1, -1, 1, -1),
                    c("success_prev", "failure_prev",
                      paste0("success_prev:modality", m),
                      paste0("failure_prev:modality", m)))
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
    k <- paste0(base_term, ":modality", m1)
    weights[k] <- 1
  }
  if (m2 != mod_reference) {
    k <- paste0(base_term, ":modality", m2)
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
      weights[paste0("success_prev:modality", m1)] <- succ_sign
      weights[paste0("failure_prev:modality", m1)] <- fail_sign
    }
    if (m2 != mod_reference) {
      weights[paste0("success_prev:modality", m2)] <- -succ_sign
      weights[paste0("failure_prev:modality", m2)] <- -fail_sign
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

# beta_S at rat r, modality m = beta_S_pop + beta_S:m (0 if VT) + b_{S,r}
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
    s_term <- paste0("success_prev:modality", m)
    f_term <- paste0("failure_prev:modality", m)
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
# evaluated at the REFERENCE MODALITY (visuotactile) - which is where the
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
