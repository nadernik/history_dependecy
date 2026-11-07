
suppressPackageStartupMessages({
  library(lme4)
})

args <- commandArgs(trailingOnly = TRUE)
model_dir     <- args[1]
mod_reference <- args[2]

cat("[", format(Sys.time()), "] Loading model.rds...\n", sep="")
model <- readRDS(file.path(model_dir, "model.rds"))
fe <- fixef(model)
vc_fixed <- as.matrix(vcov(model))
cat("  Loaded. n_fixed = ", length(fe), "\n", sep="")
cat("  Fixed-effect names:\n")
for (nm in names(fe)) cat("    ", nm, "\n", sep="")

# ============================================================
# CORRECTED HELPERS
# ============================================================

linear_contrast <- function(fe, V, weights) {
  missing_terms <- setdiff(names(weights), names(fe))
  if (length(missing_terms) > 0) {
    return(list(estimate = NA_real_, se = NA_real_, z = NA_real_,
                p = NA_real_, ci_low = NA_real_, ci_high = NA_real_))
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
       ci_low = est - crit * se, ci_high = est + crit * se)
}

wald_test <- function(fe, V, terms) {
  missing_terms <- setdiff(terms, names(fe))
  if (length(missing_terms) > 0) {
    return(list(chisq = NA_real_, df = length(terms), p = NA_real_))
  }
  b <- fe[terms]
  S <- V[terms, terms, drop = FALSE]
  W <- as.numeric(t(b) %*% solve(S) %*% b)
  p <- 1 - pchisq(W, df = length(terms))
  list(chisq = W, df = length(terms), p = p)
}

# ---- NEW: resolve interaction term names regardless of ordering ----
find_interact <- function(fe_names, base_term, m) {
  candidates <- c(
    paste0(base_term, ":modality", m),
    paste0("modality", m, ":", base_term)
  )
  found <- intersect(candidates, fe_names)
  if (length(found) > 0) return(found[1])
  return(NA_character_)
}

# Find all interaction terms of form base_term x modality<any>, either ordering
find_all_interactions_for <- function(fe_names, base_term) {
  result <- character(0)
  for (nm in fe_names) {
    parts <- strsplit(nm, ":", fixed = TRUE)[[1]]
    if (length(parts) != 2) next
    a <- parts[1]; b <- parts[2]
    if ((a == base_term && startsWith(b, "modality")) ||
        (b == base_term && startsWith(a, "modality"))) {
      result <- c(result, nm)
    }
  }
  result
}

# Detect modality levels
pat <- "^modality(.+)$"
non_ref_mods <- unique(gsub(pat, "\\1",
                             names(fe)[grepl(pat, names(fe)) & !grepl(":", names(fe))]))
mod_levels_all <- c(mod_reference, non_ref_mods)
cat("\n  Modality levels: ", paste(mod_levels_all, collapse = ", "),
    "  (reference = ", mod_reference, ")\n", sep = "")

history_map <- list(
  beta_S  = "success_prev",
  beta_F  = "failure_prev",
  beta_R  = "reward_prev",
  beta_A1 = "angle_prev_c"
)

# ============================================================
# 1. OMNIBUS WALD TESTS
# ============================================================

cat("\n[", format(Sys.time()), "] Computing omnibus Wald tests...\n", sep = "")
omnibus_rows <- list()
for (nm in names(history_map)) {
  h <- history_map[[nm]]
  interact_terms <- find_all_interactions_for(names(fe), h)
  cat("  ", h, ": found ", length(interact_terms),
      " interaction term(s): ", paste(interact_terms, collapse = ", "),
      "\n", sep = "")
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
write.csv(omnibus_df,
          file.path(model_dir, "omnibus_wald_tests.csv"),
          row.names = FALSE)
cat("\n  Wrote omnibus_wald_tests.csv\n")
print(omnibus_df)

# ============================================================
# 2. MODALITY-SPECIFIC EFFECTS
# ============================================================

cat("\n[", format(Sys.time()), "] Computing modality-specific effects...\n", sep = "")

weights_for <- function(fe_names, base_term, m, mod_reference) {
  if (m == mod_reference) {
    w <- 1
    names(w) <- base_term
  } else {
    interact <- find_interact(fe_names, base_term, m)
    if (is.na(interact)) {
      cat("    WARN: no interaction for ", base_term, " x modality", m, "\n", sep = "")
      return(NULL)
    }
    w <- c(1, 1)
    names(w) <- c(base_term, interact)
  }
  w
}

mod_effect_rows <- list()
for (m in mod_levels_all) {
  for (nm in names(history_map)) {
    base_term <- history_map[[nm]]
    w <- weights_for(names(fe), base_term, m, mod_reference)
    if (is.null(w)) next
    res <- linear_contrast(fe, vc_fixed, w)
    mod_effect_rows[[length(mod_effect_rows) + 1]] <- data.frame(
      modality = m, quantity = nm,
      estimate = res$estimate, se = res$se,
      ci_low = res$ci_low, ci_high = res$ci_high,
      z = res$z, p = res$p
    )
  }

  if (m == mod_reference) {
    w_S <- setNames(c(1, 1), c("success_prev", "failure_prev"))
    w_D <- setNames(c(1, -1), c("success_prev", "failure_prev"))
  } else {
    s_int <- find_interact(names(fe), "success_prev", m)
    f_int <- find_interact(names(fe), "failure_prev", m)
    if (is.na(s_int) || is.na(f_int)) {
      cat("    WARN: missing S/F interactions for modality", m, "\n")
      next
    }
    w_S <- setNames(c(1, 1, 1, 1),
                    c("success_prev", "failure_prev", s_int, f_int))
    w_D <- setNames(c(1, -1, 1, -1),
                    c("success_prev", "failure_prev", s_int, f_int))
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
write.csv(mod_effects_df,
          file.path(model_dir, "modality_specific_effects.csv"),
          row.names = FALSE)
cat("\n  Wrote modality_specific_effects.csv\n")
print(mod_effects_df)

# ============================================================
# 3. PAIRWISE MODALITY CONTRASTS
# ============================================================

cat("\n[", format(Sys.time()), "] Computing pairwise modality contrasts...\n", sep = "")

diff_weights_single <- function(fe_names, base_term, m1, m2, mod_reference) {
  weights <- c()
  if (m1 != mod_reference) {
    k <- find_interact(fe_names, base_term, m1)
    if (!is.na(k)) weights[k] <- 1
  }
  if (m2 != mod_reference) {
    k <- find_interact(fe_names, base_term, m2)
    if (!is.na(k)) weights[k] <- -1
  }
  weights
}

pair_rows <- list()
pairs <- combn(mod_levels_all, 2, simplify = FALSE)
for (pair in pairs) {
  m1 <- pair[1]; m2 <- pair[2]
  pair_label <- paste(m1, "-", m2)

  for (nm in names(history_map)) {
    base_term <- history_map[[nm]]
    w <- diff_weights_single(names(fe), base_term, m1, m2, mod_reference)
    if (length(w) == 0) next
    res <- linear_contrast(fe, vc_fixed, w)
    pair_rows[[length(pair_rows) + 1]] <- data.frame(
      pair = pair_label, m1 = m1, m2 = m2, quantity = nm,
      estimate = res$estimate, se = res$se,
      ci_low = res$ci_low, ci_high = res$ci_high,
      z = res$z, p = res$p
    )
  }

  for (deriv_name in c("S", "D")) {
    succ_sign <- 1
    fail_sign <- if (deriv_name == "S") 1 else -1
    weights <- c()
    if (m1 != mod_reference) {
      s1 <- find_interact(names(fe), "success_prev", m1)
      f1 <- find_interact(names(fe), "failure_prev", m1)
      if (!is.na(s1)) weights[s1] <- succ_sign
      if (!is.na(f1)) weights[f1] <- fail_sign
    }
    if (m2 != mod_reference) {
      s2 <- find_interact(names(fe), "success_prev", m2)
      f2 <- find_interact(names(fe), "failure_prev", m2)
      if (!is.na(s2)) weights[s2] <- -succ_sign
      if (!is.na(f2)) weights[f2] <- -fail_sign
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
write.csv(pair_df,
          file.path(model_dir, "modality_pairwise_contrasts.csv"),
          row.names = FALSE)
cat("\n  Wrote modality_pairwise_contrasts.csv\n")
print(pair_df)

cat("\n[", format(Sys.time()), "] Rescue complete.\n", sep = "")
