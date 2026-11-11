
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
