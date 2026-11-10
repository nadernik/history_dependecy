
library(lme4, lib.loc="C:/Users/alech/AppData/Local/R/win-library/4.4")

cat("Reading data...\n")
df <- read.csv("C:/dev/projects/Thesis_SISSA/Rats/history_dependecy/GLM_HISTORY_ANALYSIS/04_acuity_transitions/a_continuous_strength/model_data.csv")
df$rat <- as.factor(df$rat)
cat("Rows:", nrow(df), "\n")

cat("Fitting model...\n")
model <- glmer(
    action ~ angle + angle_n1 + success_n1 + failure_n1 + strength_prev + failure_x_strength + cl_success + cl_failure + (1 + success_n1 + failure_n1 | rat),
    data   = df,
    family = binomial,
    control = glmerControl(optimizer="bobyqa",
                           optCtrl=list(maxfun=2e5))
)

cat("Saving fixed effects...\n")
fe <- as.data.frame(summary(model)$coefficients)
fe$term <- rownames(fe)
write.csv(fe, "C:/dev/projects/Thesis_SISSA/Rats/history_dependecy/GLM_HISTORY_ANALYSIS/04_acuity_transitions/a_continuous_strength/model_results.csv", row.names=FALSE)

cat("Saving random effects...\n")
re <- as.data.frame(ranef(model)$rat)
re$rat <- rownames(re)
write.csv(re, "C:/dev/projects/Thesis_SISSA/Rats/history_dependecy/GLM_HISTORY_ANALYSIS/04_acuity_transitions/a_continuous_strength/model_ranef.csv", row.names=FALSE)

cat("AIC:", AIC(model), "\n")
cat("Done.\n")
