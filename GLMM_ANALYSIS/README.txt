GLMM_ANALYSIS
=============

Trial-level mixed models (GLMMs, lme4 through Rscript) for the population-level
history effects. This is NOT part of the submitted thesis (the thesis was
written without the GLMM). Per-rat GLMs live in GLM_HISTORY_ANALYSIS and in
thesis_codes/section_4_GLM.

Scripts and outputs mirror each other: scripts/<step>/ writes to outputs/<step>/.
All scripts find the project root from their own location. Run them from the
project root (history_dependecy/): the data are loaded with a relative path.
The R install path (R_HOME, RSCRIPT_EXE, R_LIBS_USER) is hardcoded at the top
of the scripts that call R.

Modality reference
------------------
All scripts use tactile as the reference modality, as the section 4 GLMs do.
Fit folders carry the reference in their name (primary_ref-tactile,
fallback_ref-tactile). Fits made earlier with visuotactile as reference are
kept in legacy_ref-visuotactile/. They have not been refitted.


Order of the analysis
---------------------

00_exploration
  glmm_hit_n_1.py   Early exploratory GLMM: history x modality interactions
                    with an extra hit_n-1 term, random effects uncorrelated,
                    session-mean centring (on by default), configurable angle
                    basis, many diagnostics (VIF, angle linearity, calibration,
                    random-effect normality, influential rats, residual
                    autocorrelation, optional DHARMa) and manual LRT refits.
                    Source of the model checks. Its outputs go to
                    outputs/00_exploration/glmm_population_*; none exist on disk.

01_glmm_I   (pooled history)
  GLMM_I.py         Question: is there a population-average signed history effect
                    (success_prev, failure_prev, reward_prev) and how much do rats
                    vary around it? History is pooled across modality. Correlated
                    (success, failure | rat) random block; primary fit with a
                    fallback to independent slopes. Fit time about 11 h.
                    Output: outputs/01_glmm_I/

02_glmm_I_validation   (does GLMM I recover the individual rats?)
  glmm_I_shrinkage_check.py
                    Section 4 per-rat GLMs (one per rat x modality, collapsed
                    across modality by inverse-variance weights) against the GLMM I
                    shrunk per-rat estimates. Reads
                    thesis_codes/section_4_GLM/fit_results/per_modality_filtered/ and the GLMM I fit.
                    Output: outputs/02_glmm_I_validation/shrinkage_plots/
  glmm_I_outlier_sensitivity.py
                    Leave-one-rat-out sensitivity of the GLMM I random-effect
                    correlation and variance ratio, against the section 4 per-rat
                    estimates (pooled, training-filtered, thesis_codes/
                    section_4_GLM/fit_results/pooled_filtered). Needs --pkl
                    pointing to glmm_I_full.pkl.
                    Output: outputs/02_glmm_I_validation/outlier_sensitivity/
  (outputs/02_glmm_I_validation/modality_contrasts/ holds within-rat modality
   contrast figures whose script is not in the repository.)

03_glmm_II   (does modality shift the population effect?)
  GLMM_II.py        GLMM I plus all history x modality interactions (18 fixed
                    terms), without the two near-zero angle random slopes.
                    One answer for the cohort: is there a shared direction of
                    shift between modalities? (Section 4 gives one answer per
                    rat.) Omnibus 2-df Wald tests, modality-specific effects,
                    pairwise modality contrasts. Fit time about 6 h.
                    Output: outputs/03_glmm_II/
  glmm_II_post_hoc.py
                    Repairs the three tables of the LEGACY GLMM II fit (broken
                    interaction names, bug since fixed in GLMM_II.py) from the
                    saved model.rds, without refitting. A fit made with the
                    current GLMM_II.py does not need it. It stops if
                    MODALITY_REFERENCE does not match the fit.


Outputs
-------
outputs/01_glmm_I/legacy_ref-visuotactile/
    primary/ fallback/   the GLMM I fit used for the shrinkage check. In GLMM I
                         the reference only recodes the modality main effects, so
                         the history and random effects are the same as in a
                         tactile fit.
    snapshot_20260906_113833/   frozen copy of an earlier state of the fit.
    glmm_I_full.pkl, glmm_I_console_log.txt, input_used.csv
outputs/03_glmm_II/legacy_ref-visuotactile/
    primary/             GLMM II fit with visuotactile as reference (modality
                         terms named modalitytactile, modalityvisual). Needs a
                         refit with GLMM_II.py to get tactile-reference results.
Most CSV, PNG and RDS files are gitignored.
