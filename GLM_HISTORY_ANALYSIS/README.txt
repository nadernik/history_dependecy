GLM_HISTORY_ANALYSIS
====================

Each top-level folder answers one conceptual question about history effects
(the influence of the previous trial on the current choice) in the rat task.
Subfolders split a question by what the scripts do.

The final, thesis-ready per-rat GLM now lives in
thesis_codes/section_4_GLM (Section_4_GLM.py). This folder holds the earlier
exploratory work, the mixed-model work, and a few pieces not replaced there.
All scripts load data through SerialDependenceAnalyzer / functions_Ale in the
project root.


01_history_structure
--------------------
Question: what is the structure of the history effect? Is it plain choice
repetition, a reward x choice interaction, or a pure reward effect?
  All scripts here fit one logistic GLM per rat (ridge, continuous regressors
  scaled, session-level structure ignored) to predict the current choice from
  the current angle and the previous trial: previous angle, previous choice
  (+1 / -1), previous outcome (hit_n-1), and the two outcome-specific choice
  terms success_n-1 = hit * choice and failure_n-1 = (1 - hit) * choice.
  Angles are centered at 45 degrees.

  exploration/
    glm_familiy1.py   Earliest, simplest sweep. Defines about 20 regressor
                      sets (current angle, modality terms, previous angle,
                      previous choice, success/failure, hit) but only runs
                      one (features_17: angle, action_n-1, angle_n-1,
                      failure_n-1, hit_n-1). Plots coefficients with bootstrap
                      CIs and prints a per-rat delta AIC / delta BIC table.
                      The bootstrap is set to 1 iteration, so the CIs are
                      placeholders. No trial exclusions.
    glm_families.py   Later, fuller version. Drops trials whose previous angle
                      is 45 (their hit/miss was recorded at random by a bug in
                      data collection). Adds categorical angle (-1 / +1),
                      ambiguity (= -difficulty) and history x ambiguity terms,
                      organised in three families: (1) current angle +
                      modality, (2) + history, (3) + history x ambiguity
                      (unfinished, not run). The models actually run are
                      features_03 and features_04: does the previous angle act
                      differently after a hit than after a miss
                      (angle_hit_n-1, angle_miss_n-1), with and without
                      success/failure. 1000 bootstraps, delta AIC / BIC per rat.
                      Note: unlike the other scripts it does not add the
                      project root to sys.path before importing functions_Ale.

  hit_side_bias_analysis/
    Is the pure-reward effect (the coefficient of hit_n-1) just side bias?
    One analysis in three steps, per rat and per modality (12 rats x 3
    modalities = 36 fits, ridge C = 1, session-level bootstrap). Run the
    steps in order, from any folder; every script finds the project root and
    the data from its own location and saves its results in this folder.
    Each step also saves a *_metadata.json (date, git commit, settings,
    library versions, trials per rat).
    Preprocessing is written out at the top of each script's main block, with
    two switches that default to True: APPLY_TRAINING_FILTER (0.80 criterion,
    functions_Ale.filter_training_trials) and DROP_PREV_ANGLE_45 (drop trials
    whose previous angle is 45, their hit/miss was recorded at random).
    Setting both to False gives the August setup that made the data files
    below.
    Step 1  glm_test_choice_reward_choice_x_reward.py
              Main model (angle, angle_n-1, success_n-1, failure_n-1), 1000
              session bootstraps. Classifies each fit as reward x choice /
              pure choice / ambiguous, and compares it with a model using only
              action_n-1 and one without angle_n-1 (delta AIC / BIC). Saves
              glm_summary.csv, forest plots and bootstrap histograms (folder
              glm_plots), and glm_summary_metadata.json.
    Step 2  hit_n_1_check.py
              Adds hit_n-1 to the model (features_18) and compares with the
              model without it (features_19): coefficient with bootstrap CI,
              delta AIC / BIC, label pure_reward_present / no_pure_reward /
              ambiguous. Saves hit_n1_test_summary.csv and
              hit_n1_test_metadata.json.
    Step 3  hit_n_1_check2.py
              Refits features_18 for the 36 fits and measures side bias two
              ways: the model intercept and the raw P(vertical) on 45-degree
              trials. Merges with glm_summary.csv and hit_n1_test_summary.csv,
              correlates beta_hit with side bias and with beta_success, lists
              the fits where hit_n-1 clearly helps (delta BIC > 2) and labels
              them repeater or switcher after a win. Saves side_bias_per_fit.csv,
              side_bias_analysis.png and side_bias_metadata.json.
    Data files kept here (all from the August run: no training filter, trials
    after a 45-degree trial kept; no metadata file)
      glm_summary.csv          Per-fit table from step 1: betas for success,
                               failure, previous angle and angle with CIs,
                               success - failure difference, probability of
                               opposite signs, delta AIC / BIC, old
                               classification. Input of step 3. Rerunning
                               step 1 overwrites it.
      hit_n1_test_summary.csv  Output of step 2. Input of step 3. Rerunning
                               step 2 overwrites it.
      side_bias_per_fit.csv, side_bias_analysis.png
                               Output of step 3. Rerunning step 3 overwrites them.
    Guard in step 3: it merges glm_summary.csv and hit_n1_test_summary.csv
    with its own fits, so all three must come from the same preprocessing.
    It reads the *_metadata.json next to each CSV and prints a warning (at the
    start and at the end, it does not stop) if a metadata file is missing or
    its two switches or its number of rows differ from the current run. The
    August files have no metadata, so the warning appears until steps 1 and 2
    are rerun with the current scripts and the same switches as step 3. Parts
    of the step 3 output that mix datasets while the warning is on: the
    beta_hit vs beta_success correlation, the delta BIC > 2 list, the
    repeater / switcher labels and the third scatter plot. The side-bias
    comparisons use only step 3's own fits and are not affected. The result
    of the check is also saved in side_bias_metadata.json.

02_learning_dynamics
--------------------
Question: do the history effects change over the course of learning?
  learning_glm_dynam.py        One GLM per session for one rat, beta
                               trajectories over sessions.
  learning_glm_dynamic_MOD.py  Same, split by modality, merging consecutive
                               sessions to reach a minimum trial count.

03_modality_modulation  (moved)
----------------------
The GLMM scripts that were here (GLMM II, its post-hoc repair, glmm_hit_n_1.py
and the GLMM I shrinkage check) now live in ../GLMM_ANALYSIS, together with
GLMM I. See ../GLMM_ANALYSIS/README.txt for the order of the analysis.

04_acuity_transitions
---------------------
Moved to ../GLMM_ANALYSIS (scripts/04_acuity_transitions and
outputs/04_acuity_transitions): the mixed models on modality transitions and
their validation against per-cell GLMs. See ../GLMM_ANALYSIS/README.txt.

_unsorted
---------
mixedlm_hit_contrasts.csv and mixedlm_hit_summary.csv: origin unknown, no
script here writes them. Move them to the right question or delete.

Shared code
-----------
functions_Ale.py (training-trial filter and helpers) and
serial_dependence_analysis.py are both in the project root. The scripts in
this folder add the root to the import path.

Notes
-----
- Fit results of all the mixed models are in history_dependecy/GLMM_ANALYSIS/outputs.
- Most CSV files are gitignored: this README is the only record of what they are.
