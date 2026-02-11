" functions to run glm analysis from the toso's like attempt. So with lapses"


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from serial_dependence_analysis import SerialDependenceAnalyzer

RAT_COL   = "rat"
ANGLE_COL = "angle"
DATE_COL  = "date"   # session identifier
TRIAL_COL = "trialID" # trial index within session 

# Analysis params
HISTORY_DEPTH = 5
MIN_TRIALS    = 200
LAPSE_PENALTY = 200.0
MAX_ITER      = 800

# Bootstrap params --> to be changed here to then run the analysis, this is not ideal, but I will change it later if I need to run it again
N_BOOT = 300
ALPHA  = 0.05

def extract_kernel(results, prefix, k):
    """
    Inputs:
    - results: output of "analyze_individual_rats_toso_lapse", which is a dict of dicts, one per rat, containing "beta" and "feature_names".
    - prefix: string prefix of the predictors to extract, e.g. "angle", "action", "hitmiss"
    - k: number of lags to extract (e.g. 5)
    Output:
    - rats: list of rat identifiers (sorted)
    - K: array of shape (n_rats, k) with the extracted coefficients for predictors named like:
      f"{prefix}_n-1", ..., f"{prefix}_n-k"

    This function extracts the coefficients for predictors.
    Extracts a (n_rats x k) matrix for predictors named like:
      f"{prefix}_n-1", ..., f"{prefix}_n-k"
    """
    rats = sorted(results.keys())
    K = np.full((len(rats), k), np.nan)
    
    for i, rat in enumerate(rats):
        beta = results[rat]["beta"]
        names = results[rat]["feature_names"]

        for lag in range(1, k + 1):
            name = f"{prefix}_n-{lag}"
            if name in names:
                idx = names.index(name)
                K[i, lag - 1] = beta[idx]

    return rats, K
# ------------------------------------------------------------
# 2) Plot kernel heatmap (your existing visualization, cleaned)
# ------------------------------------------------------------
def plot_kernel_heatmap(an, results, all_ci=None):
    """
    Plots a heatmap of the kernels (angle lags, choice lags, outcome lags) with significance stars from session bootstrap CIs (optional).
    Inputs:
- an: the SerialDependenceAnalyzer instance (used to get k and other params)
- results: output of "analyze_individual_rats_toso_lapse", which is a dict of dicts, one per rat, containing "beta" and "feature_names".
- all_ci: optional DataFrame with session bootstrap CIs and significance (as returned by the session bootstrap functions).
Output:
- A heatmap plot of the kernels with significance stars (optional).
    """
    rats = sorted(results.keys())
    k = an.k

    # current angle beta
    curr_angle_beta = []
    for rat in rats:
        names = results[rat]["feature_names"]
        beta  = results[rat]["beta"]
        if ANGLE_COL not in names:
            raise ValueError(f"'{ANGLE_COL}' not found in feature_names for rat {rat}")
        curr_angle_beta.append(beta[names.index(ANGLE_COL)])

    curr_col = np.array(curr_angle_beta)[:, None]

    # history kernels
    _, stim_kernel   = extract_kernel(results, prefix="angle",   k=k)
    _, choice_kernel = extract_kernel(results, prefix="action",  k=k)
    _, out_kernel    = extract_kernel(results, prefix="hitmiss", k=k)

    K_with_curr = np.hstack([curr_col, stim_kernel])
    K_all = np.hstack([K_with_curr, choice_kernel, out_kernel])

    xticks = (
        ["angle (curr)"] +
        [f"ang {i}" for i in range(1, k+1)] +
        [f"cho {i}" for i in range(1, k+1)] +
        [f"out {i}" for i in range(1, k+1)]
    )
    sig_mat = None
    if all_ci is not None:
        sig_wide = (
            all_ci
            .pivot(index="rat", columns="predictor", values="significant")
            .fillna(False)
        )
        sig_mat = sig_wide.to_numpy(dtype=bool)


    # -----------------------------
    # Build annotation strings (add * if significant)
    # -----------------------------
    annot = np.empty(K_all.shape, dtype=object)
    for i in range(K_all.shape[0]):
        for j in range(K_all.shape[1]):
            val = K_all[i, j]
            if np.isnan(val):
                annot[i, j] = ""
            else:
                star = ""
                if sig_mat is not None and sig_mat[i, j]:
                    star = "*"
                annot[i, j] = f"{val:.2f}{star}"

    # -----------------------------
    # Plot heatmap

    plt.figure(figsize=(1.0 * K_all.shape[1], 0.4 * len(rats)))
    sns.heatmap(
        K_all,
        cmap="coolwarm",
        center=0,
        annot=annot,
        fmt="",
        xticklabels=xticks,
        yticklabels=rats,
        cbar=True,
        annot_kws={"fontsize": 9}
    )
    plt.xlabel("Predictor (type × lag)")
    plt.ylabel("Rat")
    plt.title("History kernels: angle, choice, outcome")
    plt.tight_layout()
    plt.show()




def _prepare_session_arrays(
    an,
    rat_df,
    date_col="date",
    trial_col="trialID",
    min_trials=200,
):
    """
    Precompute per-session arrays for fast session bootstrap.
    Uses the SAME design matrix logic as the Toso lapse GLM,
    but does it ONCE per session instead of per bootstrap.
    inputs:
- an: the SerialDependenceAnalyzer instance (used to get k and other params)
- rat_df: DataFrame with data for one rat, already preprocessed and with lagged features created.
- date_col: name of the column with session identifiers (e.g. "date")
- trial_col: name of the column with trial indices within session (e.g. "trialID")
- min_trials: minimum number of trials required to include a session
outputs:
- A dict with:
  - "sessions": a list of tuples (X_cont_z, X_choice_pm, X_out, y) for each session, where:
    - X_cont_z: standardized continuous predictors (angle lags) for that session
    - X_choice_pm: choice predictors (action lags) coded as ±1 for that session
    - X_out: outcome predictors (hitmiss lags) for that session
    """
    k = int(an.k)

    # enforce correct trial order --> it might be not needed as data are already sorted by date and trial, so mmmmh
    rat_df = rat_df.sort_values([date_col, trial_col]).copy()

    # required columns (same as analyzer)
    needed = ["action", "angle"]
    for lag in range(1, k + 1):
        needed += [f"angle_n-{lag}", f"action_n-{lag}", f"hitmiss_n-{lag}"]

    existing = [c for c in needed if c in rat_df.columns]
    rat_df = rat_df.dropna(subset=existing)

    if len(rat_df) < min_trials:
        return None
    # fixed column order
    cont_cols = ["angle"] + [
        f"angle_n-{lag}" for lag in range(1, k + 1)
        if f"angle_n-{lag}" in rat_df.columns
    ]
    choice_cols = [
        f"action_n-{lag}" for lag in range(1, k + 1)
        if f"action_n-{lag}" in rat_df.columns
    ]
    out_cols = [
        f"hitmiss_n-{lag}" for lag in range(1, k + 1)
        if f"hitmiss_n-{lag}" in rat_df.columns
    ]

    feature_names = ["intercept"] + cont_cols + choice_cols + out_cols

    from sklearn.preprocessing import StandardScaler
    X_cont_all = rat_df[cont_cols].to_numpy(dtype=float, copy=False)
    scaler = StandardScaler().fit(X_cont_all)

    sessions = []
    print(f" number of sessions for rat {rat_df[RAT_COL].iloc[0]}: {rat_df[date_col].nunique()}")
    for _, g in rat_df.groupby(date_col, sort=False):
        #print(f"  session {g[date_col].iloc[0]} with {len(g)} trials")
        y = g["action"].to_numpy(dtype=int, copy=False)

        X_cont = g[cont_cols].to_numpy(dtype=float, copy=False)
        X_cont_z = scaler.transform(X_cont)

        X_choice = g[choice_cols].to_numpy(dtype=int, copy=False)
        X_choice_pm = (2 * X_choice - 1).astype(float, copy=False)

        X_out = g[out_cols].to_numpy(dtype=float, copy=False)

        sessions.append((X_cont_z, X_choice_pm, X_out, y))

    return {
        "sessions": sessions,
        "feature_names": feature_names,
    }


def bootstrap_betas_by_session_fast(
    an,
    df_processed,
    rat_id,
    n_boot=300,
    date_col="date",
    trial_col="trialID",
    min_trials=200,
    lapse_penalty=200.0,
    max_iter=4000,
    random_state=None,
):
    """
    Fast bootstrap of lapse-GLM betas by resampling whole sessions (identified by date_col) with replacement.
    inputs:
- an: the SerialDependenceAnalyzer instance (used to get k and other params)
- df_processed: the full DataFrame with preprocessed data and lagged features for all rats
- rat_id: identifier of the rat to analyze
- n_boot: number of bootstrap samples
- date_col: name of the column with session identifiers (e.g. "date")
- trial_col: name of the column with trial indices within session (e.g. "trialID")
- min_trials: minimum number of trials required to include a session
- lapse_penalty: penalty parameter for lapse fitting
- max_iter: maximum iterations for lapse fitting
- random_state: random seed for reproducibility
outputs:
- betas: array of shape (n_boot, n_predictors) with the fitted coefficients for each bootstrap sample
- feature_names: list of predictor names corresponding to the columns of betas
- failed_fits: number of failed fits in the bootstrap procedure

    """
    failed_fits = 0  
    rng = np.random.default_rng(random_state)

    rat_df = df_processed[df_processed["rat"] == rat_id].copy()

    cache = _prepare_session_arrays(
        an,
        rat_df,
        date_col=date_col,
        trial_col=trial_col,
        min_trials=min_trials,
    )
    if cache is None:
        return None, None

    sessions = cache["sessions"]
    feature_names = cache["feature_names"]

    betas = []
    n_sessions = len(sessions)

    for _ in range(n_boot): # this loop creates n_boot bootstrapped datasets and fits a model on each of them, storing the betas
        idx = rng.integers(0, n_sessions, size=n_sessions) # this line samples sessions with replacement

        Xc = np.vstack([sessions[i][0] for i in idx])
        Xh = np.vstack([sessions[i][1] for i in idx])
        Xo = np.vstack([sessions[i][2] for i in idx])
        y  = np.concatenate([sessions[i][3] for i in idx])

        intercept = np.ones((len(y), 1), dtype=float)
        X = np.hstack([intercept, Xc, Xh, Xo])

        try:
            fit = an._fit_lapse_glm(X, y, lapse_penalty=lapse_penalty, max_iter=max_iter)
            if fit["opt"].success:
                betas.append(np.asarray(fit["beta"], float))
            else:
                failed_fits += 1
        except Exception:
            continue
    
    return np.vstack(betas), feature_names, failed_fits



# ------------------------------------------------------------
# 3) Session bootstrap utilities
# ------------------------------------------------------------

def ci_table_for_rat(
    an,
    df_processed,
    rat_id,
    n_boot=300,
    alpha=0.05,
    random_state=0,
):
    """
    Computes session-bootstrap confidence intervals for the lapse-GLM coefficients of one rat.
    inputs:
- an: the SerialDependenceAnalyzer instance (used to get k and other params)
- df_processed: the full DataFrame with preprocessed data and lagged features for all rats
- rat_id: identifier of the rat to analyze
- n_boot: number of bootstrap samples
- alpha: significance level for confidence intervals
- random_state: random seed for reproducibility
outputs:
- A DataFrame with columns: "rat", "predictor", "ci_low", "ci_high", "significant", "n_boot_success", "n_boot_failed", "bootstrap"
    This function performs the session bootstrap by calling "bootstrap_betas_by_session_fast" to get the bootstrapped coefficients, 
    then computes percentile confidence intervals for each predictor, and determines significance based on whether the CI excludes zero.

    """
    betas, feature_names, failed_fits = bootstrap_betas_by_session_fast(
        an,
        df_processed,
        rat_id,
        n_boot=n_boot,
        date_col=DATE_COL,
        trial_col=TRIAL_COL,
        min_trials=MIN_TRIALS,
        lapse_penalty=LAPSE_PENALTY,
        max_iter=MAX_ITER,
        random_state=random_state,
    )

    lo, hi = np.percentile(betas, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    sig = ~((lo <= 0) & (0 <= hi))

    return pd.DataFrame({
        "rat": rat_id,
        "predictor": feature_names,
        "ci_low": lo,
        "ci_high": hi,
        "significant": sig,
        "n_boot_success": betas.shape[0],
        "n_boot_failed": failed_fits,
        "bootstrap": "session(date)",

    })


def plot_significance_heatmap(all_ci):
    """
    Heatmap: rows=rats, cols=predictors, values=1 if significant else 0.
    inputs:
- all_ci: DataFrame with session bootstrap results, including "rat", "predictor", and "significant" columns.
outputs:
- A heatmap plot where rows are rats, columns are predictors, and cells are colored if the predictor is significant for that rat based on session bootstrap CIs.
    """
    sig_wide = (
        all_ci
        .pivot(index="rat", columns="predictor", values="significant")
        .fillna(False)
    )

    plt.figure(figsize=(0.35 * sig_wide.shape[1] + 6, 0.35 * sig_wide.shape[0] + 2))
    sns.heatmap(sig_wide.astype(int), cbar=False, linewidths=0.5)
    plt.xlabel("Predictor")
    plt.ylabel("Rat")
    plt.title("Session-bootstrap significance (1 = CI excludes 0)")
    plt.tight_layout()
    plt.show()


def _ci_one_rat(rat, an, df_processed):
    """
        Helper function to compute session bootstrap CI for one rat, used for parallel processing.
        inputs:
    - rat: identifier of the rat to analyze
    - an: the SerialDependenceAnalyzer instance (used to get k and other params)
    - df_processed: the full DataFrame with preprocessed data and lagged features for all rats
    outputs:
    - A DataFrame with session bootstrap CI results for the given rat (same format as "ci_table_for_rat" output)
    """

    df_ci = ci_table_for_rat(
        an,
        df_processed,
        rat_id=rat,
        n_boot=N_BOOT,
        alpha=ALPHA,
        random_state=int(rat)  # so it stays reproducible per rat
    )
    print("Processed rat:", rat)
    return df_ci
