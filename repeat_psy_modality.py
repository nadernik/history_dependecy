'''
SAFELY CODING: 

START session: 

- check branch --> on the bottom left or run 'git branch' in terminal
-UPDATE branch --> 
git checkout main (it goes to main branch)
git pull (it updates my version of main branch to the current one if anybody changed it)
- UPDATE your working branch -->
git checkout feature/exploratory_psych_curves --> goes to your working branch
git merge main --> merges main into your working branch, so that on my branch I have the latest updates from main

START CODING


END session:
(optional) check changes made: git status
git add exploratory_psych_curves.py --> stages the changes made to this file
git commit -m "progress" --> makes a local checkpoint in your branch.
(only the first time) git push -u origin feature/exploratory_psych_curves --> pushes the changes to my remote working branch
git push --> from the second time onwards
'''









"""
A session-bootstrap psychometric analysis of repetition tendency,
split by CURRENT modality (df_processed[MOD_COL]).

- One figure per modality
- One subplot per rat
- Two curves per rat: reward_state (hitmiss n-1) = 0 vs 1
- Session-bootstrap resampling preserves within-session serial structure
"""

# -----------------------------
# IMPORTS
# -----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from serial_dependence_analysis import SerialDependenceAnalyzer
from serial_dependence_analysis import fit_psychometric_curve, cumulative_gaussian_lapse

# -----------------------------
# CONFIG / COLUMN NAMES
# -----------------------------
ANGLE_COL       = 'angle'
RAT_COL         = 'rat'
RESP_COL        = 'action'
HIT_N_1_COL     = 'hitmiss_n-1'      # 0/1 or False/True
SESSION_COL     = 'date'
MOD_COL         = 'mod'              # <-- CHANGE if needed

BOUNDARY        = 45
EVIDENCE_COL    = 'evidence_4_repetition'
REP_COL         = 'repeated_action'

# boot / fit params
N_BOOTSTRAPS    = 6
MIN_TRIALS_FIT  = 100
MIN_SESSIONS    = 2

# CI style
MAX_DEVIATION_BAND = True   # True = max-deviation band; False = pointwise 2.5/97.5%

# plotting
COLOR_MAP = {0: 'violet', 1: 'yellow'}
X_FIT = np.linspace(-45, 45, 200)

# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data()
df_processed = an.create_lagged_features()

# repetition (repeat n-1 action)
df_processed[REP_COL] = (df_processed['action_n-1'] == df_processed[RESP_COL]).astype(int)

# evidence for repetition: (angle - boundary) signed by previous action
df_processed[EVIDENCE_COL] = (df_processed[ANGLE_COL] - BOUNDARY) * (2 * df_processed['action_n-1'] - 1)

# -----------------------------
# HELPERS: session bootstrap
# -----------------------------
def bootstrap_resample_sessions(df: pd.DataFrame, session_col: str, rng: np.random.Generator) -> pd.DataFrame | None:
    """
    Resample sessions WITH replacement; keep all trials within each sampled session.
    Preserves within-session serial structure.
    """
    sessions = df[session_col].dropna().unique()
    if len(sessions) == 0:
        return None

    sampled = rng.choice(sessions, size=len(sessions), replace=True)

    parts = [df[df[session_col] == s] for s in sampled]
    out = pd.concat(parts, axis=0, ignore_index=True)
    return out

# -----------------------------
# MAIN: loop modalities -> bootstrap -> summarize -> plot
# -----------------------------
rats = sorted(pd.Series(df_processed[RAT_COL]).dropna().unique())
modalities = sorted(pd.Series(df_processed[MOD_COL]).dropna().unique(), key=str)

if len(modalities) == 0:
    raise RuntimeError(f"No modalities found in column '{MOD_COL}'. Check preprocessing and MOD_COL name.")

for md in modalities:

    print(f"\n=== Modality: {md} ===")

    # isolate data for this modality
    df_processed_md = df_processed[df_processed[MOD_COL] == md].copy()
    if df_processed_md.empty:
        print(f"[WARN] No data for modality={md} -> skipping.")
        continue

    # collect boot fits ONLY for this modality
    boot_results: list[dict] = []

    for rat in rats:
        for rs in [0, 1]:
            df_subset = df_processed_md[
                (df_processed_md[RAT_COL] == rat) &
                (df_processed_md[HIT_N_1_COL] == rs)
            ].copy()

            # session bootstrap meaningful only if multiple sessions exist
            n_sessions = df_subset[SESSION_COL].nunique(dropna=True)
            if n_sessions < MIN_SESSIONS:
                print(f"  Mod {md} | Rat {rat}, rs {rs}: only {n_sessions} sessions -> skipping.")
                continue

            # bootstrap
            for b in range(N_BOOTSTRAPS):
                # seed distinct per modality/rat/rs/bootstrap
                md_hash = abs(hash(str(md))) % 10_000_000
                seed = 10_000 + 1000 * int(rat) + 10 * int(rs) + b + md_hash
                rng = np.random.default_rng(seed)

                boot_df_tmp = bootstrap_resample_sessions(df_subset, SESSION_COL, rng)
                if boot_df_tmp is None or len(boot_df_tmp) == 0:
                    continue

                popt, ok, _, _ = fit_psychometric_curve(
                    boot_df_tmp[EVIDENCE_COL],
                    boot_df_tmp[REP_COL],
                    min_trials=MIN_TRIALS_FIT,
                    repeat_fit=True
                )
                if not ok:
                    continue

                boot_results.append({
                    RAT_COL: rat,
                    MOD_COL: md,
                    'reward_state': rs,
                    'bootstrap': b,
                    'mu':    float(popt[0]),
                    'sigma': float(popt[1]),
                    'gamma': float(popt[2]),
                    'lapse': float(popt[3]),
                })

    boot_df = pd.DataFrame(boot_results)

    if boot_df.empty:
        print(f"[WARN] No successful bootstrap fits for modality={md}. "
              f"Try lowering MIN_TRIALS_FIT or check filtering.")
        continue

    # summarize bootstrap parameters
    group_cols = [RAT_COL, 'reward_state', MOD_COL]
    g = boot_df.groupby(group_cols, dropna=False)

    summary = (
        g.agg(
            mu_med     = ('mu', 'median'),
            mu_q025    = ('mu', lambda x: x.quantile(0.025)),
            mu_q975    = ('mu', lambda x: x.quantile(0.975)),

            sigma_med  = ('sigma', 'median'),
            sigma_q025 = ('sigma', lambda x: x.quantile(0.025)),
            sigma_q975 = ('sigma', lambda x: x.quantile(0.975)),

            gamma_med  = ('gamma', 'median'),
            gamma_q025 = ('gamma', lambda x: x.quantile(0.025)),
            gamma_q975 = ('gamma', lambda x: x.quantile(0.975)),

            lapse_med  = ('lapse', 'median'),
            lapse_q025 = ('lapse', lambda x: x.quantile(0.025)),
            lapse_q975 = ('lapse', lambda x: x.quantile(0.975)),
        )
        .reset_index()
    )

    # -----------------------------
    # PLOT: one figure per modality, one subplot per rat
    # -----------------------------
    n = len(rats)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 4.5 * nrows),
        sharex=True, sharey=True,
        constrained_layout=True
    )
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, rat in zip(axes, rats):
        sub = summary[(summary[RAT_COL] == rat) & (summary[MOD_COL] == md)]

        # dashed reference curve = ALL trials of this rat within this modality (no rs split)
        raw_r = df_processed_md[df_processed_md[RAT_COL] == rat]
        popt_r, ok_r, x_fit_r, y_fit_r = fit_psychometric_curve(
            raw_r[EVIDENCE_COL],
            raw_r[REP_COL],
            min_trials=50,
            repeat_fit=True
        )
        if ok_r:
            ax.plot(x_fit_r, y_fit_r, color='black', alpha=0.8, ls='--', label='all trials')

        for _, row in sub.iterrows():
            rs = int(row['reward_state'])
            color = COLOR_MAP.get(rs, 'gray')

            # median-parameter curve
            y_fit = cumulative_gaussian_lapse(
                X_FIT,
                row['mu_med'],
                row['sigma_med'],
                row['gamma_med'],
                row['lapse_med'],
            )

            # CI band from bootstrap draws (FILTER BY MODALITY TOO)
            boot_sub = boot_df[
                (boot_df[RAT_COL] == rat) &
                (boot_df['reward_state'] == rs) &
                (boot_df[MOD_COL] == md)
            ]

            if len(boot_sub) > 10:
                y_boot = np.vstack([
                    cumulative_gaussian_lapse(X_FIT, mu, sigma, gamma, lapse)
                    for (mu, sigma, gamma, lapse) in boot_sub[['mu', 'sigma', 'gamma', 'lapse']].itertuples(index=False, name=None)
                ])

                if MAX_DEVIATION_BAND:
                    y_ref = np.quantile(y_boot, 0.50, axis=0)
                    max_dev = np.max(np.abs(y_boot - y_ref[None, :]), axis=0)
                    c = np.quantile(max_dev, 0.95)
                    y_low  = np.clip(y_ref - c, 0, 1)
                    y_high = np.clip(y_ref + c, 0, 1)
                else:
                    y_low  = np.quantile(y_boot, 0.025, axis=0)
                    y_high = np.quantile(y_boot, 0.975, axis=0)

                ax.fill_between(X_FIT, y_low, y_high, color=color, alpha=0.25, linewidth=0)

            ax.plot(X_FIT, y_fit, color=color, alpha=0.9, label=f"rs {rs}")

        # decorations
        ax.axhline(0.5, color='k', ls='--', alpha=0.4)
        ax.axvline(0,   color='k', ls='--', alpha=0.4)
        ax.set_xlim(-45, 45)
        ax.set_ylim(0, 1)

        # counts (within this modality)
        n_rs1 = df_processed_md[(df_processed_md[RAT_COL] == rat) & (df_processed_md[HIT_N_1_COL] == 1)].shape[0]
        n_rs0 = df_processed_md[(df_processed_md[RAT_COL] == rat) & (df_processed_md[HIT_N_1_COL] == 0)].shape[0]
        ax.set_title(f"Rat {rat} (rs1 N={n_rs1}, rs0 N={n_rs0})")
        ax.set_xlabel('Evidence for repetition')
        ax.set_ylabel('P(repeat)')

    # hide unused axes
    for ax in axes[len(rats):]:
        ax.set_visible(False)

    # one legend for whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            title='Curves',
            loc='upper center',
            ncol=min(6, len(handles)),
            bbox_to_anchor=(0.5, 1.02),
            fontsize=9
        )

    fig.suptitle(
        f'{md}: P(repeat) vs evidence (session-bootstrap CIs, per rat)',
        y=1.06,
        fontsize=14
    )
    plt.show()