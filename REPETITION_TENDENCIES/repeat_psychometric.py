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





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from serial_dependence_analysis import SerialDependenceAnalyzer
from serial_dependence_analysis import fit_psychometric_curve, cumulative_gaussian_lapse

# -----------------------------
# CONFIG / COLUMN NAMES
# -----------------------------
ANGLE_COL = "angle"
RAT_COL = "rat"
RESP_COL = "action"
HIT_N_1_COL = "hitmiss_n-1"  # 0/1 or False/True
SESSION_COL = "date"
MOD_COL = "mod"  # <-- CHANGE if needed

BOUNDARY = 45
EVIDENCE_COL = "evidence_4_repetition"
REP_COL = "repeated_action"

max_deviation = True

# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data()

df_processed = an.create_lagged_features()

df_processed[REP_COL] = (df_processed["action_n-1"] == df_processed[RESP_COL]).astype(int)
df_processed[EVIDENCE_COL] = (df_processed[ANGLE_COL] - BOUNDARY) * (
    2 * df_processed["action_n-1"] - 1
)

# -----------------------------
# HELPERS: session bootstrap
# -----------------------------
def bootstrap_resample_sessions(df: pd.DataFrame, session_col: str, rng: np.random.Generator):
    """
    Resample sessions WITH replacement; keep all trials within each sampled session.
    Preserves within-session serial structure.
    """
    sessions = df[session_col].dropna().unique()
    if len(sessions) == 0:
        return None

    sampled = rng.choice(sessions, size=len(sessions), replace=True)

    # concat all trials for each sampled session (duplicates allowed)
    parts = [df[df[session_col] == s] for s in sampled]
    out = pd.concat(parts, axis=0, ignore_index=True)
    return out


# -----------------------------
# BOOTSTRAPPING PSYCHOMETRIC CURVE FITS
# -----------------------------
n_bootstraps = 6  # 5 is too small for stable 95% bands
boot_results = []

rats = sorted(pd.Series(df_processed[RAT_COL]).dropna().unique())
modalities = sorted(pd.Series(df_processed[MOD_COL]).dropna().unique(), key=str)



for rat in rats:
    for rs in [0, 1]:
        df_subset = df_processed[
            (df_processed[RAT_COL] == rat) & (df_processed[HIT_N_1_COL] == rs)
        ].copy()

        # quick sanity checks
        if SESSION_COL not in df_subset.columns:
            raise ValueError(
                f"SESSION_COL='{SESSION_COL}' not found in df_subset columns."
            )

        # require at least a few sessions to make session-bootstrap meaningful
        n_sessions = df_subset[SESSION_COL].nunique(dropna=True)
        if n_sessions < 2:
            print(f"Rat {rat}, rs {rs}: only {n_sessions} sessions -> skipping.")
            continue

        for b in range(n_bootstraps):
            rng = np.random.default_rng(10_000 + 100 * int(rat) + 10 * rs + b)

            boot_df_samp = bootstrap_resample_sessions(df_subset, SESSION_COL, rng)
            if boot_df_samp is None or len(boot_df_samp) == 0:
                continue

            popt, ok, _, _ = fit_psychometric_curve(
                boot_df_samp[EVIDENCE_COL],
                boot_df_samp[REP_COL],
                min_trials=100,
                repeat_fit=True,
            )
            if not ok:
                continue

            boot_results.append(
                {
                    "rat": rat,
                    "reward_state": rs,
                    "bootstrap": b,
                    "mu": popt[0],
                    "sigma": popt[1],
                    "gamma": popt[2],
                    "lapse": popt[3],
                }
            )

boot_df = pd.DataFrame(boot_results)
if boot_df.empty:
    raise RuntimeError("No successful bootstrap fits. Lower min_trials or check data filters.")

# -----------------------------
# SUMMARIZE BOOTSTRAP PARAMS
# -----------------------------
group_cols = ["rat", "reward_state"]
g = boot_df.groupby(group_cols)

summary = (
    g.agg(
        mu_med=("mu", "median"),
        mu_q025=("mu", lambda x: x.quantile(0.025)),
        mu_q975=("mu", lambda x: x.quantile(0.975)),
        sigma_med=("sigma", "median"),
        sigma_q025=("sigma", lambda x: x.quantile(0.025)),
        sigma_q975=("sigma", lambda x: x.quantile(0.975)),
        gamma_med=("gamma", "median"),
        gamma_q025=("gamma", lambda x: x.quantile(0.025)),
        gamma_q975=("gamma", lambda x: x.quantile(0.975)),
        lapse_med=("lapse", "median"),
        lapse_q025=("lapse", lambda x: x.quantile(0.025)),
        lapse_q975=("lapse", lambda x: x.quantile(0.975)),
    )
    .reset_index()
)

# -----------------------------
# PLOT (one per rat)
# -----------------------------
color_map = {0: "violet", 1: "yellow"}

n = len(rats)
ncols = int(np.ceil(np.sqrt(n)))
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(5 * ncols, 4.5 * nrows),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)
axes = np.atleast_1d(axes).reshape(-1)

# evidence range is ~[-45, +45]
x_fit = np.linspace(-45, 45, 200)

for ax, rat in zip(axes, rats):
    sub = summary[summary[RAT_COL] == rat]

    # optional reference curve using ALL trials of that rat
    raw_r = df_processed[df_processed[RAT_COL] == rat]
    popt_r, ok_r, x_fit_r, y_fit_r = fit_psychometric_curve(
        raw_r[EVIDENCE_COL],
        raw_r[REP_COL],
        min_trials=50,
        repeat_fit=True,
    )
    if ok_r:
        ax.plot(x_fit_r, y_fit_r, color="black", alpha=0.8, ls="--")

    # plot each reward state curve + pointwise CI band
    for _, row in sub.iterrows():
        rs = row["reward_state"]
        color = color_map.get(rs, "gray")

        y_fit = cumulative_gaussian_lapse(
            x_fit,
            row["mu_med"],
            row["sigma_med"],
            row["gamma_med"],
            row["lapse_med"],
        )

        boot_sub = boot_df[(boot_df[RAT_COL] == rat) & (boot_df["reward_state"] == rs)]

        if len(boot_sub) > 10:
            y_boot = np.vstack(
                [
                    cumulative_gaussian_lapse(x_fit, mu, sigma, gamma, lapse)
                    for (mu, sigma, gamma, lapse) in boot_sub[
                        ["mu", "sigma", "gamma", "lapse"]
                    ].itertuples(index=False, name=None)
                ]
            )

            if max_deviation:
                y_ref = np.quantile(y_boot, 0.50, axis=0)
                max_dev = np.max(np.abs(y_boot - y_ref[None, :]), axis=0)
                c = np.quantile(max_dev, 0.95)
                y_low = np.clip(y_ref - c, 0, 1)
                y_high = np.clip(y_ref + c, 0, 1)
            else:
                y_low = np.quantile(y_boot, 0.025, axis=0)
                y_high = np.quantile(y_boot, 0.975, axis=0)

            ax.fill_between(x_fit, y_low, y_high, color=color, alpha=0.25, linewidth=0)

        ax.plot(x_fit, y_fit, color=color, alpha=0.9, label=f"reward_state {rs}")

    # decorations
    ax.axhline(0.5, color="k", ls="--", alpha=0.4)
    ax.axvline(0, color="k", ls="--", alpha=0.4)

    ax.set_xlim(-45, 45)
    ax.set_ylim(0, 1)

    n_rs1 = df_processed[(df_processed[RAT_COL] == rat) & (df_processed[HIT_N_1_COL] == 1)].shape[0]
    n_rs0 = df_processed[(df_processed[RAT_COL] == rat) & (df_processed[HIT_N_1_COL] == 0)].shape[0]
    ax.set_title(f"Rat {rat} (rs1 N={n_rs1}, rs0 N={n_rs0})")

    ax.set_xlabel("Evidence for repetition")
    ax.set_ylabel("P(repeat)")

# hide unused axes
for ax in axes[len(rats):]:
    ax.set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
if handles:
    fig.legend(
        handles,
        labels,
        title="Reward state",
        loc="upper center",
        ncol=min(6, len(handles)),
        bbox_to_anchor=(0.5, 1.02),
        fontsize=9,
    )

# NOTE: md here will be the *last* modality from the loop above.
fig.suptitle(
    f"P(repeat) vs evidence for repetition (session-bootstrap CIs, per rat)",
    y=1.06,
    fontsize=14,
)

plt.show()