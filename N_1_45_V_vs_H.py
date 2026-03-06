import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from serial_dependence_analysis import SerialDependenceAnalyzer
from serial_dependence_analysis import fit_psychometric_curve, cumulative_gaussian_lapse

# =============================
# CONFIG
# =============================
RAT_COL        = "rat"
ANGLE_COL      = "angle"          # trial n angle
ACTION_COL     = "action"         # trial n choice (0/1)
ANGLE_N1_COL   = "angle_n-1"      # trial n-1 angle
ACTION_N1_COL  = "action_n-1"     # trial n-1 choice (0/1)
BOUNDARY_ANGLE = 45
HITMISS_N1_COL = "hitmiss_n-1"    # trial n-1 outcome (0/1)

N_BOOT         = 100
RANDOM_SEED    = 0
MIN_TRIALS_FIT = 10               # min balanced trials per prev_action to keep the rat
use_max_dev = True                   # if True, use max-deviation method for CIs; else pointwise quantiles
# =============================
# LOAD + PREPROCESS
# =============================
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data()
df_processed = an.create_lagged_features()

# keep only well-defined rows for the columns we use
req = [RAT_COL, ANGLE_COL, ACTION_COL, ANGLE_N1_COL, ACTION_N1_COL, HITMISS_N1_COL]

dfc = df_processed.dropna(subset=req).copy()

print(f"angle n-1 distribution:\n{dfc[ANGLE_N1_COL].value_counts()}")

# datasets (only 45 used below; 40/50 are just for your sanity checks)
dfc_45 = dfc[(dfc[ANGLE_N1_COL] == BOUNDARY_ANGLE) & (dfc[HITMISS_N1_COL] == 1)].copy()
dfc_40 = dfc[dfc[ANGLE_N1_COL] == 40].copy()
dfc_50 = dfc[dfc[ANGLE_N1_COL] == 50].copy()
print(f" per rat angle distribution: \n {dfc_45.groupby([RAT_COL, ANGLE_COL]).size()}")
'''
print("number of trials per rat and previous action for 45° dataset:\n",
      dfc_45.groupby([RAT_COL, ACTION_N1_COL]).size())
print("number of trials per rat and previous action for 40° dataset:\n",
      dfc_40.groupby([RAT_COL, ACTION_N1_COL]).size())
print("number of trials per rat and previous action for 50° dataset:\n",
      dfc_50.groupby([RAT_COL, ACTION_N1_COL]).size())
'''
# =============================
# BOOTSTRAP: balance prev-action groups within each rat, fit curves on trial n
# =============================
rng = np.random.default_rng(RANDOM_SEED)
boot_rows = []
rats = sorted(dfc_45[RAT_COL].unique())

for rat in rats:
    rat_df = dfc_45[dfc_45[RAT_COL] == rat]

    g0 = rat_df[rat_df[ACTION_N1_COL] == 0]
    g1 = rat_df[rat_df[ACTION_N1_COL] == 1]

    n0, n1 = len(g0), len(g1)
    n_min = min(n0, n1)

    print(f"Rat {rat}: prev_action=0 has {n0}, prev_action=1 has {n1} -> balancing to {n_min} each")

    if n_min < MIN_TRIALS_FIT:
        print(f"  Skipping rat {rat}: not enough balanced trials, n = {n_min} (<{MIN_TRIALS_FIT}).")
        continue

    idx0 = g0.index.to_numpy()
    idx1 = g1.index.to_numpy()

    for b in range(N_BOOT):
        samp0_idx = rng.choice(idx0, size=n_min, replace=True)
        samp1_idx = rng.choice(idx1, size=n_min, replace=True)

        for prev_action, samp_idx in [(0, samp0_idx), (1, samp1_idx)]:
            samp = rat_df.loc[samp_idx]

            popt, ok, x_fit, y_fit = fit_psychometric_curve(
                samp[ANGLE_COL],
                samp[ACTION_COL],
                min_trials=MIN_TRIALS_FIT,
            )
            if not ok:
                continue

           
            boot_rows.append(
                dict(
                    rat=rat,
                    prev_action=prev_action,
                    bootstrap=b,
                    n_balanced=n_min,
                    mu=popt[0],
                    sigma=popt[1],
                    gamma=popt[2],
                    lapse=popt[3],
                )
            )

boot_df = pd.DataFrame(boot_rows)
if boot_df.empty:
    raise RuntimeError("No successful fits. Check MIN_TRIALS_FIT and your data.")

# =============================
# SUMMARIZE BOOTSTRAPS → median params
# =============================
summary = (
    boot_df.groupby([RAT_COL, "prev_action"])
    .agg(
        n_balanced=("n_balanced", "first"),
        mu_med=("mu", "median"),
        sigma_med=("sigma", "median"),
        gamma_med=("gamma", "median"),
        lapse_med=("lapse", "median"),
    )
    .reset_index()
)

# =============================
# PLOT (one subplot per rat)
# =============================
rats_ok = sorted(summary[RAT_COL].unique())
n = len(rats_ok)
ncols = int(np.ceil(np.sqrt(n)))
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), sharex=True, sharey=True, constrained_layout=True)
axes = np.atleast_1d(axes).reshape(-1)

x_grid = np.linspace(0, 90, 300)

for ax, rat in zip(axes, rats_ok):
    sub_sum = summary[summary[RAT_COL] == rat]

    # reference curve using ALL trials for that rat, conditioned on n-1 45° (not restricted to n-1=45)
    raw_rat = dfc_45[dfc_45[RAT_COL] == rat]
    popt_all, ok_all, x_fit_all, y_fit_all = fit_psychometric_curve(
        raw_rat[ANGLE_COL],
        raw_rat[ACTION_COL],
        min_trials=MIN_TRIALS_FIT,
    )
    if ok_all:
        ax.plot(x_fit_all, y_fit_all, color="black", ls="--", alpha=0.7, label="all trials (ref)")

    for prev_action in [0, 1]:
        row = sub_sum[sub_sum["prev_action"] == prev_action]
        if row.empty:
            continue
        row = row.iloc[0]

        # median curve (from median params)
        y_med = cumulative_gaussian_lapse(
            x_grid, row["mu_med"], row["sigma_med"], row["gamma_med"], row["lapse_med"]
        )

        # pointwise CI from bootstraps (param sampling)
        boot_sub = boot_df[(boot_df[RAT_COL] == rat) & (boot_df["prev_action"] == prev_action)]
        params = boot_sub[["mu", "sigma", "gamma", "lapse"]].to_numpy()

        if len(params) == 0:
            continue
        if len(boot_sub) > 0:
            if use_max_dev:
                Y = np.vstack([
                    cumulative_gaussian_lapse(x_grid, mu, sigma, gamma, lapse)
                    for (mu, sigma, gamma, lapse) in boot_sub[['mu','sigma','gamma','lapse']].itertuples(index=False, name=None)
                ])

                y_ref = np.quantile(Y, 0.50, axis=0)

                max_dev = np.max(np.abs(Y - y_ref[None, :]), axis=1)

                c = np.quantile(max_dev, 0.95)

                y_lo  = np.clip(y_ref - c, 0, 1)
                y_hi = np.clip(y_ref + c, 0, 1)
            else:
                y_boot = np.array([cumulative_gaussian_lapse(x_grid, p[0], p[1], p[2], p[3]) for p in params])
                y_lo = np.quantile(y_boot, 0.025, axis=0)
                y_hi = np.quantile(y_boot, 0.975, axis=0)

        if prev_action == 0:
            label = "prev judged H (action n-1 = 0)"
            color = "tab:blue"
        else:
            label = "prev judged V (action n-1 = 1)"
            color = "tab:orange"

        ax.fill_between(x_grid, y_lo, y_hi, color=color, alpha=0.25, linewidth=0)
        ax.plot(x_grid, y_med, color=color, lw=2, label=label)

    ax.axhline(0.5, color="k", ls="--", alpha=0.4)
    ax.axvline(45, color="k", ls="--", alpha=0.4)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 1)

    n_bal = int(sub_sum["n_balanced"].iloc[0])
    ax.set_title(f"Rat {rat} | prev-action = {n_bal}")
    ax.set_xlabel("Trial n angle (deg)")
    ax.set_ylabel("P(action=1)")

# hide unused axes
for ax in axes[len(rats_ok):]:
    ax.set_visible(False)

# one shared legend
handles, labels = axes[0].get_legend_handles_labels()
if handles:
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))

fig.suptitle(
    "Psychometric curves on trial n conditioned on rewarded boundary trial (n-1 angle=45), split by prev choice",
    y=1.06
)
plt.show()