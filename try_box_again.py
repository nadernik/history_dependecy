
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from serial_dependence_analysis import SerialDependenceAnalyzer
from serial_dependence_analysis import fit_psychometric_curve, cumulative_gaussian_lapse
import functions_Ale as fa

# -----------------------------
# CONFIG / COLUMN NAMES
# -----------------------------
BIN_COL_N_1     = 'bin_angle_n-1'     # previous-trial angle bin label
BIN_COL_N       = 'bin_angle_n'       # current-trial angle bin label
ANGLE_COL       = 'angle'             # current-trial raw angle
RESP_COL        = 'action'            # current-trial response (0/1)
HIT_COL         = 'hitmiss_n-1'       # filter on previous-trial hits (True/1) if desired
BIN_COL_N_MID   = 'bin_n_midpoint'    # numeric midpoint for current-trial bin (x-axis)
RAT_COL         = 'rat'               # subject/animal ID
trialID         = 'trialID'  
ACTION_N_1      = 'action_n-1'  # unique trial identifier
ACTION          = 'action'    # current-trial response (0/1)

# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data()
df_processed = an.create_lagged_features()

df_processed, _ = fa.bin_by_unique_angles(df_processed, name_new_col=BIN_COL_N_1, n_groups=6, angle_col='angle_n-1', exclude_angle=45)
df_processed, angle_to_label   = fa.bin_by_unique_angles(df_processed, name_new_col=BIN_COL_N,   n_groups=6, angle_col='angle', exclude_angle=45)
dfc = df_processed.dropna(subset=[BIN_COL_N_1, BIN_COL_N]).copy()
# Current-bin midpoints (for x-axis)
bin_means_cur  = {b: fa.midpoint(b) for b in pd.Series(dfc[BIN_COL_N]).dropna().unique()}

# Attach current-bin numeric midpoint to dfc so curve fits have proper x-values
dfc[BIN_COL_N_MID] = dfc[BIN_COL_N].map(bin_means_cur)
# -----------------------------
grup_col = [BIN_COL_N_1, RAT_COL]   # <-- remove HIT_COL from grouping

agg = (
    dfc
    .groupby(grup_col)
    .agg(
        right_trials   = (ACTION_N_1, 'sum'),                        # number of 1s
        left_trials = (ACTION_N_1, lambda x: x.size - x.sum()),   # number of 0s
    )
    .reset_index()
)


agg['min_side_count'] = agg[['right_trials', 'left_trials']].min(axis=1)
agg['min_side_action'] = (agg['right_trials'] < agg['left_trials']).astype(int)
worst = (
    agg.sort_values('min_side_count')
       .groupby(RAT_COL)
       .head(1)[[RAT_COL, BIN_COL_N_1, 'min_side_count', 'min_side_action']]
)
worst = worst.reset_index(drop=True)


n_bootstraps = 1000
boot_results = []

for rat in agg[RAT_COL].unique():
    # numero minimo di trial per lato (sul bin "peggiore" di quel rat)
    n_samples_per_condition = worst.loc[worst['rat'] == rat, 'min_side_count'].item()
    print(f'Rat {rat} will use {n_samples_per_condition} samples per condition (side).')
    
    # usa l'80% (arrotondato per eccesso), ma non andare sotto 1
    n_samples_per_condition_80 = max(1, int(np.ceil(n_samples_per_condition * 0.8)) - 1)
    print(f'--> Using {n_samples_per_condition_80} samples per condition (side) for bootstrapping.')
    
    for bin_label in agg[BIN_COL_N_1].unique():
        # subset per rat & prev_bin
        df_subset = dfc[(dfc[RAT_COL] == rat) & (dfc[BIN_COL_N_1] == bin_label)]
        
        # separa in "right" e "left" secondo ACTION_N_1 (azione precedente)
        right_trials = df_subset[df_subset[ACTION_N_1] == 1]
        left_trials  = df_subset[df_subset[ACTION_N_1] == 0]
        
        # se mancano dati da uno dei due lati, salta questo bin
        if (len(right_trials) == 0) or (len(left_trials) == 0):
            print(f"Skipping rat {rat}, prev_bin {bin_label}: not enough trials on at least one side.")
            continue
        
        for b in range(n_bootstraps):
            # resample con rimpiazzo
            samp_r = right_trials.sample(n_samples_per_condition_80, replace=True)
            samp_l = left_trials.sample(n_samples_per_condition_80, replace=True)
            
            samp_dataset = pd.concat([samp_r, samp_l], ignore_index=True)

            popt, ok, x_fit_dummy, y_fit_dummy = fit_psychometric_curve(
                samp_dataset[BIN_COL_N_MID],
                samp_dataset[ACTION]  # risposta corrente
            )

            if not ok:
                continue

            boot_results.append({
                'rat': rat,
                'prev_bin': bin_label,
                'bootstrap': b,
                'mu':    popt[0],
                'sigma': popt[1],
                'gamma': popt[2],
                'lapse': popt[3],
            })

# -----------------------------
# Dopo i loop: analisi bootstrap + plotting
# -----------------------------
boot_df = pd.DataFrame(boot_results)

group_cols = [RAT_COL, 'prev_bin']
g = boot_df.groupby(group_cols)

summary = (
    g.agg(
        mu_med     = ('mu', 'median'),
        mu_low     = ('mu', lambda x: x.quantile(0.025)),
        mu_high    = ('mu', lambda x: x.quantile(0.975)),
        sigma_med  = ('sigma', 'median'),
        sigma_low  = ('sigma', lambda x: x.quantile(0.025)),
        sigma_high = ('sigma', lambda x: x.quantile(0.975)),
        gamma_med  = ('gamma', 'median'),
        lapse_med  = ('lapse', 'median'),
    )
    .reset_index()
)

# plotting finale
x_fit = np.linspace(0, 90, 200)


for rat in boot_df[RAT_COL].unique():
    plt.figure()
    sub = summary[summary[RAT_COL] == rat]
    for _, row in sub.iterrows():
        
        y_fit = cumulative_gaussian_lapse(
            x_fit,
            row['mu_med'],
            row['sigma_med'],
            row['gamma_med'],
            row['lapse_med'],
        )
        label = f"rat {row[RAT_COL]}, prev_bin {row['prev_bin']}"
        plt.plot(x_fit, y_fit, label=label)

    plt.xlabel("Angle (bin midpoint)")
    plt.ylabel("P(right)")
    plt.legend()
    plt.title("Bootstrap psychometric curves by rat and previous bin")
    plt.show()

