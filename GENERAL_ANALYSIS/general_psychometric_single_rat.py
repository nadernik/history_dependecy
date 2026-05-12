# general_psychometric_rat9.py
# This script plots psychometric curves for one target rat (changeable at the top of the script). 
# there are blobs per each angle showing how many trials we have for that angle (size of the blob), and a fitted psychometric curve.
# On the possible binning size of angles (original vs binned as used in other psychometrics) I should move forward with the discussion 
# between min log likelihood methods and least squared on proportion of corrected (which is what we have been doing so far). 
# The former allows you to fit the curve on single-trial data, without binning, but it is more sensitive to noise and outliers, 
# while the latter is more robust but requires binning the data into angle groups and computing the proportion of correct responses for each group.
# This script uses the first method (fitting on single-trial data) but I can easily change it to the second method if we decide to go with that one.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf
from serial_dependence_analysis import SerialDependenceAnalyzer
from functions_Ale import filter_training_trials

RAT_ID    = 6  # <-- change this to run on a different rat
ANGLE_COL = 'angle'
RESP_COL  = 'action'
RAT_COL   = 'rat'
N_BINS    = 19  # ~5° bins across 0–90°

# --- LOAD ---
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data(fold=False)
df_processed = an.create_lagged_features()
df_processed  = filter_training_trials(df_processed, method='criterion', criterion=0.80, min_consec=2)
df_processed = df_processed[
    (df_processed[ANGLE_COL]    >= 0) & (df_processed[ANGLE_COL]    <= 90)
]
df_processed = df_processed[
    (df_processed['angle_n-1'] >= 0) & (df_processed['angle_n-1'] <= 90)
]
mod_transitions = df_processed['mod_transition_n-1'].unique()
modalities = df_processed['mod'].unique()
#modalities = sorted(modalities).copy
'''
for tr in mod_transitions:
    df_processed_tr = df_processed[df_processed['mod_transition_n-1'] == tr]
    df_rat   = df_processed_tr[df_processed_tr[RAT_COL] == RAT_ID].copy()
    angles    = df_rat[ANGLE_COL].values.astype(float)
    responses = df_rat[RESP_COL].values.astype(float)
    print(f"Rat {RAT_ID}: {len(df_rat)} trials")
'''
for md in modalities:
    print(f"\n=== Modality {md} ===")
    df_processed_md = df_processed[df_processed['mod'] == md]
    df_rat   = df_processed_md[df_processed_md[RAT_COL] == RAT_ID].copy()
   
    
#df_processed_tr = df_processed[df_processed['mod_transition_n-1'] == tr]
#df_rat   = df_processed[df_processed[RAT_COL] == RAT_ID].copy()

    angles    = df_rat[ANGLE_COL].values.astype(float)
    responses = df_rat[RESP_COL].values.astype(float)
    print(f"Rat {RAT_ID}, {len(df_rat)} trials")
    '''
    df_rat   = df_processed[df_processed[RAT_COL] == RAT_ID].copy()
    angles    = df_rat[ANGLE_COL].values.astype(float)
    responses = df_rat[RESP_COL].values.astype(float)
    print(f"Rat {RAT_ID}: {len(df_rat)} trials")
    '''
    # --- BIN into ~5° bins ---
    bin_edges   = np.linspace(0, 90, N_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx     = np.clip(np.digitize(angles, bin_edges, right=False) - 1, 0, N_BINS - 1)

    obs_x, obs_p, obs_n = [], [], []

    for b, center in enumerate(bin_centers):
        mask = bin_idx == b
        if mask.sum() < 5:
            continue
        obs_x.append(center)
        obs_p.append(np.mean(responses[mask]))
        obs_n.append(mask.sum())

    obs_x = np.array(obs_x)
    obs_p = np.array(obs_p)
    obs_n = np.array(obs_n)

    # --- FIT directly on binned proportions ---
    def cumulative_gaussian_lapse(x, mu, sigma, gamma, lambda_param):
        x = np.asarray(x, dtype=float)
        return gamma + (1 - gamma - lambda_param) * (
            0.5 * (1 + erf((x - mu) / np.sqrt(2 * sigma**2)))
        )

    p0     = [45, 15, 0.01, 0.02]
    bounds = ([20,  5,  0.0, 0.0],
            [70, 50,  0.3, 0.3])   # ← lapses unconstrained, matching your original intent

    try:
        popt, pcov = curve_fit(
            cumulative_gaussian_lapse, obs_x, obs_p,
            p0=p0, bounds=bounds, maxfev=5000
        )
        mu, sigma, gamma, lapse = popt
        fit_success = True
        print(f"  mu={mu:.1f}°  sigma={sigma:.1f}°  gamma={gamma:.4f}  lapse={lapse:.4f}")
    except Exception as e:
        fit_success = False
        print(f"Fit failed: {e}")

    # --- PLOT ---
    fig, ax = plt.subplots(figsize=(7, 5))

    sizes = (obs_n / obs_n.max()) * 350 + 40
    ax.scatter(obs_x, obs_p,
            s=sizes, color='steelblue',
            edgecolors='white', linewidths=0.8,
            zorder=5, label='Observed P(respond=1)')

    if fit_success:
        x_fit = np.linspace(0, 90, 300)
        y_fit = cumulative_gaussian_lapse(x_fit, *popt)
        ax.plot(x_fit, y_fit,
                color='tomato', linewidth=2.5, zorder=4,
                label=f'Psychometric fit\nμ={mu:.1f}°, σ={sigma:.1f}°\nγ={gamma:.3f}, λ={lapse:.3f}')

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axvline(45,  color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_xlabel('Angle (°)',       fontsize=13)
    ax.set_ylabel('P(respond = 1)', fontsize=13)
    ax.set_title(f'Psychometric Curve — Rat {RAT_ID}', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 90)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.arange(0, 91, 10))
    ax.legend(fontsize=10, loc='upper left')
    ax.annotate('Dot size ∝ trial count',
                xy=(0.98, 0.04), xycoords='axes fraction',
                ha='right', fontsize=9, color='gray')

    plt.tight_layout()
    #safe_tr = str(tr).replace("->", "-to-")

    #plt.savefig(f'psychometric_rat{RAT_ID}_Tr_{safe_tr}.png', dpi=150)
    #plt.savefig(f'psychometric_rat{RAT_ID}, Tr {tr}.png', dpi=150)
    plt.show()