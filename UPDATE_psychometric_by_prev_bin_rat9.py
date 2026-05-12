# psychometric_by_prev_bin_rat9.py
# ---------------------------------------------------------------
# Psychometric curves for rat 9, conditioned on previous-trial
# angle bin. Angles are binned before fitting to avoid sparse
# unique-angle instability.
# ---------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf

from serial_dependence_analysis import SerialDependenceAnalyzer
import functions_Ale as fa

# -----------------------------
# CONFIG
# -----------------------------
RAT_ID        = 11
BIN_COL_N_1   = 'bin_angle_n-1'
BIN_COL_N     = 'bin_angle_n'
ANGLE_COL     = 'angle'
RESP_COL      = 'action'
RAT_COL       = 'rat'
N_BINS_FIT    = 19      # ~5° bins for fitting (same as working solution)
MIN_TRIALS    = 5       # minimum trials per bin to include in fit

# -----------------------------
# PSYCHOMETRIC FUNCTION
# -----------------------------
def cumulative_gaussian_lapse(x, mu, sigma, gamma, lapse):
    x = np.asarray(x, dtype=float)
    return gamma + (1 - gamma - lapse) * (
        0.5 * (1 + erf((x - mu) / np.sqrt(2 * sigma**2)))
    )

def fit_psychometric_binned(angles, responses, n_bins=N_BINS_FIT, min_trials=MIN_TRIALS):
    """
    Bin raw trials into n_bins, compute per-bin proportions,
    then fit cumulative Gaussian via curve_fit.
    Returns (popt, x_fit, y_fit) or (None, None, None) on failure.
    """
    bin_edges   = np.linspace(0, 90, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx     = np.clip(np.digitize(angles, bin_edges, right=False) - 1, 0, n_bins - 1)

    obs_x, obs_p = [], []
    for b, center in enumerate(bin_centers):
        mask = bin_idx == b
        if mask.sum() < min_trials:
            continue
        obs_x.append(center)
        obs_p.append(np.mean(responses[mask]))

    if len(obs_x) < 4:   # need at least 4 points for 4 parameters
        return None, None, None

    obs_x = np.array(obs_x)
    obs_p = np.array(obs_p)

    try:
        popt, _ = curve_fit(
            cumulative_gaussian_lapse, obs_x, obs_p,
            p0=[45, 15, 0.01, 0.02],
            bounds=([20,  5,  0.0, 0.0],
                    [70, 50,  0.3, 0.3]),
            maxfev=5000
        )
        x_fit = np.linspace(0, 90, 300)
        y_fit = cumulative_gaussian_lapse(x_fit, *popt)
        return popt, x_fit, y_fit
    except Exception:
        return None, None, None

# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data(fold=False)
df_processed = an.create_lagged_features()
df_processed = df_processed[
    (df_processed[ANGLE_COL]    >= 0) & (df_processed[ANGLE_COL]    <= 90)
]
df_processed = df_processed[
    (df_processed['angle_n-1'] >= 0) & (df_processed['angle_n-1'] <= 90)
]
df_processed = df_processed[df_processed['mod']==3]
# -----------------------------
# BINNING
# -----------------------------
df_processed, _ = fa.bin_by_unique_angles(
    df_processed, name_new_col=BIN_COL_N_1,
    n_groups=6, angle_col='angle_n-1', exclude_angle=True
)
df_processed, angle_to_label = fa.bin_by_unique_angles(
    df_processed, name_new_col=BIN_COL_N,
    n_groups=6, angle_col=ANGLE_COL, exclude_angle=True
)

dfc = df_processed.dropna(subset=[BIN_COL_N_1, BIN_COL_N]).copy()

# -----------------------------
# COLOR MAPPING
# -----------------------------
bin_means_prev = {b: fa.midpoint(b) for b in pd.Series(dfc[BIN_COL_N_1]).dropna().unique()}
color_by_bin, _ = fa.color_bin(bin_means_prev)

# -----------------------------
# FILTER RAT 9
# -----------------------------
dfc_rat = dfc[dfc[RAT_COL] == RAT_ID].copy()
print(f"Rat {RAT_ID}: {len(dfc_rat)} trials")

# -----------------------------
# PLOT
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 5))

for b in sorted(bin_means_prev, key=bin_means_prev.get):
    color = color_by_bin[b]

    raw = dfc_rat[dfc_rat[BIN_COL_N_1].astype(str) == str(b)]
    if raw.empty:
        continue

    angles    = pd.to_numeric(raw[ANGLE_COL], errors='coerce').values
    responses = pd.to_numeric(raw[RESP_COL],  errors='coerce').values
    valid     = ~(np.isnan(angles) | np.isnan(responses))
    angles, responses = angles[valid], responses[valid]

    if len(angles) < MIN_TRIALS:
        continue

    # ── SCATTER: 6 evenly-spaced observed proportions ─────────────────
    angle_points = (raw.groupby(ANGLE_COL)[RESP_COL]
                    .agg(mean='mean', n='size')
                    .reset_index())
    angle_points = angle_points[angle_points['n'] >= MIN_TRIALS]
    angle_points['se'] = np.sqrt(
        angle_points['mean'] * (1 - angle_points['mean']) / angle_points['n']
    )
    idx = np.round(np.linspace(0, len(angle_points) - 1, 6)).astype(int)
    pts = angle_points.iloc[idx]

    ax.errorbar(pts[ANGLE_COL], pts['mean'], yerr=pts['se'],
                fmt='o', ms=4, alpha=0.8, color=color,
                label=f'prev {b}')

    # ── FIT: bin angles → stable proportions → curve_fit ──────────────
    popt, x_fit, y_fit = fit_psychometric_binned(angles, responses)
    if popt is not None:
        mu, sigma, gamma, lapse = popt
        ax.plot(x_fit, y_fit, color=color, alpha=0.9, linewidth=1.8)
        print(f"  prev-bin {b:20s} | μ={mu:.1f}°  σ={sigma:.1f}°  γ={gamma:.3f}  λ={lapse:.3f}  N={len(angles)}")
    else:
        print(f"  prev-bin {b:20s} | fit failed  N={len(angles)}")

# ── REFERENCE CURVE (all trials, gray dashed) ─────────────────────────
all_angles    = pd.to_numeric(dfc_rat[ANGLE_COL], errors='coerce').values
all_responses = pd.to_numeric(dfc_rat[RESP_COL],  errors='coerce').values
valid         = ~(np.isnan(all_angles) | np.isnan(all_responses))
popt_r, x_fit_r, y_fit_r = fit_psychometric_binned(all_angles[valid], all_responses[valid])
if popt_r is not None:
    mu_r, sigma_r, *_ = popt_r
    ax.plot(x_fit_r, y_fit_r, color='gray', alpha=0.4, ls='--', linewidth=2,
            label=f'Overall (μ={mu_r:.1f}°, σ={sigma_r:.1f}°)')
    print(f"\n  Overall              | μ={mu_r:.1f}°  σ={sigma_r:.1f}°")

# ── DECORATIONS ───────────────────────────────────────────────────────
ax.axhline(0.5, color='k', ls='--', alpha=0.4)
ax.axvline(45,  color='k', ls='--', alpha=0.4)
ax.set_xlim(0, 90)
ax.set_ylim(0, 1)
ax.set_xlabel('Current angle (°)', fontsize=12)
ax.set_ylabel('P(respond = 1)',     fontsize=12)
ax.set_title(f'Rat {RAT_ID} — Psychometric curves by previous-trial angle bin',
             fontsize=13, fontweight='bold')
ax.legend(title='Prev angle bin', fontsize=9, loc='upper left')

plt.tight_layout()
plt.savefig(f'psychometric_prev_bin_rat{RAT_ID}.png', dpi=150)
plt.show()