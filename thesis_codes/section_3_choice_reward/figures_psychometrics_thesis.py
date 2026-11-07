"""
psychometric_4analyses_all_rats.py
=====================================
Builds the psychometric-curve figures for every rat, restricted to the two
analyses that are a DIRECT, exact match to what the accurate pipeline
(Section_3_choice_reward_joint_bootstrap.py) actually fit -- no approximation,
no pooling, no gap-filling:

    Analysis 3: fake-45deg (from flanking angles 40/50), split by n-1 choice,
                conditioned on n-1 rewarded          -> cells h_rew / v_rew
    Analysis 4: fake-45deg, split by n-1 choice, conditioned on n-1 unrewarded
                                                       -> cells h_unrew / v_unrew

REVISED 2026-09-18 (per Alessandro): dropped the other two analyses from your
original script rather than approximate them --
  - Analysis 2 (fake-45, reward NOT split) previously used a trial-weighted
    pooling approximation across the rewarded/unrewarded bootstrap draws.
    Removed: not something the delivered pipeline fit directly.
  - Analysis 1 (real-45deg trials) was never buildable from anything
    delivered -- the accurate pipeline's cells are all flanking-angle (40/50)
    based, never real-45. Still not buildable; still skipped.
This script now only produces what can be built perfectly, i.e. 24 figures
(12 rats x 2 analyses), each a straight readout of one bootstrapped cell pair.

Colors are in the COLORS dict right below CONFIG -- change them there, nothing
else in the file needs to change.

Curve drawn with the cumulative Gaussian (norm.cdf) used by the actual
fitting script (neg_log_likelihood -> norm.cdf in
Section_3_choice_reward_joint_bootstrap.py). mu/sigma/gamma/lapse themselves
come straight from the bootstrap npz, fit by that script; this file only
re-draws the curve shape from those parameters.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

import bootstrap_qc as qc

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
# Real local paths (use these when running on your machine):
DATA_DIR = r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_3_choice_reward\choice_reward_joint_results\filtered\pooled\tables"
BOOTSTRAP_DIR = r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_3_choice_reward\choice_reward_joint_results\filtered\pooled\bootstraps"

# This session's uploaded copies (swap DATA_DIR/BOOTSTRAP_DIR for the two
# lines above when running locally; NPZ_FILES becomes
# {rat: f"rat_{rat}_joint_bootstrap.npz" for rat in RATS_ALL}):


OUT_DIR = r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\ale_thesis_fig_tab\images_thesis\section_3"
os.makedirs(OUT_DIR, exist_ok=True)

CELL_COMP_FILE = "cell_composition.csv"
NPZ_FILES = {
    2: "rat_2_joint_bootstrap.npz", 3: "rat_3_joint_bootstrap.npz",
    4: "rat_4_joint_bootstrap.npz", 5: "rat_5_joint_bootstrap.npz",
    6: "rat_6_joint_bootstrap.npz", 7: "rat_7_joint_bootstrap.npz",
    9: "rat_9_joint_bootstrap.npz", 10: "rat_10_joint_bootstrap.npz",
    11: "rat_11_joint_bootstrap.npz", 12: "rat_12_joint_bootstrap.npz",
    13: "rat_13_joint_bootstrap.npz", 16: "rat_16_joint_bootstrap.npz",
}

X_PLOT = np.linspace(0, 90, 200)

# ----------------------------------------------------------------------------
# COLORS -- the only thing you should need to touch to restyle this script.
# ----------------------------------------------------------------------------
COLORS = {
    "group0": "#5C36F4",   # prev-H (n-1 horizontal) curve + points
    "group1": "#F32128",   # prev-V (n-1 vertical) curve + points
    "axis_ref": "gray",    # the x=45deg / y=0.5 dotted reference lines
    "delta_point": "black",  # the delta-mu point + errorbar in panel 2
    "not_identified_note": "#e34948",  # footnote color when gamma/lapse is bound-hit
}

_cell_comp_df = pd.read_csv(os.path.join(DATA_DIR, CELL_COMP_FILE))
_cell_comp_df = _cell_comp_df[_cell_comp_df["n_trials"] > 0]
_full_fits_df = pd.read_csv(os.path.join(DATA_DIR, "full_data_cell_fits.csv"))


def psychometric(x, mu, sigma, gamma, lapse):
    """4-parameter cumulative-Gaussian psychometric function -- matches
    neg_log_likelihood() in Section_3_choice_reward_joint_bootstrap.py, the
    script that actually produced mu/sigma/gamma/lapse."""
    return gamma + (1.0 - gamma - lapse) * norm.cdf((x - mu) / sigma)


def n_trials_for(rat, cell):
    row = _cell_comp_df[(_cell_comp_df["rat"] == rat) & (_cell_comp_df["cell"] == cell)]
    return int(row["n_trials"].sum()) if len(row) else 0


# Each analysis: how to get (mu,sigma,gamma,lapse) bootstrap arrays + exclusion
# mask + n_trials for group0 (prev-H) and group1 (prev-V), plus display info.
ANALYSES = [
    {
        "key": "analysis3_rewarded",
        "kind": "direct",
        "cell0": "h_rew", "cell1": "v_rew", "contrast": "delta_mu_rew",
        "title": "Analysis 3: fake-45°, split by n-1 choice, conditioned on n-1 rewarded",
        "condition_str": "previous trial rewarded",
    },
    {
        "key": "analysis4_unrewarded",
        "kind": "direct",
        "cell0": "h_unrew", "cell1": "v_unrew", "contrast": "delta_mu_unrew",
        "title": "Analysis 4: fake-45°, split by n-1 choice, conditioned on n-1 unrewarded",
        "condition_str": "previous trial unrewarded",
    },
]


def build_figure(rat, analysis, out_dir=OUT_DIR):
    data = qc.load_bootstrap(rat, os.path.join(BOOTSTRAP_DIR, NPZ_FILES[rat]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    manifest = {"rat": rat, "analysis": analysis["key"]}

    cell0, cell1 = analysis["cell0"], analysis["cell1"]
    sigma_flag0 = qc.sigma_degenerate_mask(data, cell0)
    sigma_flag1 = qc.sigma_degenerate_mask(data, cell1)
    p0 = {p: qc.cell_param(data, cell0, p) for p in ("mu", "sigma", "gamma", "lapse")}
    p1 = {p: qc.cell_param(data, cell1, p) for p in ("mu", "sigma", "gamma", "lapse")}
    groups = [("group0", p0, sigma_flag0, n_trials_for(rat, cell0), "prev-H (n-1 horizontal)"),
              ("group1", p1, sigma_flag1, n_trials_for(rat, cell1), "prev-V (n-1 vertical)")]
    delta_mu_full = qc.cell_contrast(data, analysis["contrast"])
    not_identified_any = bool(qc.full_data_boundary_flags(_full_fits_df, rat, cell0) or
                               qc.full_data_boundary_flags(_full_fits_df, rat, cell1))

    # ---- panel 1: curves (sigma-degenerate iterations excluded from band only) ----
    for side, params, mask, n, label in groups:
        keep = ~mask
        curves = psychometric(X_PLOT[None, :], params["mu"][keep, None], params["sigma"][keep, None],
                               params["gamma"][keep, None], params["lapse"][keep, None])
        median_curve = np.median(curves, axis=0)
        ci_low = np.percentile(curves, 2.5, axis=0)
        ci_high = np.percentile(curves, 97.5, axis=0)
        color = COLORS[side]
        axes[0].plot(X_PLOT, median_curve, color=color, label=f"{label}  (n={n})", lw=2)
        axes[0].fill_between(X_PLOT, ci_low, ci_high, alpha=0.25, color=color)
        manifest[f"{side}_n_trials"] = n
        manifest[f"{side}_n_sigma_degenerate_excluded"] = int(mask.sum())

    axes[0].axvline(45, color=COLORS["axis_ref"], linestyle=":", alpha=0.5)
    axes[0].axhline(0.5, color=COLORS["axis_ref"], linestyle=":", alpha=0.5)
    axes[0].set_xlabel("Stimulus angle (°)", fontsize=12)
    axes[0].set_ylabel("P(vertical)", fontsize=12)
    axes[0].set_title(f"Rat {rat} — psychometric curves\n({analysis['condition_str']})", fontsize=10.5)
    axes[0].legend(fontsize=9)
    axes[0].set_ylim(-0.05, 1.05)

    if not_identified_any:
        axes[0].text(0.02, 0.02, "gamma/lapse not identified for one curve\n(box-constrained fit; mu unaffected)",
                     transform=axes[0].transAxes, fontsize=7.5, color=COLORS["not_identified_note"],
                     va="bottom", ha="left",
                     bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="none"))

    # ---- panel 2: delta mu, full unfiltered bootstrap ----
    med_delta = np.median(delta_mu_full)
    ci_low_d, ci_high_d = np.percentile(delta_mu_full, [2.5, 97.5])

    axes[1].errorbar(x=0, y=med_delta, yerr=[[med_delta - ci_low_d], [ci_high_d - med_delta]],
                     fmt="o", color=COLORS["delta_point"], capsize=8, markersize=10, lw=2)
    axes[1].axhline(0, color=COLORS["axis_ref"], linestyle="--", alpha=0.6)
    axes[1].set_xlim(-1, 1)
    axes[1].set_xticks([])
    axes[1].set_ylabel("Δμ  (prev-V − prev-H) [°]", fontsize=12)
    axes[1].set_title(f"Rat {rat} — shift in perceptual boundary\n95% bootstrap CI (n={data['n_boot']})", fontsize=10.5)

    data_min = min(ci_low_d, 0.0); data_max = max(ci_high_d, 0.0)
    data_range = max(data_max - data_min, 1.0)
    axes[1].set_ylim(data_min - 0.15 * data_range, data_max + 0.35 * data_range)
    axes[1].text(0, data_max + 0.20 * data_range,
                 f"Δμ = {med_delta:.2f}°\n95% CI [{ci_low_d:.2f}, {ci_high_d:.2f}]",
                 ha="center", va="center", fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.85))

    manifest["delta_mu_median"] = round(float(med_delta), 3)
    manifest["delta_mu_ci_low"] = round(float(ci_low_d), 3)
    manifest["delta_mu_ci_high"] = round(float(ci_high_d), 3)

    plt.suptitle(f"Rat {rat}  |  {analysis['title']}", fontsize=11.5, y=1.02)
    fig.tight_layout()

    path = os.path.join(out_dir, f"rat_{rat}_{analysis['key']}.svg")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    manifest["figure_path"] = path
    return manifest


def build_all(rats, out_dir=OUT_DIR, manifest_csv=None):
    rows = []
    for rat in rats:
        for analysis in ANALYSES:
            rows.append(build_figure(rat, analysis, out_dir))
    manifest = pd.DataFrame(rows)
    if manifest_csv:
        manifest.to_csv(manifest_csv, index=False)
    return manifest


if __name__ == "__main__":
    RATS_ALL = sorted(NPZ_FILES.keys())
    manifest = build_all(RATS_ALL, manifest_csv=r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_3_choice_reward\choice_reward_joint_results\filtered\pooled\psychometric_4analyses_manifest.csv")
    print(f"{len(manifest)} figures written to {OUT_DIR}")