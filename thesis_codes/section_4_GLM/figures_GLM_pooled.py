"""
Section 4 pooled-GLM figures, single script:
  Figure 1 -- PSE comparison, psychometric vs GLM, degree scale
  Figure 2 -- success/failure plane, psychometric vs GLM, with an uncertainty
              cloud per rat x method (bivariate-normal ellipse by default, or
              raw bootstrap scatter -- see CLOUD_STYLE below)

Data sources (uploaded by Alessandro, all `filtered, pooled` scope):
- bootstrap_parameter_summary.csv   -> psychometric mu (PSE) + 95% CI, per rat x cell
- contrast_summary_long.csv         -> psychometric delta_mu_rew / delta_mu_unrew point estimates
- contrast_covariance_long.csv      -> psychometric bootstrap covariance of the two contrasts
- glm_summary_long.csv              -> GLM pse_* / S_glm / D_glm point estimates + 95% CI (degrees)
- glm_bootstrap_draws.parquet       -> raw GLM bootstrap draws (5000/rat)
- rat_<N>_joint_bootstrap.npz       -> raw psychometric bootstrap draws (theta_bootstrap_x_y =
                                        delta_mu_rew, delta_mu_unrew), one file per rat. Only
                                        needed for CLOUD_STYLE = "raw"; currently uploaded for
                                        rats 9, 11, 12, 13.

GLM fits used throughout: C = "unpenalized" (MLE), matching the convention already
used for the coefficient tables in this section.

--------------------------------------------------------------------------------
FLAGS (the only things to touch for a different run)
--------------------------------------------------------------------------------
RATS_TO_PLOT   -- which rat(s) actually get drawn. Any subset of ALL_RATS, in any
                  order, e.g. [9, 11, 12, 13], ALL_RATS (full cohort), or a single
                  rat [13]. Drives both Figure 1 and Figure 2.
CI_LEVEL       -- confidence level for the Figure 2 clouds / Figure 1 error bars
                  come straight from the source files' own 95% CI columns
                  (unaffected by this flag) -- CI_LEVEL only sizes Figure 2's cloud.
CLOUD_STYLE    -- Figure 2 uncertainty-cloud style:
                    "ellipse" (default) -- bivariate-normal confidence ellipse
                             from the covariance. Needs only covariance, so works
                             for any rat in ALL_RATS.
                    "raw"              -- raw bootstrap draws plotted directly as
                             a point cloud (no distributional assumption). Needs
                             each plotted rat's joint_bootstrap.npz on the
                             psychometric side (see PSYCH_DRAWS_NPZ below) -- only
                             rats 9/11/12/13 currently have that file uploaded.
PLANE_FIXED    -- Figure 2 axis-scaling mode:
                    True  (default) -> "full plane": axes are symmetric (-L, +L)
                             on both sides, sized so EVERY rat in the full cohort
                             would fit at the current CI_LEVEL, regardless of which
                             subset is actually plotted. Keeps figures from
                             different rat selections visually comparable.
                    False -> axes crop tightly to whatever is in RATS_TO_PLOT.
N_SCATTER      -- draws plotted per cloud when CLOUD_STYLE = "raw" (subsampled
                  from the full 5000, for a readable, not-solid-ink cloud).
--------------------------------------------------------------------------------
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import chi2

# ----------------------------------------------------------------------------
# 0. Config -- the flags described above
# ----------------------------------------------------------------------------

UPLOAD_DIR = Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_3_choice_reward\choice_reward_joint_results\filtered\pooled")
PATHS = dict(
    boot_param   = f"{UPLOAD_DIR}/tables/bootstrap_parameter_summary.csv",
    contrast_sum = f"{UPLOAD_DIR}/tables/contrast_summary_long.csv",
    contrast_cov = f"{UPLOAD_DIR}/tables/contrast_covariance_long.csv",
    glm_summary  = r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_4_GLM\fit_results\pooled_filtered\glm_summary_long.csv",
    glm_draws    = r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_4_GLM\fit_results\pooled_filtered\glm_bootstrap_draws.parquet",
)
# psychometric raw joint-bootstrap draws, one file per rat (only needed for CLOUD_STYLE="raw")
PSYCH_DRAWS_NPZ = {
    2:  f"{UPLOAD_DIR}/bootstrap/rat_2_joint_bootstrap.npz",
    3:  f"{UPLOAD_DIR}/bootstrap/rat_3_joint_bootstrap.npz",
    4:  f"{UPLOAD_DIR}/bootstrap/rat_4_joint_bootstrap.npz",
    5:  f"{UPLOAD_DIR}/bootstrap/rat_5_joint_bootstrap.npz",
    6:  f"{UPLOAD_DIR}/bootstrap/rat_6_joint_bootstrap.npz",
    7:  f"{UPLOAD_DIR}/bootstrap/rat_7_joint_bootstrap.npz",
    9:  f"{UPLOAD_DIR}/bootstrap/rat_9_joint_bootstrap.npz",
    10: f"{UPLOAD_DIR}/bootstrap/rat_10_joint_bootstrap.npz",
    11: f"{UPLOAD_DIR}/bootstrap/rat_11_joint_bootstrap.npz",
    12: f"{UPLOAD_DIR}/bootstrap/rat_12_joint_bootstrap.npz",
    13: f"{UPLOAD_DIR}/bootstrap/rat_13_joint_bootstrap.npz",
    16: f"{UPLOAD_DIR}/bootstrap/rat_16_joint_bootstrap.npz",
}

GLM_C = "unpenalized"

boot_param   = pd.read_csv(PATHS["boot_param"])
contrast_sum = pd.read_csv(PATHS["contrast_sum"])
contrast_cov = pd.read_csv(PATHS["contrast_cov"])
glm_summary  = pd.read_csv(PATHS["glm_summary"])

ALL_RATS = sorted(contrast_sum.rat.unique().tolist())   # full pooled cohort: [2,3,4,5,6,7,9,10,11,12,13,16]

# ---- the flags ----
RATS_TO_PLOT = [2,3,4,5,6,7,10,16]   # <-- change to ALL_RATS, or any custom list, e.g. [13] or [9, 13]
CI_LEVEL = 95                    # <-- confidence level for the Figure 2 cloud
CLOUD_STYLE = "ellipse"          # <-- "ellipse" (default, bivariate-normal) or "raw" (bootstrap scatter)
SHOW_PSYCH_CI = False                 # <-- whether to show the psychometric cloud in Figure 2 (default True)
PLANE_FIXED = True               # <-- True = fixed "full plane" frame sized to the whole cohort
N_SCATTER = 400                  # <-- draws per cloud when CLOUD_STYLE == "raw"
RNG = np.random.default_rng(0)
# -------------------

if CLOUD_STYLE not in ("ellipse", "raw"):
    raise ValueError(f'CLOUD_STYLE must be "ellipse" or "raw", got {CLOUD_STYLE!r}')

if CLOUD_STYLE == "raw":
    missing = [r for r in RATS_TO_PLOT if r not in PSYCH_DRAWS_NPZ]
    if missing:
        raise ValueError(
            f"CLOUD_STYLE='raw' needs each plotted rat's joint_bootstrap.npz on the psychometric "
            f"side. Missing for rat(s) {missing} -- either upload those files and add them to "
            f"PSYCH_DRAWS_NPZ, drop them from RATS_TO_PLOT, or switch CLOUD_STYLE back to 'ellipse' "
            f"(which only needs the covariance summary and works for any rat)."
        )

RATS_LABEL = ("full_cohort" if sorted(RATS_TO_PLOT) == sorted(ALL_RATS)
              else "d_rats" if sorted(RATS_TO_PLOT) == sorted([9, 11, 12, 13])
              else "custom")

# 8-slot dataviz-skill validated categorical palette (light surface). Cycled if
# more than 8 rats are plotted at once -- direct rat-number labels on every
# point carry identity beyond that point anyway, so a repeated hue is fine.
PALETTE_8 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
             "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
RAT_COLOR = {rat: PALETTE_8[i % len(PALETTE_8)] for i, rat in enumerate(ALL_RATS)}
# Figure 1 colors by previous-trial OUTCOME (rewarded vs unrewarded), not by V/H.
# V and H are the previous-CHOICE category (both fit at the pooled scope -- all
# three sensory-modality contexts collapsed together -- so they are NOT visual
# vs. tactile trials; see note in psychometric_pse/glm_pse below).
REW_COLOR = {"rew": "#008300", "unrew": "#4a3aa7"}   # green / violet, validated adjacent pair

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",   # keep text as real text (not paths) in the SVG
})

def save(fig, stub):
    fig.savefig(f"{stub}.svg", bbox_inches="tight")
    fig.savefig(f"{stub}.pdf", bbox_inches="tight")

# ----------------------------------------------------------------------------
# 1. Figure 1 -- PSE comparison, psychometric vs GLM (degrees)
# ----------------------------------------------------------------------------

# condition order requested: V-rew, H-rew, V-unrew, H-unrew
CONDS = [
    ("V", "rew",   "v_rew",   "pse_V_rewarded"),
    ("H", "rew",   "h_rew",   "pse_H_rewarded"),
    ("V", "unrew", "v_unrew", "pse_V_unrewarded"),
    ("H", "unrew", "h_unrew", "pse_H_unrewarded"),
]

# NOTE on V/H: bootstrap_parameter_summary.csv is scope="pooled" -- all trials
# from all three sensory-modality contexts (visual, tactile, visuotactile)
# collapsed together before fitting. So "v_rew"/"h_rew"/... are NOT split by
# sensory modality; "V"/"H" is the previous-trial CHOICE category (crossed
# with previous-trial outcome), per the cell naming used throughout Section 3
# ("rewarded-horizontal", "unrewarded-vertical", etc. in section_3_results_draft.md).
# Figure 1 therefore encodes color by outcome (REW_COLOR), not by V/H.

def psychometric_pse(rat, cell):
    row = boot_param[(boot_param.rat == rat) & (boot_param.cell == cell) & (boot_param.parameter == "mu")]
    row = row.iloc[0]
    return row.estimate_full_data, row.ci_low, row.ci_high

def glm_pse(rat, quantity):
    row = glm_summary[(glm_summary.rat_id == rat) & (glm_summary.C == GLM_C) & (glm_summary.quantity == quantity)]
    row = row.iloc[0]
    return row.value, row.ci_low, row.ci_high

def build_figure1():
    n_rats = len(RATS_TO_PLOT)
    fig1, axes = plt.subplots(1, n_rats, figsize=(2.85 * n_rats, 3.4), sharey=True, squeeze=False)
    axes = axes[0]

    x_positions = np.arange(len(CONDS))
    dx = 0.13  # offset between the psychometric and GLM point within a condition
    all_vals = []

    for ax, rat in zip(axes, RATS_TO_PLOT):
        for xi, (mod, rew, cell, glm_q) in zip(x_positions, CONDS):
            color = REW_COLOR[rew]
            p_est, p_lo, p_hi = psychometric_pse(rat, cell)
            g_est, g_lo, g_hi = glm_pse(rat, glm_q)
            all_vals += [p_lo, p_hi, g_lo, g_hi]

            xp, xg = xi - dx, xi + dx
            ax.plot([xp, xg], [p_est, g_est], color=color, lw=0.9, alpha=0.55, zorder=1)
            ax.errorbar(xp, p_est, yerr=[[p_est - p_lo], [p_hi - p_est]], fmt="o",
                        color=color, mfc=color, mec=color, ms=5.5, mew=0.8,
                        ecolor=color, elinewidth=1.1, capsize=2.5, zorder=3)
            ax.errorbar(xg, g_est, yerr=[[g_est - g_lo], [g_hi - g_est]], fmt="s",
                        color=color, mfc="white", mec=color, ms=5.5, mew=1.2,
                        ecolor=color, elinewidth=1.1, capsize=2.5, zorder=3)

        ax.axhline(45, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=0)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{m}\n{r}" for m, r, *_ in CONDS], fontsize=8)
        ax.set_title(f"Rat {rat}", fontsize=10, fontweight="bold", pad=6)
        ax.set_xlim(-0.5, len(CONDS) - 0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=3)
        ax.grid(axis="y", color=GRID, lw=0.7, zorder=-1)

    ymin, ymax = min(all_vals), max(all_vals)
    pad = 0.08 * (ymax - ymin)
    axes[0].set_ylim(ymin - pad, ymax + pad)
    axes[0].set_ylabel("PSE (°)")

    legend_elems = [
        Line2D([0], [0], marker="o", color="none", mfc=INK, mec=INK, ms=6, label="Psychometric"),
        Line2D([0], [0], marker="s", color="none", mfc="white", mec=INK, mew=1.2, ms=6, label="GLM"),
        Line2D([0], [0], color=REW_COLOR["rew"], lw=2.5, label="Rewarded"),
        Line2D([0], [0], color=REW_COLOR["unrew"], lw=2.5, label="Unrewarded"),
        Line2D([0], [0], color=MUTED, lw=0.8, ls=(0, (4, 3)), label="45° (unbiased)"),
    ]
    fig1.legend(handles=legend_elems, loc="lower center", ncol=5, frameon=False,
                bbox_to_anchor=(0.5, -0.06), fontsize=8.5)

    fig1.tight_layout(rect=[0, 0.06, 1, 0.86])
    fig1.suptitle("Point of subjective equality: psychometric vs. GLM (degree scale)",
                  fontsize=11, y=0.99)
    fig1.text(0.5, 0.905, "rat × previous-choice/previous-outcome cell; error bars are 95% bootstrap CI",
               ha="center", fontsize=8.5, color=MUTED)

    stub = f"fig1_pse_comparison_{RATS_LABEL}"
    save(fig1, stub)
    print(f"Figure 1 saved ({stub}.svg/.pdf), rats={RATS_TO_PLOT}")
    return fig1

# ----------------------------------------------------------------------------
# 2. Figure 2 -- success/failure plane, psychometric vs GLM
# ----------------------------------------------------------------------------

def cov_entry(rat, row_name, col_name):
    r = contrast_cov[(contrast_cov.rat == rat) & (contrast_cov.contrast_row == row_name) &
                      (contrast_cov.contrast_column == col_name)]
    return r.iloc[0].covariance

def psych_mean(rat):
    s = contrast_sum[(contrast_sum.rat == rat) & (contrast_sum.contrast == "delta_mu_rew")].iloc[0].estimate_full_data
    f = contrast_sum[(contrast_sum.rat == rat) & (contrast_sum.contrast == "delta_mu_unrew")].iloc[0].estimate_full_data
    return np.array([s, f])

def psych_cov(rat):
    return np.array([
        [cov_entry(rat, "delta_mu_rew", "delta_mu_rew"), cov_entry(rat, "delta_mu_rew", "delta_mu_unrew")],
        [cov_entry(rat, "delta_mu_unrew", "delta_mu_rew"), cov_entry(rat, "delta_mu_unrew", "delta_mu_unrew")],
    ])

def psych_draws(rat):
    d = np.load(PSYCH_DRAWS_NPZ[rat], allow_pickle=True)
    theta = d["theta_bootstrap_x_y"]  # columns confirmed = (delta_mu_rew, delta_mu_unrew)
    return theta[:, 0], theta[:, 1]

_glm_draws_all = pd.read_parquet(PATHS["glm_draws"])
_glm_draws_all = _glm_draws_all[(_glm_draws_all.C == GLM_C) & (_glm_draws_all.rat_id.isin(ALL_RATS)) &
                                  (_glm_draws_all.quantity.isin(["delta_mu_rewarded", "delta_mu_unrewarded"]))]
_glm_draws_piv = _glm_draws_all.pivot_table(index=["rat_id", "boot_idx"], columns="quantity", values="value").reset_index()

def glm_mean(rat):
    s = glm_summary[(glm_summary.rat_id == rat) & (glm_summary.C == GLM_C) &
                     (glm_summary.quantity == "delta_mu_rewarded")].iloc[0].value
    f = glm_summary[(glm_summary.rat_id == rat) & (glm_summary.C == GLM_C) &
                     (glm_summary.quantity == "delta_mu_unrewarded")].iloc[0].value
    return np.array([s, f])

def glm_draws(rat):
    g = _glm_draws_piv[_glm_draws_piv.rat_id == rat]
    return g["delta_mu_rewarded"].values, g["delta_mu_unrewarded"].values

def glm_cov(rat):
    x, y = glm_draws(rat)
    return np.cov(x, y)

def ellipse_reach(mean, cov, ci_level):
    """Max |coordinate| reached by the ci_level confidence ellipse (for sizing the fixed frame)."""
    vals, _ = np.linalg.eigh(cov)
    r = np.sqrt(chi2.ppf(ci_level / 100, df=2)) * np.sqrt(vals.max())
    return max(abs(mean[0]) + r, abs(mean[1]) + r)

def draw_ellipse(ax, mean, cov, ci_level, color, linestyle, alpha):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    scale = np.sqrt(chi2.ppf(ci_level / 100, df=2))
    width, height = 2 * scale * np.sqrt(vals)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    ax.add_patch(mpatches.Ellipse(mean, width, height, angle=angle, facecolor=color,
                                    edgecolor=color, lw=1.2, ls=linestyle, alpha=alpha, zorder=1))

def draw_raw_scatter(ax, x, y, color, n_sub):
    idx = RNG.choice(len(x), size=min(n_sub, len(x)), replace=False)
    ax.scatter(x[idx], y[idx], s=5, color=color, alpha=0.10, linewidths=0, zorder=1)

def build_figure2():
    # frame is always sized from covariance (works for any rat, raw draws or not),
    # so the "full plane" is stable and comparable regardless of CLOUD_STYLE/subset
    frame_rats = ALL_RATS if PLANE_FIXED else RATS_TO_PLOT
    max_reach = max(
        max(ellipse_reach(psych_mean(r), psych_cov(r), CI_LEVEL), ellipse_reach(glm_mean(r), glm_cov(r), CI_LEVEL))
        for r in frame_rats
    )
    L = max_reach * 1.08
    xlim = ylim = (-L, L)

    fig2, ax = plt.subplots(figsize=(6.6, 6.2))

    for rat in RATS_TO_PLOT:
        color = RAT_COLOR[rat]
        p_mean = psych_mean(rat)
        g_mean = glm_mean(rat)

        if CLOUD_STYLE == "ellipse":
            
            if SHOW_PSYCH_CI:
                draw_ellipse(
                    ax, p_mean, psych_cov(rat),
                    CI_LEVEL, color, "solid", 0.16
                )

            draw_ellipse(
                ax, g_mean, glm_cov(rat),
                CI_LEVEL, color, (0, (3, 2)), 0.10
            )
            
        else:  # "raw"
            px, py = psych_draws(rat)
            gx, gy = glm_draws(rat)
            draw_raw_scatter(ax, px, py, color, N_SCATTER)
            draw_raw_scatter(ax, gx, gy, color, N_SCATTER)

        ax.plot([p_mean[0], g_mean[0]], [p_mean[1], g_mean[1]], color=color, lw=1.0,
                ls=(0, (1, 1.5)), alpha=0.8, zorder=2)
        ax.plot(*p_mean, marker="o", mfc=color, mec="white", mew=0.8, ms=8, zorder=4)
        ax.plot(*g_mean, marker="s", mfc="white", mec=color, mew=1.6, ms=8, zorder=4)
        ax.annotate(f"{rat}", p_mean, textcoords="offset points", xytext=(7, 6),
                    fontsize=9, fontweight="bold", color=color, zorder=5)

    ax.axhline(0, color=MUTED, lw=0.8, zorder=0)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=0)
    diag = np.array([-L * 1.4, L * 1.4])
    ax.plot(diag, diag, color=GRID, lw=1.0, ls=(0, (4, 3)), zorder=0)     # D = 0 (stay axis)
    ax.plot(diag, -diag, color=GRID, lw=1.0, ls=(0, (4, 3)), zorder=0)    # S = 0 (switch axis)
    lab_pos = 0.58 * L
    ax.text(lab_pos, lab_pos, "S-dominant\n(D = 0)", fontsize=7.5, color=MUTED,
            ha="center", va="center", rotation=45, clip_on=True,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))
    ax.text(lab_pos, -lab_pos, "D-dominant\n(S = 0)", fontsize=7.5, color=MUTED,
            ha="center", va="center", rotation=-45, clip_on=True,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Post-success shift, Δμ$_R$ (°)")
    ax.set_ylabel("Post-failure shift, Δμ$_U$ (°)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, lw=0.6, zorder=-2)

    legend_elems = [
        Line2D([0], [0], marker="o", color="none", mfc=INK, mec="white", mew=0.8, ms=8, label="Psychometric"),
        Line2D([0], [0], marker="s", color="none", mfc="white", mec=INK, mew=1.6, ms=8, label="GLM"),
    ] + [
        Line2D([0], [0], marker="o", color="none", mfc=RAT_COLOR[r], mec=RAT_COLOR[r], ms=8, label=f"Rat {r}")
        for r in RATS_TO_PLOT
    ]
    ncol_legend = 1 if len(RATS_TO_PLOT) <= 6 else 2
    ax.legend(handles=legend_elems, loc="upper left", frameon=False, fontsize=8.5,
              bbox_to_anchor=(1.02, 1.0), ncol=ncol_legend)

    style_label = f"{CI_LEVEL}% ellipse" if CLOUD_STYLE == "ellipse" else f"raw bootstrap draws, {N_SCATTER}/rat"
    ax.set_title(f"Success/failure plane: psychometric vs. GLM ({style_label})",
                 fontsize=11, fontweight="bold", pad=10)
    frame_note = "full-cohort frame" if PLANE_FIXED else "frame cropped to selection"
    rats_str = "full cohort" if sorted(RATS_TO_PLOT) == sorted(ALL_RATS) else f"rats {', '.join(str(r) for r in RATS_TO_PLOT)}"
    cloud_note = (f"clouds are bivariate-normal {CI_LEVEL}% regions from bootstrap covariance"
                  if CLOUD_STYLE == "ellipse" else
                  f"clouds are {N_SCATTER} raw bootstrap draws per rat×method, plotted directly")
    fig2.text(0.5, 0.005, f"{rats_str} shown; {frame_note}; {cloud_note}", ha="center", fontsize=8, color=MUTED)

    fig2.tight_layout(rect=[0, 0.02, 1, 1])
    stub = f"fig2_success_failure_plane_{RATS_LABEL}_{CLOUD_STYLE}"
    save(fig2, stub)
    print(f"Figure 2 saved ({stub}.svg/.pdf), rats={RATS_TO_PLOT}, CI={CI_LEVEL}%, "
          f"CLOUD_STYLE={CLOUD_STYLE}, PLANE_FIXED={PLANE_FIXED}, L={L:.1f}")
    return fig2

# ----------------------------------------------------------------------------
# 3. Run
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    build_figure1()
    build_figure2()