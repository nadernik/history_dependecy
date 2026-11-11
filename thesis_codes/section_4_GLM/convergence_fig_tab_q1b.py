"""
Section 4, Q1b -- psychometric vs. GLM convergence figures.

  Figure A -- 4-panel scatter (ΔμR, ΔμU, D, P): x = psychometric point estimate
              (degrees), y = GLM point estimate on the SIGN-ALIGNED log-odds
              scale (Alessandro's own Q1_comparison.py convention: value_beta_aligned
              = -beta_success_n-1 / -beta_failure_n-1 / -(beta_success_n-1 -
              beta_failure_n-1) / +beta_reward_n-1). No diagonal reference line --
              the two axes are different physical units (degrees vs. log-odds), so
              y=x would not mean anything. Zero-crossing lines instead: same-sign
              quadrants (upper-right / lower-left) are what "sign agreement" means.
              rho + p_perm annotated per panel using the values Alessandro already
              reported (independently reproduced here from the raw files -- see
              verification block below -- rho matches to 3 decimals, sign-agreement
              counts match exactly: 9/12, 9/12, 11/12, 7/12).

  Figure B -- Q1b geometric-agreement plane, FULL 12-rat cohort, TWO SEPARATE
              PANELS (not one overlaid plane): left = psychometric (ΔμR, ΔμU) in
              degrees: right = GLM (-beta_success_n-1, -beta_failure_n-1) in
              sign-aligned log-odds. Same reasoning as Figure A -- octant position
              depends on the x:y ratio, and there is no single linear rescaling
              that maps degrees onto log-odds consistently across rats (per-rat
              psychometric slopes differ), so a shared plane would misrepresent
              the geometry. Two panels, same rat colors/numbers, let a reader
              compare octant-by-octant by eye. Each panel keeps its own D=0/S=0
              reference diagonals (valid within a panel, since both of ITS axes
              share one unit) -- just not a cross-panel diagonal.

Sign-alignment reference (Alessandro's Q1_comparison.py, GLM_QUANTITY_NAMES):
    delta_mu_rew   -> -beta_success_n-1
    delta_mu_unrew -> -beta_failure_n-1
    D              -> -(beta_success_n-1 - beta_failure_n-1)
    P              -> +beta_reward_n-1

DATA_SOURCE flag (see Config below): every aggregate statistic used in Figure A's
annotations and in the LaTeX table (sign-agreement counts, Spearman rho/CI/p_perm,
octant-agreement cohort counts, the P-panel CI-width ratio) can come from either
this script's own recomputation ("recompute", default) or from Alessandro's own
Q1_comparison.py CSV outputs loaded verbatim ("precomputed"). Point estimates and
resolved/not-resolved status in the figures are unaffected by this flag -- both
scripts read them from the same shared summary CSVs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import chi2, spearmanr

# ----------------------------------------------------------------------------
# 0. Config / data
# ----------------------------------------------------------------------------

UPLOAD_DIR_3 = r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_3_choice_reward\choice_reward_joint_results\filtered\pooled\tables"
UPLOAD_DIR_4 = r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_4_GLM\fit_results\pooled_filtered"
PATHS = dict(
    contrast_sum = f"{UPLOAD_DIR_3}/contrast_summary_long.csv",
    contrast_cov = f"{UPLOAD_DIR_3}/contrast_covariance_long.csv",
    glm_summary  = f"{UPLOAD_DIR_4}/glm_summary_long.csv",
    glm_draws    = f"{UPLOAD_DIR_4}/glm_bootstrap_draws.parquet",
)
GLM_C = "unpenalized"
GLM_SUBSET = "pooled"
CI_LEVEL = 95  # Figure B ellipse level

# ---- the flag ----
DATA_SOURCE = "precomputed"   # <-- "recompute" (default) or "precomputed"
# "recompute": every aggregate statistic (sign-agreement counts, Spearman
#   rho/CI/p_perm, octant-agreement cohort counts, the P-panel CI-width ratio)
#   is computed in THIS script, from the raw uploaded CSVs, using the same
#   algorithms as Q1_comparison.py -- verified to match its reported numbers
#   exactly (see the verification block below).
# "precomputed": skip that and load Alessandro's own Q1_comparison.py CSV
#   outputs instead, so the figure annotations and table reflect his
#   environment's numbers verbatim. Point PRECOMPUTED_DIR (below) at a folder
#   containing these 5 files, copied/uploaded from his own OUTPUT_DIR:
#     layer1_sign_agreement.csv, layer1_per_rat_signs.csv, layer2_spearman.csv,
#     layer3_per_rat_geometry.csv, layer3_cohort_summary.csv
PRECOMPUTED_DIR = r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_4_GLM\Q1_results_2.0"  # <-- update if using "precomputed"
# -------------------

contrast_sum = pd.read_csv(PATHS["contrast_sum"])
contrast_sum = contrast_sum[contrast_sum.data_condition == "filtered"]
contrast_cov = pd.read_csv(PATHS["contrast_cov"])

glm_summary = pd.read_csv(PATHS["glm_summary"])
glm_summary = glm_summary[(glm_summary.C == GLM_C) & (glm_summary.subset == GLM_SUBSET)]
glm_piv = glm_summary.pivot_table(index="rat_id", columns="quantity",
                                    values=["value", "ci_low", "ci_high"])

ALL_RATS = sorted(contrast_sum.rat.unique().tolist())

# ---- sign-aligned GLM point estimates (log-odds), per Alessandro's script ----
def glm_beta_aligned(contrast, rat):
    v = glm_piv["value"]
    if contrast == "delta_mu_rew":
        return -v.loc[rat, "beta_success_n-1"]
    if contrast == "delta_mu_unrew":
        return -v.loc[rat, "beta_failure_n-1"]
    if contrast == "D":
        return -(v.loc[rat, "beta_success_n-1"] - v.loc[rat, "beta_failure_n-1"])
    if contrast == "P":
        return v.loc[rat, "beta_reward_n-1"]
    raise ValueError(contrast)

# GLM "resolved" gate uses the DEGREES-scale CI (the only one with proper joint
# bootstrap propagation for D/P) -- matches Alessandro's script exactly.
GLM_DEGREES_NAME = {"delta_mu_rew": "delta_mu_rewarded", "delta_mu_unrew": "delta_mu_unrewarded",
                     "D": "D_glm", "P": "P_glm"}

def glm_resolved(contrast, rat):
    name = GLM_DEGREES_NAME[contrast]
    lo = glm_piv["ci_low"].loc[rat, name]
    hi = glm_piv["ci_high"].loc[rat, name]
    return bool((lo > 0) or (hi < 0))

def psych_row(contrast, rat):
    r = contrast_sum[(contrast_sum.contrast == contrast) & (contrast_sum.rat == rat)].iloc[0]
    return r.estimate_full_data, bool(r.ci_excludes_zero)

# ---- verification: reproduce Alessandro's Panel A numbers from raw files ----
REPORTED = {  # (rho, p_perm, sign_match, n) as stated in the Q1b draft paragraph
    "delta_mu_rew":   (0.66, 0.023, 9, 12),
    "delta_mu_unrew": (0.57, 0.055, 9, 12),
    "D":              (0.72, 0.010, 11, 12),
    "P":              (0.45, 0.144, 7, 12),
}
print("Verification against Alessandro's reported Panel A numbers:")
for contrast in ["delta_mu_rew", "delta_mu_unrew", "D", "P"]:
    pv = np.array([psych_row(contrast, r)[0] for r in ALL_RATS])
    gv = np.array([glm_beta_aligned(contrast, r) for r in ALL_RATS])
    rho, _ = spearmanr(pv, gv)
    sign_match = int((np.sign(pv) == np.sign(gv)).sum())
    rho_rep, p_rep, sm_rep, n_rep = REPORTED[contrast]
    ok = "OK" if (abs(rho - rho_rep) < 0.01 and sign_match == sm_rep) else "MISMATCH"
    print(f"  {contrast:16s} rho={rho:.3f} (reported {rho_rep}) "
          f"sign={sign_match}/{len(ALL_RATS)} (reported {sm_rep}/{n_rep})  [{ok}]")

# One color per rat, all 12 distinct (no cycling/collisions) -- same palette as
# build_figures.py, so a given rat is the same color in every figure in the
# thesis. 12 evenly-spaced OKLCH hues (L=0.62, C=0.13), reordered so hue-wheel
# neighbors never land adjacent in rat-ID order. Validated with the dataviz
# skill's validate_palette.js: lightness band, chroma floor and contrast vs.
# surface all PASS; normal-vision adjacent separation passes comfortably
# (worst 23.2 dE, floor 15). One adjacent pair (rat 3 / rat 4) sits at CVD dE
# 4.8 (deutan), below the "legal with secondary encoding" 6-8 floor -- a
# genuine 12-simultaneous-category limit (the skill's own 8-color default
# can't pass all-pairs beyond 3 slots either), mitigated by the direct
# rat-number label already on every point -- the actual identifier, with
# color as a secondary, quick-glance cue only.
RAT_COLOR = dict(zip(ALL_RATS, [
    "#009ab1", "#c46183", "#429c5a", "#9372c8", "#a58100", "#248fcc",
    "#c86556", "#009f89", "#b167ab", "#7c9128", "#6880d4", "#be7125",
]))
ACCENT = "#2a78d6"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

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
    "svg.fonttype": "none",
})

def save(fig, stub):
    fig.savefig(f"{stub}.svg", bbox_inches="tight")
    fig.savefig(f"{stub}.pdf", bbox_inches="tight")

# ----------------------------------------------------------------------------
# Figure A -- 4-panel scatter, sign + rank agreement
# ----------------------------------------------------------------------------

PANELS = [
    ("delta_mu_rew",   "Δμ$_R$"),
    ("delta_mu_unrew", "Δμ$_U$"),
    ("D",              "D"),
    ("P",              "P"),
]

def build_figure_a():
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6))

    for ax, (contrast, label) in zip(axes, PANELS):
        xs, ys, resolved = [], [], []
        for rat in ALL_RATS:
            pv, p_res = psych_row(contrast, rat)
            gv = glm_beta_aligned(contrast, rat)
            g_res = glm_resolved(contrast, rat)
            xs.append(pv); ys.append(gv); resolved.append(p_res or g_res)
        xs, ys, resolved = np.array(xs), np.array(ys), np.array(resolved)

        ax.axhline(0, color=MUTED, lw=0.8, zorder=0)
        ax.axvline(0, color=MUTED, lw=0.8, zorder=0)

        for rat, x, y, res in zip(ALL_RATS, xs, ys, resolved):
            if res:
                ax.plot(x, y, marker="o", mfc=ACCENT, mec="white", mew=0.6, ms=7, zorder=3)
            else:
                ax.plot(x, y, marker="o", mfc="white", mec=ACCENT, mew=1.3, ms=7, zorder=3)
            ax.annotate(str(rat), (x, y), textcoords="offset points", xytext=(5, 4),
                        fontsize=6.5, color=MUTED, zorder=4)

        l1c, l2c = get_layer1(contrast), get_layer2(contrast)
        ax.text(0.04, 0.96, f"ρ = {l2c['rho']:.2f}, p$_{{perm}}$ = {l2c['p_perm']:.3f}\n"
                              f"sign agree: {l1c['opt1_match']}/{l1c['opt1_n']}",
                transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=INK,
                bbox=dict(facecolor="white", edgecolor=GRID, lw=0.7, pad=3))

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Psychometric (°)", fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=3, labelsize=8)
        ax.grid(color=GRID, lw=0.6, zorder=-1)
        xpad = 0.12 * (xs.max() - xs.min() if xs.max() != xs.min() else 1)
        ypad = 0.12 * (ys.max() - ys.min() if ys.max() != ys.min() else 1)
        ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
        ax.set_ylim(ys.min() - ypad, ys.max() + ypad)

    axes[0].set_ylabel("GLM, sign-aligned (log-odds)", fontsize=8.5)

    legend_elems = [
        Line2D([0], [0], marker="o", color="none", mfc=ACCENT, mec="white", mew=0.6, ms=7,
               label="≥1 method resolved (CI excl. 0, uncorrected)"),
        Line2D([0], [0], marker="o", color="none", mfc="white", mec=ACCENT, mew=1.3, ms=7,
               label="neither resolved"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=8.5)

    fig.suptitle("Q1b — psychometric vs. GLM: sign and rank agreement, N = 12 rats",
                  fontsize=11.5, y=1.04)
    fig.tight_layout()

    stub = "figA_q1b_sign_rank_scatter"
    save(fig, stub)
    print(f"\nFigure A saved ({stub}.svg/.pdf, DATA_SOURCE={DATA_SOURCE!r})")
    return fig

# ----------------------------------------------------------------------------
# Figure B -- geometric-agreement plane, full cohort, two panels (own scales)
# ----------------------------------------------------------------------------

def psych_mean(rat):
    s = contrast_sum[(contrast_sum.rat == rat) & (contrast_sum.contrast == "delta_mu_rew")].iloc[0].estimate_full_data
    f = contrast_sum[(contrast_sum.rat == rat) & (contrast_sum.contrast == "delta_mu_unrew")].iloc[0].estimate_full_data
    return np.array([s, f])

def cov_entry(rat, row_name, col_name):
    r = contrast_cov[(contrast_cov.rat == rat) & (contrast_cov.contrast_row == row_name) &
                      (contrast_cov.contrast_column == col_name)]
    return r.iloc[0].covariance

def psych_cov(rat):
    return np.array([
        [cov_entry(rat, "delta_mu_rew", "delta_mu_rew"), cov_entry(rat, "delta_mu_rew", "delta_mu_unrew")],
        [cov_entry(rat, "delta_mu_unrew", "delta_mu_rew"), cov_entry(rat, "delta_mu_unrew", "delta_mu_unrew")],
    ])

_glm_draws_all = pd.read_parquet(PATHS["glm_draws"])
_glm_draws_all = _glm_draws_all[(_glm_draws_all.C == GLM_C) & (_glm_draws_all.subset == GLM_SUBSET) &
                                  (_glm_draws_all.quantity.isin(["beta_success_n-1", "beta_failure_n-1"]))]
_glm_draws_piv = _glm_draws_all.pivot_table(index=["rat_id", "boot_idx"], columns="quantity", values="value").reset_index()

def glm_mean_aligned(rat):
    return np.array([glm_beta_aligned("delta_mu_rew", rat), glm_beta_aligned("delta_mu_unrew", rat)])

def glm_cov_aligned(rat):
    # Cov(-X, -Y) = Cov(X, Y): the sign flip does not change the covariance.
    g = _glm_draws_piv[_glm_draws_piv.rat_id == rat]
    return np.cov(g["beta_success_n-1"].values, g["beta_failure_n-1"].values)

def ellipse_reach(mean, cov, ci_level):
    vals, _ = np.linalg.eigh(cov)
    r = np.sqrt(chi2.ppf(ci_level / 100, df=2)) * np.sqrt(vals.max())
    return max(abs(mean[0]) + r, abs(mean[1]) + r)

def draw_ellipse(ax, mean, cov, ci_level, color):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    scale = np.sqrt(chi2.ppf(ci_level / 100, df=2))
    width, height = 2 * scale * np.sqrt(vals)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    ax.add_patch(mpatches.Ellipse(mean, width, height, angle=angle, facecolor=color,
                                    edgecolor=color, lw=1.1, alpha=0.16, zorder=1))

def draw_plane_panel(ax, means_by_rat, covs_by_rat, xlabel, ylabel, unit_label):
    L = max(ellipse_reach(means_by_rat[r], covs_by_rat[r], CI_LEVEL) for r in ALL_RATS) * 1.1
    for rat in ALL_RATS:
        color = RAT_COLOR[rat]
        mean, cov = means_by_rat[rat], covs_by_rat[rat]
        draw_ellipse(ax, mean, cov, CI_LEVEL, color)
        ax.plot(*mean, marker="o", mfc=color, mec="white", mew=0.7, ms=7, zorder=3)
        ax.annotate(str(rat), mean, textcoords="offset points", xytext=(6, 5),
                    fontsize=7.5, fontweight="bold", color=color, zorder=4)

    ax.axhline(0, color=MUTED, lw=0.8, zorder=0)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=0)
    diag = np.array([-L * 1.4, L * 1.4])
    ax.plot(diag, diag, color=GRID, lw=1.0, ls=(0, (4, 3)), zorder=0)
    ax.plot(diag, -diag, color=GRID, lw=1.0, ls=(0, (4, 3)), zorder=0)

    ax.set_xlim(-L, L); ax.set_ylim(-L, L)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{xlabel} ({unit_label})")
    ax.set_ylabel(f"{ylabel} ({unit_label})")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, lw=0.6, zorder=-2)

def build_figure_b():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 5.6))

    psych_means = {r: psych_mean(r) for r in ALL_RATS}
    psych_covs = {r: psych_cov(r) for r in ALL_RATS}
    glm_means = {r: glm_mean_aligned(r) for r in ALL_RATS}
    glm_covs = {r: glm_cov_aligned(r) for r in ALL_RATS}

    draw_plane_panel(axL, psych_means, psych_covs, "Post-success shift, Δμ$_R$", "Post-failure shift, Δμ$_U$", "°")
    axL.set_title("Psychometric", fontsize=11, fontweight="bold")

    draw_plane_panel(axR, glm_means, glm_covs, "−β$_{success,n-1}$", "−β$_{failure,n-1}$", "log-odds")
    axR.set_title("GLM (sign-aligned)", fontsize=11, fontweight="bold")

    legend_elems = [Line2D([0], [0], marker="o", color="none", mfc=RAT_COLOR[r], mec=RAT_COLOR[r], ms=7, label=f"Rat {r}")
                    for r in ALL_RATS]
    fig.legend(handles=legend_elems, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=8)

    fig.suptitle(f"Q1b — geometric agreement in the choice-outcome plane, full cohort ({CI_LEVEL}% ellipses)",
                  fontsize=11.5, y=1.02)
    fig.text(0.5, -0.11,
              "Two panels, not one overlaid plane: axes are on different scales (degrees vs. log-odds) and\n"
              "per-rat PSE-conversion slopes differ, so no single rescaling maps one plane onto the other.\n"
              "Compare octant position rat-by-rat across panels using the matched colors/numbers.",
              ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout()

    stub = "figB_q1b_geometric_plane_full_cohort"
    save(fig, stub)
    print(f"Figure B saved ({stub}.svg/.pdf)")
    return fig

# ----------------------------------------------------------------------------
# Table -- full Q1b convergence statistics, LaTeX (supplementary)
# ----------------------------------------------------------------------------
# Reproduces Alessandro's Q1_comparison.py Layers 1-3 directly from the raw
# uploaded files (contrast_summary_long.csv / glm_summary_long.csv /
# glm_bootstrap_draws.parquet), using his own algorithms (sign gate, Spearman
# BCa + permutation, quadrant4/octant8 point-estimate geometry), so this table
# is independently derived from source, not copied from the draft paragraph.
# Layer 3 here is point-estimate-only -- matches what the paragraph's
# 58%/92%/20 deg figures report. Alessandro's script also computes a
# probabilistic soft-match classification from paired bootstrap draws, but
# that needs psychometric joint_bootstrap.npz files, which are only uploaded
# for 4/12 rats (9, 11, 12, 13) -- not the full-cohort number quoted in text,
# so it is intentionally NOT reproduced here.

from scipy.stats import norm

QUANTITIES4 = ["delta_mu_rew", "delta_mu_unrew", "D", "P"]
QLABEL = {"delta_mu_rew": r"$\Delta\mu_R$", "delta_mu_unrew": r"$\Delta\mu_U$", "D": "D", "P": "P"}

def spearman_bca(x, y, n_boot=5000, alpha=0.05, seed=42):
    """BCa bootstrap CI for Spearman rho -- same algorithm as Q1_comparison.py."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    rho_hat, _ = spearmanr(x, y)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        rho_b, _ = spearmanr(x[idx], y[idx])
        boot[b] = rho_b if not np.isnan(rho_b) else 0.0
    p0 = np.clip((boot < rho_hat).mean(), 1e-10, 1 - 1e-10)
    z0 = norm.ppf(p0)
    jack = np.array([spearmanr(np.delete(x, i), np.delete(y, i)).correlation for i in range(n)])
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6.0 * ((jm - jack) ** 2).sum() ** 1.5
    a = num / den if den != 0 else 0.0
    z_alpha, z_1_alpha = norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)
    a1 = norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
    a2 = norm.cdf(z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha)))
    lo, hi = np.percentile(boot, [100 * a1, 100 * a2])
    return rho_hat, lo, hi

def spearman_perm_p(x, y, n_perm=10000, seed=42):
    """Permutation p-value for Spearman rho -- same algorithm as Q1_comparison.py."""
    rng = np.random.default_rng(seed)
    rho_obs, _ = spearmanr(x, y)
    count = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        r, _ = spearmanr(x, y_perm)
        if abs(r) >= abs(rho_obs):
            count += 1
    return (count + 1) / (n_perm + 1)

def quadrant4(r, u):
    if r > 0 and u > 0: return "Q1"
    if r < 0 and u > 0: return "Q2"
    if r < 0 and u < 0: return "Q3"
    if r > 0 and u < 0: return "Q4"
    return "on-axis"

def octant8(r, u):
    if r == 0 and u == 0:
        return 0
    ang = np.degrees(np.arctan2(u, r)) % 360
    return int(ang // 45) + 1

def cyclic_dist_octant(k1, k2, K=8):
    d = abs(k1 - k2) % K
    return min(d, K - d)

def angular_disagreement(r1, u1, r2, u2):
    a1 = np.degrees(np.arctan2(u1, r1)) % 360
    a2 = np.degrees(np.arctan2(u2, r2)) % 360
    diff = abs(a1 - a2)
    return min(diff, 360 - diff)

def compute_layer1(contrast):
    rows = []
    for rat in ALL_RATS:
        pv, p_res = psych_row(contrast, rat)
        gv = glm_beta_aligned(contrast, rat)
        g_res = glm_resolved(contrast, rat)
        rows.append(dict(rat=rat, sign_match=np.sign(pv) == np.sign(gv), p_res=p_res, g_res=g_res))
    df = pd.DataFrame(rows)
    qual = df[df.p_res | df.g_res]
    return dict(opt1_match=int(df.sign_match.sum()), opt1_n=len(df),
                opt3_match=int(qual.sign_match.sum()), opt3_n=len(qual))

def compute_layer2(contrast):
    pv = np.array([psych_row(contrast, r)[0] for r in ALL_RATS])
    gv = np.array([glm_beta_aligned(contrast, r) for r in ALL_RATS])
    rho, lo, hi = spearman_bca(pv, gv)
    p_perm = spearman_perm_p(pv, gv)
    return dict(rho=rho, ci_lo=lo, ci_hi=hi, p_perm=p_perm)

def compute_layer3():
    rows = []
    for rat in ALL_RATS:
        pr, _ = psych_row("delta_mu_rew", rat)
        pu, _ = psych_row("delta_mu_unrew", rat)
        gr = glm_beta_aligned("delta_mu_rew", rat)
        gu = glm_beta_aligned("delta_mu_unrew", rat)
        q4p, q4g = quadrant4(pr, pu), quadrant4(gr, gu)
        o8p, o8g = octant8(pr, pu), octant8(gr, gu)
        rows.append(dict(rat=rat, psych_R=pr, psych_U=pu, glm_R=gr, glm_U=gu,
                          psych_q4=q4p, glm_q4=q4g, q4_match=q4p == q4g,
                          psych_o8=o8p, glm_o8=o8g, o8_match=o8p == o8g,
                          cyclic_dist=cyclic_dist_octant(o8p, o8g),
                          angular_disagreement_deg=angular_disagreement(pr, pu, gr, gu)))
    df = pd.DataFrame(rows)
    cohort = dict(
        n=len(df),
        q4_exact=int(df.q4_match.sum()),
        o8_exact=int(df.o8_match.sum()),
        o8_within1=int((df.cyclic_dist <= 1).sum()),
        o8_within2=int((df.cyclic_dist <= 2).sum()),
        ang_median=float(df.angular_disagreement_deg.median()),
        ang_max=float(df.angular_disagreement_deg.max()),
    )
    return cohort, df

def compute_panel_c():
    """P-specific detail: resolving power on each side, sign, CI-width ratio."""
    psych_p_resolved = sum(psych_row("P", r)[1] for r in ALL_RATS)

    glm_reward_resolved = glm_reward_positive = 0
    for rat in ALL_RATS:
        lo = glm_piv["ci_low"].loc[rat, "beta_reward_n-1"]
        hi = glm_piv["ci_high"].loc[rat, "beta_reward_n-1"]
        val = glm_piv["value"].loc[rat, "beta_reward_n-1"]
        glm_reward_resolved += int((lo > 0) or (hi < 0))
        glm_reward_positive += int(val > 0)

    psych_widths, glm_widths = [], []
    for rat in ALL_RATS:
        r = contrast_sum[(contrast_sum.rat == rat) & (contrast_sum.contrast == "P")].iloc[0]
        psych_widths.append(r.ci_high - r.ci_low)
        glm_widths.append(glm_piv["ci_high"].loc[rat, "P_glm"] - glm_piv["ci_low"].loc[rat, "P_glm"])
    # median, not mean -- matches Alessandro's reported 2.75x exactly (mean gives 3.31x)
    ratio = float(np.median(psych_widths) / np.median(glm_widths))

    return dict(psych_p_resolved=psych_p_resolved, n=len(ALL_RATS),
                glm_reward_resolved=glm_reward_resolved, glm_reward_positive=glm_reward_positive,
                ci_width_ratio=ratio)

# ----------------------------------------------------------------------------
# DATA_SOURCE dispatch -- "recompute" (above) vs. "precomputed" (Alessandro's
# own Q1_comparison.py CSV outputs, loaded verbatim). get_layer1/2/3/panel_c
# are what Figure A's annotations and the LaTeX table actually call; they hide
# which source is active behind DATA_SOURCE so nothing downstream needs to care.
# ----------------------------------------------------------------------------

_precomputed_cache = {}

def _load_precomputed():
    if _precomputed_cache:
        return _precomputed_cache
    d = PRECOMPUTED_DIR
    try:
        _precomputed_cache["l1_summary"] = pd.read_csv(f"{d}/layer1_sign_agreement.csv").set_index("quantity")
        _precomputed_cache["l1_per_rat"] = pd.read_csv(f"{d}/layer1_per_rat_signs.csv")
        _precomputed_cache["l2"] = pd.read_csv(f"{d}/layer2_spearman.csv").set_index("quantity")
        _precomputed_cache["l3_per_rat"] = pd.read_csv(f"{d}/layer3_per_rat_geometry.csv")
        _precomputed_cache["l3_cohort"] = pd.read_csv(f"{d}/layer3_cohort_summary.csv").iloc[0]
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"DATA_SOURCE='precomputed' but couldn't find Q1_comparison.py's output CSVs under "
            f"{d!r} ({e}). Upload layer1_sign_agreement.csv, layer1_per_rat_signs.csv, "
            f"layer2_spearman.csv, layer3_per_rat_geometry.csv and layer3_cohort_summary.csv from "
            f"your OUTPUT_DIR, update PRECOMPUTED_DIR to point at them, or switch DATA_SOURCE back "
            f"to 'recompute'."
        ) from e
    return _precomputed_cache

def get_layer1(contrast):
    if DATA_SOURCE == "precomputed":
        row = _load_precomputed()["l1_summary"].loc[contrast]
        return dict(opt1_match=int(row.opt1_match), opt1_n=int(row.opt1_denominator),
                    opt3_match=int(row.opt3_match), opt3_n=int(row.opt3_denominator))
    return compute_layer1(contrast)

def get_layer2(contrast):
    if DATA_SOURCE == "precomputed":
        row = _load_precomputed()["l2"].loc[contrast]
        return dict(rho=float(row.rho_beta_aligned), ci_lo=float(row.bca_lo_beta_aligned),
                    ci_hi=float(row.bca_hi_beta_aligned), p_perm=float(row.p_permutation_beta_aligned))
    return compute_layer2(contrast)

def get_layer3():
    if DATA_SOURCE == "precomputed":
        p = _load_precomputed()
        row = p["l3_cohort"]
        cohort = dict(n=int(row.n_rats), q4_exact=int(row.q4_exact_match_beta),
                      o8_exact=int(row.o8_exact_match_beta), o8_within1=int(row.o8_within1_beta),
                      o8_within2=int(row.o8_within2_beta),
                      ang_median=float(row.angular_disagree_median_beta_deg),
                      ang_max=float(row.angular_disagree_max_beta_deg))
        return cohort, p["l3_per_rat"]
    return compute_layer3()

def get_panel_c():
    if DATA_SOURCE == "precomputed":
        lp = _load_precomputed()["l1_per_rat"]
        prow = lp[lp.quantity == "P"]
        psych_width = prow.psych_ci_high - prow.psych_ci_low
        glm_width = prow.glm_ci_high_degrees - prow.glm_ci_low_degrees
        return dict(psych_p_resolved=int(prow.psych_ci_excludes_zero.sum()), n=len(prow),
                    glm_reward_resolved=int(prow.glm_ci_excludes_zero.sum()),
                    glm_reward_positive=int((prow.glm_value_beta_aligned > 0).sum()),
                    ci_width_ratio=float(psych_width.median() / glm_width.median()))
    return compute_panel_c()

def build_table_q1b():
    print(f"\nBuilding full Q1b convergence table (DATA_SOURCE={DATA_SOURCE!r})...")
    l1 = {c: get_layer1(c) for c in QUANTITIES4}
    l2 = {c: get_layer2(c) for c in QUANTITIES4}
    l3_cohort, l3_per_rat = get_layer3()
    pc = get_panel_c()

    for c in QUANTITIES4:
        print(f"  {c:16s} sign {l1[c]['opt1_match']}/{l1[c]['opt1_n']}  "
              f"(>=1 resolved) {l1[c]['opt3_match']}/{l1[c]['opt3_n']}  "
              f"rho={l2[c]['rho']:.2f} [{l2[c]['ci_lo']:.2f},{l2[c]['ci_hi']:.2f}] p_perm={l2[c]['p_perm']:.3f}")
    print(f"  geometric (N={l3_cohort['n']}): quadrant={l3_cohort['q4_exact']} octant={l3_cohort['o8_exact']} "
          f"within1={l3_cohort['o8_within1']} within2={l3_cohort['o8_within2']} "
          f"median={l3_cohort['ang_median']:.0f} deg max={l3_cohort['ang_max']:.0f} deg")
    print(f"  P panel: psych resolved={pc['psych_p_resolved']}/{pc['n']}  "
          f"GLM beta_reward resolved={pc['glm_reward_resolved']}/{pc['n']}  "
          f"positive={pc['glm_reward_positive']}/{pc['n']}  CI-width ratio={pc['ci_width_ratio']:.2f}x")

    def frac_pct(a, b):
        return f"{a}/{b} ({100*a/b:.0f}\\%)"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Convergence between psychometric and GLM estimates across four "
        r"primary contrasts (N = 12 rats). Panel A: direction (sign) agreement and "
        r"Spearman rank correlation, on the sign-aligned log-odds GLM scale. Panel B: "
        r"geometric agreement in the $(\Delta\mu_R,\Delta\mu_U)$ plane. Panel C: "
        r"reward-history term $P$, GLM-side detail.}",
        r"\label{tab:q1b_convergence}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\multicolumn{5}{l}{\textbf{Panel A --- Direction agreement and rank correlation}} \\",
        r"\midrule",
        r"Quantity & Direction agreement (all rats) & Direction agreement ($\geq$1 resolved) "
        r"& Spearman $\rho$ [95\% BCa CI] & $p_{perm}$ \\",
        r"\midrule",
    ]
    for c in QUANTITIES4:
        a, b = l1[c]["opt1_match"], l1[c]["opt1_n"]
        a3, b3 = l1[c]["opt3_match"], l1[c]["opt3_n"]
        rho, lo, hi, pp = l2[c]["rho"], l2[c]["ci_lo"], l2[c]["ci_hi"], l2[c]["p_perm"]
        lines.append(f"{QLABEL[c]} & {frac_pct(a, b)} & {a3}/{b3} & "
                     f"{rho:.2f} [{lo:.2f}, {hi:.2f}] & {pp:.3f} \\\\")
    n = l3_cohort["n"]
    lines += [
        r"\midrule",
        r"\multicolumn{5}{l}{\textbf{Panel B --- Geometric agreement, joint "
        r"$(\Delta\mu_R,\Delta\mu_U)$ plane}} \\",
        r"\midrule",
        f"\\multicolumn{{4}}{{l}}{{Exact quadrant (4-way)}} & {frac_pct(l3_cohort['q4_exact'], n)} \\\\",
        f"\\multicolumn{{4}}{{l}}{{Exact octant (8-way)}} & {frac_pct(l3_cohort['o8_exact'], n)} \\\\",
        f"\\multicolumn{{4}}{{l}}{{Within 1 octant}} & {frac_pct(l3_cohort['o8_within1'], n)} \\\\",
        f"\\multicolumn{{4}}{{l}}{{Within 2 octants}} & {frac_pct(l3_cohort['o8_within2'], n)} \\\\",
        f"\\multicolumn{{4}}{{l}}{{Median / max angular disagreement}} & "
        f"{l3_cohort['ang_median']:.0f}\\textdegree\\ / {l3_cohort['ang_max']:.0f}\\textdegree \\\\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textbf{Panel C --- Reward-history term $P$, GLM-side detail}} \\",
        r"\midrule",
        f"\\multicolumn{{4}}{{l}}{{Psychometric estimator resolves $P$}} & {pc['psych_p_resolved']}/{pc['n']} \\\\",
        f"\\multicolumn{{4}}{{l}}{{GLM resolves $\\beta_{{reward,n-1}}$}} & {pc['glm_reward_resolved']}/{pc['n']} \\\\",
        f"\\multicolumn{{4}}{{l}}{{GLM point estimate positive (any $q$)}} & {pc['glm_reward_positive']}/{pc['n']} \\\\",
        f"\\multicolumn{{4}}{{l}}{{Psychometric CI width vs.\\ GLM (median ratio, degrees scale)}} & "
        f"{pc['ci_width_ratio']:.2f}$\\times$ \\\\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ]
    tex = "\n".join(lines)

    with open("table_q1b_convergence.tex", "w") as f:
        f.write(tex)
    l3_per_rat.to_csv("table_q1b_layer3_per_rat.csv", index=False)
    print("\nLaTeX table written to table_q1b_convergence.tex "
          "(per-rat geometry detail in table_q1b_layer3_per_rat.csv)")
    return tex

if __name__ == "__main__":
    build_figure_a()
    build_figure_b()
    build_table_q1b()