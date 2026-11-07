"""
Builds all three modality-interaction figures for Section 4 in one run:

  Figure 1 -- Q2/Q3 forest plot of all modality contrasts (V-T, VT-T, VT-V),
              for every history coefficient that passed the Q2 likelihood-
              ratio gate (success, failure, reward, previous-stimulus). D is
              excluded on purpose -- it gets Figure 2.

  Figure 2 -- the D figure. Left: forest plot of D_logit (Delta beta_success
              - Delta beta_failure) for the 6 rats with at least one Q2-gated
              success/failure block, all 3 modality pairs, Holm-3 resolved
              (filled) vs not (open). Right: for 4 of those 6 rats (7, 11,
              13, 16 -- rats 4 and 6 are left out of this panel only, since
              neither had anything Holm-3 resolved and their tactile-only
              points added clutter with no arrow to show for it -- they're
              still fully shown in the forest), each rat's tactile point
              (green) with a 95% bootstrap covariance ellipse, and an arrow
              out to its visual (blue) and/or visuotactile (red) point --
              only drawn when that pair (V-T or VT-T) is itself Holm-3
              resolved. Axes are the NEGATED coefficients (-beta_success,
              -beta_failure) -- see the SIGN comment below. Diagonals are
              labelled "D = 0" / "D = 1" (Alessandro's naming choice for
              this plot specifically -- Figure 3 below uses the plainer
              "S"/"D" labels for the same two lines; see the README note
              you already have if you want these unified).

  Figure 3 -- three (tactile/green, visual/blue, visuotactile/
              red), all 12 rats as numbered dots, log-odds axes
              (-beta_success_n-1S-D planes , -beta_failure_n-1), same sign convention
              as Figure 2. Diagonals labelled "S" (x=y) and "D" (x=-y).
              Rat-number labels are auto-spread with adjustText so they
              don't overlap in the crowded middle of each panel.

All statistics (Q2 gating, Q3 Holm-3 resolution, D_logit) are recomputed
from scratch below using Alessandro's own holm_correction/boot_p_value
functions (copied verbatim into STATS HELPERS), from the raw
glm_lr_tests.csv / glm_contrasts_long.csv / glm_summary_long.csv /
glm_metadata.json / bootstrap-draws parquet files -- nothing is hardcoded,
so this reproduces identically wherever those files are.

-----------------------------------------------------------------------------
SETUP
-----------------------------------------------------------------------------
pip install numpy pandas matplotlib scipy pyarrow adjustText

Edit DATA_DIR below (only thing you should need to change), then:
    python all_figures.py

DATA_DIR must contain:
    glm_lr_tests.csv
    glm_contrasts_long.csv
    glm_summary_long.csv
    glm_metadata.json
    boot_draws_q3_filtered.parquet   (your original trimmed bootstrap export)
    boot_draws_fig3_betas.parquet    (per-modality beta_success_n-1 /
                                       beta_failure_n-1 bootstrap draws,
                                       C=unpenalized, all 12 rats -- export
                                       this from your full
                                       glm_bootstrap_draws.parquet if you
                                       don't already have it: filter to
                                       C=="unpenalized", kind=="modality",
                                       quantity in
                                       ["beta_success_n-1","beta_failure_n-1"])

Figures are written as .svg into OUT_DIR (created automatically).
-----------------------------------------------------------------------------
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch
from scipy.stats import chi2 as chi2_dist
from adjustText import adjust_text

# =============================================================================
# CONFIG -- edit DATA_DIR to your results folder
# =============================================================================
DATA_DIR = r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\glm_results_thesis_filtered_interaction"
OUT_DIR = r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\ale_thesis_fig_tab\images_thesis\section_4"


# =============================================================================
# STATS HELPERS -- copied verbatim from Alessandro's own stats_helpers.py
# =============================================================================
def holm_correction(pvals, alpha=0.05):
    """Standard Holm step-down. Returns (adjusted_p, reject) as np.arrays,
    in the ORIGINAL order of pvals. adjusted_p is monotonic non-decreasing
    along the sorted order and clipped to [0, 1]."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        m = n - rank
        candidate = pvals[idx] * m
        running_max = max(running_max, candidate)
        adj[idx] = min(running_max, 1.0)
    reject = adj < alpha
    return adj, reject


def boot_p_value(draws, n_boot=None):
    """Percentile bootstrap two-sided p-value: p = 2*min(frac_pos, frac_neg),
    floored at 2/(n_boot+1) when a fraction hits exactly 0 or 1."""
    draws = np.asarray(draws, dtype=float)
    n = len(draws) if n_boot is None else n_boot
    frac_pos = np.mean(draws > 0)
    frac_neg = np.mean(draws < 0)
    p = 2 * min(frac_pos, frac_neg)
    floor = 2.0 / (n + 1)
    return max(p, floor)


# =============================================================================
# STYLE -- pair colors deliberately NOT blue/red/green (those are reserved
# for the modality colors below, used in the same figures/section)
# =============================================================================
PAIR_COLOR = {
    "V_minus_T": "#4a3aa7",   # violet
    "VT_minus_T": "#eb6834",  # orange
    "VT_minus_V": "#e87ba4",  # magenta
}
PAIR_LABEL = {
    "V_minus_T": "V − T",
    "VT_minus_T": "VT − T",
    "VT_minus_V": "VT − V",
}

MOD_COLOR = {
    "tactile": "#2ca02c",       # green
    "visual": "#1f77b4",        # blue
    "visuotactile": "#d62728",  # red
}
MOD_LABEL = {"tactile": "Tactile", "visual": "Visual", "visuotactile": "Visuotactile"}

BLOCK_LABEL = {
    "success_n-1": "Success (t−1)",
    "failure_n-1": "Failure (t−1)",
    "reward_n-1": "Reward (t−1)",
    "angle_n-1": "Previous stimulus (t−1)",
}


# =============================================================================
# GATING -- recomputes Q2 gating, Q3 Holm-3 resolution, and D_logit from the
# raw exported files
# =============================================================================
BLOCKS = ["success_n-1", "failure_n-1", "reward_n-1", "angle_n-1"]
CHOICE_OUTCOME_BLOCKS = ["success_n-1", "failure_n-1", "reward_n-1"]  # Holm-3 family
PAIRS = ["V_minus_T", "VT_minus_T", "VT_minus_V"]

BLOCK_TO_BETA_QUANTITY = {
    "success_n-1": "beta_success_n-1",
    "failure_n-1": "beta_failure_n-1",
    "reward_n-1": "beta_reward_n-1",
    "angle_n-1": "beta_angle_n-1",
}


def load_all(data_dir):
    lr = pd.read_csv(f"{data_dir}/glm_lr_tests.csv")
    contrasts = pd.read_csv(f"{data_dir}/glm_contrasts_long.csv")
    summary = pd.read_csv(f"{data_dir}/glm_summary_long.csv")
    boot = pd.read_parquet(f"{data_dir}/glm_bootstrap_draws.parquet")
    # keep only the unpenalized (MLE) fit, matching the established convention
    lr = lr[lr["C"] == "unpenalized"].copy()
    contrasts = contrasts[contrasts["C"] == "unpenalized"].copy()
    summary = summary[summary["C"] == "unpenalized"].copy()
    boot = boot[boot["C"] == "unpenalized"].copy()
    return lr, contrasts, summary, boot


def compute_q2_gating(lr):
    """Returns dict: block -> set of gated rat_ids.

    success/failure/reward: Holm-3 within rat across the 3 raw LR p-values.
    angle_n-1 (previous-stimulus): its own family, single test, raw p<0.05.
    """
    gated = {b: set() for b in BLOCKS}

    for rat_id, sub in lr.groupby("rat_id"):
        sub = sub.set_index("block")
        # choice-outcome family
        pvals = [sub.loc[b, "p_value"] for b in CHOICE_OUTCOME_BLOCKS]
        adj, reject = holm_correction(pvals, alpha=0.05)
        for b, r in zip(CHOICE_OUTCOME_BLOCKS, reject):
            if r:
                gated[b].add(int(rat_id))
        # previous-stimulus family (own family, raw p)
        if "angle_n-1" in sub.index:
            p = sub.loc["angle_n-1", "p_value"]
            if p < 0.05:
                gated["angle_n-1"].add(int(rat_id))

    return gated


def _contrast_boot_draws(boot, rat_id, pair, quantity):
    m = (
        (boot["rat_id"] == rat_id)
        & (boot["kind"] == "contrast")
        & (boot["subset_or_pair"] == pair)
        & (boot["quantity"] == quantity)
    )
    sub = boot.loc[m].sort_values("boot_idx")
    return sub["value"].to_numpy(dtype=float), sub["boot_idx"].to_numpy()


def compute_q3_resolution(contrasts, boot, gated):
    """Returns a DataFrame, one row per (rat, block, pair) for every GATED
    cell, with value/ci/p_holm/resolved -- the 48-cell Q3 table."""
    rows = []
    for block in BLOCKS:
        quantity = BLOCK_TO_BETA_QUANTITY[block]
        for rat_id in sorted(gated[block]):
            pvals = {}
            values = {}
            cis = {}
            for pair in PAIRS:
                draws, _ = _contrast_boot_draws(boot, rat_id, pair, quantity)
                pvals[pair] = boot_p_value(draws, n_boot=len(draws))
                lo, hi = np.percentile(draws, [2.5, 97.5])
                cis[pair] = (lo, hi)
                crow = contrasts[
                    (contrasts["rat_id"] == rat_id)
                    & (contrasts["contrast_pair"] == pair)
                    & (contrasts["quantity"] == quantity)
                ]
                values[pair] = float(crow["value"].iloc[0])
            adj, reject = holm_correction([pvals[p] for p in PAIRS], alpha=0.05)
            for pair, a, r in zip(PAIRS, adj, reject):
                rows.append(
                    dict(
                        rat_id=rat_id,
                        block=block,
                        pair=pair,
                        value=values[pair],
                        ci_low=cis[pair][0],
                        ci_high=cis[pair][1],
                        p_raw=pvals[pair],
                        p_holm=a,
                        resolved=bool(r),
                    )
                )
    return pd.DataFrame(rows)


def compute_d_logit(boot, d_rats):
    """D_logit = Delta_beta_success - Delta_beta_failure, per rat x pair.
    Value reported is the bootstrap MEDIAN of the paired difference draws.
    Holm-3 within rat across the 3 pairs."""
    rows = []
    for rat_id in d_rats:
        pvals = []
        rat_rows = []
        for pair in PAIRS:
            s_draws, s_idx = _contrast_boot_draws(boot, rat_id, pair, "beta_success_n-1")
            f_draws, f_idx = _contrast_boot_draws(boot, rat_id, pair, "beta_failure_n-1")
            assert np.array_equal(s_idx, f_idx), "boot_idx must be paired/aligned"
            d_draws = s_draws - f_draws
            p = boot_p_value(d_draws, n_boot=len(d_draws))
            pvals.append(p)
            lo, hi = np.percentile(d_draws, [2.5, 97.5])
            rat_rows.append(
                dict(
                    rat_id=rat_id,
                    pair=pair,
                    d_value=float(np.median(d_draws)),
                    ci_low=float(lo),
                    ci_high=float(hi),
                    p_raw=p,
                )
            )
        adj, reject = holm_correction(pvals, alpha=0.05)
        for row, a, r in zip(rat_rows, adj, reject):
            row["p_holm"] = a
            row["resolved"] = bool(r)
            rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# SHARED PLOTTING HELPER
# =============================================================================
# Sign convention shared by Figures 2 and 3: delta_mu_rewarded =
# -2*(angle_scale/beta_angle)*beta_success (and analogously for
# beta_failure/delta_mu_unrewarded) -- see compute_quantities_from_params in
# the fitting script. beta_angle is always positive (checked against
# glm_metadata.json), so the degree-scale S-D planes used elsewhere in the
# thesis are already plotting the sign-flipped coefficient. Flipping here
# keeps these log-odds planes directly comparable to those, not
# mirror-reversed.
SIGN = -1.0


def _draw_cov_ellipse(ax, x_draws, y_draws, center, color, ci_level=95, alpha=0.15, **kwargs):
    cov = np.cov(x_draws, y_draws)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    scale = np.sqrt(chi2_dist.ppf(ci_level / 100.0, df=2))
    width, height = 2 * scale * np.sqrt(vals)
    ell = Ellipse(xy=center, width=width, height=height, angle=angle,
                   facecolor=color, alpha=alpha, edgecolor=color, lw=1.0, **kwargs)
    ax.add_patch(ell)


# =============================================================================
# FIGURE 1 -- Q2/Q3 forest plot, all blocks except D
# =============================================================================
PANEL_ORDER = ["success_n-1", "failure_n-1", "reward_n-1", "angle_n-1"]
PAIR_OFFSET = {"V_minus_T": 0.24, "VT_minus_T": 0.0, "VT_minus_V": -0.24}


def build_fig1(fname_stub="fig1_q2_forest"):
    lr, contrasts, summary, boot = load_all(DATA_DIR)
    gated = compute_q2_gating(lr)
    q3 = compute_q3_resolution(contrasts, boot, gated)

    # union of rats across all four blocks, consistent row order top-to-bottom
    all_rats = sorted(set().union(*gated.values()))
    rat_to_y = {r: i for i, r in enumerate(reversed(all_rats))}  # rat 2 at top

    fig, axes = plt.subplots(1, 4, figsize=(15, 0.55 * len(all_rats) + 1.6), sharey=True)

    for ax, block in zip(axes, PANEL_ORDER):
        sub = q3[q3["block"] == block]
        for _, row in sub.iterrows():
            y = rat_to_y[row["rat_id"]] + PAIR_OFFSET[row["pair"]]
            color = PAIR_COLOR[row["pair"]]
            ax.plot([row["ci_low"], row["ci_high"]], [y, y], color=color, lw=1.3, zorder=2)
            ax.plot(
                row["value"], y,
                marker="o", ms=6.5,
                mfc=color if row["resolved"] else "white",
                mec=color, mew=1.4,
                zorder=3,
            )
        ax.axvline(0, color="0.4", lw=0.8, ls="--", zorder=1)
        ax.set_title(BLOCK_LABEL[block], fontsize=11)
        ax.set_xlabel("Modality contrast (log-odds)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_yticks(list(rat_to_y.values()))
    axes[0].set_yticklabels([f"Rat {r}" for r in reversed(all_rats)])
    axes[0].set_ylim(-0.7, len(all_rats) - 0.3)

    legend_elems = [
        Line2D([0], [0], color=PAIR_COLOR[p], marker="o", mfc=PAIR_COLOR[p],
               mec=PAIR_COLOR[p], lw=1.3, label=PAIR_LABEL[p])
        for p in PAIRS
    ] + [
        Line2D([0], [0], marker="o", mfc="black", mec="black", lw=0, label="Holm-3 resolved"),
        Line2D([0], [0], marker="o", mfc="white", mec="black", lw=0, label="not resolved"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Modality-contrast forest plot, Q2-gated coefficients\n"
        "(rows = Q2-gated rats per block; whiskers = 95% bootstrap CI; filled = Holm-3 resolved)",
        fontsize=11, y=1.03,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    fig.savefig(f"{OUT_DIR}/{fname_stub}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR}/{fname_stub}.svg")
    return q3, gated


# =============================================================================
# FIGURE 2 -- the D figure (D forest + tactile-anchored S-D plane)
# =============================================================================
D_PAIR_OFFSET = {"V_minus_T": 0.22, "VT_minus_T": 0.0, "VT_minus_V": -0.22}


def _modality_point(summary, rat_id, modality, quantity):
    row = summary[
        (summary["rat_id"] == rat_id)
        & (summary["subset"] == modality)
        & (summary["quantity"] == quantity)
    ]
    return float(row["value"].iloc[0])


def _modality_boot_draws(boot, rat_id, modality, quantity):
    m = (
        (boot["rat_id"] == rat_id)
        & (boot["kind"] == "modality")
        & (boot["subset_or_pair"] == modality)
        & (boot["quantity"] == quantity)
    )
    sub = boot.loc[m].sort_values("boot_idx")
    return sub["value"].to_numpy(dtype=float), sub["boot_idx"].to_numpy()


def _draw_arrow(ax, p_src, p_dst, color, base_scale=1.0):
    dist = float(np.hypot(p_dst[0] - p_src[0], p_dst[1] - p_src[1]))
    # Scale head size (and how much shaft it eats) with the arrow's own
    # length so short arrows don't turn into an oversized triangle.
    mutation_scale = float(np.clip(dist * 50 * base_scale + 5, 6, 15))
    shrinkA = float(np.clip(dist * 15 * base_scale, 2, 6))
    arr = FancyArrowPatch(
        p_src, p_dst, color=color, lw=2.0 * base_scale, linestyle="-",
        arrowstyle="-|>", mutation_scale=mutation_scale,
        shrinkA=shrinkA, shrinkB=2, zorder=4,
    )
    ax.add_patch(arr)


def _draw_plane_content(ax, rat_ids, summary, boot, d, label_fontsize=8.5, base_scale=1.0):
    """Draws the tactile anchor + ellipse + arrows for each rat in rat_ids
    onto ax. Returns the list of all x/y coordinates touched (dots and
    arrow tips), for axis-limit bookkeeping by the caller."""
    quantities = ("beta_success_n-1", "beta_failure_n-1")
    all_x, all_y = [], []
    for rat_id in rat_ids:
        pts = {}
        for mod in ("tactile", "visual", "visuotactile"):
            x = SIGN * _modality_point(summary, rat_id, mod, quantities[0])
            y = SIGN * _modality_point(summary, rat_id, mod, quantities[1])
            pts[mod] = (x, y)
            if mod == "tactile":
                all_x.append(x); all_y.append(y)
                xd, _ = _modality_boot_draws(boot, rat_id, mod, quantities[0])
                yd, _ = _modality_boot_draws(boot, rat_id, mod, quantities[1])
                _draw_cov_ellipse(ax, SIGN * xd, SIGN * yd, (x, y), MOD_COLOR[mod])

        d_lookup = {row["pair"]: row for _, row in d.iterrows() if row["rat_id"] == rat_id}

        # Only draw an arrow for a pair that's actually Holm-3 significant --
        # near-miss/non-significant pairs are dropped entirely to keep this
        # panel uncluttered.
        if d_lookup["V_minus_T"]["resolved"]:
            _draw_arrow(ax, pts["tactile"], pts["visual"], MOD_COLOR["visual"], base_scale)
            all_x.append(pts["visual"][0]); all_y.append(pts["visual"][1])
        if d_lookup["VT_minus_T"]["resolved"]:
            _draw_arrow(ax, pts["tactile"], pts["visuotactile"], MOD_COLOR["visuotactile"], base_scale)
            all_x.append(pts["visuotactile"][0]); all_y.append(pts["visuotactile"][1])
        # Note: VT-V (visual-to-visuotactile) arrows are dropped -- that pair
        # isn't anchored at the tactile point like the other two, and as a
        # lone third arrow it read as an unexplained stray line rather than
        # a third data series. It's still fully reported in the D forest.

        x, y = pts["tactile"]
        ax.plot(x, y, "o", ms=7 * base_scale, color=MOD_COLOR["tactile"],
                mec="white", mew=0.8, zorder=5)
        ax.annotate(f"{rat_id}", pts["tactile"], textcoords="offset points",
                    xytext=(6, 6), fontsize=label_fontsize, color="0.15",
                    fontweight="bold", zorder=6)
    return all_x, all_y


def _style_plane_axes(ax, lim, xlabel, ylabel, diag_labels=("D = 0", "D = 1"), label_fontsize=9):
    ax.plot([-lim, lim], [-lim, lim], color="0.5", lw=1.0, ls="--", zorder=1)
    ax.plot([-lim, lim], [lim, -lim], color="0.5", lw=1.0, ls=":", zorder=1)
    if diag_labels:
        label_kwargs = dict(fontsize=label_fontsize, color="0.35", ha="center", va="center",
                             rotation_mode="anchor",
                             bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.85),
                             zorder=1.5)
        ax.text(lim * 0.80, lim * 0.80, diag_labels[0], rotation=45, **label_kwargs)
        ax.text(lim * 0.80, -lim * 0.80, diag_labels[1], rotation=-45, **label_kwargs)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def build_fig2(fname_stub="fig2_d_figure"):
    lr, contrasts, summary, boot = load_all(DATA_DIR)
    gated = compute_q2_gating(lr)
    d_rats = sorted(gated["success_n-1"] | gated["failure_n-1"])
    d = compute_d_logit(boot, d_rats)

    fig, (ax_forest, ax_plane) = plt.subplots(1, 2, figsize=(14, 0.5 * len(d_rats) + 2.4))

    # ---------------- LEFT: D forest ----------------
    rat_to_y = {r: i for i, r in enumerate(reversed(d_rats))}
    for _, row in d.iterrows():
        y = rat_to_y[row["rat_id"]] + D_PAIR_OFFSET[row["pair"]]
        color = PAIR_COLOR[row["pair"]]
        ax_forest.plot([row["ci_low"], row["ci_high"]], [y, y], color=color, lw=1.3, zorder=2)
        ax_forest.plot(
            row["d_value"], y, marker="o", ms=6.5,
            mfc=color if row["resolved"] else "white", mec=color, mew=1.4, zorder=3,
        )
    ax_forest.axvline(0, color="0.4", lw=0.8, ls="--", zorder=1)
    ax_forest.set_yticks(list(rat_to_y.values()))
    ax_forest.set_yticklabels([f"Rat {r}" for r in reversed(d_rats)])
    ax_forest.set_ylim(-0.6, len(d_rats) - 0.4)
    ax_forest.set_xlabel(r"$D_{logit} = \Delta\beta_{success} - \Delta\beta_{failure}$ (log-odds)")
    ax_forest.set_title("D forest (modality contrasts)", fontsize=11)
    ax_forest.spines["top"].set_visible(False)
    ax_forest.spines["right"].set_visible(False)
    legend_elems = [
        Line2D([0], [0], color=PAIR_COLOR[p], marker="o", mfc=PAIR_COLOR[p],
               mec=PAIR_COLOR[p], lw=1.3, label=PAIR_LABEL[p])
        for p in PAIRS
    ] + [
        Line2D([0], [0], marker="o", mfc="black", mec="black", lw=0, label="Holm-3 resolved"),
        Line2D([0], [0], marker="o", mfc="white", mec="black", lw=0, label="not resolved"),
    ]
    ax_forest.legend(handles=legend_elems, loc="upper left", fontsize=8, frameon=False)

    # ---------------- RIGHT: arrows plane (4 of the 6 rats, no zoom) --------
    # Rats 4 and 6 are excluded from this panel only -- neither had a
    # Holm-3-resolved V-T or VT-T pair, so they'd show only a bare tactile
    # anchor with no arrow, which just cluttered the plane. They remain in
    # the forest on the left.
    plane_rats = [r for r in d_rats if r not in (4, 6)]
    all_x, all_y = _draw_plane_content(ax_plane, plane_rats, summary, boot, d)
    lo = min(all_x + all_y) - 0.15
    hi = max(all_x + all_y) + 0.15
    lim = max(abs(lo), abs(hi))
    _style_plane_axes(ax_plane, lim, r"$-\beta_{success}$", r"$-\beta_{failure}$")
    ax_plane.set_title("Tactile → visual / visuotactile shift", fontsize=11)
    ax_plane.spines["top"].set_visible(False)
    ax_plane.spines["right"].set_visible(False)

    mod_legend = [
        Line2D([0], [0], marker="o", color="w", mfc=MOD_COLOR["tactile"], ms=8,
               label=f"{MOD_LABEL['tactile']} (anchor point)"),
        Line2D([0], [0], color=MOD_COLOR["visual"], lw=2, label=f"→ {MOD_LABEL['visual']} (Holm-3 resolved)"),
        Line2D([0], [0], color=MOD_COLOR["visuotactile"], lw=2,
               label=f"→ {MOD_LABEL['visuotactile']} (Holm-3 resolved)"),
    ]
    ax_plane.legend(handles=mod_legend, loc="upper left", fontsize=7.5, frameon=False)

    fig.suptitle(
        "D parameter — modality-contrast forest and tactile→visual/visuotactile shift\n"
        "(ellipses = 95% bootstrap covariance; only Holm-3-significant shifts are drawn as arrows; "
        "axes sign-flipped to match this thesis's other S-D planes)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])

    fig.savefig(f"{OUT_DIR}/{fname_stub}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR}/{fname_stub}.svg")
    return d


# =============================================================================
# FIGURE 3 -- three S-D planes, one per modality, all 12 rats
# =============================================================================
MODALITIES = ["tactile", "visual", "visuotactile"]
QUANTITIES = ("beta_success_n-1", "beta_failure_n-1")


def _load_fig3_boot(data_dir):
    df = pd.read_parquet(f"{data_dir}/boot_draws_fig3_betas.parquet")
    return df[df["C"] == "unpenalized"].copy()


def _point(summary, rat_id, modality, quantity):
    row = summary[
        (summary["rat_id"] == rat_id)
        & (summary["subset"] == modality)
        & (summary["quantity"] == quantity)
    ]
    return float(row["value"].iloc[0])


def _boot_draws(boot, rat_id, modality, quantity):
    m = (
        (boot["rat_id"] == rat_id)
        & (boot["kind"] == "modality")
        & (boot["subset_or_pair"] == modality)
        & (boot["quantity"] == quantity)
    )
    sub = boot.loc[m].sort_values("boot_idx")
    return sub["value"].to_numpy(dtype=float)


def build_fig3(fname_stub="fig3_sd_planes_all_modalities"):
    lr, contrasts, summary, boot_q3 = load_all(DATA_DIR)
    boot = _load_fig3_boot(DATA_DIR)

    rats = sorted(summary["rat_id"].unique())
    missing = set(rats) - set(boot["rat_id"].unique())
    if missing:
        raise ValueError(f"boot_draws_fig3_betas.parquet is missing rats {missing}")

    data = {mod: {} for mod in MODALITIES}
    all_x, all_y = [], []
    for mod in MODALITIES:
        for rat_id in rats:
            x = SIGN * _point(summary, rat_id, mod, QUANTITIES[0])
            y = SIGN * _point(summary, rat_id, mod, QUANTITIES[1])
            xd = SIGN * _boot_draws(boot, rat_id, mod, QUANTITIES[0])
            yd = SIGN * _boot_draws(boot, rat_id, mod, QUANTITIES[1])
            data[mod][rat_id] = (x, y, xd, yd)
            all_x.append(x); all_y.append(y)
            sx, sy = np.std(xd), np.std(yd)
            all_x += [x - 3 * sx, x + 3 * sx]
            all_y += [y - 3 * sy, y + 3 * sy]

    lim = max(abs(min(all_x + all_y)), abs(max(all_x + all_y))) * 1.05

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), sharex=True, sharey=True)

    for ax, mod in zip(axes, MODALITIES):
        color = MOD_COLOR[mod]
        ax.plot([-lim, lim], [-lim, lim], color="0.5", lw=1.0, ls="--", zorder=1)
        ax.plot([-lim, lim], [lim, -lim], color="0.5", lw=1.0, ls=":", zorder=1)
        label_kwargs = dict(fontsize=10, color="0.35", ha="center", va="center",
                             rotation_mode="anchor",
                             bbox=dict(facecolor="white", edgecolor="none", pad=1.5, alpha=0.85),
                             zorder=1.5)
        ax.text(lim * 0.80, lim * 0.80, "S", rotation=45, **label_kwargs)
        ax.text(lim * 0.80, -lim * 0.80, "D", rotation=-45, **label_kwargs)

        texts = []
        for rat_id in rats:
            x, y, xd, yd = data[mod][rat_id]
            _draw_cov_ellipse(ax, xd, yd, (x, y), color, alpha=0.18)
            ax.plot(x, y, "o", ms=7, color=color, mec="white", mew=0.8, zorder=3)
            texts.append(ax.text(x, y, str(rat_id), fontsize=8.5, color="0.1",
                                  fontweight="bold", zorder=4))

        adjust_text(
            texts, ax=ax,
            expand=(1.4, 1.7),
            force_text=(0.3, 0.4),
            arrowprops=dict(arrowstyle="-", color="0.5", lw=0.6, shrinkA=0, shrinkB=2),
        )

        ax.axhline(0, color="0.85", lw=0.8, zorder=0)
        ax.axvline(0, color="0.85", lw=0.8, zorder=0)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(MOD_LABEL[mod], fontsize=12, color=color)
        ax.set_xlabel(r"$-\beta_{success}$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(r"$-\beta_{failure}$")

    fig.suptitle(
        "S-D plane by sensory modality, all 12 rats (log-odds)\n"
        "(ellipses = 95% bootstrap covariance; shared scale across panels)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    fig.savefig(f"{OUT_DIR}/{fname_stub}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR}/{fname_stub}.svg")


# =============================================================================
if __name__ == "__main__":
    print("Figure 1 (Q2 forest)...")
    build_fig1()

    print("Figure 2 (D figure)...")
    build_fig2()

    print("Figure 3 (S-D planes, all modalities)...")
    build_fig3()

    print("Done.")