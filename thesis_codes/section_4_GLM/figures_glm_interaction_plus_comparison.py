"""
Builds the modality-interaction figures for Section 4 in one run:

  Figure 1 -- Q2/Q3 forest plot of all modality contrasts (V-T, VT-T, VT-V),
              for every history coefficient that passed the Q2 likelihood-
              ratio gate (success, failure, reward, previous-stimulus). D is
              excluded on purpose -- it gets Figure 2.
              Which coefficient panels are drawn is tunable with
              FIG1_DEFAULT_SHOW (or --fig1-show / --fig1-drop on the command
              line). This only changes what is PLOTTED: Q2 gating and the
              Holm-3 resolution are always computed on all blocks, so hiding
              a panel never changes the significance shown in the others.

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

  Figure 4 -- (NEW, optional) pooled vs per-modality coefficients, all 12
              rats. One panel per history coefficient (default: failure,
              success, reward; previous stimulus can be added). Each rat row
              shows the POOLED-model estimate (black diamond) above the three
              per-modality estimates from the interaction model (tactile
              green, visual blue, visuotactile red) with 95% bootstrap CIs.
              Rows whose coefficient passed the Q2 gate (modality matters)
              are shaded, using the same compute_q2_gating as Figures 1-2.
              Coefficients are plotted on their RAW log-odds scale (positive
              beta_success / beta_failure = repeat the previous choice) --
              not sign-flipped like Figures 2-3.
              Not built by default: add it with --figures (see below).
              Panels are tunable with FIG4_DEFAULT_SHOW or --fig4-show /
              --fig4-drop. The pooled estimates come from the POOLED model,
              which lives outside DATA_DIR -- set POOLED_SUMMARY_CSV (see
              CONFIG).

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
    python all_figures.py                                # Figures 1-3, exactly as before
    python all_figures.py --fig1-drop angle_n1          # Figure 1 without previous stimulus
    python all_figures.py --fig1-show success failure   # Figure 1 with only these panels

Figure 4 (needs POOLED_SUMMARY_CSV set in CONFIG):
    python all_figures.py --figures 4                    # only Figure 4
    python all_figures.py --figures 1 2 3 4              # everything
    python all_figures.py --figures 4 --fig4-show failure success reward angle_n1
    python all_figures.py --figures 4 --fig4-drop reward

Figure 1 panel keys: success, failure, reward, angle_n1
Figure 4 panel keys: failure, success, reward, angle_n1

DATA_DIR must contain:
    glm_lr_tests.csv
    glm_contrasts_long.csv
    glm_summary_long.csv
    glm_metadata.json
    glm_bootstrap_draws.parquet      (read by load_all; Figure 4 uses its
                                       kind=="modality" rows for the per-
                                       modality CIs, so it needs the reward /
                                       previous-stimulus draws too if those
                                       panels are shown)
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
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, Patch
from scipy.stats import chi2 as chi2_dist
from adjustText import adjust_text

# =============================================================================
# CONFIG -- edit DATA_DIR to your results folder
# =============================================================================
DATA_DIR = r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_4_GLM\fit_results\interaction_filtered"
OUT_DIR = r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\ale_thesis_fig_tab\images_thesis\section_4"

# Figure 1: which coefficient panels to show (left-to-right order is fixed).
# Keys: "success", "failure", "reward", "angle_n1"
FIG1_DEFAULT_SHOW = ["success", "failure", "reward"]

# Which figures run when no --figures flag is given. Figure 4 is opt-in, so
# the default run is unchanged.
DEFAULT_FIGURES = ["1"]

# Figure 4: which coefficient panels to show (left-to-right order is fixed).
# Keys: "failure", "success", "reward", "angle_n1"
FIG4_DEFAULT_SHOW = ["failure", "success", "reward"]

# Figure 4: where the POOLED-model estimates come from.
# Point this at the glm_summary_long.csv of your POOLED (no-interaction) fit.
# Expected columns: rat_id, quantity, value (+ C, and optionally subset and
# CI columns such as ci_low/ci_high -- detected automatically, see
# POOLED_CI_COLUMN_PAIRS). If left as None, the script looks for pooled rows
# inside DATA_DIR's glm_summary_long.csv (subset in POOLED_SUBSET_NAMES) and
# stops with a clear message if there are none.
POOLED_SUMMARY_CSV = r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_4_GLM\fit_results\pooled_filtered\glm_summary_long.csv"
POOLED_SUBSET_NAMES = ("pooled", "all", "pooled_all", "all_modalities")
POOLED_CI_COLUMN_PAIRS = (("ci_low", "ci_high"), ("ci_lower", "ci_upper"),
                          ("ci_lo", "ci_hi"), ("lo", "hi"))


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

# short key -> block name used in the data
FIG1_KEYS = {
    "success": "success_n-1",
    "failure": "failure_n-1",
    "reward": "reward_n-1",
    "angle_n1": "angle_n-1",
}


def _resolve_fig1_panels(show_keys):
    bad = [k for k in show_keys if k not in FIG1_KEYS]
    if bad:
        raise SystemExit(f"Figure 1: unknown panel key(s) {bad}. Valid: {list(FIG1_KEYS)}")
    wanted = {FIG1_KEYS[k] for k in show_keys}
    panels = [b for b in PANEL_ORDER if b in wanted]  # dedupe + canonical order
    if not panels:
        raise SystemExit("Figure 1: nothing to plot, every panel was dropped.")
    return panels


def build_fig1(fname_stub="fig1_q2_forest", show=None):
    show = FIG1_DEFAULT_SHOW if show is None else show
    panels = _resolve_fig1_panels(show)

    lr, contrasts, summary, boot = load_all(DATA_DIR)
    # Statistics on ALL blocks -- panel choice affects only the plot.
    gated = compute_q2_gating(lr)
    q3 = compute_q3_resolution(contrasts, boot, gated)

    # rows = union of gated rats across the SHOWN blocks only, so hiding a
    # panel doesn't leave empty rows for rats gated only in that block
    all_rats = sorted(set().union(*(gated[b] for b in panels)))
    if not all_rats:
        raise SystemExit(f"Figure 1: no Q2-gated rats in the shown panels {panels}.")
    rat_to_y = {r: i for i, r in enumerate(reversed(all_rats))}  # lowest rat id at top

    n_panels = len(panels)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(3.75 * n_panels, 0.55 * len(all_rats) + 1.6),
        sharey=True, squeeze=False,
    )
    axes = axes[0]

    for ax, block in zip(axes, panels):
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
    fig.legend(handles=legend_elems, loc="lower center",
               ncol=5 if n_panels >= 3 else 3, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Modality-contrast forest plot, Q2-gated coefficients\n"
        "(rows = Q2-gated rats per block; whiskers = 95% bootstrap CI; filled = Holm-3 resolved)",
        fontsize=11, y=1.03,
    )
    fig.tight_layout(rect=[0, 0.05 if n_panels >= 3 else 0.10, 1, 1])

    # suffix the file name when not all panels are shown, so the full
    # version isn't overwritten
    if panels != PANEL_ORDER:
        inv = {v: k for k, v in FIG1_KEYS.items()}
        fname_stub += "_" + "-".join(inv[b] for b in panels)

    fig.savefig(f"{OUT_DIR}/{fname_stub}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR}/{fname_stub}.svg  (panels: {panels})")
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
# FIGURE 4 -- pooled vs per-modality coefficients, all 12 rats (opt-in)
# =============================================================================
# Panel order is fixed: failure first (the most modality-stable term), then
# success, reward, previous stimulus.
FIG4_PANEL_ORDER = ["failure_n-1", "success_n-1", "reward_n-1", "angle_n-1"]
FIG4_KEYS = {
    "failure": "failure_n-1",
    "success": "success_n-1",
    "reward": "reward_n-1",
    "angle_n1": "angle_n-1",
}
# Vertical position of each estimate inside a rat's row (pooled on top).
FIG4_OFFSET = {"pooled": 0.30, "tactile": 0.10, "visual": -0.10, "visuotactile": -0.30}
FIG4_XLABEL = {
    "failure_n-1": "Log-odds\n(+ = repeat after an error)",
    "success_n-1": "Log-odds\n(+ = repeat after a reward)",
    "reward_n-1": "Log-odds\n(+ = toward vertical after a reward)",
    "angle_n-1": "Log-odds\n(+ = attraction to previous stimulus)",
}
FIG4_POOLED_COLOR = "0.1"
FIG4_GATE_SHADE = "0.93"


def _resolve_fig4_panels(show_keys):
    bad = [k for k in show_keys if k not in FIG4_KEYS]
    if bad:
        raise SystemExit(f"Figure 4: unknown panel key(s) {bad}. Valid: {list(FIG4_KEYS)}")
    wanted = {FIG4_KEYS[k] for k in show_keys}
    panels = [b for b in FIG4_PANEL_ORDER if b in wanted]  # dedupe + canonical order
    if not panels:
        raise SystemExit("Figure 4: nothing to plot, every panel was dropped.")
    return panels


def _find_ci_columns(df):
    for lo, hi in POOLED_CI_COLUMN_PAIRS:
        if lo in df.columns and hi in df.columns:
            return lo, hi
    return None


def _load_pooled_estimates(summary_interaction):
    """Returns {(rat_id, quantity): (value, ci_low or None, ci_high or None)}
    for the POOLED model."""
    if POOLED_SUMMARY_CSV:
        df = pd.read_csv(POOLED_SUMMARY_CSV)
        if "C" in df.columns:
            df = df[df["C"] == "unpenalized"].copy()
        # a pooled summary may still carry a subset column; keep pooled rows
        # if any are labelled as such, otherwise use the file as it is
        if "subset" in df.columns and df["subset"].isin(POOLED_SUBSET_NAMES).any():
            df = df[df["subset"].isin(POOLED_SUBSET_NAMES)].copy()
        source = POOLED_SUMMARY_CSV
    else:
        df = summary_interaction[summary_interaction["subset"].isin(POOLED_SUBSET_NAMES)].copy()
        source = f"{DATA_DIR}/glm_summary_long.csv (subset in {POOLED_SUBSET_NAMES})"
        if df.empty:
            subsets = sorted(summary_interaction["subset"].dropna().unique().tolist())
            raise SystemExit(
                "Figure 4: no pooled-model estimates found.\n"
                f"  The interaction summary only has subsets {subsets}.\n"
                "  Set POOLED_SUMMARY_CSV in CONFIG to the glm_summary_long.csv of the POOLED fit."
            )

    dup = df.duplicated(["rat_id", "quantity"])
    if dup.any():
        raise SystemExit(
            f"Figure 4: pooled source {source} has several rows per (rat_id, quantity); "
            "filter it to one fit (e.g. one C, one subset) before using it."
        )
    ci_cols = _find_ci_columns(df)
    out = {}
    for _, row in df.iterrows():
        lo = float(row[ci_cols[0]]) if ci_cols else None
        hi = float(row[ci_cols[1]]) if ci_cols else None
        out[(int(row["rat_id"]), row["quantity"])] = (float(row["value"]), lo, hi)
    return out


def _modality_ci(boot, summary, rat_id, modality, quantity):
    """95% percentile CI from the modality bootstrap draws; falls back to CI
    columns in the summary if no draws were exported for this quantity."""
    draws, _ = _modality_boot_draws(boot, rat_id, modality, quantity)
    if len(draws):
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return float(lo), float(hi)
    ci_cols = _find_ci_columns(summary)
    if ci_cols:
        row = summary[
            (summary["rat_id"] == rat_id)
            & (summary["subset"] == modality)
            & (summary["quantity"] == quantity)
        ]
        if len(row):
            return float(row[ci_cols[0]].iloc[0]), float(row[ci_cols[1]].iloc[0])
    return None, None


def build_fig4(fname_stub="fig4_pooled_vs_modality", show=None):
    show = FIG4_DEFAULT_SHOW if show is None else show
    panels = _resolve_fig4_panels(show)

    lr, contrasts, summary, boot = load_all(DATA_DIR)
    gated = compute_q2_gating(lr)
    pooled = _load_pooled_estimates(summary)

    rats = sorted(int(r) for r in summary["rat_id"].unique())
    rat_to_y = {r: i for i, r in enumerate(reversed(rats))}  # lowest rat id at top

    n_panels = len(panels)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(3.75 * n_panels, 0.55 * len(rats) + 1.6),
        sharey=True, squeeze=False,
    )
    axes = axes[0]

    missing_ci, missing_pooled = set(), set()
    for ax, block in zip(axes, panels):
        quantity = BLOCK_TO_BETA_QUANTITY[block]
        for rat_id in rats:
            y0 = rat_to_y[rat_id]
            if rat_id in gated[block]:
                ax.axhspan(y0 - 0.45, y0 + 0.45, color=FIG4_GATE_SHADE, lw=0, zorder=0)

            for mod in MODALITIES:
                y = y0 + FIG4_OFFSET[mod]
                value = _modality_point(summary, rat_id, mod, quantity)
                lo, hi = _modality_ci(boot, summary, rat_id, mod, quantity)
                if lo is None:
                    missing_ci.add((block, mod))
                else:
                    ax.plot([lo, hi], [y, y], color=MOD_COLOR[mod], lw=1.3, zorder=2)
                ax.plot(value, y, marker="o", ms=5.5, color=MOD_COLOR[mod],
                        mec="white", mew=0.6, zorder=3)

            key = (rat_id, quantity)
            if key not in pooled:
                missing_pooled.add(key)
                continue
            value, lo, hi = pooled[key]
            y = y0 + FIG4_OFFSET["pooled"]
            if lo is not None:
                ax.plot([lo, hi], [y, y], color=FIG4_POOLED_COLOR, lw=1.3, zorder=2)
            ax.plot(value, y, marker="D", ms=6.5, color=FIG4_POOLED_COLOR,
                    mec="white", mew=0.6, zorder=4)

        ax.axvline(0, color="0.4", lw=0.8, ls="--", zorder=1)
        ax.set_title(BLOCK_LABEL[block], fontsize=11)
        ax.set_xlabel(FIG4_XLABEL[block])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if missing_pooled:
        raise SystemExit(f"Figure 4: pooled estimate missing for (rat, quantity) {sorted(missing_pooled)}")
    if missing_ci:
        print(f"  Figure 4 warning: no bootstrap draws or CI columns for {sorted(missing_ci)}; "
              "drawn as points without whiskers.")

    axes[0].set_yticks(list(rat_to_y.values()))
    axes[0].set_yticklabels([f"Rat {r}" for r in reversed(rats)])
    axes[0].set_ylim(-0.7, len(rats) - 0.3)

    legend_elems = [
        Line2D([0], [0], marker="D", color=FIG4_POOLED_COLOR, lw=0, ms=6.5, label="Pooled model"),
    ] + [
        Line2D([0], [0], color=MOD_COLOR[m], marker="o", mfc=MOD_COLOR[m], mec=MOD_COLOR[m],
               lw=1.3, label=MOD_LABEL[m])
        for m in MODALITIES
    ] + [
        Patch(facecolor=FIG4_GATE_SHADE, edgecolor="none", label="Q2 gate passed"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Pooled vs per-modality history coefficients, all rats (log-odds)\n"
        "(diamond = pooled model; dots = modality-interaction model, whiskers = 95% bootstrap CI; "
        "shaded = Q2 gate passed for that coefficient)",
        fontsize=11, y=1.03,
    )
    fig.tight_layout(rect=[0, 0.05 if n_panels >= 3 else 0.10, 1, 1])

    default_panels = _resolve_fig4_panels(FIG4_DEFAULT_SHOW)
    if panels != default_panels:
        inv = {v: k for k, v in FIG4_KEYS.items()}
        fname_stub += "_" + "-".join(inv[b] for b in panels)

    fig.savefig(f"{OUT_DIR}/{fname_stub}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_DIR}/{fname_stub}.svg  (panels: {panels})")


# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Section 4 modality-interaction figures (1-4)."
    )
    parser.add_argument("--figures", nargs="+", default=DEFAULT_FIGURES, choices=["1", "2", "3", "4"],
                        metavar="N",
                        help=f"Which figures to build (default: {' '.join(DEFAULT_FIGURES)}). "
                             "Figure 4 = pooled vs per-modality coefficients.")
    parser.add_argument("--fig1-show", nargs="+", default=FIG1_DEFAULT_SHOW, metavar="KEY",
                        help=f"Figure 1 panels to show. Keys: {list(FIG1_KEYS)}")
    parser.add_argument("--fig1-drop", nargs="+", default=[], metavar="KEY",
                        help="Figure 1 panels to remove from --fig1-show")
    parser.add_argument("--fig4-show", nargs="+", default=FIG4_DEFAULT_SHOW, metavar="KEY",
                        help=f"Figure 4 panels to show. Keys: {list(FIG4_KEYS)}")
    parser.add_argument("--fig4-drop", nargs="+", default=[], metavar="KEY",
                        help="Figure 4 panels to remove from --fig4-show")
    args = parser.parse_args()
    fig1_show = [k for k in args.fig1_show if k not in args.fig1_drop]
    fig4_show = [k for k in args.fig4_show if k not in args.fig4_drop]
    figures = set(args.figures)

    os.makedirs(OUT_DIR, exist_ok=True)

    if "1" in figures:
        print("Figure 1 (Q2 forest)...")
        build_fig1(show=fig1_show)

    if "2" in figures:
        print("Figure 2 (D figure)...")
        build_fig2()

    if "3" in figures:
        print("Figure 3 (S-D planes, all modalities)...")
        build_fig3()

    if "4" in figures:
        print("Figure 4 (pooled vs per-modality coefficients)...")
        build_fig4(show=fig4_show)

    print("Done.")