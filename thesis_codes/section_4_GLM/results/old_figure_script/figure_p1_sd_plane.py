"""Figure P1 — GLM S/D plane in log-odds, per subset.

Visual counterpart of Section 3's group SD plane (Fig A). Same figure size,
diagonals, quadrant boxes, marker size, ring convention, and legend layout;
only differences are (i) units (log-odds instead of degrees) and
(ii) the axis quantities (−β_success, −β_failure instead of Δμ_rew,
Δμ_unrew). Signs and quadrants line up 1-to-1 with Section 3.

Coloring
--------
Each rat is coloured by the psychometric ``axis_pattern`` it received in
Section 3's group_figures step (column ``axis_pattern`` in the shared CSV).
A rat classified as "D (interaction)" in the psychometric analysis is blue
in BOTH figures — so the two plots can be read as a matched pair. If the
``axis_pattern`` column is absent the loader falls back to a single blue.

Point styling
-------------
Filled circle in the psychometric-pattern colour. Black edge (lw=2, bold rat
id) if ANY of {β_success, β_failure, β_reward, S_logit, D_logit} at
``--ci-level`` excludes zero. Marginal CI segments through each point in a
light version of the point colour.

Flags
-----
--subset {pooled|tactile|visual|visuotactile}
                        which analysis subset to plot (default: pooled)
--ellipses              add bootstrap covariance ellipses on top of the
                        marginal segments (extra, not a replacement)
--overlay-psychometric  overlay Section-3 (Δμ_rew, Δμ_unrew) points as red
                        open triangles on a secondary axis pair in degrees

Concordance
-----------
Spearman ρ, p-value, and sign-agreement fractions between GLM and
psychometric contrasts are printed to stdout and saved as
``.concordance.txt`` next to the figure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.stats import chi2, spearmanr

from common_plot_features import (
    PATTERN_COLOURS, PATTERN_ORDER, PSY_COL_PATTERN,
    Q_BF, Q_BR, Q_BS,
    SubsetData,
    add_subset_arg, any_history_excludes, load_subset, resolve_paths_from_args,
)
from Portable_path_resolution_for_Section_4_GLM_figures import (
    figures_out_dir,
    glm_bootstrap_path
)

# --------------------------------------------------------------------------
# Style constants
# --------------------------------------------------------------------------
FALLBACK_POINT_COLOR = "#3B5B7A"
DIAG_S_COLOR = "#E68722"
DIAG_D_COLOR = "#2922E6"
QUAD_WARM = "#7d4a13"
QUAD_COOL = "#151785"
POINT_SIZE = 140
POINT_EDGE_LW = 2.0
LIM_FLOOR = 0.5


# --------------------------------------------------------------------------
# Ellipses (optional overlay)
# --------------------------------------------------------------------------
def _load_boot_for_ellipses(rats, subset, C_label):
    boot = pd.read_parquet(glm_bootstrap_path())
    m = (
        (boot["subset"] == subset)
        & (boot["C"] == C_label)
        & (boot["quantity"].isin([Q_BS, Q_BF]))
        & (boot["rat_id"].isin(rats))
    )
    return boot.loc[m].pivot_table(
        index=["rat_id", "boot_idx"], columns="quantity", values="value"
    ).reset_index()


def _draw_ellipse(x, y, ax, ci_level, **kw):
    if len(x) < 5:
        return None
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)):
        return None
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    scale = chi2.ppf(ci_level / 100.0, df=2)
    w, h = 2 * np.sqrt(np.maximum(vals * scale, 0.0))
    e = Ellipse((float(np.mean(x)), float(np.mean(y))),
                width=w, height=h, angle=theta,
                facecolor="none", **kw)
    ax.add_patch(e)
    return e


# --------------------------------------------------------------------------
# Concordance
# --------------------------------------------------------------------------
def _sign_agreement(gv, pv, expected):
    gv = np.asarray(gv, dtype=float)
    pv = np.asarray(pv, dtype=float)
    m = np.isfinite(gv) & np.isfinite(pv) & (gv != 0) & (pv != 0)
    if not m.any():
        return 0, 0
    if expected == -1:
        agrees = np.sign(gv[m]) == -np.sign(pv[m])
    else:
        agrees = np.sign(gv[m]) == np.sign(pv[m])
    return int(agrees.sum()), int(m.sum())


def concordance_report(data: SubsetData) -> str:
    pairs = [
        ("β_success",         data.glm_point[Q_BS],
         "Δμ_rew",            data.psy["delta_mu_rew"], -1),
        ("β_failure",         data.glm_point[Q_BF],
         "Δμ_unrew",          data.psy["delta_mu_unrew"], -1),
        ("S_logit (β_s+β_f)", data.glm_derived_point["S_logit"],
         "S (psy)",           data.psy["S"], -1),
        ("D_logit (β_s−β_f)", data.glm_derived_point["D_logit"],
         "D (psy)",           data.psy["D"], -1),
        ("P_logit (β_r)",     data.glm_derived_point["P_logit"],
         "P (psy)",           data.psy["P"], +1),
    ]
    hdr = f"{data.subset.capitalize()} GLM vs Section-3 psychometrics"
    lines = [hdr, "=" * len(hdr),
             f"Subset:   {data.subset}",
             f"N rats:   {len(data.rats)}",
             f"C:        {data.C_label}",
             f"CI level: {data.ci_level:g}%", "",
             f"{'GLM':<22} {'Psy':<12} {'ρ':>8} {'p':>8} {'sign agr':>10} {'exp':>4}",
             "-" * 68]
    for glab, gv, plab, pv, exp in pairs:
        common_rats = gv.dropna().index.intersection(pv.dropna().index)
        g = pd.to_numeric(gv.loc[common_rats], errors="coerce").to_numpy(dtype=float)
        p = pd.to_numeric(pv.loc[common_rats], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(g) & np.isfinite(p)
        g, p = g[m], p[m]
        if len(g) < 3:
            lines.append(f"{glab:<22} {plab:<12} {'—':>8} {'—':>8} {'—':>10} {'—':>4}")
            continue
        rho, pval = spearmanr(g, p)
        n_agr, n_tot = _sign_agreement(g, p, exp)
        exp_str = "−" if exp == -1 else "+"
        agr_str = f"{n_agr}/{n_tot}"
        lines.append(
            f"{glab:<22} {plab:<12} {rho:>+8.3f} {pval:>8.3f} "
            f"{agr_str:>10} {exp_str:>4}"
        )
    lines.append("")
    lines.append("Expected sign: '−' means negative Spearman ρ (opposite "
                 "conventions, so same-sign agreement counts opposite signs).")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------
def plot_sd_plane(
    data: SubsetData,
    *,
    show_ellipses: bool = False,
    overlay_psychometric: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 9))
    else:
        fig = ax.figure

    bs   = data.glm_point[Q_BS].to_numpy(dtype=float)
    bf   = data.glm_point[Q_BF].to_numpy(dtype=float)
    bs_l = data.glm_ci_lo[Q_BS].to_numpy(dtype=float)
    bs_h = data.glm_ci_hi[Q_BS].to_numpy(dtype=float)
    bf_l = data.glm_ci_lo[Q_BF].to_numpy(dtype=float)
    bf_h = data.glm_ci_hi[Q_BF].to_numpy(dtype=float)
    x, y = -bs, -bf
    xl, xh = -bs_h, -bs_l   # sign flip swaps lo/hi
    yl, yh = -bf_h, -bf_l

    all_vals = np.concatenate([x, y, xl, xh, yl, yh])
    all_vals = all_vals[np.isfinite(all_vals)]
    max_abs = float(np.max(np.abs(all_vals))) if len(all_vals) else LIM_FLOOR
    lim = max(LIM_FLOOR, np.ceil(max_abs * 12) / 10)

    ax.axhline(0, color="k", lw=0.6, zorder=1)
    ax.axvline(0, color="k", lw=0.6, zorder=1)

    ax.plot([-lim, lim], [-lim, lim], ":", color=DIAG_S_COLOR, lw=1.6,
            label="y = x   (pure choice — S axis)", zorder=2)
    ax.plot([-lim, lim], [lim, -lim], ":", color=DIAG_D_COLOR, lw=1.6,
            label="y = −x  (reward × choice — D axis)", zorder=2)

    box = dict(boxstyle="round,pad=0.3", facecolor="white",
               edgecolor="none", alpha=0.75)
    ax.text(0.97, 0.97, "both switch\n(win-switch\n+ lose-switch)",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color=QUAD_WARM, bbox=box, linespacing=1.15)
    ax.text(0.03, 0.03, "both stay\n(win-stay\n+ lose-stay)",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, color=QUAD_WARM, bbox=box, linespacing=1.15)
    ax.text(0.03, 0.97, "win-stay\n+ lose-switch",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8.5, color=QUAD_COOL, bbox=box, linespacing=1.15)
    ax.text(0.97, 0.03, "win-switch\n+ lose-stay",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, color=QUAD_COOL, bbox=box, linespacing=1.15)

    sig = any_history_excludes(data)
    patterns = data.psy[PSY_COL_PATTERN] if PSY_COL_PATTERN in data.psy.columns else None
    if patterns is None:
        print(f"  [warn] '{PSY_COL_PATTERN}' column not in psychometric CSV; "
              f"points will use a single fallback colour.")

    for i, rat in enumerate(data.rats):
        colour = (
            PATTERN_COLOURS.get(patterns.loc[rat], "#95a5a6")
            if patterns is not None else FALLBACK_POINT_COLOR
        )
        xi, yi = x[i], y[i]
        if np.isfinite(xl[i]) and np.isfinite(xh[i]):
            ax.plot([xl[i], xh[i]], [yi, yi],
                    color=colour, lw=1.0, alpha=0.35, zorder=3)
        if np.isfinite(yl[i]) and np.isfinite(yh[i]):
            ax.plot([xi, xi], [yl[i], yh[i]],
                    color=colour, lw=1.0, alpha=0.35, zorder=3)

        is_sig_rat = bool(sig.loc[rat])
        ax.scatter(
            xi, yi, s=POINT_SIZE, c=colour,
            edgecolors="black" if is_sig_rat else "none",
            linewidths=POINT_EDGE_LW if is_sig_rat else 0,
            zorder=5,
        )
        ax.annotate(
            f"{int(rat)}", (xi, yi),
            xytext=(5, 5), textcoords="offset points",
            fontsize=9, fontweight="bold" if is_sig_rat else "normal",
            zorder=6,
        )

    if show_ellipses:
        boot_wide = _load_boot_for_ellipses(
            list(data.rats), data.subset, data.C_label
        )
        for rat in data.rats:
            sub = boot_wide.loc[boot_wide["rat_id"] == rat]
            if sub.empty:
                continue
            colour = (
                PATTERN_COLOURS.get(patterns.loc[rat], "#95a5a6")
                if patterns is not None else FALLBACK_POINT_COLOR
            )
            xs = -sub[Q_BS].to_numpy(dtype=float)
            ys = -sub[Q_BF].to_numpy(dtype=float)
            _draw_ellipse(
                xs, ys, ax, ci_level=data.ci_level,
                edgecolor=colour, linewidth=0.9, alpha=0.55, zorder=4,
            )

    if overlay_psychometric:
        px = pd.to_numeric(data.psy["delta_mu_rew"],  errors="coerce").to_numpy(dtype=float)
        py = pd.to_numeric(data.psy["delta_mu_unrew"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(px) & np.isfinite(py)
        if finite.any():
            plim = max(1.0, float(np.nanmax(np.abs(np.concatenate(
                [px[finite], py[finite]])))) * 1.25)
            scale = lim / plim
            ax.scatter(px * scale, py * scale, s=45,
                       facecolors="none", edgecolors="#B04040",
                       linewidths=1.0, marker="^", zorder=2,
                       label="Section-3 (Δμ, rescaled)")
            ax_top   = ax.twiny(); ax_right = ax.twinx()
            ax_top.set_xlim(-plim, plim); ax_right.set_ylim(-plim, plim)
            ax_top.set_xlabel("Δμ_rew  [deg, Section 3]",
                              color="#B04040", fontsize=9)
            ax_right.set_ylabel("Δμ_unrew  [deg, Section 3]",
                                color="#B04040", fontsize=9)
            for _ax, side in [(ax_top, "top"), (ax_right, "right")]:
                _ax.tick_params(axis="x" if side == "top" else "y",
                                colors="#B04040", labelsize=8)
                _ax.spines[side].set_color("#B04040")

    if patterns is not None:
        counts = patterns.value_counts()
        handles = [
            plt.Line2D([0], [0], marker="o", linestyle="",
                       markerfacecolor=PATTERN_COLOURS[p], markeredgecolor="none",
                       markersize=10,
                       label=f"{p} (n={int(counts.get(p, 0))})")
            for p in PATTERN_ORDER
        ]
        handles.append(plt.Line2D(
            [0], [0], marker="o", linestyle="",
            markerfacecolor="white", markeredgecolor="black",
            markersize=11, markeredgewidth=POINT_EDGE_LW,
            label=r"dark ring: sig. in ≥1 of S, D, P, $\beta_s$, $\beta_f$",
        ))
        ax.legend(handles=handles, loc="upper left",
                  bbox_to_anchor=(1.02, 1.0), fontsize=9, frameon=False,
                  title="Axis pattern (Section 3)")

    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    if not overlay_psychometric:
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$-\beta_{\mathrm{success}}$  [log-odds]  "
                  r"(previous choice effect | prev rewarded)", fontsize=11)
    ax.set_ylabel(r"$-\beta_{\mathrm{failure}}$  [log-odds]  "
                  r"(previous choice effect | prev unrewarded)", fontsize=11)
    n_sig = int(sig.sum())
    ax.set_title(
        f"Group SD plane — {data.subset} scope, GLM (log-odds, C = {data.C_label})\n"
        f"Medians $(-\\beta_s, -\\beta_f)$ per rat with marginal "
        f"{data.ci_level:g}% CIs  (n = {len(data.rats)}, {n_sig} sig.)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.2)
    return fig


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_subset_arg(parser)
    parser.add_argument("--ellipses", action="store_true")
    parser.add_argument("--overlay-psychometric", action="store_true")
    parser.add_argument("--ci-level", type=float, default=95.0)
    parser.add_argument("--C", default="unpenalized")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    summary, bootstrap, psy = resolve_paths_from_args(args)
    data = load_subset(
        subset=args.subset, C_label=args.C, ci_level=args.ci_level,
        summary_path=summary, bootstrap_path=bootstrap, psy_path=psy,
    )
    fig = plot_sd_plane(
        data,
        show_ellipses=args.ellipses,
        overlay_psychometric=args.overlay_psychometric,
    )
    report = concordance_report(data)
    print(report)

    out_svg = Path(args.out) if args.out else (
        figures_out_dir(args.subset)
        / f"figure_p1_sd_plane_{args.subset}.svg"
    )
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, bbox_inches="tight")
    fig.savefig(out_svg.with_suffix(".png"), dpi=200, bbox_inches="tight")
    out_svg.with_suffix(".concordance.txt").write_text(report, encoding="utf-8")
    print(f"\nSaved: {out_svg}")
    print(f"Saved: {out_svg.with_suffix('.png')}")
    print(f"Saved: {out_svg.with_suffix('.concordance.txt')}")

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())