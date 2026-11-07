"""Figure P3 — Pooled GLM: forest of S/D/P (log-odds contrasts).

Three panels, styled identically to Section 3's ``figure_B_forest``:

    S_logit = β_s + β_f    orange   (matches S diagonal in P1)
    D_logit = β_s − β_f    blue     (matches D diagonal in P1)
    P_logit = β_r          green

Each panel's title carries the Spearman ρ, p-value and sign-agreement fraction
against the corresponding psychometric contrast, so the reader gets the
concordance numbers without having to consult the .txt file.

Psychometric overlay (behind ``--overlay-psychometric``):

    S and D panels (strategy B):
        A secondary top x-axis in degrees; each rat's psychometric estimate
        is drawn as an open red triangle on that axis at the same y-row.

    P panel (strategy C):
        Each rat's psychometric P estimate + CI is printed as text next to
        its row (P has small values in degrees; a secondary axis would sit
        tightly at zero and add little).

Row sort order (--forest-order): 'effect' (default) sorts within each panel
by GLM point estimate; 'rat' uses a common rat-id order across panels.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common_plot_features import (
    CONTRAST_COLOURS,
    PSY_D_COL, PSY_D_EX, PSY_D_HI, PSY_D_LO,
    PSY_P_COL, PSY_P_EX, PSY_P_HI, PSY_P_LO,
    PSY_S_COL, PSY_S_EX, PSY_S_HI, PSY_S_LO,
    SubsetData,
    add_subset_arg, excludes_zero, glm_contrast_excludes, load_subset,
    resolve_paths_from_args,
)
from Portable_path_resolution_for_Section_4_GLM_figures import figures_out_dir

# --------------------------------------------------------------------------
# Concordance
# --------------------------------------------------------------------------
def _sign_agreement(glm_vals: np.ndarray, psy_vals: np.ndarray,
                    expected: int) -> tuple[int, int]:
    """Count sign agreements. ``expected = -1`` flips the GLM sign before
    comparing (S and D have opposite sign conventions between GLM and
    psychometric); ``expected = +1`` compares directly (P)."""
    m = np.isfinite(glm_vals) & np.isfinite(psy_vals) & \
        (glm_vals != 0) & (psy_vals != 0)
    if not m.any():
        return 0, 0
    g = glm_vals[m]
    p = psy_vals[m]
    agrees = (np.sign(g) == -np.sign(p)) if expected == -1 else (np.sign(g) == np.sign(p))
    return int(agrees.sum()), int(m.sum())


def _concordance(glm_vals: pd.Series, psy_vals: pd.Series,
                 expected_sign: int) -> tuple[float, float, int, int]:
    """Spearman ρ, p-value, and sign-agreement counts on common finite rats."""
    common = glm_vals.dropna().index.intersection(psy_vals.dropna().index)
    g = pd.to_numeric(glm_vals.loc[common], errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(psy_vals.loc[common], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(g) & np.isfinite(p)
    g, p = g[m], p[m]
    if len(g) < 3:
        return float("nan"), float("nan"), 0, 0
    rho, pval = spearmanr(g, p)
    n_agr, n_tot = _sign_agreement(g, p, expected_sign)
    return float(rho), float(pval), n_agr, n_tot


# --------------------------------------------------------------------------
# Panel drawing
# --------------------------------------------------------------------------
def _plot_forest_panel(
    ax: plt.Axes,
    values: np.ndarray,
    ci_lo: np.ndarray,
    ci_hi: np.ndarray,
    rat_ids: np.ndarray,
    excludes: np.ndarray,
    colour: str,
    name: str,
    order: str,
    ci_level: float,
    concordance: tuple[float, float, int, int] | None,
) -> np.ndarray:
    """Draw the panel and return the row-order index (so overlays can align)."""
    if order == "effect":
        idx = np.argsort(values, kind="mergesort")
    elif order == "rat":
        idx = np.argsort(rat_ids, kind="mergesort")
    else:
        raise ValueError(f"Unknown order {order!r}")

    v  = values[idx]
    lo = ci_lo[idx]
    hi = ci_hi[idx]
    r  = rat_ids[idx]
    ex = excludes[idx]

    for i in range(len(v)):
        face = colour if ex[i] else "white"
        xerr_lo = max(0.0, float(v[i] - lo[i]))
        xerr_hi = max(0.0, float(hi[i] - v[i]))
        ax.errorbar(
            v[i], i,
            xerr=[[xerr_lo], [xerr_hi]],
            fmt="o", ecolor=colour, elinewidth=1.4,
            markerfacecolor=face, markeredgecolor=colour,
            markeredgewidth=1.8, markersize=9, capsize=3,
        )

    ax.axvline(0, color="k", lw=0.8, linestyle="--", alpha=0.7)
    ax.set_yticks(range(len(v)))
    ax.set_yticklabels([f"rat {int(x)}" for x in r], fontsize=9)
    ax.set_xlabel(r"$\beta$  [log-odds]", fontsize=10)

    n_sig = int(ex.sum())
    lines = [name, f"{n_sig}/{len(v)} rats: {ci_level:g}% CI excludes 0"]
    if concordance is not None:
        rho, pval, n_agr, n_tot = concordance
        rho_str = f"ρ = {rho:+.2f}" if np.isfinite(rho) else "ρ = n/a"
        p_str   = f"p = {pval:.3f}" if np.isfinite(pval) else "p = n/a"
        lines.append(f"psy: {rho_str}, {p_str}, sign agr {n_agr}/{n_tot}")
    ax.set_title("\n".join(lines), fontsize=9.5)
    ax.grid(True, axis="x", alpha=0.2)
    return idx


# --------------------------------------------------------------------------
# Overlays
# --------------------------------------------------------------------------
_OVERLAY_COLOR = "#B04040"


def _overlay_secondary_axis(
    ax: plt.Axes,
    row_idx: np.ndarray,
    rat_ids: np.ndarray,
    psy_values: np.ndarray,
    psy_lo: np.ndarray,
    psy_hi: np.ndarray,
    psy_ex: np.ndarray,
    axis_name: str,
) -> None:
    """Strategy B — add a top x-axis in degrees and plot psychometric points
    as open triangles at each row's y position."""
    finite = np.isfinite(psy_values) & np.isfinite(psy_lo) & np.isfinite(psy_hi)
    if not finite.any():
        return
    span = float(np.nanmax(np.abs(np.concatenate([
        psy_values[finite], psy_lo[finite], psy_hi[finite]
    ])))) * 1.15
    span = max(span, 1.0)

    ax_top = ax.twiny()
    ax_top.set_xlim(-span, span)
    ax_top.set_ylim(ax.get_ylim())
    ax_top.tick_params(axis="x", colors=_OVERLAY_COLOR, labelsize=8)
    ax_top.spines["top"].set_color(_OVERLAY_COLOR)
    ax_top.set_xlabel(f"{axis_name} [deg, Section 3]",
                      color=_OVERLAY_COLOR, fontsize=9)
    ax_top.axvline(0, color=_OVERLAY_COLOR, lw=0.5, linestyle=":", alpha=0.5)

    for i_row, k in enumerate(row_idx):
        if not finite[k]:
            continue
        face = _OVERLAY_COLOR if psy_ex[k] else "none"
        ax_top.errorbar(
            psy_values[k], i_row,
            xerr=[[psy_values[k] - psy_lo[k]], [psy_hi[k] - psy_values[k]]],
            fmt="^", ecolor=_OVERLAY_COLOR, elinewidth=0.9,
            markerfacecolor=face, markeredgecolor=_OVERLAY_COLOR,
            markeredgewidth=1.2, markersize=6, capsize=2, alpha=0.85,
        )


def _overlay_text_annotation(
    ax: plt.Axes,
    row_idx: np.ndarray,
    psy_values: np.ndarray,
    psy_lo: np.ndarray,
    psy_hi: np.ndarray,
    psy_ex: np.ndarray,
) -> None:
    """Strategy C — annotate each row with 'ψ: +v° [lo, hi]' at the right edge."""
    finite = np.isfinite(psy_values) & np.isfinite(psy_lo) & np.isfinite(psy_hi)
    if not finite.any():
        return
    xlim = ax.get_xlim()
    x_text = xlim[0] + (xlim[1] - xlim[0]) * 1.02
    for i_row, k in enumerate(row_idx):
        if not finite[k]:
            continue
        weight = "bold" if psy_ex[k] else "normal"
        txt = f"ψ: {psy_values[k]:+.2f}° [{psy_lo[k]:+.2f}, {psy_hi[k]:+.2f}]"
        ax.text(x_text, i_row, txt, fontsize=7.5, va="center",
                color=_OVERLAY_COLOR, fontweight=weight)


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
_PANELS: tuple[tuple[str, str, str, int, str, str, str, str], ...] = (
    # (glm_contrast, psy_col, psy_lo, psy_hi, expected_sign,
    #  overlay_strategy, panel_title, degree_axis_name)
    ("S_logit", PSY_S_COL, PSY_S_LO, PSY_S_HI, -1,
     "secondary", r"S (main effect of previous choice)", "S"),
    ("D_logit", PSY_D_COL, PSY_D_LO, PSY_D_HI, -1,
     "secondary", r"D (previous-reward × previous-choice)", "D"),
    ("P_logit", PSY_P_COL, PSY_P_LO, PSY_P_HI, +1,
     "annotation", r"P (main effect of previous outcome)", "P"),
)


def plot_sdp_forest(
    data: SubsetData,
    order: str = "effect",
    overlay_psychometric: bool = False,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    rat_arr = np.array(data.rats, dtype=int)

    for ax, (contrast, psy_col, psy_lo_col, psy_hi_col, exp_sign,
             strategy, title, deg_name) in zip(axes, _PANELS):
        v  = data.glm_derived_point[contrast].to_numpy(dtype=float)
        lo = data.glm_derived_ci_lo[contrast].to_numpy(dtype=float)
        hi = data.glm_derived_ci_hi[contrast].to_numpy(dtype=float)
        ex = glm_contrast_excludes(data, contrast).to_numpy(dtype=bool)

        psy_v  = pd.to_numeric(data.psy[psy_col],    errors="coerce").to_numpy(dtype=float)
        psy_lo = pd.to_numeric(data.psy[psy_lo_col], errors="coerce").to_numpy(dtype=float)
        psy_hi = pd.to_numeric(data.psy[psy_hi_col], errors="coerce").to_numpy(dtype=float)
        psy_ex = excludes_zero(psy_lo, psy_hi)

        concordance = _concordance(
            data.glm_derived_point[contrast],
            pd.to_numeric(data.psy[psy_col], errors="coerce"),
            expected_sign=exp_sign,
        )
        idx = _plot_forest_panel(
            ax, v, lo, hi, rat_arr, ex,
            colour=CONTRAST_COLOURS[contrast], name=title,
            order=order, ci_level=data.ci_level,
            concordance=concordance,
        )

        if overlay_psychometric:
            if strategy == "secondary":
                _overlay_secondary_axis(
                    ax, idx, rat_arr,
                    psy_v, psy_lo, psy_hi, psy_ex,
                    axis_name=rf"$\Delta$ {deg_name}",
                )
            elif strategy == "annotation":
                _overlay_text_annotation(ax, idx, psy_v, psy_lo, psy_hi, psy_ex)

    fig.suptitle(
        f"Forest — GLM contrasts vs Section-3 psychometrics "
        f"({data.subset} scope, C = {data.C_label}, n = {len(data.rats)} rats)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_subset_arg(parser)
    parser.add_argument("--overlay-psychometric", action="store_true",
                        help="Overlay Section-3 estimates (secondary axis for "
                             "S/D, text annotation for P).")
    parser.add_argument("--forest-order", choices=["effect", "rat"],
                        default="effect")
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
    fig = plot_sdp_forest(
        data, order=args.forest_order,
        overlay_psychometric=args.overlay_psychometric,
    )

    out_svg = Path(args.out) if args.out else (
        figures_out_dir(args.subset)
        / f"figure_p3_sdp_forest_{args.subset}.svg"
    )
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, bbox_inches="tight")
    fig.savefig(out_svg.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {out_svg}")
    print(f"Saved: {out_svg.with_suffix('.png')}")

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())