"""Figure P2 — Pooled GLM: forest of the six coefficients.

Two rows of three panels; each panel is a forest plot of one coefficient
across rats, styled identically to Section 3's ``figure_B_forest``:

    row 1 (task):     β0 (intercept)    β_θ (current angle)    β_θ,n-1 (previous angle)
    row 2 (history):  β_r,n-1 (reward)   β_s,n-1 (prev succ)   β_f,n-1 (prev fail)

Point + CI in log-odds. Filled marker if the CI at ``--ci-level`` excludes
zero, hollow otherwise. Reference dashed line at 0. Row sort order:
--forest-order effect (default, sorted by that panel's point estimate) or
'rat' (rat id ascending, common across all panels).

No psychometric overlay — P2 is a GLM internal-consistency view.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_plot_features import (
    COEF_COLOURS, COEF_LABELS, COEF_ORDER,
    SubsetData,
    add_subset_arg, glm_coef_excludes, load_subset, resolve_paths_from_args,
)
from Portable_path_resolution_for_Section_4_GLM_figures import figures_out_dir


# --------------------------------------------------------------------------
# Panel drawing (mirrors Section_3_group_figures.figure_B_forest)
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
    xlabel: str,
) -> None:
    if order == "effect":
        idx = np.argsort(values, kind="mergesort")  # stable
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
    ax.set_xlabel(xlabel, fontsize=10)
    n_sig = int(ex.sum())
    ax.set_title(f"{name}\n{n_sig}/{len(v)} rats: {ci_level:g}% CI excludes 0",
                 fontsize=10)
    ax.grid(True, axis="x", alpha=0.2)


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def plot_coefficient_forest(
    data: SubsetData,
    order: str = "effect",
) -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(14, 12))

    rat_arr = np.array(data.rats, dtype=int)
    for ax, q in zip(axes.ravel(), COEF_ORDER):
        if q not in data.glm_point.columns:
            ax.set_axis_off()
            ax.set_title(f"{COEF_LABELS[q]}\n(not in summary)", fontsize=10)
            continue
        v  = data.glm_point[q].to_numpy(dtype=float)
        lo = data.glm_ci_lo[q].to_numpy(dtype=float)
        hi = data.glm_ci_hi[q].to_numpy(dtype=float)
        ex = glm_coef_excludes(data, q).to_numpy(dtype=bool)
        _plot_forest_panel(
            ax, v, lo, hi, rat_arr, ex,
            colour=COEF_COLOURS[q], name=COEF_LABELS[q],
            order=order, ci_level=data.ci_level,
            xlabel=r"$\beta$  [log-odds]",
        )

    fig.suptitle(
        f"Forest — GLM coefficients ({data.subset} scope, C = {data.C_label}, "
        f"n = {len(data.rats)} rats)",
        fontsize=12, y=1.005,
    )
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_subset_arg(parser)
    parser.add_argument("--forest-order", choices=["effect", "rat"],
                        default="effect",
                        help="Sort rats within each panel by effect size "
                             "(default) or by rat id.")
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
    fig = plot_coefficient_forest(data, order=args.forest_order)

    out_svg = Path(args.out) if args.out else (
        figures_out_dir(args.subset)
        / f"figure_p2_coefficient_forest_{args.subset}.svg"
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