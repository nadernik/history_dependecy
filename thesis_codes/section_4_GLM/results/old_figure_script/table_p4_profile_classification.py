"""Table P4 — Pooled profile classification: GLM vs Section-3 psychometric.

Compares the per-rat "profile" (which of choice/interaction/reward has a
non-zero CI at the chosen level) between the two analyses. Same
CI-excludes-0 rule applied on both sides so the labels are directly
comparable — 'choice+interaction' from GLM means the same thing as
'choice+interaction' from the psychometric.

Emits three artefacts:
    - CSV table: rat, profile_psy, profile_glm, agreement
    - PNG/SVG figure: color-coded matplotlib table
    - Console summary: N exact / N partial / N none, plus a confusion matrix
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_plot_features import (
    SubsetData,
    add_subset_arg, glm_profile_series, load_subset, profile_agreement,
    psy_profile_series, resolve_paths_from_args,
)
from Portable_path_resolution_for_Section_4_GLM_figures import figures_out_dir

# Cell background colors for the agreement column
AGREE_COLORS = {
    "exact":   "#C8E6C9",   # light green
    "partial": "#FFE0B2",   # light amber
    "none":    "#FFCDD2",   # light red
}
AGREE_ORDER = ["exact", "partial", "none"]


# --------------------------------------------------------------------------
# Table assembly
# --------------------------------------------------------------------------
def build_profile_table(data: SubsetData) -> pd.DataFrame:
    """Return DataFrame with columns rat, profile_psy, profile_glm, agreement."""
    psy = psy_profile_series(data)
    glm = glm_profile_series(data)
    agr = [profile_agreement(psy.loc[r], glm.loc[r]) for r in data.rats]
    df = pd.DataFrame({
        "rat":          list(data.rats),
        "profile_psy":  psy.to_numpy(),
        "profile_glm":  glm.to_numpy(),
        "agreement":    agr,
    })
    # Sort by agreement quality first (exact → partial → none), then rat id
    agr_rank = {name: i for i, name in enumerate(AGREE_ORDER)}
    df["_rank"] = df["agreement"].map(agr_rank)
    df = df.sort_values(["_rank", "rat"]).drop(columns="_rank").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Matplotlib figure
# --------------------------------------------------------------------------
def render_table_figure(df: pd.DataFrame, data: SubsetData) -> plt.Figure:
    n_rows = len(df)
    row_h_in    = 0.32
    header_h_in = 0.75   # for suptitle
    fig_h = header_h_in + row_h_in * (n_rows + 1) + 0.2
    top_frac = 1.0 - header_h_in / fig_h
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.set_axis_off()

    cell_text = [
        [f"{int(row['rat'])}", row["profile_psy"], row["profile_glm"], row["agreement"]]
        for _, row in df.iterrows()
    ]
    col_labels = ["Rat", "profile (Section 3)", "profile (GLM)", "agreement"]

    cell_colours = []
    for _, row in df.iterrows():
        bg = AGREE_COLORS.get(row["agreement"], "#FFFFFF")
        row_tint = _lighten(bg, 0.55)
        cell_colours.append([row_tint, row_tint, row_tint, bg])

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colours,
        colColours=["#ECEFF1"] * 4,
        colWidths=[0.10, 0.35, 0.35, 0.20],
        cellLoc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    for r in range(n_rows + 1):
        for c in range(4):
            cell = tbl[r, c]
            cell.set_edgecolor("#B0BEC5")
            if r == 0:
                cell.set_text_props(fontweight="bold")

    total = len(df)
    n_exact   = int((df["agreement"] == "exact").sum())
    n_partial = int((df["agreement"] == "partial").sum())
    n_none    = int((df["agreement"] == "none").sum())
    subtitle = (
        f"{n_exact}/{total} exact match   ·   "
        f"{n_partial}/{total} partial overlap   ·   "
        f"{n_none}/{total} no overlap"
    )
    fig.suptitle(
        f"Profile agreement — {data.subset} Section 3 vs GLM  "
        f"(C = {data.C_label}, {data.ci_level:g}% CI)\n{subtitle}",
        fontsize=11, y=1.0 - 0.02,
    )
    plt.subplots_adjust(top=top_frac - 0.02,
                        bottom=0.02, left=0.02, right=0.98)
    return fig


def _lighten(hex_color: str, factor: float) -> str:
    """Blend hex_color toward white by `factor` in [0, 1]."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f"#{r:02X}{g:02X}{b:02X}"


# --------------------------------------------------------------------------
# Console summary + confusion matrix
# --------------------------------------------------------------------------
def print_summary(df: pd.DataFrame, data: SubsetData) -> str:
    total = len(df)
    n_exact   = int((df["agreement"] == "exact").sum())
    n_partial = int((df["agreement"] == "partial").sum())
    n_none    = int((df["agreement"] == "none").sum())

    lines = [
        f"{data.subset.capitalize()} profile agreement (Section 3 vs GLM)",
        "=" * 44,
        f"Subset:   {data.subset}",
        f"N rats:   {total}",
        f"C:        {data.C_label}",
        f"CI level: {data.ci_level:g}%",
        "",
        f"exact match     : {n_exact:>3}/{total}",
        f"partial overlap : {n_partial:>3}/{total}",
        f"no overlap      : {n_none:>3}/{total}",
        "",
        "Confusion (rows = psychometric, cols = GLM):",
    ]
    conf = pd.crosstab(df["profile_psy"], df["profile_glm"], dropna=False)
    lines.append(conf.to_string())
    lines.append("")
    lines.append("Per-rat table:")
    lines.append(df.to_string(index=False))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_subset_arg(parser)
    parser.add_argument("--ci-level", type=float, default=95.0)
    parser.add_argument("--C", default="unpenalized")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="Output SVG path (CSV and .txt saved alongside)")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    summary, bootstrap, psy = resolve_paths_from_args(args)
    data = load_subset(
        subset=args.subset, C_label=args.C, ci_level=args.ci_level,
        summary_path=summary, bootstrap_path=bootstrap, psy_path=psy,
    )
    df = build_profile_table(data)
    fig = render_table_figure(df, data)
    report = print_summary(df, data)
    print(report)

    out_svg = Path(args.out) if args.out else (
        figures_out_dir(args.subset)
        / f"table_p4_profile_comparison_{args.subset}.svg"
    )
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_svg, bbox_inches="tight")
    fig.savefig(out_svg.with_suffix(".png"), dpi=200, bbox_inches="tight")
    df.to_csv(out_svg.with_suffix(".csv"), index=False)
    out_svg.with_suffix(".txt").write_text(report, encoding="utf-8")
    print(f"\nSaved: {out_svg}")
    print(f"Saved: {out_svg.with_suffix('.png')}")
    print(f"Saved: {out_svg.with_suffix('.csv')}")
    print(f"Saved: {out_svg.with_suffix('.txt')}")

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())