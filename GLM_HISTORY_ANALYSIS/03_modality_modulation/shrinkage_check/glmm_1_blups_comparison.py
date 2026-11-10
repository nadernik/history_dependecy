"""
Shrinkage plot: §4 individual per-rat GLM estimates vs GLMM I shrunk BLUPs.

Scientific purpose
------------------
Validate that GLMM I's random structure is genuinely recovering individual
rat-level history effects rather than being an artifact of the (correlated)
random-slope covariance. A well-behaved shrinkage plot shows:

    - points scattered around y = x
    - slope < 1: extreme rats pulled toward the horizontal population-mean line
    - more shrinkage for rats whose §4 estimate had larger uncertainty

Four panels in log-odds units:
    beta_S   post-success bias           (beta_success)
    beta_F   post-failure bias           (beta_failure)
    S        choice-following            (beta_S + beta_F)
    D        reward x choice interaction (beta_S - beta_F)

Two output figures
------------------
    shrinkage_v1_drop_rat11_tactile.png    rat 11 tactile cell excluded from
                                           the precision-weighted collapse
                                           (matches §4 degree-scale reporting rule)
    shrinkage_v2_include_all.png           all (rat x modality) cells included

Companion CSV table with per-rat unpooled and shrunk values for both versions.

Modality collapsing
-------------------
GLMM I pools across modality in its history fixed effects. §4 estimates one set
of history coefficients per (rat x modality). To compare on the same axis we
collapse §4's three per-modality estimates per rat into one number using
inverse-variance (precision) weighting. SE per modality is the bootstrap SD.

    combined_point = sum_m (w_m * point_m) / sum_m w_m
    combined_se    = sqrt(1 / sum_m w_m)
    w_m            = 1 / se_m^2

This matches what GLMM I is implicitly doing (pooling with weight proportional
to information content of each modality).

Everything is done in log-odds. S/D are computed from beta_success and
beta_failure directly, so §4's pse_on_scale gate (which was about physical
realizability of degree-scale PSE reconstructions) does not apply here. Version
A still excludes rat 11 tactile to stay consistent with §4's own convention.

Expected input directory structure
----------------------------------
    <PROJECT_ROOT>/
        glm_plots_final/
            glm_bootstrap_draws.parquet     (§4 bootstrap, long format)
            glm_summary_long.csv            (§4 MLE point estimates)
        thesis_fits/
            glmm_I/
                glmm_I_primary/
                    per_rat_coords.csv      (GLMM I shrunk per-rat coords)
                    fixed_effects.csv       (GLMM I population fixed effects)
                    contrasts.csv           (GLMM I S_beta, D_beta contrasts)

Output goes to <PROJECT_ROOT>/thesis_fits/glmm_I/shrinkage_plots/

Porting to another machine: edit PROJECT_ROOT only. The relative subpath
constants below assume the same layout the analysis pipeline produces.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================================
# CONFIGURATION - edit PROJECT_ROOT only when moving to a new machine
# =========================================================================

PROJECT_ROOT = Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy")

# Fixed relative structure (change only if the analysis outputs move)
GLM_SUBDIR       = Path("glm_plots_final")
GLMM_I_SUBDIR    = Path("thesis_fits") / "glmm_I" / "glmm_I_primary"
OUTPUT_SUBDIR    = Path("thesis_fits") / "glmm_I" / "shrinkage_plots"

# Rat x modality cell to exclude in Version A. Rat 11 tactile is the §4
# pse_on_scale failure. Match the rat_id string as it appears in the parquet.
EXCLUDE_CELL = ("11", "tactile")

# Which §4 fits to compare against. "unpenalized" matches GLMM I; the ridge
# C values in §4 are only sensitivity checks and are not used here.
GLM_C_LABEL = "unpenalized"

# Bootstrap draws per (rat x modality) fit in §4 (only used for a sanity print)
EXPECTED_N_BOOT = 5000


# =========================================================================
# DERIVED PATHS (do not edit)
# =========================================================================

GLM_DIR    = PROJECT_ROOT / GLM_SUBDIR
GLMM_I_DIR = PROJECT_ROOT / GLMM_I_SUBDIR
OUTPUT_DIR = PROJECT_ROOT / OUTPUT_SUBDIR

BOOT_PARQUET   = GLM_DIR / "glm_bootstrap_draws.parquet"
SUMMARY_CSV    = GLM_DIR / "glm_summary_long.csv"
PER_RAT_CSV    = GLMM_I_DIR / "per_rat_coords.csv"
FIXED_CSV      = GLMM_I_DIR / "fixed_effects.csv"
CONTRASTS_CSV  = GLMM_I_DIR / "contrasts.csv"


# =========================================================================
# LOAD
# =========================================================================

def load_all():
    for p in [BOOT_PARQUET, SUMMARY_CSV, PER_RAT_CSV, FIXED_CSV, CONTRASTS_CSV]:
        if not p.exists():
            raise FileNotFoundError(f"Missing expected input: {p}")

    print("Loading inputs...")
    boot    = pd.read_parquet(BOOT_PARQUET)
    summary = pd.read_csv(SUMMARY_CSV)
    per_rat = pd.read_csv(PER_RAT_CSV)
    fixed   = pd.read_csv(FIXED_CSV)
    contr   = pd.read_csv(CONTRASTS_CSV)

    # Cast rat IDs to string for consistent joins across sources
    boot["rat_id"]    = boot["rat_id"].astype(str)
    summary["rat_id"] = summary["rat_id"].astype(str)
    per_rat["rat_id"] = per_rat["rat_id"].astype(str)

    print(f"  bootstrap: {len(boot):,} rows, "
          f"{boot['rat_id'].nunique()} rats, "
          f"C values: {sorted(boot['C'].unique())}")
    print(f"  summary:   {len(summary):,} rows")
    print(f"  per_rat:   {len(per_rat)} rats")

    return boot, summary, per_rat, fixed, contr


# =========================================================================
# §4 SIDE: per (rat x modality) point + SE, then precision-weighted collapse
# =========================================================================

def compute_per_cell(boot, summary):
    """Return one row per (rat_id, subset) with MLE point + bootstrap SE for
    beta_S, beta_F, S = beta_S + beta_F, D = beta_S - beta_F."""

    # ---- Bootstrap: pivot to get beta_S and beta_F per draw, then derive S/D
    b = boot[
        (boot["C"] == GLM_C_LABEL)
        & (boot["subset"].isin(["tactile", "visual", "visuotactile"]))
        & (boot["quantity"].isin(["beta_success_n-1", "beta_failure_n-1"]))
    ].copy()

    b_wide = b.pivot_table(
        index=["rat_id", "subset", "boot_idx"],
        columns="quantity",
        values="value",
    ).reset_index()
    b_wide = b_wide.rename(columns={
        "beta_success_n-1": "beta_S",
        "beta_failure_n-1": "beta_F",
    })
    b_wide["S"] = b_wide["beta_S"] + b_wide["beta_F"]
    b_wide["D"] = b_wide["beta_S"] - b_wide["beta_F"]

    # Sanity: expected number of draws per cell
    n_per_cell = b_wide.groupby(["rat_id", "subset"]).size()
    if (n_per_cell != EXPECTED_N_BOOT).any():
        odd = n_per_cell[n_per_cell != EXPECTED_N_BOOT]
        print(f"  WARNING: {len(odd)} cells have != {EXPECTED_N_BOOT} bootstrap draws")

    boot_se = b_wide.groupby(["rat_id", "subset"]).agg(
        beta_S_se=("beta_S", lambda x: x.std(ddof=1)),
        beta_F_se=("beta_F", lambda x: x.std(ddof=1)),
        S_se     =("S",      lambda x: x.std(ddof=1)),
        D_se     =("D",      lambda x: x.std(ddof=1)),
    ).reset_index()

    # ---- MLE point estimates from §4's summary_long
    s = summary[
        (summary["C"] == GLM_C_LABEL)
        & (summary["subset"].isin(["tactile", "visual", "visuotactile"]))
        & (summary["quantity"].isin(["beta_success_n-1", "beta_failure_n-1"]))
    ].copy()
    mle = s.pivot_table(
        index=["rat_id", "subset"],
        columns="quantity",
        values="value",
    ).reset_index()
    mle = mle.rename(columns={
        "beta_success_n-1": "beta_S_mle",
        "beta_failure_n-1": "beta_F_mle",
    })
    mle["S_mle"] = mle["beta_S_mle"] + mle["beta_F_mle"]
    mle["D_mle"] = mle["beta_S_mle"] - mle["beta_F_mle"]

    cell = mle.merge(boot_se, on=["rat_id", "subset"], how="inner")
    print(f"  §4 per-cell stats: {len(cell)} (rat x modality) cells "
          f"across {cell['rat_id'].nunique()} rats")
    return cell


def precision_weighted(points, ses):
    """Inverse-variance weighted combination. Returns (point, se, n_used)."""
    points = np.asarray(points, dtype=float)
    ses    = np.asarray(ses,    dtype=float)
    ok = np.isfinite(points) & np.isfinite(ses) & (ses > 0)
    if not ok.any():
        return np.nan, np.nan, 0
    w = 1.0 / (ses[ok] ** 2)
    p = np.sum(w * points[ok]) / np.sum(w)
    s = np.sqrt(1.0 / np.sum(w))
    return float(p), float(s), int(ok.sum())


def collapse_across_modality(cell_stats):
    """Return one row per rat with collapsed point + SE for each quantity."""
    rows = []
    for rat_id, grp in cell_stats.groupby("rat_id"):
        row = {"rat_id": rat_id,
               "modalities_used": ",".join(sorted(grp["subset"].tolist()))}
        for q in ["beta_S", "beta_F", "S", "D"]:
            p, s, n = precision_weighted(grp[f"{q}_mle"], grp[f"{q}_se"])
            row[f"{q}_collapsed"]    = p
            row[f"{q}_collapsed_se"] = s
            row["n_modalities"]      = n
        rows.append(row)
    return pd.DataFrame(rows)


# =========================================================================
# GLMM I SIDE: shrunk per-rat coordinates + population means
# =========================================================================

def get_population_means(fixed, contr):
    def _lookup(df, col_name, col_val, val_col):
        row = df[df[col_name] == col_val]
        if row.empty:
            raise KeyError(f"{col_val} not found in {col_name}")
        return float(row[val_col].iloc[0])

    pop = {
        "beta_S": _lookup(fixed, "term",     "success_prev", "estimate"),
        "beta_F": _lookup(fixed, "term",     "failure_prev", "estimate"),
        "S":      _lookup(contr, "contrast", "S_beta",       "estimate"),
        "D":      _lookup(contr, "contrast", "D_beta",       "estimate"),
    }
    print(f"  GLMM I population means (log-odds):")
    for k, v in pop.items():
        print(f"    {k:8s} = {v:+.4f}")
    return pop


def prepare_glmm_side(per_rat):
    """Rename GLMM I columns to line up with the collapsed §4 side."""
    g = per_rat[["rat_id", "beta_S_plus_b", "beta_F_plus_b", "S", "D"]].copy()
    g = g.rename(columns={
        "beta_S_plus_b": "beta_S_glmm",
        "beta_F_plus_b": "beta_F_glmm",
        "S":             "S_glmm",
        "D":             "D_glmm",
    })
    return g


# =========================================================================
# PLOTTING
# =========================================================================

PANEL_SPECS = [
    # (quantity_key, panel_title)
    ("beta_S", r"$\beta_S$   (post-success)"),
    ("beta_F", r"$\beta_F$   (post-failure)"),
    ("S",      r"$S = \beta_S + \beta_F$   (choice-following)"),
    ("D",      r"$D = \beta_S - \beta_F$   (reward $\times$ choice)"),
]


def make_shrinkage_figure(merged, pop_means, subtitle, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    axes = axes.flatten()

    # Consistent per-rat colors (sorted numerically where possible)
    def _sort_key(s):
        try:
            return (0, int(s))
        except ValueError:
            return (1, s)
    rat_ids = sorted(merged["rat_id"].unique(), key=_sort_key)
    cmap = plt.get_cmap("tab20")
    rat_colors = {r: cmap(i % 20) for i, r in enumerate(rat_ids)}

    for ax, (q, panel_title) in zip(axes, PANEL_SPECS):
        x_col = f"{q}_collapsed"
        y_col = f"{q}_glmm"
        se_col = f"{q}_collapsed_se"
        pop_val = pop_means[q]

        xs = merged[x_col].dropna().values
        ys = merged[y_col].dropna().values
        finite_pool = np.concatenate([xs, ys, [pop_val, 0.0]])
        span = finite_pool.max() - finite_pool.min()
        pad = 0.08 * (span if span > 0 else 1.0)
        lo = finite_pool.min() - pad
        hi = finite_pool.max() + pad

        # Reference geometry
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.35, linewidth=1,
                label="y = x (no shrinkage)")
        ax.axhline(pop_val, color="crimson", linestyle=":", alpha=0.7,
                   linewidth=1.2,
                   label=f"GLMM I pop. mean = {pop_val:+.3f}")
        ax.axhline(0, color="gray", alpha=0.25, linewidth=0.5)
        ax.axvline(0, color="gray", alpha=0.25, linewidth=0.5)

        # Points
        for _, row in merged.iterrows():
            rat = row["rat_id"]
            x, y = row[x_col], row[y_col]
            se = row[se_col]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            ax.errorbar(x, y,
                        xerr=se if np.isfinite(se) else None,
                        fmt="o", color=rat_colors[rat],
                        markersize=8, markeredgecolor="black",
                        markeredgewidth=0.6, capsize=3, alpha=0.9,
                        zorder=3)
            ax.annotate(rat, (x, y), xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8, alpha=0.75)

        ax.set_xlabel("§4 per-rat estimate\n(precision-weighted across modality)")
        ax.set_ylabel("GLMM I shrunk per-rat coordinate\n" +
                      r"($\beta + b_{\mathrm{rat}}$)")
        ax.set_title(panel_title, fontsize=11)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best", fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.2)

    fig.suptitle(
        "BLUP shrinkage: §4 individual vs GLMM I shrunk estimates\n" + subtitle,
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 72)
    print("Shrinkage plot: §4 individual GLMs vs GLMM I BLUPs")
    print("=" * 72)
    print(f"Project root: {PROJECT_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    boot, summary, per_rat, fixed, contr = load_all()

    print("\nComputing §4 per (rat x modality) point + SE...")
    cell_stats = compute_per_cell(boot, summary)

    # Version A: exclude rat 11 tactile
    ex_rat, ex_mod = EXCLUDE_CELL
    mask_ex = (cell_stats["rat_id"] == ex_rat) & (cell_stats["subset"] == ex_mod)
    n_ex = int(mask_ex.sum())
    if n_ex == 0:
        print(f"\n  NOTE: no cell matched EXCLUDE_CELL={EXCLUDE_CELL}; "
              f"Version A will equal Version B")
    else:
        print(f"\n  Version A: dropping {n_ex} cell(s) matching "
              f"rat={ex_rat}, modality={ex_mod}")
    cell_A = cell_stats[~mask_ex].copy()
    cell_B = cell_stats.copy()

    print("\nPrecision-weighted collapse across modality...")
    collapsed_A = collapse_across_modality(cell_A)
    collapsed_B = collapse_across_modality(cell_B)
    print(f"  Version A: {len(collapsed_A)} rats")
    print(f"  Version B: {len(collapsed_B)} rats")

    print("\nLoading GLMM I population means and per-rat coords...")
    pop_means = get_population_means(fixed, contr)
    glmm_side = prepare_glmm_side(per_rat)
    print(f"  {len(glmm_side)} rats in GLMM I per_rat_coords")

    # Merge §4 collapsed with GLMM I shrunk coords
    merged_A = collapsed_A.merge(glmm_side, on="rat_id", how="inner")
    merged_B = collapsed_B.merge(glmm_side, on="rat_id", how="inner")
    print(f"\nMerged for plotting:")
    print(f"  Version A: {len(merged_A)} rats (join keys matched)")
    print(f"  Version B: {len(merged_B)} rats (join keys matched)")

    # Warn if any rats didn't join
    rat_glmm  = set(glmm_side["rat_id"])
    rat_glmA  = set(collapsed_A["rat_id"])
    rat_glmB  = set(collapsed_B["rat_id"])
    unmatched = (rat_glmm ^ rat_glmA) | (rat_glmm ^ rat_glmB)
    if unmatched:
        print(f"  WARNING: rat IDs not shared between GLMM and §4 sides: "
              f"{sorted(unmatched)}")

    print("\nRendering figures...")
    fig_A = OUTPUT_DIR / "shrinkage_v1_drop_rat11_tactile.png"
    fig_B = OUTPUT_DIR / "shrinkage_v2_include_all.png"
    make_shrinkage_figure(
        merged_A, pop_means,
        subtitle=f"Version A: {EXCLUDE_CELL[0]} {EXCLUDE_CELL[1]} excluded from collapse",
        output_path=fig_A,
    )
    make_shrinkage_figure(
        merged_B, pop_means,
        subtitle="Version B: all (rat x modality) cells included",
        output_path=fig_B,
    )

    # Companion CSV
    def _tag(df, ver):
        out = df.copy()
        out["version"] = ver
        return out
    table = pd.concat([
        _tag(merged_A, "drop_rat11_tactile"),
        _tag(merged_B, "include_all"),
    ], ignore_index=True)

    keep_cols = [
        "version", "rat_id", "modalities_used", "n_modalities",
        "beta_S_collapsed", "beta_S_collapsed_se", "beta_S_glmm",
        "beta_F_collapsed", "beta_F_collapsed_se", "beta_F_glmm",
        "S_collapsed",      "S_collapsed_se",      "S_glmm",
        "D_collapsed",      "D_collapsed_se",      "D_glmm",
    ]
    table = table[[c for c in keep_cols if c in table.columns]]
    table_path = OUTPUT_DIR / "shrinkage_comparison_table.csv"
    table.to_csv(table_path, index=False)
    print(f"  Saved: {table_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()