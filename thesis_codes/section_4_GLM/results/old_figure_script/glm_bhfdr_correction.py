#!/usr/bin/env python3
"""
glm_bhfdr_correction.py
========================

Reproduces the Section 4 / Q1a numbers for the pooled-GLM history-effects
analysis:

  * a two-sided percentile-bootstrap p-value for each (rat, contrast) cell,
    computed straight from the GLM pipeline's own raw bootstrap replicates
    (not approximated from the summary CI table),
  * a Benjamini-Hochberg FDR correction applied jointly across the pooled
    "primary family" (n_rats x 4 contrasts: delta_mu_rewarded [dmuR],
    delta_mu_unrewarded [dmuU], D_glm [D], P_glm [P]) -- the same family
    definition used for the psychometric analysis in Section 3, so the two
    analyses stay comparable,
  * the before/after-correction resolved-rat counts and rollup stats quoted
    in the results text,
  * the per-rat AIC/BIC model-comparison table from the fit metadata.

Point of this script: your GLM pipeline (whichever branch/commit produced
it) only needs to emit its three standard artifacts --

  1. a "long" summary table  (one row per rat x C x quantity, with point
     estimate + 95% CI)   -- e.g. glm_summary_long.csv
  2. a run-metadata JSON   (config + per-fit AIC/BIC/etc.)
                            -- e.g. glm_metadata.json
  3. the raw bootstrap replicates (one row per rat x C x quantity x
     boot_idx)              -- e.g. glm_bootstrap_draws.parquet

-- and this script will rebuild the correction and every derived number
from scratch, every time, so results-section values never have to be
hand-copied out of a notebook again.

Usage
-----
    python glm_bhfdr_correction.py \\
        --summary-csv   glm_summary_long.csv \\
        --metadata-json glm_metadata.json \\
        --bootstrap-parquet glm_bootstrap_draws.parquet \\
        --out-dir results_section_4

Optional:
    --c-value unpenalized        # which regularization fit to correct (default: unpenalized)
    --subset pooled              # which trial subset to correct (default: pooled)
    --alpha 0.05                 # BH-FDR level (default: 0.05)
    --psychometric-resolved 11   # rat id(s) resolved by the Section-3 psychometric
                                  # analysis at the *same* alpha/family size, for the
                                  # cross-method comparison table. Repeatable, e.g.
                                  # --psychometric-resolved 11 --psychometric-resolved 4

Outputs (written to --out-dir)
-------------------------------
    bh_fdr_results.csv        one row per (rat, contrast): frac_pos, ci_low/high,
                               p_boot, rank, q_bh, significant
    correction_summary.csv    per-contrast resolved-count before/after correction
    per_rat_summary.csv       per-rat resolved-contrast count + psychometric flag
    model_comparison.csv      per-rat AIC/BIC deltas + pipeline 'profile' label
    results_section_summary.txt   plain-text block of the numbers, phrased the
                               way the Q1a paragraph uses them, ready to paste
                               into the draft and update by hand.

Dependencies: pandas, numpy, pyarrow (for the parquet file).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# The four contrasts that make up the pooled "primary family" (Section 3's
# family: ΔμR, ΔμU, D, P). If your GLM pipeline renames these columns, only
# this dict needs to change -- nothing downstream references the raw names.
# ---------------------------------------------------------------------------
PRIMARY_CONTRASTS = {
    "delta_mu_rewarded": "dmuR",
    "delta_mu_unrewarded": "dmuU",
    "D_glm": "D",
    "P_glm": "P",
}


# ---------------------------------------------------------------------------
# Bootstrap p-value
# ---------------------------------------------------------------------------
def bootstrap_p_value(draws: np.ndarray) -> tuple[float, float]:
    """Two-sided percentile-bootstrap p-value for H0: parameter == 0.

    p = 2 * min(P(draw > 0), P(draw < 0)), floored at 2/(n_boot+1) when all
    draws land on one side (the smallest p a finite bootstrap can resolve --
    reporting a bare 0 would overstate precision).

    This matches the convention the GLM pipeline already uses internally
    for `beta_angle_boot_frac_positive`.

    Returns (p_value, frac_positive).
    """
    n = len(draws)
    frac_pos = float(np.mean(draws > 0))
    p_raw = 2 * min(frac_pos, 1 - frac_pos)
    p_floor = 2.0 / (n + 1)
    p = p_floor if p_raw == 0 else p_raw
    return min(p, 1.0), frac_pos


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR (standard step-up procedure; matches
# statsmodels.stats.multitest.multipletests(method="fdr_bh"))
# ---------------------------------------------------------------------------
def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    """Return BH-adjusted q-values, same order as the input p-values."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranks = np.arange(1, m + 1)
    p_sorted = p[order]

    p_adj_raw = p_sorted * m / ranks
    # enforce monotonicity: q(i) = min(p_adj_raw(i), p_adj_raw(i+1), ..., p_adj_raw(m))
    q_sorted = np.minimum.accumulate(p_adj_raw[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)

    q = np.empty(m)
    q[order] = q_sorted
    return q


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_inputs(summary_csv: Path, metadata_json: Path, bootstrap_parquet: Path):
    summary = pd.read_csv(summary_csv)
    with open(metadata_json) as f:
        metadata = json.load(f)
    per_fit = pd.DataFrame(metadata["per_fit"])
    draws = pd.read_parquet(bootstrap_parquet)
    return summary, metadata, per_fit, draws


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------
def compute_bh_fdr_table(
    draws: pd.DataFrame,
    c_value: str,
    subset: str,
    alpha: float,
) -> pd.DataFrame:
    """One row per (rat, contrast) with the bootstrap p-value and BH q-value."""
    mask = (draws["C"] == c_value) & draws["quantity"].isin(PRIMARY_CONTRASTS)
    if "subset" in draws.columns:
        mask &= draws["subset"] == subset
    sub = draws.loc[mask]

    if sub.empty:
        raise ValueError(
            f"No bootstrap draws found for C={c_value!r}, subset={subset!r}, "
            f"quantities in {list(PRIMARY_CONTRASTS)}. Check --c-value/--subset "
            f"against the actual values in your bootstrap parquet."
        )

    rows = []
    for (rat_id, quantity), g in sub.groupby(["rat_id", "quantity"]):
        vals = g["value"].to_numpy()
        p_boot, frac_pos = bootstrap_p_value(vals)
        ci_low, ci_high = np.percentile(vals, [2.5, 97.5])
        rows.append(
            dict(
                rat_id=int(rat_id),
                quantity=quantity,
                contrast=PRIMARY_CONTRASTS[quantity],
                n_boot=len(vals),
                frac_pos=frac_pos,
                ci_low=float(ci_low),
                ci_high=float(ci_high),
                ci_excludes_zero=bool((ci_low > 0) or (ci_high < 0)),
                p_boot=p_boot,
            )
        )
    table = pd.DataFrame(rows)

    table["q_bh"] = benjamini_hochberg(table["p_boot"].to_numpy())
    table["significant"] = table["q_bh"] < alpha
    table = table.sort_values(["p_boot", "rat_id", "quantity"]).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    return table


def build_correction_summary(bh_table: pd.DataFrame) -> pd.DataFrame:
    """Per-contrast resolved count, raw CI vs. BH-corrected."""
    g = bh_table.groupby("contrast")
    summary = pd.DataFrame(
        {
            "resolved_raw_CI": g["ci_excludes_zero"].sum(),
            "resolved_BH_corrected": g["significant"].sum(),
        }
    )
    summary["n_rats"] = g.size()
    summary["difference"] = summary["resolved_raw_CI"] - summary["resolved_BH_corrected"]
    # keep a stable, familiar row order if present
    order = [c for c in PRIMARY_CONTRASTS.values() if c in summary.index]
    return summary.loc[order].reset_index()


def build_per_rat_summary(bh_table: pd.DataFrame, psychometric_resolved: set[int]) -> pd.DataFrame:
    per_rat = (
        bh_table.groupby("rat_id")["significant"]
        .sum()
        .rename("n_glm_contrasts_resolved")
        .reset_index()
    )
    per_rat["resolved_by_psychometric"] = per_rat["rat_id"].isin(psychometric_resolved)
    return per_rat.sort_values("rat_id").reset_index(drop=True)


def build_model_comparison(per_fit: pd.DataFrame, c_value: str, subset: str) -> pd.DataFrame:
    """Per-rat AIC/BIC deltas (only populated at the fit where the pipeline
    computed them -- typically the unpenalized fit)."""
    mask = per_fit["C"] == c_value
    if "subset" in per_fit.columns:
        mask &= per_fit["subset"] == subset
    cols = [c for c in per_fit.columns if c.startswith("delta_") or c in (
        "rat_id", "n_trials", "n_sessions", "profile"
    )]
    return per_fit.loc[mask, cols].sort_values("rat_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Results-section text block
# ---------------------------------------------------------------------------
def render_results_text(
    bh_table: pd.DataFrame,
    correction_summary: pd.DataFrame,
    per_rat: pd.DataFrame,
    alpha: float,
) -> str:
    n_rats = per_rat["rat_id"].nunique()
    lines = []
    lines.append(f"Pooled GLM, {n_rats} rats, BH-FDR corrected jointly across "
                  f"{len(bh_table)} tests ({n_rats} rats x {len(PRIMARY_CONTRASTS)} "
                  f"contrasts), alpha={alpha}.")
    lines.append("")
    lines.append("Resolved counts after correction:")
    for _, row in correction_summary.iterrows():
        lines.append(
            f"  {row['contrast']}: {int(row['resolved_BH_corrected'])} of {int(row['n_rats'])} "
            f"(raw CI, uncorrected: {int(row['resolved_raw_CI'])}; "
            f"difference: {int(row['difference'])})"
        )
    lines.append("")
    n_any = int((per_rat["n_glm_contrasts_resolved"] >= 1).sum())
    lines.append(f"Rats resolved on >=1 contrast: {n_any} of {n_rats}.")

    non_psych = per_rat.loc[~per_rat["resolved_by_psychometric"]]
    n_non_psych = len(non_psych)
    n_non_psych_any = int((non_psych["n_glm_contrasts_resolved"] >= 1).sum())
    n_non_psych_2plus = int((non_psych["n_glm_contrasts_resolved"] >= 2).sum())
    lines.append(
        f"Of the {n_non_psych} rats not resolved by the psychometric analysis: "
        f"{n_non_psych_any} resolved by the GLM on >=1 contrast, "
        f"{n_non_psych_2plus} on >=2."
    )

    psych = per_rat.loc[per_rat["resolved_by_psychometric"]]
    for _, row in psych.iterrows():
        lines.append(
            f"Rat {int(row['rat_id'])} (resolved by both methods): "
            f"{int(row['n_glm_contrasts_resolved'])} of {len(PRIMARY_CONTRASTS)} "
            f"GLM contrasts resolved."
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument("--bootstrap-parquet", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("section4_q1a_outputs"))
    parser.add_argument("--c-value", type=str, default="unpenalized",
                         help="Which regularization fit to correct (default: unpenalized).")
    parser.add_argument("--subset", type=str, default="pooled",
                         help="Which trial subset to correct (default: pooled).")
    parser.add_argument("--alpha", type=float, default=0.05, help="BH-FDR level.")
    parser.add_argument(
        "--psychometric-resolved", type=int, action="append", default=None,
        help="Rat id resolved by the Section-3 psychometric analysis at matched "
             "correction. Repeat the flag for multiple rats. Default: [11] "
             "(matches the current Section 3 draft) -- override if that changes.",
    )
    args = parser.parse_args(argv)

    psychometric_resolved = set(args.psychometric_resolved) if args.psychometric_resolved else {11}

    summary, metadata, per_fit, draws = load_inputs(
        args.summary_csv, args.metadata_json, args.bootstrap_parquet
    )

    bh_table = compute_bh_fdr_table(draws, args.c_value, args.subset, args.alpha)
    correction_summary = build_correction_summary(bh_table)
    per_rat = build_per_rat_summary(bh_table, psychometric_resolved)
    model_comparison = build_model_comparison(per_fit, args.c_value, args.subset)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bh_table.to_csv(args.out_dir / "bh_fdr_results.csv", index=False)
    correction_summary.to_csv(args.out_dir / "correction_summary.csv", index=False)
    per_rat.to_csv(args.out_dir / "per_rat_summary.csv", index=False)
    model_comparison.to_csv(args.out_dir / "model_comparison.csv", index=False)

    text = render_results_text(bh_table, correction_summary, per_rat, args.alpha)
    (args.out_dir / "results_section_summary.txt").write_text(text + "\n")

    print(text)
    print(f"\nWrote CSVs + results_section_summary.txt to {args.out_dir}/")


if __name__ == "__main__":
    main()