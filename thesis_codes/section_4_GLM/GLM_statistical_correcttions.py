"""Canonical pooled-GLM inference for the thesis.

Reads the pooled GLM full-data summary and saved bootstrap draws, computes
two-sided equal-tail percentile-bootstrap p-values, and applies one
Benjamini-Hochberg correction across the declared 48-test family:

    12 rats x {delta_mu_rewarded, delta_mu_unrewarded, D_glm, P_glm}

The script writes both a complete canonical table and the legacy-compatible
``glm_contrast_significance_unpenalized.csv`` expected by the existing
psychometric-vs-GLM table script. Figures and tables should read these outputs
and should not recompute p- or q-values.

Example
-------
python canonical_pooled_glm_inference.py \
    --glm-summary glm_summary_long.csv \
    --bootstrap-draws glm_bootstrap_draws.parquet \
    --out-dir pooled_glm_inference
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# PATH CONFIGURATION — EDIT THESE PATHS
# =============================================================================

GLM_RESULTS_DIR = Path(
    r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy"
    r"\thesis_codes\section_4_GLM\glm_results_thesis_filtered"
)

GLM_SUMMARY_PATH = GLM_RESULTS_DIR / "glm_summary_long.csv"

GLM_BOOTSTRAP_PATH = GLM_RESULTS_DIR / "glm_bootstrap_draws.parquet"

# Canonical inference outputs will be written here.
OUTPUT_DIR = GLM_RESULTS_DIR / "canonical_inference"


PRIMARY_CONTRASTS = (
    "delta_mu_rewarded",
    "delta_mu_unrewarded",
    "D_glm",
    "P_glm",
)
EXPECTED_RATS = 12
EXPECTED_FAMILY_SIZE = EXPECTED_RATS * len(PRIMARY_CONTRASTS)
REFERENCE_C = "unpenalized"
REFERENCE_SUBSET = "pooled"
ALPHA = 0.05




def read_bootstrap_draws(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if suffixes.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported bootstrap file type: {path}")


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def equal_tail_bootstrap_p(draws: np.ndarray, null: float = 0.0) -> dict:
    """Two-sided equal-tail percentile-bootstrap p-value.

    Uses strict tails so an exact-null draw is not counted in both tails.
    The plus-one correction prevents a simulated p-value of exactly zero.
    """
    values = np.asarray(draws, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite bootstrap draws")

    n_negative = int(np.count_nonzero(values < null))
    n_positive = int(np.count_nonzero(values > null))
    n_equal = int(np.count_nonzero(values == null))
    p_value = min(
        1.0,
        2.0 * (min(n_negative, n_positive) + 1.0) / (values.size + 1.0),
    )
    return {
        "p_boot": float(p_value),
        "n_boot": int(values.size),
        "n_negative": n_negative,
        "n_positive": n_positive,
        "n_equal_zero": n_equal,
        "frac_negative": n_negative / values.size,
        "frac_positive": n_positive / values.size,
    }


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values in the original input order."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or not np.all(np.isfinite(p)):
        raise ValueError("BH input must be a finite one-dimensional array")
    
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("BH input p-values must lie in [0, 1]")

    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    adjusted_ranked = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def select_primary_summary(summary: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        summary,
        {
            "rat_id", "subset", "C", "quantity", "scale", "value",
            "ci_low", "ci_high",
        },
        "GLM summary",
    )
    selected = summary[
        (summary["subset"] == REFERENCE_SUBSET)
        & (summary["C"] == REFERENCE_C)
        & (summary["scale"] == "degrees")
        & (summary["quantity"].isin(PRIMARY_CONTRASTS))
    ].copy()
    return selected


def select_primary_draws(draws: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        draws,
        {"rat_id", "subset", "C", "quantity", "boot_idx", "value"},
        "Bootstrap draws",
    )
    selected = draws[
        (draws["subset"] == REFERENCE_SUBSET)
        & (draws["C"] == REFERENCE_C)
        & (draws["quantity"].isin(PRIMARY_CONTRASTS))
    ].copy()
    return selected


def validate_family(summary: pd.DataFrame, draws: pd.DataFrame) -> list[int]:
    if summary.empty or draws.empty:
        raise ValueError("No pooled unpenalized primary contrasts were found")

    if summary[["rat_id", "quantity"]].duplicated().any():
        duplicated = summary.loc[
            summary[["rat_id", "quantity"]].duplicated(False),
            ["rat_id", "quantity"],
        ]
        raise ValueError(f"Duplicate summary rows:\n{duplicated}")

    rats = sorted(int(r) for r in summary["rat_id"].unique())
    if len(rats) != EXPECTED_RATS:
        raise ValueError(f"Expected {EXPECTED_RATS} rats, found {len(rats)}: {rats}")
    expected = pd.MultiIndex.from_product(
        [rats, PRIMARY_CONTRASTS], names=["rat_id", "quantity"]
    )
    summary_index = pd.MultiIndex.from_frame(summary[["rat_id", "quantity"]])
    draw_index = pd.MultiIndex.from_frame(
        draws[["rat_id", "quantity"]].drop_duplicates()
    )
    missing_summary = expected.difference(summary_index)
    missing_draws = expected.difference(draw_index)
    extra_summary = summary_index.difference(expected)
    extra_draws = draw_index.difference(expected)
    if len(missing_summary) or len(missing_draws) or len(extra_summary) or len(extra_draws):
        raise ValueError(
            "Primary family mismatch. "
            f"Missing summary={list(missing_summary)}; "
            f"missing draws={list(missing_draws)}; "
            f"extra summary={list(extra_summary)}; "
            f"extra draws={list(extra_draws)}"
        )
    if len(summary) != EXPECTED_FAMILY_SIZE:
        raise ValueError(
            f"Expected exactly {EXPECTED_FAMILY_SIZE} summary rows, found {len(summary)}"
        )

    draw_counts = draws.groupby(["rat_id", "quantity"])["boot_idx"].nunique()
    if (draw_counts < 100).any():
        raise ValueError(f"Too few bootstrap draws in cells:\n{draw_counts[draw_counts < 100]}")
    return rats

def build_inference(summary: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    rats = validate_family(summary, draws)
    summary_lookup = summary.set_index(["rat_id", "quantity"])
    rows = []

    for rat_id in rats:
        for quantity in PRIMARY_CONTRASTS:
            summary_row = summary_lookup.loc[(rat_id, quantity)]
            values = draws.loc[
                (draws["rat_id"] == rat_id) & (draws["quantity"] == quantity),
                "value",
            ].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                raise ValueError(f"No finite draws for rat {rat_id}, {quantity}")

            ci_low, ci_high = np.percentile(values, [2.5, 97.5])
            tail = equal_tail_bootstrap_p(values)
            rows.append({
                "rat_id": int(rat_id),
                "contrast": quantity,
                "estimate_full_data": float(summary_row["value"]),
                "bootstrap_mean": float(np.mean(values)),
                "bootstrap_median": float(np.median(values)),
                "bootstrap_sd": float(np.std(values, ddof=1)),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
                **tail,
                "summary_ci_low": float(summary_row["ci_low"]),
                "summary_ci_high": float(summary_row["ci_high"]),
            })

    result = pd.DataFrame(rows)
    if len(result) != EXPECTED_FAMILY_SIZE:
        raise AssertionError("Internal error: incomplete primary family")

    result["q_bh"] = benjamini_hochberg(result["p_boot"].to_numpy())
    result["reject_bh_05"] = result["q_bh"] <= ALPHA
    result["family"] = "pooled_glm_primary_48"
    result["family_size"] = EXPECTED_FAMILY_SIZE
    result["C"] = REFERENCE_C
    result["subset"] = REFERENCE_SUBSET

    result["ci_matches_summary"] = (
        # Bootstrap values were stored as float32 in the parquet file, while
        # the summary CSV may retain float64 percentiles. Allow only the tiny
        # rounding discrepancy implied by that storage conversion.
        np.isclose(result["ci_low"], result["summary_ci_low"], atol=1e-4, rtol=1e-6)
        & np.isclose(result["ci_high"], result["summary_ci_high"], atol=1e-4, rtol=1e-6)
    )
    if not result["ci_matches_summary"].all():
        mismatches = result.loc[
            ~result["ci_matches_summary"],
            ["rat_id", "contrast", "ci_low", "ci_high", "summary_ci_low", "summary_ci_high"],
        ]
        raise ValueError(
            "Recomputed bootstrap CIs do not match glm_summary_long.csv. "
            "Check that both inputs came from the same run.\n"
            f"{mismatches.to_string(index=False)}"
        )
    return result


def main() -> None:
    print("Canonical pooled-GLM inference")
    print(f"  GLM summary:     {GLM_SUMMARY_PATH}")
    print(f"  Bootstrap draws: {GLM_BOOTSTRAP_PATH}")
    print(f"  Output folder:   {OUTPUT_DIR}")
    print()

    if not GLM_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"GLM summary file not found:\n{GLM_SUMMARY_PATH}"
        )

    if not GLM_BOOTSTRAP_PATH.exists():
        raise FileNotFoundError(
            f"GLM bootstrap file not found:\n{GLM_BOOTSTRAP_PATH}"
        )

    summary = pd.read_csv(GLM_SUMMARY_PATH)
    draws = read_bootstrap_draws(GLM_BOOTSTRAP_PATH)

    primary_summary = select_primary_summary(summary)
    primary_draws = select_primary_draws(draws)

    inference = build_inference(
        primary_summary,
        primary_draws,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    canonical_path = (
        OUTPUT_DIR / "pooled_glm_primary_inference_48.csv"
    )

    legacy_path = (
        OUTPUT_DIR /
        "glm_contrast_significance_unpenalized.csv"
    )

    manifest_path = (
        OUTPUT_DIR /
        "pooled_glm_primary_inference_manifest.json"
    )

    inference.to_csv(
        canonical_path,
        index=False,
        float_format="%.10g",
    )

    # Backward-compatible file for the existing table scripts.
    legacy_columns = [
        "rat_id",
        "contrast",
        "estimate_full_data",
        "bootstrap_mean",
        "bootstrap_median",
        "ci_low",
        "ci_high",
        "ci_excludes_zero",
        "n_boot",
        "n_negative",
        "n_positive",
        "n_equal_zero",
        "p_boot",
        "q_bh",
        "reject_bh_05",
    ]

    inference[legacy_columns].to_csv(
        legacy_path,
        index=False,
        float_format="%.10g",
    )

    manifest = {
        "family": "pooled_glm_primary_48",
        "family_size": int(len(inference)),
        "contrasts": list(PRIMARY_CONTRASTS),
        "subset": REFERENCE_SUBSET,
        "C": REFERENCE_C,
        "alpha": ALPHA,
        "p_value_method": (
            "Two-sided equal-tail percentile-bootstrap "
            "p-value with plus-one correction"
        ),
        "multiple_comparison_method": (
            "Benjamini-Hochberg across all 48 tests"
        ),
        "inputs": {
            "glm_summary": str(GLM_SUMMARY_PATH),
            "bootstrap_draws": str(GLM_BOOTSTRAP_PATH),
        },
        "outputs": {
            "canonical": str(canonical_path),
            "legacy_compatible": str(legacy_path),
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    significant = inference[inference["reject_bh_05"]]

    print(f"Saved: {canonical_path}")
    print(f"Saved: {legacy_path}")
    print(f"Saved: {manifest_path}")
    print(
        f"BH-FDR significant contrasts: "
        f"{len(significant)}/{len(inference)}"
    )

    if not significant.empty:
        print()
        print(
            significant[
                [
                    "rat_id",
                    "contrast",
                    "estimate_full_data",
                    "p_boot",
                    "q_bh",
                ]
            ].to_string(index=False)
        )

if __name__ == "__main__":
    main()