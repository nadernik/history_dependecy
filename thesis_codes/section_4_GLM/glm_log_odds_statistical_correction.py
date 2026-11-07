"""Canonical pooled-GLM inference for log-odds history coefficients.

The correction scheme contains two explicitly separate BH-FDR families:

1. choice_outcome_36:
   12 rats x {beta_reward_n-1, beta_success_n-1, beta_failure_n-1}
2. stimulus_history_12:
   12 rats x {beta_angle_n-1}

The intercept and current-trial beta_angle are descriptive/reference model
parameters and are deliberately excluded from inferential correction.

Outputs in --out-dir:
  pooled_glm_log_odds_inference_48.csv
  glm_coefficient_significance_unpenalized.csv
  pooled_glm_log_odds_inference_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RESULTS_DIR = Path(
    r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy"
    r"\thesis_codes\section_4_GLM\glm_results_thesis_filtered"
)

CHOICE_OUTCOME_COEFFICIENTS = (
    "beta_reward_n-1",
    "beta_success_n-1",
    "beta_failure_n-1",
)
STIMULUS_HISTORY_COEFFICIENTS = ("beta_angle_n-1",)
INFERENTIAL_COEFFICIENTS = (
    *STIMULUS_HISTORY_COEFFICIENTS,
    *CHOICE_OUTCOME_COEFFICIENTS,
)

EXPECTED_RATS = 12
CHOICE_OUTCOME_FAMILY_SIZE = EXPECTED_RATS * len(CHOICE_OUTCOME_COEFFICIENTS)
STIMULUS_HISTORY_FAMILY_SIZE = EXPECTED_RATS * len(STIMULUS_HISTORY_COEFFICIENTS)
EXPECTED_TOTAL_ROWS = CHOICE_OUTCOME_FAMILY_SIZE + STIMULUS_HISTORY_FAMILY_SIZE
REFERENCE_C = "unpenalized"
REFERENCE_SUBSET = "pooled"
REFERENCE_SCALE = "log_odds"
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
    """Two-sided equal-tail bootstrap p-value with plus-one correction."""
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
    """BH-adjusted p-values returned in the original input order."""
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


def select_inputs(summary: pd.DataFrame, draws: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_columns(
        summary,
        {"rat_id", "subset", "C", "quantity", "scale", "value", "ci_low", "ci_high"},
        "GLM summary",
    )
    require_columns(
        draws,
        {"rat_id", "subset", "C", "quantity", "boot_idx", "value"},
        "Bootstrap draws",
    )

    selected_summary = summary.loc[
        (summary["subset"] == REFERENCE_SUBSET)
        & (summary["C"] == REFERENCE_C)
        & (summary["scale"] == REFERENCE_SCALE)
        & (summary["quantity"].isin(INFERENTIAL_COEFFICIENTS))
    ].copy()

    selected_draws = draws.loc[
        (draws["subset"] == REFERENCE_SUBSET)
        & (draws["C"] == REFERENCE_C)
        & (draws["quantity"].isin(INFERENTIAL_COEFFICIENTS))
    ].copy()
    return selected_summary, selected_draws


def validate_inputs(summary: pd.DataFrame, draws: pd.DataFrame) -> list[int]:
    if summary.empty or draws.empty:
        raise ValueError("No pooled unpenalized log-odds history coefficients were found")

    duplicate_mask = summary[["rat_id", "quantity"]].duplicated(keep=False)
    if duplicate_mask.any():
        duplicated = summary.loc[duplicate_mask, ["rat_id", "quantity"]]
        raise ValueError(f"Duplicate summary rows:\n{duplicated.to_string(index=False)}")

    rats = sorted(int(rat) for rat in summary["rat_id"].unique())
    if len(rats) != EXPECTED_RATS:
        raise ValueError(f"Expected {EXPECTED_RATS} rats, found {len(rats)}: {rats}")

    expected = pd.MultiIndex.from_product(
        [rats, INFERENTIAL_COEFFICIENTS], names=["rat_id", "quantity"]
    )
    summary_index = pd.MultiIndex.from_frame(summary[["rat_id", "quantity"]])
    draws_index = pd.MultiIndex.from_frame(
        draws[["rat_id", "quantity"]].drop_duplicates()
    )

    missing_summary = expected.difference(summary_index)
    missing_draws = expected.difference(draws_index)
    extra_summary = summary_index.difference(expected)
    extra_draws = draws_index.difference(expected)
    if any(map(len, [missing_summary, missing_draws, extra_summary, extra_draws])):
        raise ValueError(
            "Inferential coefficient grid mismatch. "
            f"Missing summary={list(missing_summary)}; missing draws={list(missing_draws)}; "
            f"extra summary={list(extra_summary)}; extra draws={list(extra_draws)}"
        )

    if len(summary) != EXPECTED_TOTAL_ROWS:
        raise ValueError(
            f"Expected exactly {EXPECTED_TOTAL_ROWS} summary rows, found {len(summary)}"
        )

    draw_counts = draws.groupby(["rat_id", "quantity"])["boot_idx"].nunique()
    if (draw_counts < 100).any():
        raise ValueError(f"Too few bootstrap draws in cells:\n{draw_counts[draw_counts < 100]}")
    return rats


def build_inference(summary: pd.DataFrame, draws: pd.DataFrame) -> pd.DataFrame:
    rats = validate_inputs(summary, draws)
    summary_lookup = summary.set_index(["rat_id", "quantity"])
    rows = []

    for rat_id in rats:
        for coefficient in INFERENTIAL_COEFFICIENTS:
            summary_row = summary_lookup.loc[(rat_id, coefficient)]
            values = draws.loc[
                (draws["rat_id"] == rat_id) & (draws["quantity"] == coefficient),
                "value",
            ].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                raise ValueError(f"No finite draws for rat {rat_id}, {coefficient}")

            ci_low, ci_high = np.percentile(values, [2.5, 97.5])
            family = (
                "stimulus_history_12"
                if coefficient in STIMULUS_HISTORY_COEFFICIENTS
                else "choice_outcome_36"
            )
            family_size = (
                STIMULUS_HISTORY_FAMILY_SIZE
                if family == "stimulus_history_12"
                else CHOICE_OUTCOME_FAMILY_SIZE
            )
            rows.append(
                {
                    "rat_id": rat_id,
                    "coefficient": coefficient,
                    "estimate_full_data": float(summary_row["value"]),
                    "bootstrap_mean": float(np.mean(values)),
                    "bootstrap_median": float(np.median(values)),
                    "bootstrap_sd": float(np.std(values, ddof=1)),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "ci_excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
                    **equal_tail_bootstrap_p(values),
                    "summary_ci_low": float(summary_row["ci_low"]),
                    "summary_ci_high": float(summary_row["ci_high"]),
                    "family": family,
                    "family_size": family_size,
                }
            )

    result = pd.DataFrame(rows)
    if len(result) != EXPECTED_TOTAL_ROWS:
        raise AssertionError("Internal error: incomplete inference table")

    result["q_bh"] = np.nan
    for family_name, expected_size in (
        ("choice_outcome_36", CHOICE_OUTCOME_FAMILY_SIZE),
        ("stimulus_history_12", STIMULUS_HISTORY_FAMILY_SIZE),
    ):
        mask = result["family"] == family_name
        if int(mask.sum()) != expected_size:
            raise AssertionError(
                f"{family_name}: expected {expected_size} rows, found {int(mask.sum())}"
            )
        result.loc[mask, "q_bh"] = benjamini_hochberg(
            result.loc[mask, "p_boot"].to_numpy()
        )

    result["significant_bh"] = result["q_bh"] <= ALPHA
    result["reject_bh_05"] = result["significant_bh"]
    result["C"] = REFERENCE_C
    result["subset"] = REFERENCE_SUBSET
    result["scale"] = REFERENCE_SCALE
    result["ci_matches_summary"] = (
        np.isclose(result["ci_low"], result["summary_ci_low"], atol=1e-4, rtol=1e-6)
        & np.isclose(result["ci_high"], result["summary_ci_high"], atol=1e-4, rtol=1e-6)
    )

    if not result["ci_matches_summary"].all():
        mismatches = result.loc[
            ~result["ci_matches_summary"],
            [
                "rat_id", "coefficient", "ci_low", "ci_high",
                "summary_ci_low", "summary_ci_high",
            ],
        ]
        raise ValueError(
            "Recomputed bootstrap CIs do not match glm_summary_long.csv. "
            "Check that both inputs came from the same run.\n"
            f"{mismatches.to_string(index=False)}"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--glm-summary",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "glm_summary_long.csv",
    )
    parser.add_argument(
        "--bootstrap-draws",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "glm_bootstrap_draws.parquet",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "canonical_inference",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Canonical pooled log-odds inference")
    print(f"  GLM summary:     {args.glm_summary}")
    print(f"  Bootstrap draws: {args.bootstrap_draws}")
    print(f"  Output folder:   {args.out_dir}")

    if not args.glm_summary.exists():
        raise FileNotFoundError(f"GLM summary file not found:\n{args.glm_summary}")
    if not args.bootstrap_draws.exists():
        raise FileNotFoundError(f"GLM bootstrap file not found:\n{args.bootstrap_draws}")

    summary = pd.read_csv(args.glm_summary)
    draws = read_bootstrap_draws(args.bootstrap_draws)
    selected_summary, selected_draws = select_inputs(summary, draws)
    inference = build_inference(selected_summary, selected_draws)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = args.out_dir / "pooled_glm_log_odds_inference_48.csv"
    legacy_path = args.out_dir / "glm_coefficient_significance_unpenalized.csv"
    manifest_path = args.out_dir / "pooled_glm_log_odds_inference_manifest.json"

    inference.to_csv(canonical_path, index=False, float_format="%.10g")

    compatibility_columns = [
        "rat_id", "coefficient", "estimate_full_data", "bootstrap_mean",
        "bootstrap_median", "ci_low", "ci_high", "ci_excludes_zero",
        "n_boot", "n_negative", "n_positive", "n_equal_zero", "p_boot",
        "q_bh", "significant_bh", "reject_bh_05", "family", "family_size",
    ]
    inference[compatibility_columns].to_csv(
        legacy_path, index=False, float_format="%.10g"
    )

    manifest = {
        "analysis": "pooled_glm_log_odds_history_inference",
        "total_inferential_rows": int(len(inference)),
        "families": {
            "choice_outcome_36": {
                "size": CHOICE_OUTCOME_FAMILY_SIZE,
                "coefficients": list(CHOICE_OUTCOME_COEFFICIENTS),
                "correction": "Benjamini-Hochberg within these 36 tests",
            },
            "stimulus_history_12": {
                "size": STIMULUS_HISTORY_FAMILY_SIZE,
                "coefficients": list(STIMULUS_HISTORY_COEFFICIENTS),
                "correction": "Benjamini-Hochberg within these 12 tests",
            },
        },
        "excluded_from_inference": ["intercept", "beta_angle"],
        "subset": REFERENCE_SUBSET,
        "C": REFERENCE_C,
        "scale": REFERENCE_SCALE,
        "alpha": ALPHA,
        "p_value_method": "Two-sided equal-tail bootstrap p-value with plus-one correction",
        "inputs": {
            "glm_summary": str(args.glm_summary),
            "bootstrap_draws": str(args.bootstrap_draws),
        },
        "outputs": {
            "canonical": str(canonical_path),
            "table_compatible": str(legacy_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved: {canonical_path}")
    print(f"Saved: {legacy_path}")
    print(f"Saved: {manifest_path}")
    for family_name in ("choice_outcome_36", "stimulus_history_12"):
        family = inference[inference["family"] == family_name]
        print(
            f"{family_name}: {int(family['significant_bh'].sum())}/{len(family)} "
            "BH-FDR significant"
        )


if __name__ == "__main__":
    main()
