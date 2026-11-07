"""Check multicollinearity for pooled and modality-interaction GLMs.

This companion script imports the preprocessing and interaction design-matrix
functions from the Section 6 analysis script. For every rat, it checks both:

1. pooled: the five base predictors without modality terms;
2. modality_interaction: the five base predictors, two modality dummies, and
   all predictor-by-modality interactions.

Outputs
-------
vif_by_rat.csv
    One row per model, rat, and design-matrix column.

vif_matrix_diagnostics.csv
    One row per model and rat with matrix rank, condition number, maximum VIF,
    and counts above commonly used descriptive thresholds (5 and 10).

Notes
-----
VIF is a property of the predictor matrix, not of the binary response, the
regularization strength C, or the bootstrap. Therefore it is computed once
per model and rat, rather than once per C or bootstrap sample.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG - EDIT THESE PATHS IF NEEDED
# =============================================================================

PROJECT_ROOT = Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy")

# Set this to the filename/path of the interaction-GLM script supplied with
# this VIF checker. An absolute path is also accepted.
ANALYSIS_SCRIPT = PROJECT_ROOT / "thesis_codes" / "section_4_GLM" / "section_4_GLM_interaction_pooled_modality.py"

OUT_DIR = PROJECT_ROOT / "thesis_codes" / "section_4_GLM" / "vif_diagnostics"
MIN_TRIALS = 200


def load_analysis_module(script_path: Path):
    """Load the analysis script so its exact preprocessing/design code is used."""
    script_path = script_path.resolve()
    if not script_path.exists():
        raise FileNotFoundError(
            f"Analysis script not found: {script_path}\n"
            "Edit ANALYSIS_SCRIPT at the top of this file."
        )

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    spec = importlib.util.spec_from_file_location("interaction_glm_analysis", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load analysis script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_pooled_design_matrix(
    rat_data: pd.DataFrame,
    base_features: list[str],
    continuous_regressors: set[str],
) -> Tuple[np.ndarray, list[str]]:
    """Build the five-predictor pooled design matrix.

    Continuous predictors are divided by their SD, matching the interaction
    analysis's ``StandardScaler(with_mean=False)`` transformation. Scaling or
    centering does not change VIF, but matching it makes the diagnostic matrix
    directly comparable to the fitted matrix.
    """
    columns = []
    for name in base_features:
        values = rat_data[name].to_numpy(dtype=float)
        if name in continuous_regressors:
            scale = float(np.std(values, ddof=0))
            if scale > 0:
                values = values / scale
        columns.append(values.reshape(-1, 1))

    return np.column_stack(columns), list(base_features)


def vif_for_matrix(
    X: np.ndarray,
    column_names: list[str],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute VIF for every column using auxiliary OLS regressions.

    Each predictor is regressed on all remaining predictors plus an intercept:
        VIF_j = 1 / (1 - R_j^2).

    A constant predictor or perfect linear dependence is assigned infinite VIF.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(column_names):
        raise ValueError("X dimensions do not match column_names.")
    if not np.isfinite(X).all():
        raise ValueError("The design matrix contains NaN or infinite values.")

    n, p = X.shape
    rows = []

    for j, name in enumerate(column_names):
        y_j = X[:, j]
        variance = float(np.var(y_j))

        if np.isclose(variance, 0.0):
            r_squared = np.nan
            vif = np.inf
            status = "zero_variance"
        else:
            others = np.delete(X, j, axis=1)
            auxiliary_X = np.column_stack([np.ones(n), others])
            fitted = auxiliary_X @ np.linalg.lstsq(auxiliary_X, y_j, rcond=None)[0]

            ss_res = float(np.sum((y_j - fitted) ** 2))
            ss_tot = float(np.sum((y_j - y_j.mean()) ** 2))
            r_squared = 1.0 - ss_res / ss_tot

            # Guard against tiny numerical excursions outside [0, 1].
            r_squared = float(np.clip(r_squared, 0.0, 1.0))
            vif = np.inf if (1.0 - r_squared) <= 1e-12 else 1.0 / (1.0 - r_squared)
            status = "perfect_collinearity" if not np.isfinite(vif) else "ok"

        rows.append({
            "predictor": name,
            "variance": variance,
            "auxiliary_r_squared": r_squared,
            "vif": vif,
            "tolerance": 0.0 if not np.isfinite(vif) else 1.0 / vif,
            "vif_gt_5": bool(vif > 5.0),
            "vif_gt_10": bool(vif > 10.0),
            "status": status,
        })

    # Condition number is computed after centering and scaling every
    # nonconstant predictor, so it is not inflated merely by measurement units.
    sd = X.std(axis=0, ddof=0)
    usable = sd > 1e-12
    Z = (X[:, usable] - X[:, usable].mean(axis=0)) / sd[usable]
    singular_values = np.linalg.svd(Z, compute_uv=False) if Z.size else np.array([])
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if len(singular_values) and singular_values[-1] > 1e-12
        else np.inf
    )

    vif_values = np.asarray([row["vif"] for row in rows], dtype=float)
    diagnostics = {
        "n_trials": int(n),
        "n_predictors": int(p),
        "matrix_rank": int(np.linalg.matrix_rank(X)),
        "full_column_rank": bool(np.linalg.matrix_rank(X) == p),
        "condition_number_scaled": condition_number,
        "max_vif": float(np.max(vif_values)),
        "n_vif_gt_5": int(np.sum(vif_values > 5.0)),
        "n_vif_gt_10": int(np.sum(vif_values > 10.0)),
        "n_infinite_vif": int(np.sum(~np.isfinite(vif_values))),
    }
    return pd.DataFrame(rows), diagnostics


def main() -> None:
    analysis = load_analysis_module(ANALYSIS_SCRIPT)

    from serial_dependence_analysis import SerialDependenceAnalyzer

    analyzer = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
    analyzer.load_and_preprocess_data()
    df = analyzer.create_lagged_features()
    df = analysis.filter_training_trials(df, **analysis.FILTER_KWARGS)
    df = analysis.prepare_history_predictors(df)

    required = [
        "rat", "date", "action", "mod", "action_n-1",
        *analysis.FEATURES_FULL,
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    vif_frames = []
    diagnostic_rows = []

    for rat_id in sorted(df["rat"].dropna().unique()):
        rat_df = df[df["rat"] == rat_id].dropna(subset=required[1:]).copy()
        if len(rat_df) < MIN_TRIALS:
            print(f"Rat {rat_id}: skipped ({len(rat_df)} trials)")
            continue

        X_pooled, pooled_names = build_pooled_design_matrix(
            rat_df,
            analysis.FEATURES_FULL,
            analysis.CONTINUOUS_REGRESSORS,
        )
        X_interaction, _, interaction_names, _ = (
            analysis.build_design_matrix_interaction(
                rat_df,
                analysis.FEATURES_FULL,
                exclude_blocks=None,
            )
        )

        design_matrices = {
            "pooled": (X_pooled, pooled_names),
            "modality_interaction": (X_interaction, interaction_names),
        }

        for model_name, (X, column_names) in design_matrices.items():
            vif_table, diagnostics = vif_for_matrix(X, column_names)
            vif_table.insert(0, "model", model_name)
            vif_table.insert(1, "rat_id", int(rat_id))
            vif_table.insert(2, "n_trials", len(rat_df))
            vif_frames.append(vif_table)

            diagnostics.update({
                "model": model_name,
                "rat_id": int(rat_id),
                "n_sessions": int(rat_df["date"].nunique()),
            })
            diagnostic_rows.append(diagnostics)

            print(
                f"Rat {rat_id}, {model_name}: "
                f"max VIF={diagnostics['max_vif']:.3g}, "
                f">5: {diagnostics['n_vif_gt_5']}, "
                f">10: {diagnostics['n_vif_gt_10']}, "
                f"rank={diagnostics['matrix_rank']}/{diagnostics['n_predictors']}"
            )

    if not vif_frames:
        raise RuntimeError("No rats met MIN_TRIALS; no VIF results were produced.")

    vif_all = pd.concat(vif_frames, ignore_index=True)
    diagnostics_all = pd.DataFrame(diagnostic_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vif_path = OUT_DIR / "vif_by_rat.csv"
    diagnostics_path = OUT_DIR / "vif_matrix_diagnostics.csv"
    vif_all.to_csv(vif_path, index=False)
    diagnostics_all.to_csv(diagnostics_path, index=False)

    print(f"\nSaved: {vif_path}")
    print(f"Saved: {diagnostics_path}")
    print("\nInterpretation: VIF 1 means no linear redundancy; values above 5 or 10")
    print("are common descriptive warning thresholds, not formal hypothesis tests.")
    print("Compare 'pooled' with 'modality_interaction' to distinguish collinearity")
    print("among the base predictors from that introduced by modality terms.")


if __name__ == "__main__":
    main()
