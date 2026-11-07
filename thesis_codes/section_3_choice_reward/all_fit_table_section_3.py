"""
Create the Section 3 appendix transparency table of raw fitted parameters.

The table contains one row per rat x previous-choice x previous-outcome cell
(12 rats x 4 cells = 48 rows). It reports the complete-data estimate and the
95% joint-bootstrap percentile interval for each fitted psychometric parameter:

    mu, sigma, gamma, lambda (the script's internal name is "lapse")

No significance markers are used: inferential tests in Section 3 concern the
planned derived contrasts, not these individual raw fit parameters.

Inputs (already produced by Section_3_choice_reward_joint_bootstrap.py)
------------------------------------------------------------------------
RESULTS_DIR/tables/full_data_cell_fits.csv
RESULTS_DIR/tables/bootstrap_parameter_summary.csv

Outputs
-------
section3_raw_fitted_parameters.tex
section3_raw_fitted_parameters.docx
section3_raw_fitted_parameters.csv

Keep this script next to table_utils.py, or otherwise make table_utils.py
importable on PYTHONPATH.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from table_utils import (
    Cell,
    Table,
    fmt_est_ci,
    latex_est_ci,
    table_to_docx,
    table_to_latex_longtable,
)


# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Change "filtered" to "unfiltered" if that is the Section 3 run reported in
# the thesis. The directory must contain tables/ and bootstraps/ from one run.
RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "choice_reward_joint_results"
    / "filtered"
    / "pooled"
)

OUTPUT_DIR = Path("C:/dev/projects/Thesis_SISSA/Rats/history_dependecy/thesis_codes/ale_thesis_fig_tab/tables/section_3")

DECIMALS_MU_SIGMA = 2
DECIMALS_GAMMA_LAPSE = 3

CAPTION = (
    "Raw psychometric parameters fitted to the four previous-choice by "
    "previous-outcome cells for each rat."
)
LABEL = "tab:section3-raw-fitted-parameters"
NOTE = (
    "Entries are complete-data maximum-likelihood estimates with 95% "
    "percentile intervals from the aligned joint session bootstrap in square "
    "brackets. H and V denote the previous horizontal and vertical choices; "
    "R and U denote rewarded and unrewarded previous trials. mu is the point "
    "of subjective equality in degrees, sigma is the psychometric spread in "
    "degrees, gamma is the lower asymptote, and lambda is the lapse (upper-"
    "asymptote) parameter. n is the number of trials retained by the fitting "
    "function after its per-angle minimum-count rule. These raw parameters are "
    "reported for transparency; significance testing was performed on the "
    "planned derived contrasts rather than on the individual cell parameters."
)


# =============================================================================
# INPUT DEFINITIONS
# =============================================================================

FULL_FITS_PATH = RESULTS_DIR / "tables" / "full_data_cell_fits.csv"
BOOTSTRAP_SUMMARY_PATH = (
    RESULTS_DIR / "tables" / "bootstrap_parameter_summary.csv"
)

CELL_ORDER = ("h_rew", "v_rew", "h_unrew", "v_unrew")
CELL_RANK = {cell: index for index, cell in enumerate(CELL_ORDER)}
CELL_DISPLAY = {
    "h_rew": ("H", "R"),
    "v_rew": ("V", "R"),
    "h_unrew": ("H", "U"),
    "v_unrew": ("V", "U"),
}
PARAMETER_ORDER = ("mu", "sigma", "gamma", "lapse")


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required Section 3 output not found: {path}")


def require_columns(frame: pd.DataFrame, columns: set[str], filename: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise KeyError(f"Missing columns in {filename}: {sorted(missing)}")


def load_and_merge() -> pd.DataFrame:
    """Load saved results and make one checked row per rat x cell x parameter."""
    require_file(FULL_FITS_PATH)
    require_file(BOOTSTRAP_SUMMARY_PATH)

    fits = pd.read_csv(FULL_FITS_PATH)
    boot = pd.read_csv(BOOTSTRAP_SUMMARY_PATH)

    require_columns(
        fits,
        {"rat", "cell", "success", "n_used"} | set(PARAMETER_ORDER),
        FULL_FITS_PATH.name,
    )
    require_columns(
        boot,
        {
            "rat", "cell", "parameter", "estimate_full_data",
            "ci_low", "ci_high", "ci_valid", "n_boot_valid",
        },
        BOOTSTRAP_SUMMARY_PATH.name,
    )

    fits = fits.copy()
    boot = boot.copy()
    fits["rat"] = fits["rat"].astype(int)
    boot["rat"] = boot["rat"].astype(int)

    fits = fits[fits["cell"].isin(CELL_ORDER)].copy()
    boot = boot[
        boot["cell"].isin(CELL_ORDER)
        & boot["parameter"].isin(PARAMETER_ORDER)
    ].copy()

    if fits.duplicated(["rat", "cell"]).any():
        duplicate = fits.loc[
            fits.duplicated(["rat", "cell"], keep=False), ["rat", "cell"]
        ]
        raise ValueError(f"Duplicate full-fit rows found:\n{duplicate}")
    if boot.duplicated(["rat", "cell", "parameter"]).any():
        duplicate = boot.loc[
            boot.duplicated(["rat", "cell", "parameter"], keep=False),
            ["rat", "cell", "parameter"],
        ]
        raise ValueError(f"Duplicate bootstrap-summary rows found:\n{duplicate}")

    failed = fits.loc[~fits["success"].map(as_bool), ["rat", "cell"]]
    if not failed.empty:
        raise ValueError(
            "The appendix table requires successful full-data fits. "
            f"Failed cells:\n{failed.to_string(index=False)}"
        )

    invalid_ci = boot.loc[~boot["ci_valid"].map(as_bool), ["rat", "cell", "parameter"]]
    if not invalid_ci.empty:
        raise ValueError(
            "The appendix table requires valid bootstrap intervals. "
            f"Invalid cells/parameters:\n{invalid_ci.to_string(index=False)}"
        )

    # Verify that the estimate exported in the bootstrap summary is the same
    # complete-data estimate saved in the full-fit table.
    long_fits = fits.melt(
        id_vars=["rat", "cell", "n_used"],
        value_vars=list(PARAMETER_ORDER),
        var_name="parameter",
        value_name="estimate_from_full_fit",
    )
    merged = boot.merge(
        long_fits,
        on=["rat", "cell", "parameter"],
        how="inner",
        validate="one_to_one",
    )
    expected_rows = len(fits) * len(PARAMETER_ORDER)
    if len(merged) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} merged parameter rows but found {len(merged)}. "
            "Check that both input CSVs came from the same completed run."
        )

    discrepancies = ~np.isclose(
        merged["estimate_full_data"],
        merged["estimate_from_full_fit"],
        rtol=1e-7,
        atol=1e-8,
        equal_nan=False,
    )
    if discrepancies.any():
        bad = merged.loc[
            discrepancies,
            [
                "rat", "cell", "parameter", "estimate_full_data",
                "estimate_from_full_fit",
            ],
        ]
        raise ValueError(
            "Full-data estimates disagree between the two input files:\n"
            + bad.to_string(index=False)
        )

    merged["cell_rank"] = merged["cell"].map(CELL_RANK)
    return merged.sort_values(["rat", "cell_rank", "parameter"]).reset_index(drop=True)


def as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def integer_digits(values: pd.Series) -> int:
    finite = np.abs(pd.to_numeric(values, errors="coerce").to_numpy(dtype=float))
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return 1
    return max(1, max(len(str(int(np.floor(value)))) for value in finite))


def make_wide_data(merged: pd.DataFrame) -> pd.DataFrame:
    """One row per rat x cell, keeping estimate and CI columns per parameter."""
    id_columns = ["rat", "cell", "cell_rank", "n_used", "n_boot_valid"]
    value_columns = ["estimate_full_data", "ci_low", "ci_high"]
    wide = merged.pivot_table(
        index=id_columns,
        columns="parameter",
        values=value_columns,
        aggfunc="first",
    ).reset_index()
    wide.columns = [
        "_".join(part for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in wide.columns
    ]
    return wide.sort_values(["rat", "cell_rank"]).reset_index(drop=True)


# =============================================================================
# TABLE CONSTRUCTION
# =============================================================================

def header_rows() -> list[list[Cell]]:
    return [[
        Cell("Rat"),
        Cell("Previous choice"),
        Cell("Previous outcome"),
        Cell("n"),
        Cell("mu (degrees)"),
        Cell("sigma (degrees)"),
        Cell("gamma"),
        Cell("lambda"),
    ]]


def build_plain_table(wide: pd.DataFrame) -> Table:
    """Plain-text cells for the native Word table."""
    rows = []
    for record in wide.itertuples(index=False):
        choice, outcome = CELL_DISPLAY[record.cell]
        rows.append([
            Cell(str(record.rat)),
            Cell(choice),
            Cell(outcome),
            Cell(f"{int(record.n_used):,}"),
            Cell(fmt_est_ci(
                record.estimate_full_data_mu, record.ci_low_mu,
                record.ci_high_mu, DECIMALS_MU_SIGMA,
            )),
            Cell(fmt_est_ci(
                record.estimate_full_data_sigma, record.ci_low_sigma,
                record.ci_high_sigma, DECIMALS_MU_SIGMA,
            )),
            Cell(fmt_est_ci(
                record.estimate_full_data_gamma, record.ci_low_gamma,
                record.ci_high_gamma, DECIMALS_GAMMA_LAPSE,
            )),
            Cell(fmt_est_ci(
                record.estimate_full_data_lapse, record.ci_low_lapse,
                record.ci_high_lapse, DECIMALS_GAMMA_LAPSE,
            )),
        ])
    return Table(
        header_rows=header_rows(),
        body_rows=rows,
        n_cols=8,
        caption=CAPTION,
        label=LABEL,
        note=NOTE,
    )


def build_latex_table(wide: pd.DataFrame) -> Table:
    """Same underlying rows, with math-aware aligned numeric LaTeX cells."""
    digit_width = {}
    for parameter in PARAMETER_ORDER:
        values = pd.concat([
            wide[f"estimate_full_data_{parameter}"],
            wide[f"ci_low_{parameter}"],
            wide[f"ci_high_{parameter}"],
        ], ignore_index=True)
        digit_width[parameter] = integer_digits(values)

    rows = []
    previous_rat = None
    for record in wide.itertuples(index=False):
        choice, outcome = CELL_DISPLAY[record.cell]
        # Show the rat number once at the start of each four-row block; this
        # makes the long appendix table easier to scan without adding blank
        # separator rows that would look awkward in Word.
        rat_text = str(record.rat) if record.rat != previous_rat else ""
        previous_rat = record.rat
        rows.append([
            Cell(rat_text),
            Cell(choice),
            Cell(outcome),
            Cell(f"{int(record.n_used):,}"),
            Cell(latex_est_ci(
                record.estimate_full_data_mu, record.ci_low_mu,
                record.ci_high_mu, DECIMALS_MU_SIGMA,
                pad_sign=True, int_digits=digit_width["mu"],
            )),
            Cell(latex_est_ci(
                record.estimate_full_data_sigma, record.ci_low_sigma,
                record.ci_high_sigma, DECIMALS_MU_SIGMA,
                pad_sign=True, int_digits=digit_width["sigma"],
            )),
            Cell(latex_est_ci(
                record.estimate_full_data_gamma, record.ci_low_gamma,
                record.ci_high_gamma, DECIMALS_GAMMA_LAPSE,
                pad_sign=True, int_digits=digit_width["gamma"],
            )),
            Cell(latex_est_ci(
                record.estimate_full_data_lapse, record.ci_low_lapse,
                record.ci_high_lapse, DECIMALS_GAMMA_LAPSE,
                pad_sign=True, int_digits=digit_width["lapse"],
            )),
        ])

    return Table(
        header_rows=header_rows(),
        body_rows=rows,
        n_cols=8,
        caption=CAPTION,
        label=LABEL,
        note=NOTE,
    )


def export_companion_csv(wide: pd.DataFrame, path: Path) -> None:
    """Machine-readable version containing the exact values used in the table."""
    export = wide.copy()
    export[["previous_choice", "previous_outcome"]] = pd.DataFrame(
        export["cell"].map(CELL_DISPLAY).tolist(), index=export.index
    )
    columns = ["rat", "cell", "previous_choice", "previous_outcome", "n_used"]
    for parameter in PARAMETER_ORDER:
        columns.extend([
            f"estimate_full_data_{parameter}",
            f"ci_low_{parameter}",
            f"ci_high_{parameter}",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    export[columns].to_csv(path, index=False, float_format="%.8g")


# =============================================================================
# RUN
# =============================================================================

def main() -> None:
    merged = load_and_merge()
    wide = make_wide_data(merged)

    expected_cells = len(wide["rat"].unique()) * len(CELL_ORDER)
    if len(wide) != expected_cells:
        raise ValueError(
            f"Expected four cells for every rat ({expected_cells} total), "
            f"but found {len(wide)}."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tex_path = OUTPUT_DIR / "section3_raw_fitted_parameters.tex"
    docx_path = OUTPUT_DIR / "section3_raw_fitted_parameters.docx"
    csv_path = OUTPUT_DIR / "section3_raw_fitted_parameters.csv"

    latex_table = build_latex_table(wide)
    tex_path.write_text(
        table_to_latex_longtable(
            latex_table,
            landscape=True,
            fontsize=r"\scriptsize",
            tabcolsep_pt=3.0,
        ),
        encoding="utf-8",
    )

    word_table = build_plain_table(wide)
    table_to_docx(word_table, docx_path, font_size_pt=8, landscape=True)
    export_companion_csv(wide, csv_path)

    print(f"Wrote LaTeX table: {tex_path}")
    print(f"Wrote Word table:  {docx_path}")
    print(f"Wrote source CSV:  {csv_path}")
    print(
        f"Rows: {len(wide)} "
        f"({wide['rat'].nunique()} rats x {len(CELL_ORDER)} cells)"
    )


if __name__ == "__main__":
    main()
