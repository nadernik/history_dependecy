"""Build pooled log-odds coefficient and ridge-sensitivity LaTeX tables.

Primary-table inference follows two separate correction families:
  - choice/outcome history: BH across 36 tests (12 rats x reward/success/failure)
  - previous-stimulus history: BH across 12 beta_angle_n-1 tests

Intercept and current-trial beta_angle are shown descriptively, without stars.
The significance CSV is produced by canonical_pooled_log_odds_inference.py.
"""

import argparse
from pathlib import Path

import pandas as pd

from table_utils import (
    Cell,
    Table,
    fmt_est_ci,
    latex_est_ci,
    latex_num,
    stars_for_q,
    table_to_latex,
)


LOG_ODDS_COEFFICIENTS = [
    "intercept",
    "beta_angle",
    "beta_angle_n-1",
    "beta_reward_n-1",
    "beta_success_n-1",
    "beta_failure_n-1",
]
INFERENTIAL_COEFFICIENTS = {
    "beta_angle_n-1",
    "beta_reward_n-1",
    "beta_success_n-1",
    "beta_failure_n-1",
}
REFERENCE_COEFFICIENTS = {"intercept", "beta_angle"}

COEF_LABELS = {
    "intercept": ("Intercept", r"Intercept"),
    "beta_angle": ("β(angle)", r"$\beta_{\mathrm{angle}}$"),
    "beta_angle_n-1": ("β(angle, n−1)", r"$\beta_{\mathrm{angle},\,n-1}$"),
    "beta_reward_n-1": ("β(reward, n−1)", r"$\beta_{\mathrm{reward},\,n-1}$"),
    "beta_success_n-1": ("β(success, n−1)", r"$\beta_{\mathrm{success},\,n-1}$"),
    "beta_failure_n-1": ("β(failure, n−1)", r"$\beta_{\mathrm{failure},\,n-1}$"),
}

RIDGE_LEVELS = ["C=0.1", "C=1.0", "C=10.0"]
DECIMALS = 3


def require_columns(frame, required, label):
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def load_inputs(summary_csv: Path, sig_csv: Path):
    summary = pd.read_csv(summary_csv)
    require_columns(
        summary,
        {"rat_id", "subset", "C", "scale", "quantity", "value", "ci_low", "ci_high"},
        "GLM summary",
    )
    coef = summary.loc[
        (summary["subset"] == "pooled")
        & (summary["scale"] == "log_odds")
        & (summary["quantity"].isin(LOG_ODDS_COEFFICIENTS))
    ].copy()

    sig = pd.read_csv(sig_csv)
    require_columns(
        sig,
        {"rat_id", "coefficient", "q_bh", "significant_bh", "family"},
        "Coefficient significance table",
    )
    sig = sig.loc[sig["coefficient"].isin(INFERENTIAL_COEFFICIENTS)].copy()

    if coef.empty:
        raise ValueError("No pooled log-odds coefficient rows found")
    if sig.empty:
        raise ValueError("No inferential coefficient rows found in significance CSV")

    rats = sorted(int(rat) for rat in coef["rat_id"].unique())
    expected_sig = pd.MultiIndex.from_product(
        [rats, sorted(INFERENTIAL_COEFFICIENTS)], names=["rat_id", "coefficient"]
    )
    observed_sig = pd.MultiIndex.from_frame(sig[["rat_id", "coefficient"]])
    if observed_sig.has_duplicates:
        raise ValueError("Significance CSV has duplicate rat/coefficient rows")
    missing = expected_sig.difference(observed_sig)
    extra = observed_sig.difference(expected_sig)
    if len(missing) or len(extra):
        raise ValueError(
            f"Significance grid mismatch. Missing={list(missing)}; extra={list(extra)}"
        )

    family_by_coefficient = sig.groupby("coefficient")["family"].unique().to_dict()
    expected_families = {
        "beta_angle_n-1": "stimulus_history_12",
        "beta_reward_n-1": "choice_outcome_36",
        "beta_success_n-1": "choice_outcome_36",
        "beta_failure_n-1": "choice_outcome_36",
    }
    for coefficient, expected_family in expected_families.items():
        observed = list(family_by_coefficient.get(coefficient, []))
        if observed != [expected_family]:
            raise ValueError(
                f"{coefficient}: expected family {expected_family!r}, found {observed}"
            )
    return coef, sig


def build_primary_table(coef: pd.DataFrame, sig: pd.DataFrame):
    unpen = coef.loc[coef["C"] == "unpenalized"].set_index(["rat_id", "quantity"])
    sig_idx = sig.set_index(["rat_id", "coefficient"])
    rats = sorted(int(rat) for rat in coef["rat_id"].unique())

    expected_unpen = pd.MultiIndex.from_product(
        [rats, LOG_ODDS_COEFFICIENTS], names=["rat_id", "quantity"]
    )
    missing = expected_unpen.difference(unpen.index)
    if len(missing):
        raise ValueError(f"Missing unpenalized summary rows: {list(missing)}")

    plain_rows, latex_rows, csv_rows = [], [], []
    for rat in rats:
        plain_cells = [Cell(str(rat), align="left")]
        latex_cells = [Cell(str(rat), align="left")]
        csv_row = {"rat_id": rat}

        for coefficient in LOG_ODDS_COEFFICIENTS:
            row = unpen.loc[(rat, coefficient)]
            if coefficient in INFERENTIAL_COEFFICIENTS:
                qrow = sig_idx.loc[(rat, coefficient)]
                q_bh = float(qrow["q_bh"])
                significant = bool(qrow["significant_bh"])
                family = qrow["family"]
                stars = stars_for_q(q_bh)
            else:
                q_bh = float("nan")
                significant = False
                family = "reference_no_test"
                stars = ""

            plain_text = fmt_est_ci(
                row["value"], row["ci_low"], row["ci_high"], DECIMALS
            )
            if stars:
                plain_text += f" {stars}"
            plain_cells.append(Cell(plain_text, bold=False))
            latex_cells.append(
                Cell(
                    latex_est_ci(
                        row["value"],
                        row["ci_low"],
                        row["ci_high"],
                        DECIMALS,
                        stars,
                        bold=False,
                        pad_sign=True,
                        max_stars=3,
                    )
                )
            )

            csv_row[f"{coefficient}_value"] = row["value"]
            csv_row[f"{coefficient}_ci_low"] = row["ci_low"]
            csv_row[f"{coefficient}_ci_high"] = row["ci_high"]
            csv_row[f"{coefficient}_q_bh"] = q_bh
            csv_row[f"{coefficient}_significant_bh"] = significant
            csv_row[f"{coefficient}_family"] = family

        plain_rows.append(plain_cells)
        latex_rows.append(latex_cells)
        csv_rows.append(csv_row)

    header_plain = [Cell("Rat", align="left")] + [
        Cell(COEF_LABELS[q][0]) for q in LOG_ODDS_COEFFICIENTS
    ]
    header_latex = [Cell("Rat", align="left")] + [
        Cell(COEF_LABELS[q][1]) for q in LOG_ODDS_COEFFICIENTS
    ]

    note = (
        "Values are point estimate [95% bootstrap CI], log-odds scale, unpenalized "
        "(MLE) pooled fit. Asterisks mark BH-FDR corrected significance. Reward, "
        "success, and failure history form one 36-test family (12 rats x 3 "
        "coefficients); previous-stimulus history forms a separate 12-test family "
        "(12 rats x 1 coefficient): * q<.05, ** q<.01, *** q<.001. The intercept "
        "and current-trial stimulus slope are descriptive reference parameters and "
        "were not included in either inferential family."
    )

    plain_table = Table(
        header_rows=[header_plain],
        body_rows=plain_rows,
        n_cols=len(header_plain),
        caption="Pooled GLM: unpenalized log-odds coefficients, by rat",
        note=note,
    )
    latex_table = Table(
        header_rows=[header_latex],
        body_rows=latex_rows,
        n_cols=len(header_latex),
        caption="Pooled GLM: unpenalized log-odds coefficients, by rat.",
        label="tab:glm-coef-primary",
        note=note,
    )
    return plain_table, latex_table, pd.DataFrame(csv_rows)


def build_ridge_table(coef: pd.DataFrame):
    ridge = coef.loc[coef["C"].isin(RIDGE_LEVELS)].set_index(
        ["rat_id", "C", "quantity"]
    )
    rats = sorted(int(rat) for rat in coef["rat_id"].unique())

    expected = pd.MultiIndex.from_product(
        [rats, RIDGE_LEVELS, LOG_ODDS_COEFFICIENTS],
        names=["rat_id", "C", "quantity"],
    )
    missing = expected.difference(ridge.index)
    if len(missing):
        raise ValueError(f"Missing ridge summary rows: {list(missing)[:20]}")

    plain_rows, csv_rows = [], []
    for rat in rats:
        for c_level in RIDGE_LEVELS:
            plain_cells = [Cell(str(rat), align="left"), Cell(c_level, align="left")]
            csv_row = {"rat_id": rat, "C": c_level}
            for coefficient in LOG_ODDS_COEFFICIENTS:
                row = ridge.loc[(rat, c_level, coefficient)]
                plain_cells.append(
                    Cell(fmt_est_ci(row["value"], row["ci_low"], row["ci_high"], DECIMALS))
                )
                csv_row[f"{coefficient}_value"] = row["value"]
                csv_row[f"{coefficient}_ci_low"] = row["ci_low"]
                csv_row[f"{coefficient}_ci_high"] = row["ci_high"]
            plain_rows.append(plain_cells)
            csv_rows.append(csv_row)

    header_plain = [Cell("Rat", align="left"), Cell("Penalization", align="left")] + [
        Cell(COEF_LABELS[q][0]) for q in LOG_ODDS_COEFFICIENTS
    ]
    plain_table = Table(
        header_rows=[header_plain],
        body_rows=plain_rows,
        n_cols=len(header_plain),
        caption="Pooled GLM: ridge-penalized log-odds coefficients",
    )

    note = (
        "Values are point estimate [95% bootstrap CI], log-odds scale. No "
        "significance test was run at ridge-penalized fits; compare with the "
        "primary unpenalized table for inferential results."
    )
    latex_tables = []
    for index, c_level in enumerate(RIDGE_LEVELS, start=1):
        latex_rows = []
        for rat in rats:
            cells = [Cell(str(rat), align="left")]
            for coefficient in LOG_ODDS_COEFFICIENTS:
                row = ridge.loc[(rat, c_level, coefficient)]
                cells.append(
                    Cell(
                        latex_est_ci(
                            row["value"], row["ci_low"], row["ci_high"],
                            DECIMALS, pad_sign=True,
                        )
                    )
                )
            latex_rows.append(cells)

        header_latex = [Cell("Rat", align="left")] + [
            Cell(COEF_LABELS[q][1]) for q in LOG_ODDS_COEFFICIENTS
        ]
        latex_tables.append(
            Table(
                header_rows=[header_latex],
                body_rows=latex_rows,
                n_cols=len(header_latex),
                caption=f"Pooled GLM: ridge-penalized log-odds coefficients, by rat ({c_level}).",
                label=f"tab:glm-coef-ridge-{index}",
                note=note,
            )
        )
    return plain_table, latex_tables, pd.DataFrame(csv_rows)


def build_shift_summary_table(coef: pd.DataFrame):
    wide = coef.pivot_table(
        index=["rat_id", "quantity"], columns="C", values="value"
    ).reset_index()

    required_levels = {"unpenalized", *RIDGE_LEVELS}
    missing_levels = required_levels.difference(wide.columns)
    if missing_levels:
        raise ValueError(f"Shift summary is missing C levels: {sorted(missing_levels)}")

    plain_rows, latex_rows, csv_rows = [], [], []
    for coefficient in LOG_ODDS_COEFFICIENTS:
        sub = wide.loc[wide["quantity"] == coefficient]
        plain_cells = [Cell(COEF_LABELS[coefficient][0], align="left")]
        latex_cells = [Cell(COEF_LABELS[coefficient][1], align="left")]
        csv_row = {"coefficient": coefficient}
        for c_level in RIDGE_LEVELS:
            shift = (sub[c_level] - sub["unpenalized"]).abs()
            median_shift, max_shift = shift.median(), shift.max()
            plain_cells.extend([Cell(f"{median_shift:.4f}"), Cell(f"{max_shift:.4f}")])
            latex_cells.extend([Cell(latex_num(median_shift, 4)), Cell(latex_num(max_shift, 4))])
            csv_row[f"{c_level}_median_abs_shift"] = median_shift
            csv_row[f"{c_level}_max_abs_shift"] = max_shift
        plain_rows.append(plain_cells)
        latex_rows.append(latex_cells)
        csv_rows.append(csv_row)

    def header_for(median_label, max_label):
        top = [Cell("Coefficient", align="left", colspan=1)]
        bottom = [Cell("", align="left")]
        for c_level in RIDGE_LEVELS:
            top.append(Cell(c_level, colspan=2))
            bottom.extend([Cell(median_label), Cell(max_label)])
        return [top, bottom]

    note = (
        "Shift is the absolute difference between the coefficient at each "
        "penalization level and its unpenalized value, summarized across rats."
    )
    plain_table = Table(
        header_rows=header_for("Median |Δ|", "Max |Δ|"),
        body_rows=plain_rows,
        n_cols=1 + 2 * len(RIDGE_LEVELS),
        caption="Ridge-penalization sensitivity: coefficient shifts",
        note=note,
    )
    latex_table = Table(
        header_rows=header_for(r"Median $|\Delta|$", r"Max $|\Delta|$"),
        body_rows=latex_rows,
        n_cols=1 + 2 * len(RIDGE_LEVELS),
        caption="Ridge-penalization sensitivity: per-coefficient shift from the unpenalized fit.",
        label="tab:glm-coef-shift-summary",
        note=note,
    )
    return plain_table, latex_table, pd.DataFrame(csv_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--sig-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    coef, sig = load_inputs(args.summary_csv, args.sig_csv)

    _, latex_table, csv_df = build_primary_table(coef, sig)
    (args.out_dir / "table_glm_coef_primary.tex").write_text(
        table_to_latex(latex_table, fit_to_width=True), encoding="utf-8"
    )
    csv_df.to_csv(args.out_dir / "table_glm_coef_primary_data.csv", index=False)

    _, latex_tables, csv_df = build_ridge_table(coef)
    ridge_tex = "\n\n\\clearpage\n\n".join(
        table_to_latex(table, fit_to_width=True) for table in latex_tables
    )
    (args.out_dir / "table_glm_coef_ridge.tex").write_text(ridge_tex, encoding="utf-8")
    csv_df.to_csv(args.out_dir / "table_glm_coef_ridge_data.csv", index=False)

    _, latex_table, csv_df = build_shift_summary_table(coef)
    (args.out_dir / "table_glm_coef_shift_summary.tex").write_text(
        table_to_latex(latex_table), encoding="utf-8"
    )
    csv_df.to_csv(args.out_dir / "table_glm_coef_shift_summary_data.csv", index=False)

    print(f"Wrote 3 tables (.tex + _data.csv each) to {args.out_dir}")


if __name__ == "__main__":
    main()
