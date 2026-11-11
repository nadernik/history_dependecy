"""
Build the two log-odds coefficient tables (primary unpenalized + ridge
sensitivity) and the coefficient shift-summary table, as LaTeX only, from:
  - glm_summary_long.csv                          (point estimates + CIs, all C levels)
  - glm_coefficient_significance_unpenalized.csv   (BH-FDR q-values, unpenalized only —
                                                     produced by glm_coefficient_significance.py)

Produces, in --out-dir:
  table_glm_coef_primary.tex     — unpenalized coefficients, 12 rats x 6 coefficients
  table_glm_coef_ridge.tex       — ridge sensitivity, 12 rats x 3 C-levels x 6 coefficients
  table_glm_coef_shift_summary.tex — median/max |shift| per coefficient per C-level
  (plus a _data.csv per table with the exact numbers behind it)

Significance (primary table only): asterisks from q_bh (*** q<.001,
** q<.01, * q<.05), per the BH-FDR correction already computed across the
72-test (12 rat x 6 coefficient) family — no bold (dropped per request).
The ridge table and the shift-summary table report magnitudes only — no
hypothesis test was run at the ridge C-levels, so nothing there is marked
significant/non-significant.

Note: .docx output was dropped (2026-09-17, per request) — Google Docs
paste-in is no longer needed, LaTeX only. `table_to_docx` still exists in
table_utils.py if that's ever wanted back, it's just not called here.
"""
#python "C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_4_GLM\pooled_log_odds_coefficient_tables.py" `
#  --summary-csv "C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_4_GLM\fit_results\pooled_filtered\glm_summary_long.csv" `
#  --sig-csv "C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_4_GLM\fit_results\pooled_filtered\canonical_inference\glm_contrast_significance_unpenalized.csv" `
#  --out-dir "C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\ale_thesis_fig_tab\tables_thesis\section_4"

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from table_utils import (
    Cell,
    Table,
    fmt_est_ci,
    latex_est_ci,
    latex_num,
    stars_for_q,
    table_to_latex,
    table_to_latex_longtable,
)

LOG_ODDS_COEFFICIENTS = [
    "intercept",
    "beta_angle",
    "beta_angle_n-1",
    "beta_reward_n-1",
    "beta_success_n-1",
    "beta_failure_n-1",
]

# display labels: (plain text for docx/CSV, LaTeX math-mode label)
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


def load_inputs(summary_csv: Path, sig_csv: Path):
    summary = pd.read_csv(summary_csv)
    coef = summary[
        (summary["scale"] == "log_odds") & (summary["quantity"].isin(LOG_ODDS_COEFFICIENTS))
    ].copy()
    sig = pd.read_csv(sig_csv)
    return coef, sig


# ---------------------------------------------------------------------------
# Table 1: primary (unpenalized) coefficients — rows = rat, cols = 6 coefficients
# ---------------------------------------------------------------------------

def build_primary_table(coef: pd.DataFrame, sig: pd.DataFrame):
    unpen = coef[coef["C"] == "unpenalized"].set_index(["rat_id", "quantity"])
    sig_idx = sig.set_index(["rat_id", "coefficient"])
    rats = sorted(coef["rat_id"].unique())

    plain_rows, latex_rows, csv_rows = [], [], []
    for rat in rats:
        plain_cells = [Cell(str(rat), align="left")]
        latex_cells = [Cell(str(rat), align="left")]
        csv_row = {"rat_id": rat}
        for q in LOG_ODDS_COEFFICIENTS:
            row = unpen.loc[(rat, q)]
            qrow = sig_idx.loc[(rat, q)]
            stars = stars_for_q(qrow["q_bh"])
            bold = bool(qrow["significant_bh"])
            plain_text = fmt_est_ci(row["value"], row["ci_low"], row["ci_high"], DECIMALS)
            plain_text_with_stars = plain_text + (f" {stars}" if stars else "")
            plain_cells.append(Cell(plain_text_with_stars, bold=bold))
            latex_cells.append(
                Cell(
                    latex_est_ci(
                        row["value"],
                        row["ci_low"],
                        row["ci_high"],
                        DECIMALS,
                        stars,
                        bold=False,  # no bold in the LaTeX table — significance is asterisks only
                        pad_sign=True,  # keep +/- values the same width so CI brackets line up
                        max_stars=3,  # pad 0-3 stars to a fixed width, same reason
                    ),
                )
            )
            csv_row[f"{q}_value"] = row["value"]
            csv_row[f"{q}_ci_low"] = row["ci_low"]
            csv_row[f"{q}_ci_high"] = row["ci_high"]
            csv_row[f"{q}_q_bh"] = qrow["q_bh"]
            csv_row[f"{q}_significant_bh"] = qrow["significant_bh"]
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
        "Values are point estimate [95% bootstrap CI], log-odds scale, unpenalized (MLE) fit. "
        "Asterisks mark BH-FDR corrected significance across the 72-test "
        "(12 rats x 6 coefficients) family: * q<.05, ** q<.01, *** q<.001."
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


# ---------------------------------------------------------------------------
# Table 2: ridge sensitivity — rows = rat x C level, cols = 6 coefficients
# ---------------------------------------------------------------------------

def build_ridge_table(coef: pd.DataFrame):
    """Returns (plain_table_full, list_of_latex_tables, csv_df_full).

    The CSV keeps the full 12 rats x 3 C-levels x 6 coefficients grid in
    one file — it's data, width doesn't matter.

    The LaTeX rendering is split into three tables, one per penalization
    level (C=0.1 / C=1.0 / C=10.0), each shaped exactly like the primary
    table: 12 rats x 6 coefficients, 7 columns, rendered with the same
    non-long `table_to_latex(..., fit_to_width=True)` path (resizebox).

    This replaces an earlier design that kept all 3 C-levels in one
    36-row `longtable` split by coefficient group instead of by C-level.
    That approach needed `landscape` + shrunk fonts + reduced \\tabcolsep
    and still overflowed the page by 40-50pt in practice (confirmed by
    test-compiling it inside a realistic document, i.e. preceded by other
    content — `resizebox` doesn't work with `longtable`, so every fix had
    to be a manual width guess, and none of them reliably cleared the
    overflow once `pad_sign` padding was added for CI-bracket alignment).
    Splitting by C-level instead means every table is short enough
    (12 rows) to not need `longtable` pagination at all, and narrow enough
    (7 columns, same as the primary table) that `resizebox` guarantees a
    fit regardless of content width — no more manual tuning."""
    ridge = coef[coef["C"].isin(RIDGE_LEVELS)].set_index(["rat_id", "C", "quantity"])
    rats = sorted(coef["rat_id"].unique())

    # full-width plain/csv (all 3 C-levels, all 6 coefficients in one table) — data only
    plain_rows, csv_rows = [], []
    for rat in rats:
        for c in RIDGE_LEVELS:
            plain_cells = [Cell(str(rat), align="left"), Cell(c, align="left")]
            csv_row = {"rat_id": rat, "C": c}
            for q in LOG_ODDS_COEFFICIENTS:
                row = ridge.loc[(rat, c, q)]
                plain_cells.append(
                    Cell(fmt_est_ci(row["value"], row["ci_low"], row["ci_high"], DECIMALS))
                )
                csv_row[f"{q}_value"] = row["value"]
                csv_row[f"{q}_ci_low"] = row["ci_low"]
                csv_row[f"{q}_ci_high"] = row["ci_high"]
            plain_rows.append(plain_cells)
            csv_rows.append(csv_row)
    header_plain = [Cell("Rat", align="left"), Cell("Penalization", align="left")] + [
        Cell(COEF_LABELS[q][0]) for q in LOG_ODDS_COEFFICIENTS
    ]
    plain_table = Table(
        header_rows=[header_plain],
        body_rows=plain_rows,
        n_cols=len(header_plain),
        caption="Pooled GLM: ridge-penalized log-odds coefficients, by rat and penalization strength",
    )

    # one 12x6 LaTeX table per C-level, same shape/renderer as the primary table
    note = (
        "Values are point estimate [95% bootstrap CI], log-odds scale. No significance test "
        "was run at ridge-penalized fits; compare against the primary (unpenalized) table for "
        "the operative result and its BH-FDR-corrected significance."
    )
    latex_tables = []
    for i, c in enumerate(RIDGE_LEVELS):
        latex_rows = []
        for rat in rats:
            latex_cells = [Cell(str(rat), align="left")]
            for q in LOG_ODDS_COEFFICIENTS:
                row = ridge.loc[(rat, c, q)]
                latex_cells.append(
                    Cell(
                        latex_est_ci(
                            row["value"], row["ci_low"], row["ci_high"], DECIMALS, pad_sign=True
                        )
                    )
                )
            latex_rows.append(latex_cells)
        header_latex = [Cell("Rat", align="left")] + [
            Cell(COEF_LABELS[q][1]) for q in LOG_ODDS_COEFFICIENTS
        ]
        latex_tables.append(
            Table(
                header_rows=[header_latex],
                body_rows=latex_rows,
                n_cols=len(header_latex),
                caption=f"Pooled GLM: ridge-penalized log-odds coefficients, by rat ({c}).",
                label=f"tab:glm-coef-ridge-{i + 1}",
                note=note,
            )
        )
    return plain_table, latex_tables, pd.DataFrame(csv_rows)


# ---------------------------------------------------------------------------
# Table 3: shift summary — rows = 6 coefficients, cols = 3 ridge levels x {median, max} |shift|
# ---------------------------------------------------------------------------

def build_shift_summary_table(coef: pd.DataFrame):
    wide = coef.pivot_table(index=["rat_id", "quantity"], columns="C", values="value").reset_index()

    plain_rows, latex_rows, csv_rows = [], [], []
    for q in LOG_ODDS_COEFFICIENTS:
        sub = wide[wide["quantity"] == q]
        plain_cells = [Cell(COEF_LABELS[q][0], align="left")]
        latex_cells = [Cell(COEF_LABELS[q][1], align="left")]
        csv_row = {"coefficient": q}
        for c in RIDGE_LEVELS:
            shift = (sub[c] - sub["unpenalized"]).abs()
            med, mx = shift.median(), shift.max()
            plain_cells.append(Cell(f"{med:.4f}"))
            plain_cells.append(Cell(f"{mx:.4f}"))
            latex_cells.append(Cell(latex_num(med, 4)))
            latex_cells.append(Cell(latex_num(mx, 4)))
            csv_row[f"{c}_median_abs_shift"] = med
            csv_row[f"{c}_max_abs_shift"] = mx
        plain_rows.append(plain_cells)
        latex_rows.append(latex_cells)
        csv_rows.append(csv_row)

    def header_for(median_label: str, max_label: str):
        top = [Cell("Coefficient", align="left", colspan=1)]
        bottom = [Cell("", align="left")]
        for c in RIDGE_LEVELS:
            top.append(Cell(c, colspan=2))
            bottom.append(Cell(median_label))
            bottom.append(Cell(max_label))
        return [top, bottom]

    # Plain (docx/CSV) header can use the Unicode delta directly; the LaTeX
    # header uses the math-mode $\Delta$ command instead, since a raw Unicode
    # character in plain LaTeX text needs font/encoding support we can't
    # assume ("plain LaTeX only" was the agreed baseline).
    header_plain = header_for("Median |Δ|", "Max |Δ|")
    header_latex = header_for(r"Median $|\Delta|$", r"Max $|\Delta|$")

    note = (
        "Shift is the absolute difference between the coefficient at this penalization level "
        "and its unpenalized value, log-odds scale, summarized across the 12 rats (median and "
        "max of the 12 per-rat absolute shifts). This is the per-coefficient breakdown behind "
        "the aggregate robustness statement in the text (median/largest log-odds shift, and the resulting "
        "degree-scale contrast shift)."
    )

    plain_table = Table(
        header_rows=header_plain,
        body_rows=plain_rows,
        n_cols=1 + 2 * len(RIDGE_LEVELS),
        caption="Ridge-penalization sensitivity: per-coefficient shift from the unpenalized fit",
        note=note,
    )
    latex_table = Table(
        header_rows=header_latex,
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

    # --- primary table ---
    plain, latex_t, csv_df = build_primary_table(coef, sig)
    (args.out_dir / "table_glm_coef_primary.tex").write_text(
        table_to_latex(latex_t, fit_to_width=True)
    )
    csv_df.to_csv(args.out_dir / "table_glm_coef_primary_data.csv", index=False)

    # --- ridge sensitivity table (split into 3 tables, one per C-level — see build_ridge_table) ---
    plain, latex_tables, csv_df = build_ridge_table(coef)
    ridge_tex = "\n\n\\clearpage\n\n".join(
        table_to_latex(lt, fit_to_width=True) for lt in latex_tables
    )
    (args.out_dir / "table_glm_coef_ridge.tex").write_text(ridge_tex)
    csv_df.to_csv(args.out_dir / "table_glm_coef_ridge_data.csv", index=False)

    # --- shift summary table ---
    plain, latex_t, csv_df = build_shift_summary_table(coef)
    (args.out_dir / "table_glm_coef_shift_summary.tex").write_text(table_to_latex(latex_t))
    csv_df.to_csv(args.out_dir / "table_glm_coef_shift_summary_data.csv", index=False)

    print(f"Wrote 3 tables (.tex + _data.csv each) to {args.out_dir}")


if __name__ == "__main__":
    main()