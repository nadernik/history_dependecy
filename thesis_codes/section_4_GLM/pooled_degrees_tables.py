"""
Build the psychometric-vs-GLM comparison tables (degree scale, pooled scope)
for Section 4 Q1a, as LaTeX only, from:
  - contrast_summary_long.csv                    (psychometric point estimates + CIs + q_bh_primary_family)
  - glm_summary_long.csv                          (GLM point estimates + CIs, all C levels)
  - glm_contrast_significance_unpenalized.csv     (GLM BH-FDR q-values, unpenalized only --
                                                     produced by glm_contrast_significance.py)

Produces, in --out-dir:
  table_comparison_main.tex    -- main-text table, 15 rows (10 confirmatory + 5 discovery)
  table_comparison_full.tex    -- appendix table, all 12 rats x 4 contrasts = 48 rows
  (plus a _data.csv per table with the exact numbers behind it)

Row selection for the main-text table (see section_4_q1a_merged_tables.md):
  Confirmatory block (10 rows) -- rat/contrast pairs already named in the
  current Section 4 draft prose: rat 13 (dmuR), rat 9 (dmuU, D), rat 11
  (dmuU, D), rat 12 (dmuU, D), rat 4 (P, dmuU, D).
  Discovery block (5 rows) -- rat/contrast pairs where the GLM resolves a
  large effect the psychometric analysis missed entirely (q~.97, essentially
  null), flagged in section_4_q1a_merged_tables.md's "two things this
  surfaces" section: rat 7 (dmuU, D), rat 10 (D), rat 16 (dmuR, dmuU).

Significance: asterisks from q_bh (*** q<.001, ** q<.01, * q<.05), computed
per-column (psychometric q from its own 48-test primary family; GLM q from
the matching 48-test GLM contrast family) -- no bold, asterisks only, same
convention as the coefficient tables (table_utils.stars_for_q).

CI ratio column = width(GLM CI) / width(psychometric CI), unitless.

Both tables' two row-blocks (confirmatory/discovery in the main table;
main-text-rats/remaining-cohort in the appendix table) are separated by a
plain horizontal rule (an empty row in the Table model, rendered as
\midrule by table_utils) rather than a labeled row -- the description of
what the rule separates lives in the table's note text instead.

Every number in both tables (point estimate and both CI bounds, psychometric
and GLM columns alike) is left-padded to a shared integer-digit width via
`latex_num`'s `int_digits` (computed once from the full 48-row combination
space by `compute_global_int_digits`), on top of the existing sign padding.
Without this, e.g. "9.17" and "17.13" in the same column differ in width
by one digit even with sign padding alone, so the CI's "[", the comma, and
the "]" all drift out of alignment down the column -- this is what actually
causes a column of mixed-magnitude values to look misaligned.
"""

import argparse
from pathlib import Path

import pandas as pd

from table_utils import (
    Cell,
    Table,
    fmt_est_ci,
    latex_est_ci,
    stars_for_q,
    table_to_latex,
    table_to_latex_longtable,
)

# psychometric contrast label -> GLM quantity name
CONTRAST_MAP = {
    "delta_mu_rew": "delta_mu_rewarded",
    "delta_mu_unrew": "delta_mu_unrewarded",
    "D": "D_glm",
    "P": "P_glm",
}
CONTRAST_ORDER = ["delta_mu_rew", "delta_mu_unrew", "D", "P"]
CONTRAST_LABELS = {
    "delta_mu_rew": ("ΔμR", r"$\Delta\mu_{R}$"),
    "delta_mu_unrew": ("ΔμU", r"$\Delta\mu_{U}$"),
    "D": ("D", r"$D$"),
    "P": ("P", r"$P$"),
}

DECIMALS = 2

# main-text table row selection: (rat, contrast) pairs, in the order they
# should appear, split into confirmatory / discovery blocks.
CONFIRMATORY_ROWS = [
    (13, "delta_mu_rew"),
    (9, "delta_mu_unrew"),
    (9, "D"),
    (11, "delta_mu_unrew"),
    (11, "D"),
    (12, "delta_mu_unrew"),
    (12, "D"),
    (4, "P"),
    (4, "delta_mu_unrew"),
    (4, "D"),
]
DISCOVERY_ROWS = [
    (7, "delta_mu_unrew"),
    (7, "D"),
    (10, "D"),
    (16, "delta_mu_rew"),
    (16, "delta_mu_unrew"),
]


def load_inputs(contrast_csv: Path, glm_summary_csv: Path, glm_sig_csv: Path):
    psych = pd.read_csv(contrast_csv)
    psych = psych[
        (psych["data_condition"] == "filtered")
        & (psych["scope"] == "pooled")
        & (psych["formal_primary"] == True)  # noqa: E712
    ].copy()

    glm_summary = pd.read_csv(glm_summary_csv)
    glm = glm_summary[
        (glm_summary["C"] == "unpenalized")
        & (glm_summary["scale"] == "degrees")
        & (glm_summary["quantity"].isin(CONTRAST_MAP.values()))
    ].copy()

    glm_sig = pd.read_csv(glm_sig_csv)

    return psych, glm, glm_sig


def _ci_width(row) -> float:
    return float(row["ci_high"]) - float(row["ci_low"])


def build_row_data(rat: int, contrast: str, psych: pd.DataFrame, glm: pd.DataFrame, glm_sig: pd.DataFrame) -> dict:
    p_row = psych[(psych["rat"] == rat) & (psych["contrast"] == contrast)].iloc[0]
    glm_quantity = CONTRAST_MAP[contrast]
    g_row = glm[(glm["rat_id"] == rat) & (glm["quantity"] == glm_quantity)].iloc[0]
    g_sig = glm_sig[(glm_sig["rat_id"] == rat) & (glm_sig["contrast"] == glm_quantity)].iloc[0]

    psych_q = p_row["q_bh_primary_family"]
    glm_q = g_sig["q_bh"]

    ci_ratio = _ci_width(g_row) / _ci_width(p_row)

    return {
        "rat": rat,
        "contrast": contrast,
        "psych_value": p_row["estimate_full_data"],
        "psych_ci_low": p_row["ci_low"],
        "psych_ci_high": p_row["ci_high"],
        "psych_q": psych_q,
        "glm_value": g_row["value"],
        "glm_ci_low": g_row["ci_low"],
        "glm_ci_high": g_row["ci_high"],
        "glm_q": glm_q,
        "ci_ratio": ci_ratio,
    }


def compute_global_int_digits(psych: pd.DataFrame, glm: pd.DataFrame, glm_sig: pd.DataFrame) -> int:
    """Max integer-digit width across every value/CI-bound cell that goes into
    either comparison table (psych value/CI, GLM value/CI, all 12 rats x 4
    contrasts). Used to left-pad every number in both tables to the same
    integer-digit width, so decimal points -- and therefore the CI brackets,
    commas, and closing brackets -- line up vertically down each column
    regardless of whether a given row's numbers happen to be 1 or 2 digits
    before the decimal point. Computed once from the full 48-row combination
    space (a superset of both tables' rows) so a single width is shared
    consistently across the main-text and appendix tables."""
    rats = sorted(psych["rat"].unique())
    max_digits = 0
    for rat in rats:
        for contrast in CONTRAST_ORDER:
            d = build_row_data(rat, contrast, psych, glm, glm_sig)
            for key in (
                "psych_value", "psych_ci_low", "psych_ci_high",
                "glm_value", "glm_ci_low", "glm_ci_high",
            ):
                max_digits = max(max_digits, len(str(int(abs(d[key])))))
    return max_digits


def _render_row(d: dict, latex: bool, int_digits: int = 0) -> list:
    psych_stars = stars_for_q(d["psych_q"])
    glm_stars = stars_for_q(d["glm_q"])
    label_plain, label_latex = CONTRAST_LABELS[d["contrast"]]

    if latex:
        return [
            Cell(str(d["rat"]), align="left"),
            Cell(label_latex),
            Cell(
                latex_est_ci(
                    d["psych_value"], d["psych_ci_low"], d["psych_ci_high"], DECIMALS,
                    psych_stars, bold=False, pad_sign=True, max_stars=3, int_digits=int_digits,
                )
            ),
            Cell(
                latex_est_ci(
                    d["glm_value"], d["glm_ci_low"], d["glm_ci_high"], DECIMALS,
                    glm_stars, bold=False, pad_sign=True, max_stars=3, int_digits=int_digits,
                )
            ),
            Cell(f"{d['ci_ratio']:.2f}"),
        ]
    else:
        psych_text = fmt_est_ci(d["psych_value"], d["psych_ci_low"], d["psych_ci_high"], DECIMALS)
        psych_text += f" {psych_stars}" if psych_stars else ""
        glm_text = fmt_est_ci(d["glm_value"], d["glm_ci_low"], d["glm_ci_high"], DECIMALS)
        glm_text += f" {glm_stars}" if glm_stars else ""
        return [
            Cell(str(d["rat"]), align="left"),
            Cell(label_plain),
            Cell(psych_text),
            Cell(glm_text),
            Cell(f"{d['ci_ratio']:.2f}"),
        ]


HEADER_PLAIN = [
    Cell("Rat", align="left"),
    Cell("Contrast"),
    Cell("Psychometric ° [95% CI]"),
    Cell("GLM ° [95% CI]"),
    Cell("CI ratio\n(GLM/Psych)"),
]
HEADER_LATEX = [
    Cell("Rat", align="left"),
    Cell("Contrast"),
    Cell("Psychometric $^{\\circ}$ [95\\% CI]"),
    Cell("GLM $^{\\circ}$ [95\\% CI]"),
    Cell("CI ratio (GLM/Psych)"),
]

NOTE = (
    "Values are point estimate [95% bootstrap CI], degree scale, pooled scope, filtered data. "
    "Asterisks mark BH-FDR corrected significance, each column within its own 48-test "
    "(12 rats x 4 contrasts) family -- psychometric q from the primary-contrast family "
    "(contrast_summary_long.csv, q_bh_primary_family), GLM q from the matching pooled-GLM "
    "contrast family (glm_contrast_significance_unpenalized.csv): * q<.05, ** q<.01, *** q<.001. "
    "CI ratio is CI width(GLM) / CI width(psychometric); values <1 indicate a narrower GLM interval."
)

NOTE_MAIN = NOTE + (
    " The rule partway down the table separates two blocks of rows: above it, the confirmatory "
    "block (10 rows) already discussed in the main Section 4 text; below it, the discovery block "
    "(5 rows) -- rat/contrast pairs where the psychometric estimate is essentially null but the "
    "GLM resolves a large, significant effect (rats 7, 10, 16; see main text)."
)

NOTE_FULL = NOTE + (
    " The rule partway down the table separates two blocks of rows: above it, the rats discussed "
    "in the main text (13, 9, 11, 12, 4); below it, the remaining cohort, in rat-ID order."
)


def build_main_table(psych: pd.DataFrame, glm: pd.DataFrame, glm_sig: pd.DataFrame, int_digits: int = 0):
    confirmatory = [build_row_data(r, c, psych, glm, glm_sig) for r, c in CONFIRMATORY_ROWS]
    discovery = [build_row_data(r, c, psych, glm, glm_sig) for r, c in DISCOVERY_ROWS]

    plain_rows = [_render_row(d, latex=False) for d in confirmatory]
    plain_rows.append([])  # dividing rule, no text -- see NOTE_MAIN for what it separates
    plain_rows += [_render_row(d, latex=False) for d in discovery]

    latex_rows = [_render_row(d, latex=True, int_digits=int_digits) for d in confirmatory]
    latex_rows.append([])  # dividing rule, no text -- see NOTE_MAIN for what it separates
    latex_rows += [_render_row(d, latex=True, int_digits=int_digits) for d in discovery]

    csv_rows = confirmatory + discovery
    for d, block in zip(csv_rows, ["confirmatory"] * len(confirmatory) + ["discovery"] * len(discovery)):
        d["block"] = block

    plain_table = Table(
        header_rows=[HEADER_PLAIN],
        body_rows=plain_rows,
        n_cols=5,
        caption="Psychometric vs. GLM primary contrasts: main-text comparison (pooled, filtered)",
        note=NOTE_MAIN,
    )
    latex_table = Table(
        header_rows=[HEADER_LATEX],
        body_rows=latex_rows,
        n_cols=5,
        caption="Psychometric vs. GLM primary contrasts: main-text comparison (pooled, filtered).",
        label="tab:comparison-main",
        note=NOTE_MAIN,
    )
    return plain_table, latex_table, pd.DataFrame(csv_rows)


def build_full_table(psych: pd.DataFrame, glm: pd.DataFrame, glm_sig: pd.DataFrame, int_digits: int = 0):
    rats = sorted(psych["rat"].unique())
    main_rats = [13, 9, 11, 12, 4]  # rats discussed in the main text, listed first
    remaining_rats = [r for r in rats if r not in main_rats]
    ordered_rats = main_rats + remaining_rats

    all_rows = []
    for rat in ordered_rats:
        for contrast in CONTRAST_ORDER:
            all_rows.append(build_row_data(rat, contrast, psych, glm, glm_sig))

    plain_rows = []
    latex_rows = []
    for i, rat in enumerate(ordered_rats):
        block = [d for d in all_rows if d["rat"] == rat]
        for d in block:
            plain_rows.append(_render_row(d, latex=False))
            latex_rows.append(_render_row(d, latex=True, int_digits=int_digits))
        if i == len(main_rats) - 1 and remaining_rats:
            plain_rows.append([])  # dividing rule, no text -- see NOTE_FULL for what it separates
            latex_rows.append([])  # dividing rule, no text -- see NOTE_FULL for what it separates

    plain_table = Table(
        header_rows=[HEADER_PLAIN],
        body_rows=plain_rows,
        n_cols=5,
        caption="Psychometric vs. GLM primary contrasts: full appendix comparison, all 12 rats (pooled, filtered)",
        note=NOTE_FULL,
    )
    latex_table = Table(
        header_rows=[HEADER_LATEX],
        body_rows=latex_rows,
        n_cols=5,
        caption="Psychometric vs. GLM primary contrasts: full appendix comparison, all 12 rats (pooled, filtered).",
        label="tab:comparison-full",
        note=NOTE_FULL,
    )
    return plain_table, latex_table, pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contrast-csv", type=Path, required=True)
    parser.add_argument("--glm-summary-csv", type=Path, required=True)
    parser.add_argument("--glm-sig-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    psych, glm, glm_sig = load_inputs(args.contrast_csv, args.glm_summary_csv, args.glm_sig_csv)

    # Shared integer-digit width so the point estimate and CI bounds line up
    # vertically both within a table and between the main-text and appendix
    # tables (same underlying values, just different row subsets/order).
    int_digits = compute_global_int_digits(psych, glm, glm_sig)

    # --- main-text table (15 rows) ---
    plain, latex_t, csv_df = build_main_table(psych, glm, glm_sig, int_digits=int_digits)
    (args.out_dir / "table_comparison_main.tex").write_text(table_to_latex(latex_t, fit_to_width=True))
    csv_df.to_csv(args.out_dir / "table_comparison_main_data.csv", index=False)

    # --- full appendix table (48 rows) ---
    plain, latex_t, csv_df = build_full_table(psych, glm, glm_sig, int_digits=int_digits)
    (args.out_dir / "table_comparison_full.tex").write_text(
        table_to_latex_longtable(latex_t)
    )
    csv_df.to_csv(args.out_dir / "table_comparison_full_data.csv", index=False)

    print(f"Wrote 2 tables (.tex + _data.csv each) to {args.out_dir}")


if __name__ == "__main__":
    main()


