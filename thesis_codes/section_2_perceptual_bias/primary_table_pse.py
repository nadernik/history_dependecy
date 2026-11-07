"""
Section 2 — perceptual-history (3-bin) primary results table.

Builds the single consolidated LaTeX table (36 rows: rat x modality) from
the `perceptual_bias_test_revised_thesis.py` pipeline outputs, using
`table_utils.py` for booktabs rendering. Columns: Rat, Modality, N trials,
then three DeltaPSE(deg) [95% CI] columns in the order 1->3 (primary,
extreme-bin contrast, FDR-starred), 1->2 (hard->medium), 2->3
(medium->easy) — the two secondary columns are starred on the raw
uncorrected bootstrap p-value (no FDR family is defined for them).

Run: `python3 section2_build_table_via_table_utils.py`
Requires: pandas, and table_utils.py in the same directory (or on PYTHONPATH).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from table_utils import Cell, Table, table_to_latex, stars_for_q, latex_est_ci

# ---------------------------------------------------------------------------
# Config — real local paths first, falling back to the uploaded-files
# directory next to this script so it runs unchanged in either environment.
# ---------------------------------------------------------------------------
DATA_DIR_CANDIDATES = [
    Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_2_perceptual_bias\2.0_tables_perceptual_bias_revised"),
    Path(__file__).resolve().parent / "data",
]
DATA_DIR = Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_2_perceptual_bias\2.0_tables_perceptual_bias_revised")
#next((p for p in DATA_DIR_CANDIDATES if p.exists()), DATA_DIR_CANDIDATES[-1])
OUT_DIR = Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\ale_thesis_fig_tab\tables_thesis\section_2")
#OUT_DIR.mkdir(exist_ok=True, parents=True)

MOD_LABEL = {1: "Tactile", 2: "Visual", 3: "Visuotactile"}

# column order: primary (1->3) first, as the main comparison, then the two
# secondary steps
PAIR_ORDER = ["1->3", "1->2", "2->3"]
PAIR_TITLE = {
    "1->3": "$\\Delta$PSE (°) hard$\\to$easy [95\\% CI]",
    "1->2": "$\\Delta$PSE (°) hard$\\to$medium [95\\% CI]",
    "2->3": "$\\Delta$PSE (°) medium$\\to$easy [95\\% CI]",
}


def pair_code(pair_text: str) -> str:
    """Map details.csv's free-text pair label to a 1->2 / 1->3 / 2->3 code."""
    if "hard" in pair_text and "medium" in pair_text:
        return "1->2"
    if "hard" in pair_text and "easy" in pair_text:
        return "1->3"
    return "2->3"


def build_table() -> Table:
    summary = pd.read_csv(DATA_DIR / "summary_perceptual_bias.csv")
    details = pd.read_csv(DATA_DIR / "details_perceptual_bias.csv")
    qc = pd.read_csv(DATA_DIR / "analysis_qc.csv")

    details = details.copy()
    details["pair_code"] = details["pair"].apply(pair_code)
    det = details.set_index(["rat", "modality", "pair_code"])
    n_trials = qc.set_index(["rat", "modality"])["n_trials"]

    # integer-digit width per pair-column, computed once so each column's
    # own numbers line up (the three pair-columns have different typical
    # magnitudes, so a single global width would over-pad the narrow ones)
    int_digits = {}
    for pc in PAIR_ORDER:
        sub = det.xs(pc, level="pair_code")
        vals = pd.concat(
            [sub["delta_pse_hat"], sub["delta_pse_ci_lo"], sub["delta_pse_ci_hi"]]
        )
        int_digits[pc] = len(str(int(vals.abs().max())))

    header_rows = [
        [
            Cell("Rat", bold=True),
            Cell("Modality", bold=True),
            Cell("N trials", bold=True),
        ]
        + [Cell(PAIR_TITLE[pc], bold=True) for pc in PAIR_ORDER]
    ]

    body_rows = []
    rats = sorted(summary["rat"].unique())
    for i, rat in enumerate(rats):
        if i > 0:
            body_rows.append([])  # midrule between rats
        for mod in sorted(summary["modality"].unique()):
            key = (rat, mod)
            row_cells = [
                Cell(str(rat)),
                Cell(MOD_LABEL[mod]),
                Cell(f"{int(n_trials.loc[key]):,}"),
            ]
            for pc in PAIR_ORDER:
                d = det.loc[(rat, mod, pc)]
                if pc == "1->3":
                    # primary contrast: FDR-corrected q-value drives the stars
                    stars = stars_for_q(d["p_fdr_bh_primary_only"])
                else:
                    # secondary steps: no FDR family defined for these;
                    # stars here mark the RAW (uncorrected) bootstrap
                    # p-value instead
                    stars = stars_for_q(d["p_boot_unadjusted"])
                cell_text = latex_est_ci(
                    d["delta_pse_hat"], d["delta_pse_ci_lo"], d["delta_pse_ci_hi"],
                    decimals=2, stars=stars, bold=False,
                    pad_sign=True, max_stars=3, int_digits=int_digits[pc],
                )
                row_cells.append(Cell(cell_text))
            body_rows.append(row_cells)

    note = (
        "First PSE column (hard to easy) is the primary, FDR-corrected extreme-bin "
        "contrast: stars mark Benjamini-Hochberg FDR significance across the 36 "
        "primary contrasts, alpha=.05 (*** q<.001, ** q<.01, * q<.05). The other "
        "two columns (hard to medium, medium to easy) are secondary steps with no "
        "FDR family defined; their stars mark the raw, uncorrected bootstrap "
        "p-value at the same thresholds (*** p<.001, ** p<.01, * p<.05) -- treat "
        "them as descriptive, not corrected for multiple comparisons. Bootstrap: "
        "session-level, joint across bins, n=5000 (n=4999 for rat 3 modality 3 "
        "and rat 5 modality 1, where one iteration was invalid), 95 percent "
        "percentile CI. Bin ranges: hard 5-15 deg, medium 20-30 deg, easy 35-45 "
        "deg (deg = distance from the boundary 45\u00b0 in degrees)."
    )

    return Table(
        header_rows=header_rows,
        body_rows=body_rows,
        n_cols=3 + len(PAIR_ORDER),
        caption=(
            "Perceptual-history PSE shift by rat and modality, all three "
            "previous-trial-difficulty bin-pairs. First column is the primary "
            "(extreme-bin) contrast."
        ),
        label="tab:perceptual_bias_pse_all_pairs",
        note=note,
    )


if __name__ == "__main__":
    table = build_table()
    latex = table_to_latex(table, fit_to_width=True)
    out_path = OUT_DIR / "table_primary_by_modality.tex"
    out_path.write_text(latex, encoding="utf-8")
    print(f"Wrote {out_path} ({sum(1 for r in table.body_rows if r)} data rows)")