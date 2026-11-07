"""Create and render Panel A of the pooled-to-modality GLM bridge table.

The PNG is produced by compiling LaTeX, not by recreating the table with
Matplotlib.  The script also writes a booktabs fragment for the thesis and a
native Word table using the project's table_utils.py helpers.
"""

from pathlib import Path
import shutil
import subprocess

import numpy as np
import pandas as pd

from table_utils import Cell, Table, table_to_docx


# =============================================================================
# CONFIG — edit these paths
# =============================================================================
POOLED_SUMMARY = Path("C:\\dev\\projects\\Thesis_SISSA\\Rats\\history_dependecy\\thesis_codes\\section_4_GLM\\glm_results_thesis_filtered\\glm_summary_long.csv")
MODALITY_SUMMARY = Path("C:\\dev\\projects\\Thesis_SISSA\\Rats\\history_dependecy\\glm_results_thesis_filtered_interaction\\glm_summary_long.csv")
OUT_DIR = Path("C:\\dev\\projects\\Thesis_SISSA\\Rats\\history_dependecy\\thesis_codes\\ale_thesis_fig_tab\\tables_thesis\\section_4")
REFERENCE_C = "unpenalized"


HISTORY_COEFFICIENTS = [
    "beta_angle_n-1",
    "beta_reward_n-1",
    "beta_success_n-1",
    "beta_failure_n-1",
]

LABELS = {
    "beta_angle_n-1": "Previous stimulus",
    "beta_reward_n-1": "Reward",
    "beta_success_n-1": "Post-success",
    "beta_failure_n-1": "Post-failure",
}


def resolved_by_ci(df):
    return (df["ci_low"] > 0) | (df["ci_high"] < 0)


def load_data():
    pooled = pd.read_csv(POOLED_SUMMARY).query(
        "C == @REFERENCE_C and scale == 'log_odds' and quantity in @HISTORY_COEFFICIENTS"
    ).copy()
    modality = pd.read_csv(MODALITY_SUMMARY).query(
        "C == @REFERENCE_C and scale == 'log_odds' and quantity in @HISTORY_COEFFICIENTS"
    ).copy()
    pooled["resolved"] = resolved_by_ci(pooled)
    modality["resolved"] = resolved_by_ci(modality)

    if len(pooled) != 48:
        raise ValueError(f"Expected 48 pooled rows, found {len(pooled)}")
    if len(modality) != 144:
        raise ValueError(f"Expected 144 modality rows, found {len(modality)}")
    return pooled, modality


def summarize(pooled, modality):
    rows = []
    for quantity in HISTORY_COEFFICIENTS + ["TOTAL"]:
        if quantity == "TOTAL":
            pool_scope = pooled
            mod_scope = modality
            label = "Total"
        else:
            pool_scope = pooled[pooled.quantity == quantity]
            mod_scope = modality[modality.quantity == quantity]
            label = LABELS[quantity]

        resolved_pool = pool_scope[pool_scope.resolved]
        sign_same = 0
        distribution = {0: 0, 1: 0, 2: 0, 3: 0}

        for _, pr in resolved_pool.iterrows():
            mr = modality[
                (modality.rat_id == pr.rat_id)
                & (modality.quantity == pr.quantity)
            ]
            if len(mr) != 3:
                raise ValueError(
                    f"Expected three modalities for rat {pr.rat_id}, {pr.quantity}"
                )
            sign_same += int(np.all(np.sign(mr.value) == np.sign(pr.value)))
            distribution[int(mr.resolved.sum())] += 1

        rows.append(
            {
                "label": label,
                "pooled": f"{len(resolved_pool)}/{len(pool_scope)}",
                "same_sign": f"{sign_same}/{len(resolved_pool)}",
                "modality": f"{int(mod_scope.resolved.sum())}/{len(mod_scope)}",
                "n0": str(distribution[0]),
                "n1": str(distribution[1]),
                "n2": str(distribution[2]),
                "n3": str(distribution[3]),
            }
        )
    return rows


def latex_fragment(rows):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Agreement between pooled and modality-specific GLM history coefficients}",
        r"\label{tab:pooled-modality-bridge}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Coefficient & \multicolumn{3}{c}{Overall agreement} & \multicolumn{4}{c}{Modalities resolved} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}",
        r" & Pooled resolved & Same sign in all 3 & Modality cells resolved & 0 & 1 & 2 & 3 \\",
        r"\midrule",
    ]
    for row in rows:
        vals = [
            row["label"], row["pooled"], row["same_sign"], row["modality"],
            row["n0"], row["n1"], row["n2"], row["n3"],
        ]
        if row["label"] == "Total":
            lines.append(r"\midrule")
            vals = [rf"\textbf{{{v}}}" for v in vals]
        lines.append(" & ".join(vals) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{4pt}",
        r"\par\noindent",
        r"\begin{minipage}{\linewidth}\small",
        r"\textit{Note.} Resolution was defined descriptively as an uncorrected 95\% session-bootstrap confidence interval excluding zero. ``Same sign in all 3'' was evaluated only among pooled-resolved effects. Columns 0--3 give the number of pooled-resolved effects that were separately resolved in zero, one, two, or all three modalities. These agreement summaries do not test whether modalities differ; formal differences require direct modality contrasts.",
        r"\end{minipage}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def standalone_preview(rows):
    """Standalone LaTeX source used only to render the PNG preview."""
    body = [
        r"\documentclass[border=8pt]{standalone}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{caption}",
        r"\begin{document}",
        # A fixed canvas is intentional here: the natural booktabs table is
        # wider than standalone.cls's default \textwidth.  Using \textwidth
        # would clip the last columns in the rendered preview.
        r"\begin{minipage}{720pt}",
        r"\centering",
        r"\captionof{table}{Agreement between pooled and modality-specific GLM history coefficients}",
        r"\vspace{3pt}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Coefficient & \multicolumn{3}{c}{Overall agreement} & \multicolumn{4}{c}{Modalities resolved} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}",
        r" & \shortstack{Pooled\\resolved} & \shortstack{Same sign\\in all 3} & \shortstack{Modality cells\\resolved} & 0 & 1 & 2 & 3 \\",
        r"\midrule",
    ]
    for row in rows:
        vals = [row["label"], row["pooled"], row["same_sign"], row["modality"], row["n0"], row["n1"], row["n2"], row["n3"]]
        if row["label"] == "Total":
            body.append(r"\midrule")
            vals = [rf"\textbf{{{v}}}" for v in vals]
        body.append(" & ".join(vals) + r" \\")
    body += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{6pt}",
        r"\par\small\raggedright",
        r"\textit{Note.} Resolution was defined descriptively as an uncorrected 95\% session-bootstrap confidence interval excluding zero. ``Same sign in all 3'' was evaluated only among pooled-resolved effects. Columns 0--3 give the number of pooled-resolved effects separately resolved in zero, one, two, or all three modalities. These summaries do not test whether modalities differ; formal differences require direct modality contrasts.",
        r"\end{minipage}",
        r"\end{document}",
    ]
    return "\n".join(body)


def docx_model(rows):
    headers = [
        [
            Cell("Coefficient"),
            Cell("Overall agreement", colspan=3),
            Cell("Resolved modalities per pooled-resolved effect", colspan=4),
        ],
        [
            Cell(""), Cell("Pooled resolved"), Cell("Same sign in all 3"),
            Cell("Modality cells resolved"), Cell("0"), Cell("1"), Cell("2"), Cell("3"),
        ],
    ]
    body = []
    for r in rows:
        bold = r["label"] == "Total"
        body.append([
            Cell(r["label"], bold=bold, align="left"),
            Cell(r["pooled"], bold=bold), Cell(r["same_sign"], bold=bold),
            Cell(r["modality"], bold=bold), Cell(r["n0"], bold=bold),
            Cell(r["n1"], bold=bold), Cell(r["n2"], bold=bold), Cell(r["n3"], bold=bold),
        ])
    return Table(
        header_rows=headers,
        body_rows=body,
        n_cols=8,
        caption="Agreement between pooled and modality-specific GLM history coefficients",
        note=(
            "Note. Resolution was defined descriptively as an uncorrected 95% "
            "session-bootstrap confidence interval excluding zero. 'Same sign in all 3' "
            "was evaluated only among pooled-resolved effects. Columns 0-3 give the "
            "number resolved in zero, one, two, or three modalities. Formal modality "
            "differences require direct contrasts."
        ),
    )


def render_latex_png(preview_tex_path):
    if shutil.which("pdflatex") is None or shutil.which("pdftoppm") is None:
        raise RuntimeError("Rendering requires pdflatex and pdftoppm on PATH")
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", preview_tex_path.name],
        cwd=preview_tex_path.parent,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    pdf_path = preview_tex_path.with_suffix(".pdf")
    png_prefix = preview_tex_path.with_suffix("")
    subprocess.run(
        ["pdftoppm", "-png", "-r", "220", "-singlefile", str(pdf_path), str(png_prefix)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return png_prefix.with_suffix(".png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pooled, modality = load_data()
    rows = summarize(pooled, modality)

    fragment_path = OUT_DIR / "pooled_modality_bridge_panel_a.tex"
    preview_path = OUT_DIR / "pooled_modality_bridge_panel_a_preview.tex"
    docx_path = OUT_DIR / "pooled_modality_bridge_panel_a.docx"

    fragment_path.write_text(latex_fragment(rows), encoding="utf-8")
    preview_path.write_text(standalone_preview(rows), encoding="utf-8")
    #table_to_docx(docx_model(rows), docx_path, font_size_pt=9, landscape=True)
    #png_path = render_latex_png(preview_path)

    print(f"LaTeX fragment: {fragment_path.resolve()}")
    #print(f"Rendered PNG:  {png_path.resolve()}")


if __name__ == "__main__":
    main()
