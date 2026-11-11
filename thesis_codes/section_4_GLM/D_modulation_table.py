"""Create the Section 4 D_logit modality-modulation table.

D_logit for a modality pair is
    (Delta beta_success) - (Delta beta_failure).
The difference is formed within each bootstrap replicate, preserving the
covariance between its two components. The three pairwise tests are corrected
with Holm's method separately for each rat.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shutil
import subprocess

from stats_helper import boot_p_value, holm_correction
from table_utils import Cell, Table, fmt_est_ci, stars_for_q, table_to_docx, table_to_latex


HERE = Path(__file__).resolve().parent
#INPUT_DIR = HERE / "upload"
BOOT_PATH = Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_4_GLM\fit_results\interaction_filtered\glm_bootstrap_draws.parquet")
SUMMARY_PATH = Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_4_GLM\fit_results\interaction_filtered\glm_summary_long.csv")
OUT_DIR = Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\ale_thesis_fig_tab\tables_thesis\section_4")

RATS = [4, 6, 7, 11, 13, 16]
MODALITIES = {"T": "tactile", "V": "visual", "VT": "visuotactile"}
PAIRS = [("V-T", "V", "T"), ("VT-T", "VT", "T"), ("VT-V", "VT", "V")]
QUANTITIES = ["beta_success_n-1", "beta_failure_n-1"]


def load_inputs():
    filters = [
        ("rat_id", "in", RATS), ("C", "=", "unpenalized"),
        ("kind", "=", "modality"), ("quantity", "in", QUANTITIES),
    ]
    draws = pq.read_table(BOOT_PATH, filters=filters).to_pandas()
    summary = pd.read_csv(SUMMARY_PATH)
    summary = summary[
        summary["rat_id"].isin(RATS)
        & summary["C"].eq("unpenalized")
        & summary["subset"].isin(MODALITIES.values())
        & summary["quantity"].isin(QUANTITIES)
        & summary["scale"].eq("log_odds")
    ]
    return draws, summary


def derive_results(draws, summary):
    point = summary.pivot_table(index=["rat_id", "subset"], columns="quantity", values="value")
    wide = draws.pivot_table(
        index=["rat_id", "subset_or_pair", "boot_idx"], columns="quantity", values="value"
    )
    wide["within_D"] = wide["beta_success_n-1"] - wide["beta_failure_n-1"]

    rows = []
    for rat in RATS:
        rat_rows = []
        for pair, left, right in PAIRS:
            left_name, right_name = MODALITIES[left], MODALITIES[right]
            estimate = (
                point.loc[(rat, left_name), "beta_success_n-1"]
                - point.loc[(rat, right_name), "beta_success_n-1"]
                - point.loc[(rat, left_name), "beta_failure_n-1"]
                + point.loc[(rat, right_name), "beta_failure_n-1"]
            )
            dl = wide.loc[(rat, left_name), "within_D"]
            dr = wide.loc[(rat, right_name), "within_D"]
            # Align by boot_idx explicitly before subtraction.
            ddraw = dl.sort_index().to_numpy() - dr.sort_index().to_numpy()
            low, high = np.percentile(ddraw, [2.5, 97.5])
            rat_rows.append({"rat_id": rat, "contrast": pair, "D_logit": estimate,
                             "ci_low": low, "ci_high": high,
                             "p_raw": boot_p_value(ddraw), "n_boot": len(ddraw)})
        adjusted, rejected = holm_correction([x["p_raw"] for x in rat_rows])
        for row, p_holm, reject in zip(rat_rows, adjusted, rejected):
            row["p_Holm"] = p_holm
            row["significant"] = bool(reject)
            rows.append(row)
    return pd.DataFrame(rows)


def build_table(results):
    def fmt_p(x):
        return "<.001" if x < 0.001 else f"{x:.3f}".lstrip("0")

    body = []
    for _, r in results.iterrows():
        stars = stars_for_q(r.p_Holm)
        body.append([
            Cell(str(int(r.rat_id)), align="left"), Cell(r.contrast),
            Cell(fmt_est_ci(r.D_logit, r.ci_low, r.ci_high, 3)),
            Cell(fmt_p(r.p_raw)), Cell(fmt_p(r.p_Holm) + stars),
        ])
    return Table(
        header_rows=[[Cell("Rat"), Cell("Modality contrast"), Cell("D_logit [95% CI]"),
                      Cell("p_raw"), Cell("p_Holm")]],
        body_rows=body, n_cols=5,
        caption="Modality modulation of the success-failure difference (D_logit).",
        label="tab:D-logit-modality-modulation",
        note=("D_logit = Delta beta_success - Delta beta_failure, in log-odds units. "
              "CIs and two-sided p-values are percentile-bootstrap estimates (5,000 session-level draws). "
              "Holm correction was applied across the three modality contrasts within each rat. "
              "Asterisks mark Holm-adjusted significance: * p<.05, ** p<.01, *** p<.001."),
    )


def latex_fragment(results):
    """Return the thesis-ready booktabs fragment with sign-aligned CIs."""
    def p_tex(x):
        return r"$<.001$" if x < 0.001 else f"{x:.3f}".lstrip("0")

    def num(x):
        # Reserve one sign-width for positive values so all CI brackets,
        # commas, and closing brackets align despite mixed signs.
        return f"{x:.3f}" if x < 0 else rf"\phantom{{-}}{x:.3f}"

    rows = []
    for _, r in results.iterrows():
        stars = stars_for_q(r.p_Holm)
        pair = r.contrast.replace("-", "--")
        est = rf"${num(r.D_logit)}\;[{num(r.ci_low)},\,{num(r.ci_high)}]$"
        p_holm = p_tex(r.p_Holm)
        if bool(r.significant):
            p_holm = rf"${f'{r.p_Holm:.3f}'.lstrip('0')}^{{{stars}}}$"
        rows.append(f"{int(r.rat_id)} & {pair} & {est} & {p_tex(r.p_raw)} & {p_holm} \\\\")

    return "\n".join([
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Modality modulation of the success--failure difference ($D_{\mathrm{logit}}$).}",
        r"\smallskip", r"\begin{tabular}{llccc}", r"\toprule",
        r"Rat & Modality contrast & $D_{\mathrm{logit}}$ [95\% CI] & $p_{\mathrm{raw}}$ & $p_{\mathrm{Holm}}$ \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", r"\vspace{5pt}",
        r"\footnotesize\raggedright \textit{Note.} $D_{\mathrm{logit}}=\Delta\beta_{\mathrm{success}}-\Delta\beta_{\mathrm{failure}}$, in log-odds units. CIs and two-sided $p$-values are percentile-bootstrap estimates (5,000 session-level draws). Holm correction was applied across the three modality contrasts within each rat. Asterisks indicate Holm-adjusted significance: $^{*}p<.05$, $^{**}p<.01$, $^{***}p<.001$.",
        r"\end{table}",
    ])


def render_png(results, out_path):
    """Compile a genuine booktabs LaTeX table and rasterise its PDF."""
    if not shutil.which("pdflatex") or not shutil.which("pdftoppm"):
        raise RuntimeError("Rendering the PNG requires pdflatex and pdftoppm on PATH")

    fragment = latex_fragment(results)
    # For a tightly cropped preview, replace the float wrapper with a minipage.
    preview_body = fragment.replace(r"\begin{table}[htbp]", r"\begin{minipage}{0.98\textwidth}")
    preview_body = preview_body.replace(r"\end{table}", r"\end{minipage}")
    preview_body = preview_body.replace(r"\caption{", r"\captionof{table}{")
    tex = "\n".join([
        r"\documentclass[10pt,border=10pt]{standalone}",
        r"\usepackage{booktabs}", r"\usepackage{caption}", r"\begin{document}",
        preview_body, r"\end{document}",
    ])
    render_tex = out_path.parent / "render_table.tex"
    render_tex.write_text(tex, encoding="utf-8")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", render_tex.name],
                   cwd=out_path.parent, check=True, capture_output=True, text=True)
    pdf = render_tex.with_suffix(".pdf")
    subprocess.run(["pdftoppm", "-png", "-r", "240", "-singlefile", pdf.name, out_path.stem],
                   cwd=out_path.parent, check=True, capture_output=True, text=True)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    draws, summary = load_inputs()
    results = derive_results(draws, summary)
    results.to_csv(OUT_DIR / "D_logit_modality_modulation.csv", index=False)
    table = build_table(results)
    (OUT_DIR / "D_logit_modality_modulation.tex").write_text(latex_fragment(results), encoding="utf-8")
    #table_to_docx(table, OUT_DIR / "D_logit_modality_modulation.docx")
    #render_png(results, OUT_DIR / "D_logit_modality_modulation.png")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
