"""
Shared rendering helpers: build a table once as plain data, render it to
both a LaTeX (booktabs) fragment and a native Word table (python-docx),
so the two output formats never drift apart.

Table model
-----------
A table is:
  - `header_rows`: list of header rows, each a list of Cell (for simple
    single-row headers, just one row; use two rows for grouped headers
    with a top label spanning several sub-columns via `colspan`).
  - `body_rows`: list of body rows, each a list of Cell.

Cell is a small dataclass: text, bold, colspan. Kept deliberately simple —
no cell needs more than "this text, maybe bold, maybe spanning N columns"
for the tables in this project.

Significance convention (fixed across every table in this project):
  bold the cell AND append asterisks — *** q<.001, ** q<.01, * q<.05 —
  computed from a q-value with `stars_for_q`.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt


@dataclass
class Cell:
    text: str
    bold: bool = False
    colspan: int = 1
    align: str = "center"  # "center" | "left" | "right"


@dataclass
class Table:
    header_rows: List[List[Cell]]
    body_rows: List[List[Cell]]
    n_cols: int
    caption: str = ""
    label: str = ""  # LaTeX \label{...} key, e.g. "tab:glm-coef-primary"
    note: str = ""  # footnote text under the table (both formats)


def stars_for_q(q: Optional[float]) -> str:
    """*** q<.001, ** q<.01, * q<.05, '' otherwise. `q=None` -> ''."""
    if q is None:
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def fmt_est_ci(value: float, ci_low: float, ci_high: float, decimals: int = 3) -> str:
    """Plain-text 'value [ci_low, ci_high]', fixed decimals, for docx/CSV."""
    return f"{value:.{decimals}f} [{ci_low:.{decimals}f}, {ci_high:.{decimals}f}]"


# ---------------------------------------------------------------------------
# LaTeX rendering (booktabs; no siunitx, no amsmath — only core commands)
# ---------------------------------------------------------------------------

_LATEX_ESCAPE = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
}


def _latex_escape_plain(s: str) -> str:
    """Escape LaTeX special characters in plain (non-math) text."""
    out = []
    for ch in s:
        out.append(_LATEX_ESCAPE.get(ch, ch))
    return "".join(out)


def latex_num(
    x: float,
    decimals: int = 3,
    stars: str = "",
    bold: bool = False,
    pad_sign: bool = False,
    max_stars: int = 0,
    int_digits: int = 0,
) -> str:
    """Number formatted in math mode so '-' renders as a proper minus sign.
    Stars (already-computed significance markers) are added as a superscript
    inside the same math group, e.g. '$0.144^{***}$'. `bold=True` wraps the
    whole math group in \\mathbf so it renders bold in the PDF — bolding is
    applied here, at construction time, rather than by post-processing the
    string later (a cell can contain several separate $...$ groups, e.g. an
    estimate plus a CI, so string-level bold-wrapping after the fact is
    unreliable).

    `pad_sign=True` prepends \\hphantom{-} to non-negative numbers, so a
    positive and a negative number in the same column occupy the same
    width. Without this, a missing minus sign shifts everything that
    follows (notably the CI's opening bracket) slightly right for positive
    rows, so a column with a mix of signs reads as misaligned even though
    every individual number is typeset correctly — this is the fix for
    that.

    `int_digits>0` additionally left-pads the integer part with invisible
    \\hphantom{0...} up to that many digits, so e.g. "9.17" and "17.13" in
    the same column both reserve 2 integer-digit slots and their decimal
    points (and, in a CI, the "[", "," and "]") line up vertically. Without
    this, `pad_sign` alone only fixes the sign column; numbers that differ
    in magnitude (1 digit vs. 2 digits before the decimal) still shift the
    rest of the row, which is what actually causes CI brackets to look
    misaligned down a column of mixed-magnitude values. Pass the same
    `int_digits` to every number sharing a column (typically computed once
    from the max integer-digit width across the whole table) so the
    padding is consistent.

    `max_stars>0` does the same for the significance-stars superscript:
    pads with invisible \\hphantom{*} up to that many characters, so a cell
    with one star and a cell with three stars still end at the same
    horizontal position (otherwise the varying star count reintroduces the
    same misalignment sign-padding just fixed, just one token later)."""
    formatted = f"{abs(x):.{decimals}f}"
    int_str, frac_str = formatted.split(".")
    pad_int = max(0, int_digits - len(int_str))
    int_pad = r"\hphantom{" + "0" * pad_int + "}" if pad_int > 0 else ""
    if x < 0:
        sign_disp = "-"
    elif pad_sign:
        sign_disp = r"\hphantom{-}"
    else:
        sign_disp = ""
    body = f"{sign_disp}{int_pad}{int_str}.{frac_str}"
    if max_stars > 0:
        pad = max_stars - len(stars)
        stars_disp = stars + (r"\hphantom{" + "*" * pad + "}" if pad > 0 else "")
    else:
        stars_disp = stars
    inner = f"{body}^{{{stars_disp}}}" if stars_disp else body
    if bold:
        inner = rf"\mathbf{{{inner}}}"
    return f"${inner}$"


def latex_est_ci(
    value: float,
    ci_low: float,
    ci_high: float,
    decimals: int = 3,
    stars: str = "",
    bold: bool = False,
    pad_sign: bool = False,
    max_stars: int = 0,
    int_digits: int = 0,
) -> str:
    """'$\\mathbf{0.144^{***}}$ [$\\mathbf{0.064}$, $\\mathbf{0.225}$]' when
    bold=True — each number in its own math group so minus signs render
    correctly throughout, and bold (when significant) covers the estimate
    and both CI bounds. `pad_sign`/`max_stars`/`int_digits` are forwarded to
    every number in the triple (see `latex_num`) — the CI bounds never
    carry stars themselves, so `max_stars` only applies to the point
    estimate. Passing `int_digits` here keeps the point estimate and both
    CI bounds at the same integer-digit width as each other, in addition to
    lining up with the same-column values in other rows."""
    v = latex_num(
        value, decimals, stars, bold=bold, pad_sign=pad_sign, max_stars=max_stars, int_digits=int_digits
    )
    lo = latex_num(ci_low, decimals, bold=bold, pad_sign=pad_sign, int_digits=int_digits)
    hi = latex_num(ci_high, decimals, bold=bold, pad_sign=pad_sign, int_digits=int_digits)
    return f"{v} [{lo}, {hi}]"


def table_to_latex(t: Table, fit_to_width: bool = False) -> str:
    """`fit_to_width=True` wraps the tabular in \\resizebox{\\linewidth}{!}{...}
    (requires \\usepackage{graphicx} — near-universal, usually already loaded
    for figures) so a table with many/wide columns is shrunk to fit the page
    instead of running off the right margin. Use it for tables you've seen
    overflow; leave it off for tables that already fit (resizebox would just
    stretch them to full width, changing font size for no reason)."""
    col_spec = "l" + "c" * (t.n_cols - 1)
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    if t.caption:
        lines.append(rf"\caption{{{_latex_escape_plain(t.caption)}}}")
    if t.label:
        lines.append(rf"\label{{{t.label}}}")
    if fit_to_width:
        lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    for hr in t.header_rows:
        lines.append(_latex_row(hr) + r" \\")
    lines.append(r"\midrule")
    for br in t.body_rows:
        if not br:  # empty row = plain dividing rule between sub-groups, no text
            lines.append(r"\midrule")
        else:
            lines.append(_latex_row(br) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if fit_to_width:
        lines.append(r"}")
    if t.note:
        # \par + an explicit \linewidth-wide minipage forces a real
        # paragraph break and constrains the note to wrap under the full
        # table width. Without this, text placed right after \end{tabular}
        # can end up sharing the table's last line box instead of starting
        # a fresh line below it, which is what produced the note running
        # off to the side instead of appearing under the table.
        lines.append(r"\vspace{4pt}")
        lines.append(r"\par\noindent")
        lines.append(
            r"\begin{minipage}{\linewidth}\small "
            + _latex_escape_plain(t.note)
            + r"\end{minipage}"
        )
    lines.append(r"\end{table}")
    return "\n".join(lines)


def table_to_latex_longtable(
    t: Table,
    landscape: bool = False,
    fontsize: str = r"\small",
    tabcolsep_pt: float = None,
) -> str:
    """Same table model, rendered with the `longtable` package instead of
    `table`+`tabular`, for bodies long enough to span a page (e.g. the
    36-row ridge-sensitivity table or the 48-row appendix comparison
    table). Requires \\usepackage{longtable}.

    `landscape=True` additionally wraps the table in a `landscape`
    environment (requires \\usepackage{pdflscape}) and applies `fontsize`
    (default \\small) — use this for tables with enough columns that they
    won't fit a portrait page even in a small font (a `resizebox` trick
    like the one used for the non-long tables doesn't work with longtable,
    since longtable's whole point is breaking across pages, which a fixed-
    size resized box can't do).

    `tabcolsep_pt`, if set, shrinks LaTeX's default 6pt \\tabcolsep (the
    padding added on both sides of every column). This is a fixed length,
    not scaled by font size — so at \\scriptsize or smaller it's often the
    single biggest remaining contributor to a table still not fitting the
    page, well past the point where shrinking the font further stops
    helping much.
    """
    col_spec = "l" + "c" * (t.n_cols - 1)
    lines = []
    if landscape:
        lines.append(r"\begin{landscape}")
        lines.append(fontsize)
    if tabcolsep_pt is not None:
        lines.append(rf"\setlength{{\tabcolsep}}{{{tabcolsep_pt}pt}}")
    lines.append(rf"\begin{{longtable}}{{{col_spec}}}")
    if t.caption:
        lines.append(rf"\caption{{{_latex_escape_plain(t.caption)}}}")
    if t.label:
        lines.append(rf"\label{{{t.label}}} \\")
    lines.append(r"\toprule")
    for hr in t.header_rows:
        lines.append(_latex_row(hr) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\multicolumn{" + str(t.n_cols) + r"}{l}{\textit{(continued)}} \\")
    lines.append(r"\toprule")
    for hr in t.header_rows:
        lines.append(_latex_row(hr) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    lines.append(r"\bottomrule")
    lines.append(r"\endfoot")
    # \endlastfoot's content (shown only after the final body row, on the
    # last page) must be declared here, BEFORE the body rows — longtable
    # replays whatever sits between \endfoot and \endlastfoot only once,
    # at the very end, but the body rows themselves always come after this
    # block, not before it. The note is deliberately NOT put here as a
    # `\multicolumn{n}{p{\linewidth}}{...}` row: `\linewidth` inside the
    # longtable is the surrounding page's full text width, and a p-column
    # cell that wide forces the whole table (and therefore its last column,
    # which absorbs whatever width the other columns' natural sizes don't
    # already account for) to stretch out to match — pushing the last
    # column visibly away from the rest of the table and making the table
    # wider than its content needs. Rendering the note as a separate
    # minipage after \end{longtable} instead (matching table_to_latex's
    # non-long version) keeps the table's own width governed purely by its
    # cell content.
    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")
    for br in t.body_rows:
        if not br:  # empty row = plain dividing rule between sub-groups, no text
            lines.append(r"\midrule")
        else:
            lines.append(_latex_row(br) + r" \\")
    lines.append(r"\end{longtable}")
    if landscape:
        lines.append(r"\end{landscape}")
    if t.note:
        lines.append(r"\vspace{4pt}")
        lines.append(r"\par\noindent")
        lines.append(
            r"\begin{minipage}{\linewidth}\small "
            + _latex_escape_plain(t.note)
            + r"\end{minipage}"
        )
    return "\n".join(lines)


def _latex_row(cells: List[Cell]) -> str:
    parts = []
    for c in cells:
        text = c.text if c.text.startswith("$") or "\\" in c.text else _latex_escape_plain(c.text)
        # Math-mode cells (built by latex_num/latex_est_ci) carry their own
        # bold via \mathbf, applied at construction time — see table_utils
        # docstring. Only plain-text cells (e.g. the "Rat" label column) are
        # bolded here.
        if c.bold and not text.startswith("$"):
            text = rf"\textbf{{{text}}}"
        if c.colspan > 1:
            align = {"center": "c", "left": "l", "right": "r"}[c.align]
            text = rf"\multicolumn{{{c.colspan}}}{{{align}}}{{{text}}}"
        parts.append(text)
    return " & ".join(parts)


# ---------------------------------------------------------------------------
# DOCX rendering (native Word table, for copy-paste into Google Docs)
# ---------------------------------------------------------------------------

def table_to_docx(
    t: Table, out_path: Path, font_size_pt: int = 9, landscape: bool = False
) -> None:
    """`landscape=True` sets the page to landscape orientation (swapping
    page width/height) before the table is added — useful for a table with
    enough columns that portrait width forces awkward mid-number wrapping.
    This only matters if you open/print the .docx itself; when you copy the
    table out and paste it into an existing Google Doc, the doc's own page
    orientation applies instead, so this is a "look right if opened
    standalone" convenience, not something that carries through the paste.
    """
    doc = Document()

    if landscape:
        section = doc.sections[0]
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width, section.page_height = section.page_height, section.page_width

    if t.caption:
        p = doc.add_paragraph()
        run = p.add_run(t.caption)
        run.bold = True
        run.font.size = Pt(font_size_pt + 1)

    n_header_rows = len(t.header_rows)
    n_rows = n_header_rows + len(t.body_rows)
    table = doc.add_table(rows=n_rows, cols=t.n_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"

    def write_row(row_idx: int, cells: List[Cell], header: bool):
        col = 0
        for c in cells:
            docx_cell = table.cell(row_idx, col)
            if c.colspan > 1:
                merge_target = table.cell(row_idx, col + c.colspan - 1)
                docx_cell = docx_cell.merge(merge_target)
            docx_cell.text = ""
            p = docx_cell.paragraphs[0]
            p.alignment = {
                "center": WD_ALIGN_PARAGRAPH.CENTER,
                "left": WD_ALIGN_PARAGRAPH.LEFT,
                "right": WD_ALIGN_PARAGRAPH.RIGHT,
            }[c.align]
            run = p.add_run(c.text)
            run.bold = c.bold or header
            run.font.size = Pt(font_size_pt)
            col += c.colspan

    for i, hr in enumerate(t.header_rows):
        write_row(i, hr, header=True)
    for j, br in enumerate(t.body_rows):
        write_row(n_header_rows + j, br, header=False)

    if t.note:
        p = doc.add_paragraph()
        run = p.add_run(t.note)
        run.italic = True
        run.font.size = Pt(font_size_pt - 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(out_path))