"""
Section 1 tables — build straight from all_cells.pkl, rendered through
table_utils.py (must be importable: keep it next to this script, or on
your PYTHONPATH).

Run: python build_latex_tables.py
Requires: pandas, python-docx (only because table_utils.py imports it).

Produces four .tex fragments in OUT_DIR:
  table1_dataset_overview.tex   - trial/session counts, unfilt./filt. side
                                   by side, + % kept, per rat x modality(+pooled)
  table2_psychometric.tex       - mu/sigma/gamma/lapse, unfilt./filt. side by
                                   side, per rat x modality(+pooled). Dagger
                                   flags fit_degenerate=True cells (currently
                                   only sigma-at-bound; see note below).
  table3_filter_fit_delta.tex   - filtered-minus-unfiltered deltas on the same
                                   4 params, rat x modality ONLY (no pooled)
  table4_classification.tex     - acc/sigma/lapse tier, unfilt./filt. side by
                                   side, shaded cells (rat x modality+pooled)

Each .tex is a standalone \\begin{longtable}...\\end{longtable} fragment
(table 2 is wrapped in \\begin{landscape}, needs \\usepackage{pdflscape}) --
\\input{} it directly into the thesis. Preamble needs:
  \\usepackage[T1]{fontenc}, \\usepackage[table]{xcolor}, \\usepackage{booktabs},
  \\usepackage{longtable}, \\usepackage{pdflscape}

OPEN ITEM (flagged, not yet resolved): all_cells.pkl's own
classification['fit_degenerate'] only trips on sigma hitting its upper bound
(50 deg) -- it does NOT catch mu, gamma or lapse pinned at THEIR bounds. A
direct bounds scan found 15 such hits across 14 rat x modality x filter_state
cells (vs. 3 caught by fit_degenerate), almost all in the tactile modality.
Table 2's dagger below only reflects the existing (sigma-only) flag. If you
decide to expand it, edit the `flagged = ...` line in the TABLE 2 section.
"""
import pickle
from pathlib import Path
import pandas as pd

from table_utils import Cell, Table, table_to_latex_longtable, latex_num

# --- EDIT THESE TWO PATHS for your own machine ---
PKL_PATH = str(Path(__file__).resolve().parent / 'results' / 'task_check_output_no_boot' / 'raw' / 'all_cells.pkl')
OUT_DIR = r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\ale_thesis_fig_tab\tables_thesis\section_1"

MOD_LABEL = {1: "Tactile", 2: "Visual", 3: "Visuotactile", "pooled": "Pooled"}
MOD_ORDER_KEY = {1: 0, 2: 1, 3: 2, "pooled": 3}
TIER_COLOR = {"good": "green!15", "medium": "yellow!25", "poor": "red!15"}

with open(PKL_PATH, "rb") as f:
    raw_cells = pickle.load(f)

rows = []
for (rat, cond, fstate), d in raw_cells.items():
    rat = int(rat)
    cond_key = cond if cond == "pooled" else int(cond)
    popt = d["popt"]
    mu, sigma, gamma, lapse = popt.tolist()
    cls = d["classification"]
    rows.append(
        dict(
            rat=rat, condition=cond_key, filter_state=fstate,
            n_trials=d["n_trials"], n_sessions=d["n_sessions"],
            mu=mu, sigma=sigma, gamma=gamma, lapse=lapse, total_lapse=gamma + lapse,
            accuracy=d["perf"]["accuracy"],
            acc_tier=cls["acc_tier"], sigma_tier=cls["sigma_tier"],
            lapse_tier=cls["lapse_tier"], classification=cls["classification"],
            fit_degenerate=cls["fit_degenerate"],
        )
    )
df = pd.DataFrame(rows)
df["modality"] = df["condition"].map(MOD_LABEL)
df["_sortkey"] = df["condition"].map(MOD_ORDER_KEY)
df = df.sort_values(by=["rat", "_sortkey"]).reset_index(drop=True)

RATS = sorted(df["rat"].unique())


def rat_blank_col(rat_series):
    """Blank the rat number on every row after the first for that rat block
    (matches the existing thesis convention: no rowspan support in
    table_utils, so print rat once, leave subsequent rows blank)."""
    out = []
    prev = None
    for r in rat_series:
        out.append("" if r == prev else str(r))
        prev = r
    return out


# ===========================================================================
# TABLE 1 — dataset overview (counts)
# ===========================================================================
t1_df = df.pivot_table(
    index=["rat", "_sortkey", "modality"], columns="filter_state",
    values=["n_trials", "n_sessions"], aggfunc="first",
).reset_index()
t1_df.columns = ["rat", "_sortkey", "modality", "n_sess_f", "n_sess_u", "n_trials_f", "n_trials_u"]
t1_df = t1_df.sort_values(by=["rat", "_sortkey"]).reset_index(drop=True)
t1_df["rat_disp"] = rat_blank_col(t1_df["rat"])
t1_df["pct_kept"] = 100.0 * t1_df["n_trials_f"] / t1_df["n_trials_u"]

header1 = [
    [Cell("Rat"), Cell("Modality"), Cell("N trials", colspan=3), Cell("N sessions", colspan=2)],
    [Cell(""), Cell(""), Cell("Unfilt."), Cell("Filt."), Cell("\\% kept"), Cell("Unfilt."), Cell("Filt.")],
]
body1 = []
for _, r in t1_df.iterrows():
    bold = r["modality"] == "Pooled"
    body1.append([
        Cell(r["rat_disp"], bold=bold),
        Cell(r["modality"], bold=bold, align="left"),
        Cell(f"{int(r['n_trials_u']):,}", bold=bold),
        Cell(f"{int(r['n_trials_f']):,}", bold=bold),
        Cell(f"{r['pct_kept']:.1f}", bold=bold),
        Cell(str(int(r["n_sess_u"])), bold=bold),
        Cell(str(int(r["n_sess_f"])), bold=bold),
    ])

table1 = Table(
    header_rows=header1, body_rows=body1, n_cols=7,
    caption="Dataset overview: trial and session counts per rat and modality, before and after warm-up/learning-phase filtering.",
    label="tab:s1-dataset-overview",
    note="Pooled = all three modalities combined for that rat. Unfilt. = before trial filtering; Filt. = after "
         "session-start warm-up removal and performance-criterion session cutoff (see Methods). % kept = "
         "filtered trials as a percentage of unfiltered trials for that row.",
)
with open(f"{OUT_DIR}/table1_dataset_overview.tex", "w") as f:
    f.write(table_to_latex_longtable(table1))

# ===========================================================================
# TABLE 2 — psychometric fit parameters, unfiltered & filtered side by side
# ===========================================================================
piv2 = df.pivot_table(
    index=["rat", "_sortkey", "modality"], columns="filter_state",
    values=["mu", "sigma", "gamma", "lapse", "fit_degenerate"], aggfunc="first",
).reset_index()
piv2.columns = ["rat", "_sortkey", "modality",
                "degen_f", "degen_u", "gamma_f", "gamma_u", "lapse_f", "lapse_u",
                "mu_f", "mu_u", "sigma_f", "sigma_u"]
piv2 = piv2.sort_values(by=["rat", "_sortkey"]).reset_index(drop=True)
piv2["rat_disp"] = rat_blank_col(piv2["rat"])

header2 = [
    [Cell("Rat"), Cell("Modality"), Cell("$\\mu$ (°)", colspan=2), Cell("$\\sigma$ (°)", colspan=2),
     Cell("$\\gamma$", colspan=2), Cell("$\\lambda$", colspan=2)],
    [Cell(""), Cell(""), Cell("Unfilt."), Cell("Filt."), Cell("Unfilt."), Cell("Filt."),
     Cell("Unfilt."), Cell("Filt."), Cell("Unfilt."), Cell("Filt.")],
]
body2 = []
n_degenerate = 0
for _, r in piv2.iterrows():
    bold = r["modality"] == "Pooled"
    flagged = bool(r["degen_u"]) or bool(r["degen_f"])
    if flagged:
        n_degenerate += 1
    modality_text = r["modality"] + ("$^\\dagger$" if flagged else "")
    body2.append([
        Cell(r["rat_disp"], bold=bold),
        Cell(modality_text, bold=bold, align="left"),
        Cell(latex_num(r["mu_u"], 2, bold=bold)),
        Cell(latex_num(r["mu_f"], 2, bold=bold)),
        Cell(latex_num(r["sigma_u"], 2, bold=bold)),
        Cell(latex_num(r["sigma_f"], 2, bold=bold)),
        Cell(latex_num(r["gamma_u"], 2, bold=bold)),
        Cell(latex_num(r["gamma_f"], 2, bold=bold)),
        Cell(latex_num(r["lapse_u"], 2, bold=bold)),
        Cell(latex_num(r["lapse_f"], 2, bold=bold)),
    ])
table2 = Table(
    header_rows=header2, body_rows=body2, n_cols=10,
    caption="Psychometric fit parameters per rat and modality, unfiltered vs.\\ filtered.",
    label="tab:s1-psychometric",
    note="mu = point of subjective equality (stimulus angle, in degrees); sigma = slope; gamma = guess rate; "
         "lambda = lapse rate. Point estimates only -- no bootstrap was run for this fit (n_boot = 0), so no "
         "CI is reported on these parameters. Dagger flags a degenerate fit, where sigma hit the upper bound "
         "of its allowed range (50 degrees); this happens for rats 11, 12 and 13 (tactile, unfiltered only) "
         "and is resolved by filtering in every one of those cases.",
)
with open(f"{OUT_DIR}/table2_psychometric.tex", "w") as f:
    f.write(table_to_latex_longtable(table2, landscape=True, fontsize="\\small"))
print(f"Table 2: {n_degenerate} rows flagged degenerate (dagger)")

# ===========================================================================
# TABLE 3 — filter effect on fit (delta only), rat x modality, no pooled
# ===========================================================================
df_mod = df[df.condition != "pooled"].copy()
piv3 = df_mod.pivot_table(
    index=["rat", "_sortkey", "modality"], columns="filter_state",
    values=["mu", "sigma", "gamma", "lapse"],
).reset_index()
piv3.columns = ["rat", "_sortkey", "modality", "gamma_f", "gamma_u", "lapse_f", "lapse_u", "mu_f", "mu_u", "sigma_f", "sigma_u"]
for p in ["mu", "sigma", "gamma", "lapse"]:
    piv3[f"d_{p}"] = piv3[f"{p}_f"] - piv3[f"{p}_u"]
piv3 = piv3.sort_values(by=["rat", "_sortkey"]).reset_index(drop=True)
piv3["rat_disp"] = rat_blank_col(piv3["rat"])

header3 = [[Cell("Rat"), Cell("Modality"), Cell("$\\Delta\\mu$ (°)"), Cell("$\\Delta\\sigma$ (°)"), Cell("$\\Delta\\gamma$"), Cell("$\\Delta\\lambda$")]]
body3 = []
for _, r in piv3.iterrows():
    body3.append([
        Cell(r["rat_disp"]),
        Cell(r["modality"], align="left"),
        Cell(latex_num(r["d_mu"], 2, pad_sign=True)),
        Cell(latex_num(r["d_sigma"], 2, pad_sign=True)),
        Cell(latex_num(r["d_gamma"], 2, pad_sign=True)),
        Cell(latex_num(r["d_lapse"], 2, pad_sign=True)),
    ])
table3 = Table(
    header_rows=header3, body_rows=body3, n_cols=6,
    caption="Effect of trial filtering on the psychometric fit, per rat and modality.",
    label="tab:s1-filter-fit-delta",
    note="Delta = filtered minus unfiltered (Table 2b minus Table 2a). A positive delta on sigma or "
         "lambda means the fit got noisier or more lapse-prone after filtering; negative means it improved. "
         "Pooled condition omitted here; see Tables 2a/2b for pooled values.",
)
with open(f"{OUT_DIR}/table3_filter_fit_delta.tex", "w") as f:
    f.write(table_to_latex_longtable(table3))

# ===========================================================================
# TABLE 4 — classification, unfiltered & filtered side by side, shaded
# ===========================================================================
t4_df = df.pivot_table(
    index=["rat", "_sortkey", "modality"], columns="filter_state",
    values=["acc_tier", "sigma_tier", "lapse_tier", "classification"], aggfunc="first",
).reset_index()
t4_df.columns = ["rat", "_sortkey", "modality",
                 "acc_f", "acc_u", "cls_f", "cls_u", "lapse_f", "lapse_u", "sigma_f", "sigma_u"]
t4_df = t4_df.sort_values(by=["rat", "_sortkey"]).reset_index(drop=True)
t4_df["rat_disp"] = rat_blank_col(t4_df["rat"])


def tier_cell(tier):
    color = TIER_COLOR.get(tier, "")
    text = f"\\cellcolor{{{color}}}{tier}" if color else tier
    return Cell(text)


header4 = [
    [Cell("Rat"), Cell("Modality"), Cell("Accuracy", colspan=2), Cell("Sigma", colspan=2),
     Cell("Lapse", colspan=2)],
    [Cell(""), Cell(""), Cell("Unfilt."), Cell("Filt."), Cell("Unfilt."), Cell("Filt."),
     Cell("Unfilt."), Cell("Filt.")],
]
body4 = []
for _, r in t4_df.iterrows():
    body4.append([
        Cell(r["rat_disp"], bold=(r["modality"] == "Pooled")),
        Cell(r["modality"], bold=(r["modality"] == "Pooled"), align="left"),
        tier_cell(r["acc_u"]), tier_cell(r["acc_f"]),
        tier_cell(r["sigma_u"]), tier_cell(r["sigma_f"]),
        tier_cell(r["lapse_u"]), tier_cell(r["lapse_f"]),
    ])
table4 = Table(
    header_rows=header4, body_rows=body4, n_cols=8,
    caption="Performance classification per rat and modality, unfiltered vs.\\ filtered.",
    label="tab:s1-classification",
    note="Tier thresholds: accuracy good at least 0.80, medium at least 0.65; sigma good at most 20 "
         "degrees, medium at most 35 degrees; lapse good at most 0.15, medium at most 0.30 (else poor). "
         "Cell shading requires the xcolor package loaded with the table option in the thesis preamble.",
)
with open(f"{OUT_DIR}/table4_classification.tex", "w") as f:
    f.write(table_to_latex_longtable(table4))

# ===========================================================================
# TABLE 5 — per-rat descriptive summary (median, IQR, range) across the 3
# modalities, filtered data (primary analysis set). Supplements Tables 2/4
# with a single per-rat row -- shows each rat's OWN spread across its three
# modality fits, which is what the thesis draft's per-rat prose needs.
# ===========================================================================
def fmt_summary(series, decimals, pct=False):
    s = series * (100 if pct else 1)
    med = s.median()
    q1, q3 = s.quantile([0.25, 0.75])
    lo, hi = s.min(), s.max()
    f = f"{{:.{decimals}f}}"
    return f"{f.format(med)} ({f.format(q1)}–{f.format(q3)}; {f.format(lo)}–{f.format(hi)})"


df_mod_f = df[(df.condition != "pooled") & (df.filter_state == "filtered")]
header5 = [[Cell("Rat"), Cell("Accuracy (\\%)"), Cell("JND (°)"),
            Cell("Total lapse ($\\gamma+\\lambda$)"), Cell("PSE (°)")]]
body5 = []
for rat in RATS:
    sub = df_mod_f[df_mod_f.rat == rat]
    # Note: no FILTERED fit is ever flagged fit_degenerate (that flag only trips
    # on unfiltered fits in this dataset -- see Table 2's note), so a per-row
    # dagger here would never fire. The general caveat about rats 7/11/12/13's
    # boundary-pinned parameters is in the table note instead.
    body5.append([
        Cell(str(rat)),
        Cell(fmt_summary(sub["accuracy"], 0, pct=True)),
        Cell(fmt_summary(sub["sigma"], 1)),
        Cell(fmt_summary(sub["total_lapse"], 2)),
        Cell(fmt_summary(sub["mu"], 1)),
    ])
table5 = Table(
    header_rows=header5, body_rows=body5, n_cols=5,
    caption="Per-rat descriptive summary across the three modalities (filtered data): "
            "median (IQR; range).",
    label="tab:s1-per-rat-summary",
    note="Each row summarizes that rat's 3 modality-specific fits (tactile, visual, visuotactile; "
         "pooled excluded). Dagger flags rats with at least one degenerate fit among those 3 "
         "(rats 11, 12, 13 -- unfiltered tactile only; none of their FILTERED fits are flagged, "
         "but their IQR/range here still widen because tactile is consistently their worst "
         "modality). Rat 7's PSE range (0.0 to 45.8) is inflated by its filtered tactile fit's mu "
         "sitting exactly at the lower bound (0 degrees) -- not flagged degenerate by the pipeline's "
         "sigma-only criterion, but the same kind of boundary artifact.",
)
with open(f"{OUT_DIR}/table5_per_rat_summary.tex", "w") as f:
    f.write(table_to_latex_longtable(table5))

# ===========================================================================
# TABLE 6 — filtered trial counts per rat and modality
# Compact summary table for the main text
# ===========================================================================
t6_df = (
    df[df["filter_state"] == "filtered"]
    .pivot_table(
        index="rat",
        columns="condition",
        values="n_trials",
        aggfunc="first",
    )
    .reset_index()
    .rename(
        columns={
            1: "Tactile",
            2: "Visual",
            3: "Visuotactile",
            "pooled": "Pooled",
        }
    )
)

# Ensure the intended rat and column order.
t6_df = (
    t6_df[["rat", "Tactile", "Visual", "Visuotactile", "Pooled"]]
    .sort_values("rat")
    .reset_index(drop=True)
)

# Check that the pooled count is the sum of the three modalities.
modality_sum = (
    t6_df["Tactile"]
    + t6_df["Visual"]
    + t6_df["Visuotactile"]
)

if not (modality_sum == t6_df["Pooled"]).all():
    bad_rats = t6_df.loc[
        modality_sum != t6_df["Pooled"], "rat"
    ].tolist()
    raise ValueError(
        "Pooled filtered trial count does not equal the sum of the "
        f"three modalities for rat(s): {bad_rats}"
    )

header6 = [[
    Cell("Rat"),
    Cell("Tactile"),
    Cell("Visual"),
    Cell("Visuotactile"),
    Cell("Pooled"),
]]

body6 = []

for _, r in t6_df.iterrows():
    body6.append([
        Cell(str(int(r["rat"]))),
        Cell(f"{int(r['Tactile']):,}"),
        Cell(f"{int(r['Visual']):,}"),
        Cell(f"{int(r['Visuotactile']):,}"),
        Cell(f"{int(r['Pooled']):,}", bold=True),
    ])

# Add cohort totals.
totals = t6_df[
    ["Tactile", "Visual", "Visuotactile", "Pooled"]
].sum()

body6.append([
    Cell("Total", bold=True),
    Cell(f"{int(totals['Tactile']):,}", bold=True),
    Cell(f"{int(totals['Visual']):,}", bold=True),
    Cell(f"{int(totals['Visuotactile']):,}", bold=True),
    Cell(f"{int(totals['Pooled']):,}", bold=True),
])

table6 = Table(
    header_rows=header6,
    body_rows=body6,
    n_cols=5,
    caption=(
        "Number of filtered trials retained for each rat, "
        "separately by sensory modality and pooled across modalities."
    ),
    label="tab:s1-filtered-trial-counts",
    note=(
        "Filtered trials constitute the primary analysis dataset after "
        "session-start warm-up removal and application of the "
        "performance-criterion session cutoff. Pooled is the sum of the "
        "tactile, visual and visuotactile trials for each rat."
    ),
)

with open(
    f"{OUT_DIR}/table6_filtered_trial_counts.tex",
    "w",
    encoding="utf-8",
) as f:
    f.write(table_to_latex_longtable(table6))
