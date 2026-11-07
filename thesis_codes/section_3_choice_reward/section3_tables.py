"""
section3_tables.py
=====================================
Rebuilds the three section-3 LaTeX tables using Alessandro's own shared
`table_utils.py` (Table/Cell data model + latex_num/latex_est_ci/stars_for_q),
the same rendering pipeline already used for the section-4 GLM tables, so
every table in the thesis is stylistically consistent and never hand-strung
together with ad hoc f-strings.

Same three tables, same underlying numbers as `section3_supplementary_tables.py`
(not recomputed, not re-verified -- this only changes HOW they're rendered):

  1. table_omnibus_2df_3df.tex   -- omnibus 2-df / 3-df tests, all rats.
  2. table_bootstrap_qc.tex      -- sigma-degenerate / gamma-lapse-near-ceiling
                                     census per (rat, cell), all rats.
  3. table_sigma_filter_shift.tex -- median/CI shift under sigma-only bootstrap
                                      filtering, all rats.

What changed vs. the ad hoc version, to match the project's established
table_utils.py convention (see section_4_q2_q3_tables_built.md /
section_4_pooled_glm_tables_plan.md):

  - Significance stars now use the project's real three-tier `stars_for_q`
    (*** q<.001, ** q<.01, * q<.05), not a single-tier "*" if q<.05. This is
    a real, visible correction in Table 1: rat 11's q_BH = 0.002 now renders
    as ** (it's < .01), not * as the earlier ad hoc script showed.
  - Stars render as a LaTeX superscript directly on the number
    ($0.002^{**}$), not a separate trailing asterisk column -- removes the
    two blank "sig" columns Table 1 used to carry.
  - CI-bracket alignment: `pad_sign` + `int_digits` applied to every numeric
    column (T-statistics, q-values, deltas, CI bounds), computed once per
    column from the actual data, matching the fix already applied to every
    other table in the thesis.
  - Shift values in Table 3 (e.g. "+0.01", "-0.21") are now wrapped in LaTeX
    math mode so the minus sign renders as a real minus, not a hyphen --
    the earlier ad hoc script used an unwrapped f-string there.
  - Bare "<" (Table 1's "$<$.001", Table 2's "sigma<3") no longer needs the
    \\!<\\! math-mode workaround -- \\usepackage[T1]{fontenc} (already
    required elsewhere in the thesis for the same reason) fixes the OT1
    ligature bug directly, so this version emits plain "<" text.
  - `table_to_latex`'s built-in "empty row = \\midrule, no text" convention
    is used for the focus/rest rat grouping instead of manually appending a
    raw `\\midrule` string.

CONFIG below points at Alessandro's real local directories by default; this
run used the uploaded copies (see UPLOADS_DIR override).
"""
import os
import numpy as np
import pandas as pd

import bootstrap_qc as qc
from table_utils import Cell, Table, table_to_latex, stars_for_q, latex_num, latex_est_ci

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DATA_DIR = r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_3_choice_reward\choice_reward_joint_results\filtered\pooled\tables"
BOOTSTRAP_DIR = r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_3_choice_reward\choice_reward_joint_results\filtered\pooled\bootstraps"

OUT_DIR = r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\ale_thesis_fig_tab\tables_thesis\section_3"
os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    "omnibus": "omnibus_summary.csv",
    "cell_fits": "full_data_cell_fits.csv",
    "cell_composition": "cell_composition.csv",
}
NPZ_FILES = {
    2: "rat_2_joint_bootstrap.npz", 3: "rat_3_joint_bootstrap.npz",
    4: "rat_4_joint_bootstrap.npz", 5: "rat_5_joint_bootstrap.npz",
    6: "rat_6_joint_bootstrap.npz", 7: "rat_7_joint_bootstrap.npz",
    9: "rat_9_joint_bootstrap.npz", 10: "rat_10_joint_bootstrap.npz",
    11: "rat_11_joint_bootstrap.npz", 12: "rat_12_joint_bootstrap.npz",
    13: "rat_13_joint_bootstrap.npz", 16: "rat_16_joint_bootstrap.npz",
}
# When running locally: NPZ_FILES = {rat: f"rat_{rat}_joint_bootstrap.npz" for rat in RATS_ALL}

DATA_CONDITION = "filtered"
SCOPE = "pooled"
FOCUS_RATS = [9, 11, 12, 13]


def restrict(df):
    return df.loc[(df["data_condition"] == DATA_CONDITION) & (df["scope"] == SCOPE)].copy()


def load_csv(name):
    return restrict(pd.read_csv(os.path.join(DATA_DIR, FILES[name])))


def fmt_p(p):
    """Plain text (T1 fontenc handles the bare '<' correctly -- no math-mode
    workaround needed, see module docstring)."""
    if p < 0.001:
        return "<.001"
    s = f"{p:.4f}"
    return s[1:] if s.startswith("0.") else s


def ordered_rats(rats_all, focus_rats):
    rest = sorted(r for r in rats_all if r not in focus_rats)
    return list(focus_rats) + rest


def int_digits_for(values):
    """Max integer-digit width across a list of floats (min 1) -- computed
    once per column/table and passed to every latex_num/latex_est_ci call
    sharing that column, per the project's established alignment fix."""
    return max(1, max(len(f"{abs(v):.0f}") for v in values))


def max_stars_for(qs):
    return max((len(stars_for_q(q)) for q in qs), default=0)


def latex_int(x, int_digits=0):
    """Integer counts (n_trials, n_excluded) rendered in math mode, with the
    same \\hphantom digit-padding convention as latex_num -- but plain
    integers have no decimal point, so latex_num's split('.') doesn't apply
    and this is a small, separate helper instead."""
    s = str(int(x))
    pad = max(0, int_digits - len(s))
    pad_str = r"\hphantom{" + "0" * pad + "}" if pad > 0 else ""
    return f"${pad_str}{s}$"


# ============================================================================
# 1. OMNIBUS TABLE
# ============================================================================
print("Building omnibus table...")
omnibus = load_csv("omnibus")
RATS_ALL = sorted(omnibus["rat"].unique().tolist())
rat_order = ordered_rats(RATS_ALL, FOCUS_RATS)


def omnibus_slice(df, omnibus_name, df_n):
    sub = df.loc[df["omnibus"] == omnibus_name, [
        "rat", "wald_T", "p_omnibus_bootstrap", "q_bh_omnibus_family"
    ]].copy()
    return sub.rename(columns={
        "wald_T": f"T_{df_n}df", "p_omnibus_bootstrap": f"p_boot_{df_n}df",
        "q_bh_omnibus_family": f"q_BH_{df_n}df",
    }).set_index("rat")


two_df = omnibus_slice(omnibus, "choice_plane_2df", 2)
three_df = omnibus_slice(omnibus, "full_factorial_3df", 3)
omnibus_table = two_df.join(three_df, how="outer")

T_int_digits = int_digits_for(pd.concat([omnibus_table["T_2df"], omnibus_table["T_3df"]]).tolist())
q_max_stars = max_stars_for(pd.concat([omnibus_table["q_BH_2df"], omnibus_table["q_BH_3df"]]).tolist())

header_rows_1 = [
    [Cell("", colspan=1), Cell("Choice-plane test (2 df)", colspan=3),
     Cell("Full-factorial test (3 df)", colspan=3)],
    [Cell("Rat"), Cell("$T$"), Cell("$p_{\\mathrm{boot}}$"), Cell("$q_{\\mathrm{BH}}$"),
     Cell("$T$"), Cell("$p_{\\mathrm{boot}}$"), Cell("$q_{\\mathrm{BH}}$")],
]
body_rows_1 = []
for i, rat in enumerate(rat_order):
    r = omnibus_table.loc[rat]
    row = [
        Cell(str(rat)),
        Cell(latex_num(r["T_2df"], decimals=2, pad_sign=True, int_digits=T_int_digits)),
        Cell(fmt_p(r["p_boot_2df"])),
        Cell(latex_num(r["q_BH_2df"], decimals=3, stars=stars_for_q(r["q_BH_2df"]),
                        pad_sign=True, max_stars=q_max_stars, int_digits=1)),
        Cell(latex_num(r["T_3df"], decimals=2, pad_sign=True, int_digits=T_int_digits)),
        Cell(fmt_p(r["p_boot_3df"])),
        Cell(latex_num(r["q_BH_3df"], decimals=3, stars=stars_for_q(r["q_BH_3df"]),
                        pad_sign=True, max_stars=q_max_stars, int_digits=1)),
    ]
    body_rows_1.append(row)
    if i + 1 == len(FOCUS_RATS):
        body_rows_1.append([])  # empty row -> plain \midrule (table_utils convention)

table1 = Table(
    header_rows=header_rows_1,
    body_rows=body_rows_1,
    n_cols=7,
    caption="Omnibus tests of joint history-dependence of the PSE (pooled scope), all rats.",
    label="tab:omnibus_2df_3df",
    note=(
        "2-df choice-plane test: H0: delta_mu_rew = delta_mu_unrew = 0. 3-df full-factorial "
        "test: H0: delta_mu_rew = delta_mu_unrew = P = 0. q_BH = Benjamini-Hochberg corrected "
        "q-value within each test's own 12-rat family. Stars: *** q<.001, ** q<.01, * q<.05 "
        "(no bold). Rows: focus rats (9, 11, 12, 13) first, separated by a rule from the rest."
    ),
)
with open(os.path.join(OUT_DIR, "table_omnibus_2df_3df.tex"), "w") as f:
    f.write(table_to_latex(table1))
print("  wrote table_omnibus_2df_3df.tex")


# ============================================================================
# 2. BOOTSTRAP QC TABLE
# ============================================================================
print("Building bootstrap QC table...")
qc_report = qc.build_qc_report_all(NPZ_FILES, BOOTSTRAP_DIR).set_index(["rat", "cell"])

# "rew" (3 chars) vs "unrew" (5 chars) differ only by the leading "un", so
# padding the shorter word with \hphantom{un} makes both cells occupy the
# same rendered width. table_to_latex's Cell column is centered (col_spec
# hardcoded as "l" + "c"*(n-1) inside table_utils.py, not a Table/Cell
# parameter), so unequal cell widths center on different amounts and the
# "H |"/"V |" prefix -- and the "|" itself -- drift left/right row to row.
# Equal-width cells fix that without needing to touch table_utils.py itself.
CELL_LABEL = {
    "h_rew": "H | rew\\hphantom{un}", "v_rew": "V | rew\\hphantom{un}",
    "h_unrew": "H | unrew", "v_unrew": "V | unrew",
}
CELLS = ["h_rew", "v_rew", "h_unrew", "v_unrew"]

all_pct_sigma = qc_report["pct_sigma_degenerate"].tolist()
all_pct_gl = qc_report["pct_gamma_lapse_near_ceiling"].tolist()
pct_int_digits = int_digits_for(all_pct_sigma + all_pct_gl)
n_trials_int_digits = int_digits_for(qc_report["n_trials"].tolist())

header_rows_2 = [
    [Cell("Rat"), Cell("Cell"), Cell("$n_{\\mathrm{trials}}$"), Cell("\\%$\\sigma<3$\\textdegree"),
     Cell("\\%$\\gamma$/$\\lambda \\geq$.29"), Cell("")],
]
body_rows_2 = []
for i, rat in enumerate(rat_order):
    for j, cell in enumerate(CELLS):
        row = qc_report.loc[(rat, cell)]
        boundary = bool(row["full_data_boundary_params"])
        body_rows_2.append([
            Cell(str(rat) if j == 0 else ""),
            Cell(CELL_LABEL[cell]),
            Cell(latex_int(row["n_trials"], int_digits=n_trials_int_digits)),
            Cell(latex_num(row["pct_sigma_degenerate"], decimals=2, int_digits=pct_int_digits)),
            Cell(latex_num(row["pct_gamma_lapse_near_ceiling"], decimals=2, int_digits=pct_int_digits)),
            Cell("*" if boundary else ""),
        ])
    if i + 1 == len(FOCUS_RATS):
        body_rows_2.append([])

table2 = Table(
    header_rows=header_rows_2,
    body_rows=body_rows_2,
    n_cols=6,
    caption="Session-bootstrap boundary-degeneracy census, all rats.",
    label="tab:bootstrap_qc",
    note=(
        "pct sigma<3deg = share of session-bootstrap iterations whose fitted sigma collapsed "
        "toward the optimizer's lower bound (cosmetic curve-band artifact). pct gamma/lapse>=.29 "
        "= share of iterations with an elevated floor/ceiling parameter near the 0.30 upper "
        "bound. * marks the one cell (rat 13, h_unrew) whose full-data point estimate is itself "
        "pinned at a box constraint (gamma_at_upper in full_data_cell_fits.csv) -- gamma/lapse "
        "not identified for that cell; mu is unaffected (verified)."
    ),
)
with open(os.path.join(OUT_DIR, "table_bootstrap_qc.tex"), "w") as f:
    f.write(table_to_latex(table2))
print("  wrote table_bootstrap_qc.tex")


# ============================================================================
# 3. SIGMA-ONLY FILTER SHIFT TABLE
# ============================================================================
print("Building sigma-only filter shift table...")

shift_rows = []
for rat in RATS_ALL:
    data = qc.load_bootstrap(rat, os.path.join(BOOTSTRAP_DIR, NPZ_FILES[rat]))
    for outcome in ("rew", "unrew"):
        cell0, cell1 = qc.OUTCOME_CELLS[outcome]
        sigma_flag = qc.sigma_degenerate_mask(data, cell0) | qc.sigma_degenerate_mask(data, cell1)
        delta = qc.cell_contrast(data, qc.OUTCOME_CONTRAST[outcome])

        med_full = np.median(delta)
        lo_full, hi_full = np.percentile(delta, [2.5, 97.5])
        clean = delta[~sigma_flag]
        med_clean = np.median(clean)
        lo_clean, hi_clean = np.percentile(clean, [2.5, 97.5])
        sig_full = not (lo_full <= 0 <= hi_full)

        shift_rows.append({
            "rat": rat, "outcome": outcome, "n_boot": data["n_boot"],
            "n_excluded": int(sigma_flag.sum()), "pct_excluded": 100 * sigma_flag.mean(),
            "median_full": med_full, "median_clean": med_clean, "shift": med_clean - med_full,
            "ci_full_lo": lo_full, "ci_full_hi": hi_full,
            "ci_clean_lo": lo_clean, "ci_clean_hi": hi_clean, "sig_full": sig_full,
        })

shift_df = pd.DataFrame(shift_rows).set_index(["rat", "outcome"])
shift_df.to_csv(os.path.join(OUT_DIR, "table_sigma_filter_shift.csv"))

all_deltas = (
    shift_df["median_full"].tolist() + shift_df["ci_full_lo"].tolist() + shift_df["ci_full_hi"].tolist()
    + shift_df["median_clean"].tolist() + shift_df["ci_clean_lo"].tolist() + shift_df["ci_clean_hi"].tolist()
)
delta_int_digits = int_digits_for(all_deltas)
pct_excl_int_digits = int_digits_for(shift_df["pct_excluded"].tolist())
n_excl_int_digits = int_digits_for(shift_df["n_excluded"].tolist())

OUTCOME_LABEL = {"rew": "rewarded", "unrew": "unrewarded"}
header_rows_3 = [
    [Cell(""), Cell(""), Cell(""), Cell(""), Cell("Full", colspan=1),
     Cell("Filtered ($\\sigma$-only)", colspan=1), Cell("")],
    [Cell("Rat"), Cell("Outcome"), Cell("$n_{\\mathrm{excl}}$"), Cell("%"),
     Cell("Full estimate [95% CI]"), Cell("Filtered estimate [95% CI]"), Cell("Shift")],
]
body_rows_3 = []
for i, rat in enumerate(rat_order):
    for j, outcome in enumerate(("rew", "unrew")):
        row = shift_df.loc[(rat, outcome)]
        stars = "*" if row["sig_full"] else ""
        full_cell = latex_est_ci(
            row["median_full"], row["ci_full_lo"], row["ci_full_hi"], decimals=2,
            stars=stars, pad_sign=True, max_stars=1, int_digits=delta_int_digits,
        )
        filtered_cell = latex_est_ci(
            row["median_clean"], row["ci_clean_lo"], row["ci_clean_hi"], decimals=2,
            pad_sign=True, int_digits=delta_int_digits,
        )
        shift_cell = f"${row['shift']:+.2f}$"
        body_rows_3.append([
            Cell(str(rat) if j == 0 else ""),
            Cell(OUTCOME_LABEL[outcome]),
            Cell(latex_int(row["n_excluded"], int_digits=n_excl_int_digits)),
            Cell(latex_num(row["pct_excluded"], decimals=2, int_digits=pct_excl_int_digits)),
            Cell(full_cell),
            Cell(filtered_cell),
            Cell(shift_cell),
        ])
    if i + 1 == len(FOCUS_RATS):
        body_rows_3.append([])

table3 = Table(
    header_rows=header_rows_3,
    body_rows=body_rows_3,
    n_cols=7,
    caption="Median/CI shift in delta-mu under sigma-only bootstrap filtering, all rats.",
    label="tab:sigma_filter_shift",
    note=(
        "Effect of excluding sigma-only degenerate bootstrap iterations (sigma<3deg; the same "
        "criterion used to build the curve-band figures) on delta_mu_rew/delta_mu_unrew's "
        "median and 95% CI. 'Full' = all bootstrap iterations (the numbers already reported "
        "elsewhere); 'Filtered' = sigma-only exclusion. * marks the full-bootstrap estimate's "
        "95% CI excluding zero, for reference -- not a BH q-value and not recomputed on the "
        "filtered sample."
    ),
)
with open(os.path.join(OUT_DIR, "table_sigma_filter_shift.tex"), "w") as f:
    f.write(table_to_latex(table3, fit_to_width=True))
print("  wrote table_sigma_filter_shift.tex")

print("\nAll three LaTeX tables written to", OUT_DIR, "(via table_utils.py)")