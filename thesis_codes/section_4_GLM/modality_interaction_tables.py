"""
Single build script for all three Section 4.x.3-4.x.4 tables (LR gating,
log-odds-by-modality, modality-contrast resolution), built against
Alessandro's ACTUAL table_utils.py (Cell/Table dataclass model,
python-docx + LaTeX dual rendering) rather than a reimplementation.

table_utils.py is used completely unmodified here. Statistics that don't
belong in a rendering-only module (Holm step-down correction, percentile
bootstrap p-values, the int-digit-padding helper) live in stats_helpers.py
instead -- see that file's docstring for why.

Two things worth knowing before reading the code below:

1. Significance convention. table_utils.py's own docstring says the fixed
   project convention is "bold the cell AND append asterisks." But later
   project notes (section_4_pooled_glm_tables_plan.md, "Fixes applied this
   round") record that bold was explicitly REMOVED from the primary/ridge/
   comparison tables in favor of asterisks-only, specifically to avoid
   fighting the CI-bracket-alignment padding -- and Alessandro re-confirmed
   asterisks-only for these three tables this session. So every call below
   uses bold=False. The stale docstring is worth updating in the canonical
   file if these two are meant to stay in sync.

2. Notes cannot contain LaTeX markup. table_to_latex/table_to_latex_longtable
   run `_latex_escape_plain` unconditionally on `t.note` (unlike cell text,
   which is only escaped when it does NOT start with "$" or contain a
   backslash). That means \ref{...}, $...$ math, or \% inside a note gets
   mangled (underscores/braces/dollar-signs turned into literal \_ \{ \}
   \$). So every table below refers to the others by a PLAIN, hardcoded
   table number in its note text ("Table 1", "Table 4", etc.) rather than
   \ref -- those numbers match this script's own test_document.tex build
   order, but will NOT auto-update if the tables land in a different order
   inside the actual thesis chapter. Worth checking by hand once they're
   placed, or patching table_utils.py's note-escaping to use the same
   bypass rule cell text already gets, if this comes up again.

Usage: python3 build_all_tables.py
Inputs (must be in the working directory): glm_lr_tests.csv,
glm_contrasts_long.csv, glm_summary_long.csv, boot_draws_q3_filtered.parquet.

Outputs:
  table1_lr_gating.tex             + table1_lr_gating_data.csv
  table2a_modality_coeffs.tex      \\_ table2_modality_coeffs_data.csv
  table2b_modality_coeffs.tex      /
  table3_contrasts.tex             + table3_contrasts_data.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / 'results' / 'modality_interaction_tables'
OUT_DIR.mkdir(parents=True, exist_ok=True)

from table_utils import Cell, Table, stars_for_q, latex_num, latex_est_ci, table_to_latex, table_to_latex_longtable
from stats_helper import holm_correction, boot_p_value, compute_global_int_digits

# ---------------------------------------------------------------------------
# Load inputs once, shared across all three tables
# ---------------------------------------------------------------------------
lr = pd.read_csv(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_4_GLM\fit_results\interaction_filtered\glm_lr_tests.csv")
lr = lr[lr["C"] == "unpenalized"]

contrasts = pd.read_csv(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_4_GLM\fit_results\interaction_filtered\glm_contrasts_long.csv")
contrasts = contrasts[contrasts["C"] == "unpenalized"]

summ = pd.read_csv(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_4_GLM\fit_results\interaction_filtered\glm_summary_long.csv")
summ = summ[summ["C"] == "unpenalized"]

boot = pd.read_parquet(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_4_GLM\fit_results\interaction_filtered\glm_bootstrap_draws.parquet")
boot = boot[(boot["C"] == "unpenalized") & (boot["kind"] == "contrast")]

RATS = sorted(lr["rat_id"].unique())
FAMILY_BLOCKS = ["success_n-1", "failure_n-1", "reward_n-1"]
BLOCK_LABEL = {"success_n-1": "success", "failure_n-1": "failure",
               "reward_n-1": "reward", "angle_n-1": "prev-stimulus"}
BLOCK_ORDER = {"success_n-1": 0, "failure_n-1": 1, "reward_n-1": 2, "angle_n-1": 3}


# ---------------------------------------------------------------------------
# Shared gating computation (used by Table 1, and reused by Tables 2 and 3
# so the gate decision is computed exactly once, not re-derived per table)
# ---------------------------------------------------------------------------
def compute_lr_gating():
    rows = {}
    gated_cells = []
    for rat in RATS:
        sub = lr[lr["rat_id"] == rat].set_index("block")
        raw_p = [sub.loc[b, "p_value"] for b in FAMILY_BLOCKS]
        adj_p, reject = holm_correction(raw_p, alpha=0.05)
        cell = {}
        for b, p_raw, p_adj, rej in zip(FAMILY_BLOCKS, raw_p, adj_p, reject):
            cell[b] = {"p": p_raw, "p_holm": p_adj, "reject": bool(rej)}
            if rej:
                gated_cells.append((rat, b))
        p_prev = sub.loc["angle_n-1", "p_value"]
        prev_reject = bool(sub.loc["angle_n-1", "reject_at_5pct"])
        assert prev_reject == (p_prev < 0.05)
        cell["angle_n-1"] = {"p": p_prev, "p_holm": np.nan, "reject": prev_reject}
        if prev_reject:
            gated_cells.append((rat, "angle_n-1"))
        rows[rat] = cell

    gated_cells.sort(key=lambda x: (BLOCK_ORDER[x[1]], x[0]))
    return rows, gated_cells


def validate_gating(gating_rows, gated_cells):
    def rats_for(block):
        return [r for r, b in gated_cells if b == block]

    expected = {
        "success_n-1": [4, 6, 11, 13, 16],
        "failure_n-1": [7, 13, 16],
        "reward_n-1": [7, 12, 16],
        "angle_n-1": [2, 4, 7, 9, 13],
    }
    for block, exp in expected.items():
        got = sorted(rats_for(block))
        assert got == exp, (block, got, exp)
    any_choice_outcome = sorted(set(rats_for("success_n-1")) | set(rats_for("failure_n-1")) | set(rats_for("reward_n-1")))
    assert any_choice_outcome == [4, 6, 7, 11, 12, 13, 16], any_choice_outcome
    assert len(gated_cells) == 16, len(gated_cells)
    print("Gating validation OK: matches project notes exactly.")
    print("  success:", rats_for("success_n-1"))
    print("  failure:", rats_for("failure_n-1"))
    print("  reward:", rats_for("reward_n-1"))
    print("  prev-stimulus:", rats_for("angle_n-1"))
    print("  any choice-outcome rejection (7 rats):", any_choice_outcome)


# ---------------------------------------------------------------------------
# Table 1 -- LR-test gating
# ---------------------------------------------------------------------------
def build_table1(gating_rows):
    def fmt_p(p):
        return "$<$0.001" if p < 0.001 else f"{p:.3f}"

    def cell_str(p, rejected):
        s = stars_for_q(p) if rejected else ""  # stars_for_q's thresholds are generic; reused for raw/Holm p here
        return fmt_p(p) + s

    header_rows = [
        [Cell("Rat", colspan=1),
         Cell("Choice-outcome family (Holm-3 within rat)", colspan=3),
         Cell("Previous-stim. (raw p<.05, own family)", colspan=1)],
        [Cell(""), Cell("success"), Cell("failure"), Cell("reward"), Cell("")],
    ]
    body_rows = []
    for rat in RATS:
        c = gating_rows[rat]
        body_rows.append([
            Cell(str(rat)),
            Cell(cell_str(c["success_n-1"]["p"], c["success_n-1"]["reject"])),
            Cell(cell_str(c["failure_n-1"]["p"], c["failure_n-1"]["reject"])),
            Cell(cell_str(c["reward_n-1"]["p"], c["reward_n-1"]["reject"])),
            Cell(cell_str(c["angle_n-1"]["p"], c["angle_n-1"]["reject"])),
        ])

    note = ("Within-rat likelihood-ratio gating for the interaction-mode GLM. Raw LR-test "
            "p-values shown throughout; stars mark significance (* p<.05, ** p<.01, *** p<.001) "
            "under the correction stated for that family. Success/failure/reward form one "
            "3-test family per rat, corrected jointly (Holm step-down, alpha=.05); "
            "previous-stimulus is its own family (single 2-df test per rat), so no correction "
            "is applied. Only rats/effects starred here are carried forward into the "
            "modality-contrast analysis (Table 4).")
    t = Table(header_rows=header_rows, body_rows=body_rows, n_cols=5,
              caption="Within-rat likelihood-ratio gating for the interaction-mode GLM.",
              label="tab:lr_gating", note=note)
    with open(OUT_DIR / "table1_lr_gating.tex", "w") as f:
        f.write(table_to_latex(t, fit_to_width=False))

    out_rows = []
    for rat in RATS:
        c = gating_rows[rat]
        out_rows.append((rat, c["success_n-1"]["p"], c["success_n-1"]["reject"],
                          c["failure_n-1"]["p"], c["failure_n-1"]["reject"],
                          c["reward_n-1"]["p"], c["reward_n-1"]["reject"],
                          c["angle_n-1"]["p"], c["angle_n-1"]["reject"]))
    out = pd.DataFrame(out_rows, columns=[
        "rat_id", "success_p", "success_reject", "failure_p", "failure_reject",
        "reward_p", "reward_reject", "prev_stim_p", "prev_stim_reject"])
    out.to_csv(OUT_DIR / "table1_lr_gating_data.csv", index=False)
    print("Wrote table1_lr_gating.tex and table1_lr_gating_data.csv")
    return out.set_index("rat_id")


# ---------------------------------------------------------------------------
# Table 3 (rendered as "Table 4" in the test document, after 2a/2b) --
# modality-contrast resolution, gated cells only
# ---------------------------------------------------------------------------
def build_table3(gated_cells):
    pairs = ["V_minus_T", "VT_minus_T", "VT_minus_V"]
    quantity_map = {
        "success_n-1": "beta_success_n-1", "failure_n-1": "beta_failure_n-1",
        "reward_n-1": "beta_reward_n-1", "angle_n-1": "beta_angle_n-1",
    }

    n_resolved_total = 0
    n_resolved_by_block = {b: 0 for b in quantity_map}
    n_tests_by_block = {b: 0 for b in quantity_map}
    rows_out = []

    for rat, block in gated_cells:
        qty = quantity_map[block]
        raw_ps, point_ests = [], []
        for pair in pairs:
            draws = boot[(boot["rat_id"] == rat) & (boot["subset_or_pair"] == pair) & (boot["quantity"] == qty)]["value"].values
            assert len(draws) > 0, (rat, block, pair)
            raw_ps.append(boot_p_value(draws))
            pe = contrasts[(contrasts["rat_id"] == rat) & (contrasts["contrast_pair"] == pair) & (contrasts["quantity"] == qty)]["value"].values
            assert len(pe) == 1
            point_ests.append(pe[0])
        adj_p, reject = holm_correction(raw_ps, alpha=0.05)
        n_tests_by_block[block] += 3
        n_resolved_by_block[block] += int(reject.sum())
        n_resolved_total += int(reject.sum())
        rows_out.append((rat, block, point_ests, adj_p, reject))

    print("Table 3 (modality contrasts): total resolved:", n_resolved_total, "/ 48")
    for b in ["success_n-1", "failure_n-1", "reward_n-1", "angle_n-1"]:
        print(f"  {BLOCK_LABEL[b]}: {n_resolved_by_block[b]}/{n_tests_by_block[b]}")
    assert n_resolved_total == 21
    assert n_resolved_by_block["success_n-1"] == 8
    assert n_resolved_by_block["failure_n-1"] == 2
    assert n_resolved_by_block["reward_n-1"] == 4
    assert n_resolved_by_block["angle_n-1"] == 7
    print("Table 3 validation OK (matches project note: 21/48, 8/15/2/9/4/9/7/15).")

    all_vals = [v for _, _, pe, _, _ in rows_out for v in pe]
    int_digits = compute_global_int_digits(all_vals)

    header_rows = [[Cell("Rat"), Cell("Block"), Cell("V-T"), Cell("VT-T"), Cell("VT-V")]]
    body_rows = []
    for rat, block, point_ests, adj_p, reject in rows_out:
        row = [Cell(str(rat)), Cell(BLOCK_LABEL[block])]
        for pe, p, rej in zip(point_ests, adj_p, reject):
            stars = stars_for_q(p) if rej else ""
            row.append(Cell(latex_num(pe, decimals=3, stars=stars, bold=False,
                                       pad_sign=True, max_stars=3, int_digits=int_digits)))
        body_rows.append(row)

    note = ("Log-odds contrasts (fitted, unpenalized) for the 16 rat x block cells gated by "
            "Table 1. Holm-3 step-down correction (percentile bootstrap p, n_boot=5000, "
            "floored at 2/5001) within each rat x block cell, jointly across the three pairwise "
            "modality contrasts. * p_Holm<.05, ** p_Holm<.01, *** p_Holm<.001; unstarred values "
            "did not resolve. 21 of 48 tests resolved (success 8/15, failure 2/9, reward 4/9, "
            "previous-stimulus 7/15).")
    t = Table(header_rows=header_rows, body_rows=body_rows, n_cols=5,
              caption="Pairwise modality contrasts (V-T, VT-T, VT-V) for the log-odds history "
                      "coefficients, restricted to rat x block cells that passed the Table 1 gate.",
              label="tab:q3_contrasts", note=note)
    with open(OUT_DIR / "table3_contrasts.tex", "w") as f:
        f.write(table_to_latex(t, fit_to_width=True))

    long_rows = []
    for i, pair in enumerate(pairs):
        for rat, block, point_ests, adj_p, reject in rows_out:
            long_rows.append((rat, block, pair, point_ests[i], adj_p[i], bool(reject[i])))
    out = pd.DataFrame(long_rows, columns=["rat_id", "block", "contrast_pair", "value", "p_holm", "resolved"])
    out.to_csv(OUT_DIR / "table3_contrasts_data.csv", index=False)
    print("Wrote table3_contrasts.tex and table3_contrasts_data.csv")


# ---------------------------------------------------------------------------
# Table 2 (2a/2b, rendered as "Table 2"/"Table 3" in the test document) --
# log-odds by modality, LR-gate shading
# ---------------------------------------------------------------------------
def build_table2(gating_table1_df):
    modalities = ["tactile", "visual", "visuotactile"]
    mod_label = {"tactile": "tactile", "visual": "visual", "visuotactile": "visuotact."}

    coeffs_2a = ["intercept", "beta_angle", "beta_angle_n-1"]
    coeffs_2b = ["beta_reward_n-1", "beta_success_n-1", "beta_failure_n-1"]
    coeff_header = {
        "intercept": "intercept",
        "beta_angle": r"$\beta_{\mathrm{angle}}$",
        "beta_angle_n-1": r"$\beta_{\mathrm{angle}(n-1)}$",
        "beta_reward_n-1": r"$\beta_{\mathrm{reward}(n-1)}$",
        "beta_success_n-1": r"$\beta_{\mathrm{success}(n-1)}$",
        "beta_failure_n-1": r"$\beta_{\mathrm{failure}(n-1)}$",
    }
    # Plain-text names for use inside `note` strings. table_utils.py escapes
    # note text unconditionally (unlike cell text, which is skipped when it
    # starts with "$" or contains a backslash) -- so the math-mode
    # coeff_header strings above CANNOT go into a note (they'd come out as
    # "\$\beta\_\{...\}\$" and pdflatex chokes on the stray "\$"). This
    # mapping is what notes use instead.
    coeff_plain = {
        "intercept": "intercept",
        "beta_angle": "the stimulus-slope coefficient",
        "beta_angle_n-1": "the previous-stimulus coefficient",
        "beta_reward_n-1": "the reward-history coefficient",
        "beta_success_n-1": "the success-history coefficient",
        "beta_failure_n-1": "the failure-history coefficient",
    }
    gate_col = {
        "intercept": None, "beta_angle": None,
        "beta_angle_n-1": "prev_stim_reject", "beta_reward_n-1": "reward_reject",
        "beta_success_n-1": "success_reject", "beta_failure_n-1": "failure_reject",
    }

    def build_half(coeffs, out_stub, label_suffix, table_number_str):
        all_vals = []
        for rat in RATS:
            for mod in modalities:
                sub = summ[(summ["rat_id"] == rat) & (summ["subset"] == mod)].set_index("quantity")
                for c in coeffs:
                    all_vals.extend([sub.loc[c, "value"], sub.loc[c, "ci_low"], sub.loc[c, "ci_high"]])
        int_digits = compute_global_int_digits(all_vals)

        header_rows = [[Cell("Rat"), Cell("Modality")] + [Cell(coeff_header[c]) for c in coeffs]]
        body_rows = []
        for rat in RATS:
            gated_flags = {c: (gate_col[c] is not None and bool(gating_table1_df.loc[rat, gate_col[c]])) for c in coeffs}
            for i, mod in enumerate(modalities):
                # Cell has no rowspan (only colspan), so the rat number is
                # printed once (first modality row) and left blank on the
                # other two, rather than \multirow-spanned.
                row = [Cell(str(rat) if i == 0 else ""), Cell(mod_label[mod])]
                sub = summ[(summ["rat_id"] == rat) & (summ["subset"] == mod)].set_index("quantity")
                for c in coeffs:
                    text = latex_est_ci(sub.loc[c, "value"], sub.loc[c, "ci_low"], sub.loc[c, "ci_high"],
                                         decimals=3, bold=False, pad_sign=True, int_digits=int_digits)
                    if gated_flags[c]:
                        text = r"\cellcolor{gray!15}" + text
                    row.append(Cell(text))
                body_rows.append(row)
            body_rows.append([])
        if body_rows and body_rows[-1] == []:
            body_rows.pop()

        gate_note = ""
        if any(gate_col[c] is not None for c in coeffs):
            gate_note = ("Shaded cells: rat/coefficient combinations where the modality-interaction "
                         "LR test was significant (Table 1) -- shading applies to all three modality "
                         "rows for that rat, since the test is on whether modality matters at all for "
                         "that coefficient, not a per-modality result. ")
        no_test_cols = [coeff_plain[c] for c in coeffs if gate_col[c] is None]
        no_test_note = ""
        if no_test_cols:
            no_test_note = ("No LR-gating test is defined for " + " or ".join(no_test_cols) +
                            " in glm_lr_tests.csv, so these columns are never shaded. ")
        note = ("Fitted log-odds coefficient [95% bootstrap CI], unpenalized fit, per rat x "
                "modality. Values and CIs are as reported directly in glm_summary_long.csv -- no "
                "correction has been applied to these per-modality estimates individually (the "
                "formal tests are the LR gate, Table 1, and the pairwise modality contrasts, "
                "Table 4). " + gate_note + no_test_note)

        t = Table(header_rows=header_rows, body_rows=body_rows, n_cols=2 + len(coeffs),
                  caption=f"Log-odds coefficients by sensory modality, unpenalized fit ({label_suffix}).",
                  label=f"tab:{out_stub}", note=note)
        with open(OUT_DIR / f"{out_stub}.tex", "w") as f:
            f.write(table_to_latex_longtable(t))
        print(f"Wrote {out_stub}.tex ({len(RATS)} rats x {len(modalities)} modalities x {len(coeffs)} coeffs)")

    build_half(coeffs_2a, "table2a_modality_coeffs", "intercept, stimulus slope, previous-stimulus", "2")
    build_half(coeffs_2b, "table2b_modality_coeffs", "reward/success/failure history", "3")

    rows = []
    for rat in RATS:
        for mod in modalities:
            sub = summ[(summ["rat_id"] == rat) & (summ["subset"] == mod)].set_index("quantity")
            for c in coeffs_2a + coeffs_2b:
                gated = gate_col[c] is not None and bool(gating_table1_df.loc[rat, gate_col[c]])
                rows.append((rat, mod, c, sub.loc[c, "value"], sub.loc[c, "ci_low"], sub.loc[c, "ci_high"], gated))
    out = pd.DataFrame(rows, columns=["rat_id", "modality", "coefficient", "value", "ci_low", "ci_high", "lr_gated"])
    out.to_csv(OUT_DIR / "table2_modality_coeffs_data.csv", index=False)
    print("Wrote table2_modality_coeffs_data.csv")


# ---------------------------------------------------------------------------
# Run all three
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gating_rows, gated_cells = compute_lr_gating()
    validate_gating(gating_rows, gated_cells)

    gating_df = build_table1(gating_rows)
    build_table3(gated_cells)
    build_table2(gating_df)

    print("\nAll three tables built successfully.")