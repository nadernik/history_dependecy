"""
Extracts a small, upload-friendly summary from the FULL modality-interaction
bootstrap-draws parquet (the one too large to upload directly), for the one
quantity we're missing from `boot_draws_fig3_betas.parquet`:
`beta_angle_n-1` (the previous-stimulus-angle / perceptual-history GLM
coefficient), per rat x modality.

`boot_draws_fig3_betas.parquet` only ever contained beta_success_n-1 and
beta_failure_n-1 (it was scoped to a specific figure) -- this script assumes
your full/raw bootstrap parquet has the same 7-column schema
(rat_id, C, kind, subset_or_pair, quantity, boot_idx, value) but with ALL
quantities included, and pulls out just what's needed for a percentile-
bootstrap p-value per cell: the fraction of draws above/below zero. That
collapses ~180,000 rows (12 rats x 3 modalities x 5000 draws) down to 36
summary rows -- small enough to upload here.

USAGE
-----
1. Edit BOOT_PATH below to point at your full bootstrap parquet file.
2. Run: python extract_angle_n1_boot_summary.py
3. Upload the resulting angle_n1_boot_summary.csv back to Claude.

If your file's column names differ from what's assumed here, the script
prints the schema it actually found before failing, so you can see what to
adjust (look for the QUANTITY_COL / KIND_COL / SUBSET_COL / C_COL /
RAT_COL / VALUE_COL constants below).
"""
import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ---- EDIT THIS ------------------------------------------------------------
BOOT_PATH = r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\glm_results_thesis_filtered_interaction\glm_bootstrap_draws.parquet"
# ----------------------------------------------------------------------------

# Column names as seen in boot_draws_fig3_betas.parquet -- adjust if your
# full file uses different names.
RAT_COL = "rat_id"
C_COL = "C"
KIND_COL = "kind"           # e.g. "modality"
SUBSET_COL = "subset_or_pair"  # e.g. "tactile" / "visual" / "visuotactile"
QUANTITY_COL = "quantity"
VALUE_COL = "value"

TARGET_QUANTITY = "beta_angle_n-1"
TARGET_C = "unpenalized"
TARGET_KIND = "modality"


def boot_p_value(draws, n_boot=None):
    """Alessandro's percentile-bootstrap two-sided p-value (verbatim)."""
    draws = np.asarray(draws, dtype=float)
    n = len(draws) if n_boot is None else n_boot
    frac_pos = np.mean(draws > 0)
    frac_neg = np.mean(draws < 0)
    p = 2 * min(frac_pos, frac_neg)
    floor = 2.0 / (n + 1)
    return max(p, floor)


def main():
    if not os.path.isfile(BOOT_PATH):
        print(f"ERROR: BOOT_PATH does not exist: {BOOT_PATH}")
        print("Edit BOOT_PATH at the top of this script to point at your full bootstrap parquet.")
        sys.exit(1)

    schema = pq.read_schema(BOOT_PATH)
    cols = schema.names
    print("Columns found in file:", cols)

    missing = [c for c in [RAT_COL, C_COL, KIND_COL, SUBSET_COL, QUANTITY_COL, VALUE_COL] if c not in cols]
    if missing:
        print(f"\nERROR: expected column(s) not found: {missing}")
        print("Edit the *_COL constants near the top of this script to match your actual schema, then re-run.")
        sys.exit(1)

    # Predicate pushdown: only pull the rows we need, not the whole file.
    filters = [
        (QUANTITY_COL, "==", TARGET_QUANTITY),
        (C_COL, "==", TARGET_C),
        (KIND_COL, "==", TARGET_KIND),
    ]
    table = pq.read_table(BOOT_PATH, columns=[RAT_COL, SUBSET_COL, VALUE_COL], filters=filters)
    df = table.to_pandas()

    if df.empty:
        print(f"\nWARNING: no rows matched quantity='{TARGET_QUANTITY}', C='{TARGET_C}', kind='{TARGET_KIND}'.")
        print("Unique values actually present in the file (reading full columns, may be slow):")
        full_cols = pq.read_table(BOOT_PATH, columns=[C_COL, KIND_COL, QUANTITY_COL]).to_pandas()
        print("  C:", sorted(full_cols[C_COL].unique().tolist()))
        print("  kind:", sorted(full_cols[KIND_COL].unique().tolist()))
        print("  quantity:", sorted(full_cols[QUANTITY_COL].unique().tolist()))
        print("\nAdjust TARGET_QUANTITY / TARGET_C / TARGET_KIND above to match, then re-run.")
        sys.exit(1)

    rows = []
    for (rat, mod), g in df.groupby([RAT_COL, SUBSET_COL]):
        vals = g[VALUE_COL].values
        n = len(vals)
        frac_pos = float(np.mean(vals > 0))
        frac_neg = float(np.mean(vals < 0))
        p_boot = boot_p_value(vals, n_boot=n)
        rows.append({
            "rat_id": rat,
            "modality": mod,
            "quantity": TARGET_QUANTITY,
            "C": TARGET_C,
            "n_boot": n,
            "mean_draw": float(np.mean(vals)),
            "median_draw": float(np.median(vals)),
            "frac_positive": frac_pos,
            "frac_negative": frac_neg,
            "p_boot": p_boot,
        })

    out = pd.DataFrame(rows).sort_values(["rat_id", "modality"]).reset_index(drop=True)
    out_path = os.path.join(os.path.dirname(BOOT_PATH), "angle_n1_boot_summary.csv")
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out)} rows to: {out_path}")
    print(out.to_string(index=False))
    print(f"\nExpected 36 rows (12 rats x 3 modalities); got {len(out)}.")


if __name__ == "__main__":
    main()