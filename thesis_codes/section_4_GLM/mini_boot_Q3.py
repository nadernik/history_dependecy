"""
Run this ON YOUR MACHINE, in the folder that has glm_bootstrap_draws.parquet
(the full, un-trimmed bootstrap file the interaction-mode GLM script writes
to fit_results/interaction_filtered/).

Figure 3 switched from the degree-scale delta_mu_rewarded/unrewarded to the
log-odds beta_success_n-1/beta_failure_n-1 coefficients, so it needs THIS
export instead of (not in addition to) the earlier
boot_draws_fig3_deltamu.parquet -- same idea, different quantities: the
per-modality (kind="modality") bootstrap draws for beta_success_n-1 and
beta_failure_n-1, C=unpenalized, all 12 rats.

(boot_draws_q3_filtered.parquet, the very first file you sent, already has
these two quantities but only for 9 of the 12 rats -- this fills in the
missing rats 3, 5, 10 too, by just exporting all of them fresh.)

Usage:
    python export_boot_draws_fig3_betas.py

Adjust IN_DIR below if you run it from somewhere else. Send back the
resulting boot_draws_fig3_betas.parquet (should be well under 1 MB).
"""
import pandas as pd

IN_DIR = r"fit_results/interaction_filtered"
IN_PATH = f"{IN_DIR}/glm_bootstrap_draws.parquet"
OUT_PATH = f"{IN_DIR}/boot_draws_fig3_betas.parquet"

QUANTITIES = ["beta_success_n-1", "beta_failure_n-1"]

df = pd.read_parquet(IN_PATH)
print(f"Full file: {len(df):,} rows, {df['rat_id'].nunique()} rats")

trimmed = df[
    (df["C"] == "unpenalized")
    & (df["kind"] == "modality")
    & (df["quantity"].isin(QUANTITIES))
].copy()

print(f"Trimmed: {len(trimmed):,} rows, {trimmed['rat_id'].nunique()} rats "
      f"({sorted(trimmed['rat_id'].unique())})")

trimmed.to_parquet(OUT_PATH, index=False, compression="snappy")
print(f"Saved {OUT_PATH}")