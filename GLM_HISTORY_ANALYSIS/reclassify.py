"""
Reclassify GLM fits from the summary CSV using a revised rule.

Old rule (too strict):
    reward_x_choice iff P(opposite signs) >= 0.95
Missed: cases where one of {beta_success, beta_failure} is ~0 and the other is
clearly nonzero, which is also a reward x choice signature (see project notes).

Revised rule:
    reward_x_choice iff (beta_success - beta_failure) CI excludes 0
                    AND NOT (both coefficients clearly same-sign with similar magnitude)
    pure_choice     iff diff CI includes 0 AND at least one coefficient is clearly
                    nonzero with same-sign majority
    ambiguous       otherwise

Both rules are computed and printed side by side for comparison.
"""

import pandas as pd
from scipy.stats import binomtest

CSV_PATH = "glm_summary.csv"        # path to the summary CSV
OUT_PATH = "glm_summary_reclassified.csv"


def old_rule(row, opp_thresh=0.95):
    p_opp = row["p_opposite_signs"]
    diff_includes_zero = row["diff_ci_low"] <= 0 <= row["diff_ci_high"]
    if p_opp >= opp_thresh:
        return "reward_x_choice"
    if diff_includes_zero and p_opp < 0.5:
        return "pure_choice"
    return "ambiguous"


def new_rule(row):
    """Classification based on the diff CI + coefficient CIs.

    Logic:
      - diff CI excludes 0 -> the two coefficients differ meaningfully.
        This is compatible with (a) opposite signs, or (b) one ~0 and one nonzero.
        Both are reward x choice signatures. So: diff CI excludes 0 -> reward_x_choice,
        unless BOTH coefficient CIs exclude 0 with the same sign (that would be
        pure choice with different magnitudes, not reward x choice).
      - diff CI includes 0 AND at least one coefficient CI excludes 0
        AND same-sign majority (both estimates same sign) -> pure_choice
      - otherwise -> ambiguous
    """
    diff_low = row["diff_ci_low"]
    diff_high = row["diff_ci_high"]
    diff_excludes_zero = (diff_low > 0) or (diff_high < 0)

    s_low, s_high = row["ci_low_success_n-1"], row["ci_high_success_n-1"]
    f_low, f_high = row["ci_low_failure_n-1"], row["ci_high_failure_n-1"]
    s = row["beta_success_n-1"]
    f = row["beta_failure_n-1"]

    s_excludes_zero = (s_low > 0) or (s_high < 0)
    f_excludes_zero = (f_low > 0) or (f_high < 0)
    same_sign = (s * f) > 0

    if diff_excludes_zero:
        # both CIs exclude 0 AND same sign -> pure choice, different magnitudes
        if s_excludes_zero and f_excludes_zero and same_sign:
            return "pure_choice"
        return "reward_x_choice"

    # diff CI includes 0
    if (s_excludes_zero or f_excludes_zero) and same_sign:
        return "pure_choice"

    return "ambiguous"


def main():
    df = pd.read_csv(CSV_PATH)
    df["classification_old"] = df["classification"]
    df["classification_new"] = df.apply(new_rule, axis=1)

    # side-by-side
    cols = ["rat_id", "subset", "classification_old", "classification_new",
            "beta_success_n-1", "ci_low_success_n-1", "ci_high_success_n-1",
            "beta_failure_n-1", "ci_low_failure_n-1", "ci_high_failure_n-1",
            "diff_s_minus_f", "diff_ci_low", "diff_ci_high", "p_opposite_signs"]
    print("\n=== Per-fit classification (old vs new rule) ===")
    print(df[cols].to_string(index=False))

    # counts per subset
    print("\n=== Class counts per subset (new rule) ===")
    counts = (df.groupby(["subset", "classification_new"])
                .size().unstack(fill_value=0))
    print(counts.to_string())

    # binomial tests
    print("\n=== Binomial test: fraction reward_x_choice vs 50% (new rule) ===")
    for sub in df["subset"].unique():
        sub_df = df[df["subset"] == sub]
        n = len(sub_df)
        k = int((sub_df["classification_new"] == "reward_x_choice").sum())
        p = binomtest(k, n, p=0.5, alternative="greater").pvalue
        print(f"  {sub:>12}: {k}/{n} ({k/n:.2%}), p = {p:.4f}")

    # also: fraction with diff_ci excluding 0 (a looser, purely quantitative check)
    print("\n=== Fraction of fits with diff CI excluding 0 (per subset) ===")
    for sub in df["subset"].unique():
        sub_df = df[df["subset"] == sub]
        diff_excl = ((sub_df["diff_ci_low"] > 0) | (sub_df["diff_ci_high"] < 0)).sum()
        n = len(sub_df)
        print(f"  {sub:>12}: {diff_excl}/{n} ({diff_excl/n:.2%})")

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved reclassified summary to {OUT_PATH}")


if __name__ == "__main__":
    main()
