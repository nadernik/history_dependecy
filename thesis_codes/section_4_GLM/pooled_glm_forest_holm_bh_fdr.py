"""
Forest plot of pooled-GLM log-odds coefficients, one row per rat.

Choose which panels to show with --show / --drop, e.g.
    python glm_coefficient_forest.py                        # all 5 panels
    python glm_coefficient_forest.py --drop angle_n1        # no beta_angle(n-1)
    python glm_coefficient_forest.py --show reward success failure

Panel keys:
    angle     -> beta_angle         (current-trial slope; reference, never tested)
    angle_n1  -> beta_angle_n-1
    reward    -> beta_reward_n-1
    success   -> beta_success_n-1
    failure   -> beta_failure_n-1

Multiple-comparison correction: Benjamini-Hochberg FDR by default
(--correction holm switches back to Holm). The family (--family) is set
SEPARATELY from the panels shown.
By default it is the 4 history coefficients x all rats (48 tests), whether or not
every one of them is plotted, so hiding a panel does not change the significance
of the others. Pass --family to redefine the family explicitly.

Significance tiers:
    grey                          : 95% bootstrap CI includes 0
    open marker, coloured bar     : CI excludes 0, not FDR-significant
    filled marker, coloured bar   : CI excludes 0 and FDR-significant (q < .05)

Outputs (CSV, SVG, PDF) go to OUTPUT_DIR; the file names carry a suffix
listing the shown panels when not all five are shown, plus '_holm' when
--correction holm is used. CSV columns: p_raw, p_adj (BH q-value), reject.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# ---- Defaults (edit these if you run the script from the IDE) --------------

DEFAULT_SHOW = ["angle", "reward", "success", "failure"]
DEFAULT_FAMILY = ["angle_n1", "reward", "success", "failure"]
DEFAULT_CORRECTION = "bh"   # "bh" (Benjamini-Hochberg FDR) or "holm"

ALPHA = 0.05
FIT = "unpenalized"
SUBSET = "pooled"

CANDIDATE_DATA_DIRS = [
    r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_4_GLM\fit_results\pooled_filtered",
    str(Path(__file__).resolve().parent),
]
OUTPUT_DIR = Path(
    r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes"
    r"\ale_thesis_fig_tab\images_thesis\section_4"
)


# ---- Coefficient registry ---------------------------------------------------

COEF_KEYS = {
    "angle": "beta_angle",
    "angle_n1": "beta_angle_n-1",
    "reward": "beta_reward_n-1",
    "success": "beta_success_n-1",
    "failure": "beta_failure_n-1",
}
ORDER = list(COEF_KEYS)  # canonical left-to-right panel order

PANEL_LABELS = {
    "beta_angle": r"$\beta_{angle}$ (current trial, reference)",
    "beta_angle_n-1": r"$\beta_{angle}$ (n-1)",
    "beta_reward_n-1": r"$\beta_{reward}$ (n-1)",
    "beta_success_n-1": r"$\beta_{success}$ (n-1)",
    "beta_failure_n-1": r"$\beta_{failure}$ (n-1)",
}
PANEL_COLOR = {
    "beta_angle": "#e87ba4",
    "beta_angle_n-1": "#2a78d6",
    "beta_reward_n-1": "#eb6834",
    "beta_success_n-1": "#1baf7a",
    "beta_failure_n-1": "#eda100",
}
GREY = "#9a9892"
TEXT_SECONDARY = "#52514e"
TEXT_PRIMARY = "#0b0b0b"


# ---- Statistical helpers ----------------------------------------------------

def holm_correction(pvals, alpha=0.05):
    """Holm-Bonferroni step-down adjusted p-values."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, pvals[idx] * (n - rank))
        adj[idx] = min(running_max, 1.0)
    return adj, adj < alpha


def bh_fdr_correction(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR adjusted p-values (q-values)."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order] * n / np.arange(1, n + 1)
    # step-up: enforce monotonicity from the largest p downwards
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n, dtype=float)
    adj[order] = np.minimum(ranked, 1.0)
    return adj, adj < alpha


CORRECTIONS = {
    "bh": (bh_fdr_correction, "BH-FDR", "FDR-significant"),
    "holm": (holm_correction, "Holm", "Holm-significant"),
}


def boot_p_value(draws, n_boot=None):
    """Two-sided percentile-bootstrap p-value, floored at 2/(n+1)."""
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return np.nan
    n = len(draws) if n_boot is None else n_boot
    p = 2 * min(np.mean(draws > 0), np.mean(draws < 0))
    return min(max(p, 2.0 / (n + 1)), 1.0)


# ---- I/O --------------------------------------------------------------------

def _find(filename_substr):
    for directory in CANDIDATE_DATA_DIRS:
        if not os.path.isdir(directory):
            continue
        matches = sorted(f for f in os.listdir(directory) if filename_substr in f)
        if matches:
            return os.path.join(directory, matches[0])
    raise FileNotFoundError(
        f"No file containing {filename_substr!r} in: {CANDIDATE_DATA_DIRS}"
    )


def _resolve(keys, arg_name):
    bad = [k for k in keys if k not in COEF_KEYS]
    if bad:
        raise SystemExit(f"{arg_name}: unknown key(s) {bad}. Valid: {ORDER}")
    return [COEF_KEYS[k] for k in ORDER if k in keys]  # dedupe + canonical order


# ---- Main -------------------------------------------------------------------

def main(show_keys, family_keys, correction=DEFAULT_CORRECTION):
    panel_coefs = _resolve(show_keys, "--show/--drop")
    family_coefs = _resolve(family_keys, "--family")
    if not panel_coefs:
        raise SystemExit("Nothing to plot: every panel was dropped.")
    needed_coefs = sorted(set(panel_coefs) | set(family_coefs))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(_find("glm_summary_long.csv"))
    draws = pd.read_parquet(_find("glm_bootstrap_draws.parquet"))
    with open(_find("glm_metadata.json"), encoding="utf-8") as f:
        meta = json.load(f)
    n_boot = int(meta["config"]["n_boot"])

    rats = sorted(
        summary.loc[summary["subset"] == SUBSET, "rat_id"].dropna().unique().tolist()
    )
    if not rats:
        raise ValueError(f"No rats found for subset={SUBSET!r}.")

    sub = summary.loc[
        (summary["subset"] == SUBSET)
        & (summary["C"] == FIT)
        & (summary["quantity"].isin(needed_coefs))
    ]
    if sub.duplicated(["quantity", "rat_id"]).any():
        raise ValueError("Duplicate summary rows for some coefficient/rat pairs.")

    dsub = draws.loc[
        (draws["subset"] == SUBSET)
        & (draws["C"] == FIT)
        & (draws["quantity"].isin(needed_coefs))
    ]

    # -- assemble table (all coefficients needed for plot OR correction family)
    rows = []
    for coef in needed_coefs:
        for rat in rats:
            m = sub.loc[(sub["quantity"] == coef) & (sub["rat_id"] == rat)]
            if m.empty:
                continue
            s = m.iloc[0]
            boot = dsub.loc[
                (dsub["quantity"] == coef) & (dsub["rat_id"] == rat), "value"
            ].to_numpy(dtype=float)
            if boot.size == 0:
                raise ValueError(f"No bootstrap draws for rat={rat}, coef={coef!r}.")
            rows.append(dict(
                coef=coef, rat_id=rat,
                value=float(s["value"]),
                ci_low=float(s["ci_low"]), ci_high=float(s["ci_high"]),
                p_raw=boot_p_value(boot, n_boot=n_boot),
            ))
    res = pd.DataFrame(rows)

    # -- multiple-comparison correction across the chosen family
    family = res.loc[res["coef"].isin(family_coefs)].copy()
    n_tests = len(family)
    expected = len(family_coefs) * len(rats)
    if n_tests != expected:
        raise ValueError(f"Expected {expected} family tests, got {n_tests}.")
    if family["p_raw"].isna().any():
        raise ValueError("Missing p-values in the correction family.")
    corr_fn, corr_name, sig_label = CORRECTIONS[correction]
    if n_tests:
        family["p_adj"], family["reject"] = corr_fn(
            family["p_raw"].to_numpy(), alpha=ALPHA
        )
        res = res.merge(
            family[["coef", "rat_id", "p_adj", "reject"]],
            on=["coef", "rat_id"], how="left", validate="one_to_one",
        )
    else:
        res["p_adj"] = np.nan
        res["reject"] = False
    res["in_family"] = res["coef"].isin(family_coefs)
    res["reject"] = res["reject"].astype("boolean").fillna(False).astype(bool)

    res["excludes_zero"] = (res["ci_low"] > 0) | (res["ci_high"] < 0)
    res["tier"] = np.select(
        [~res["excludes_zero"], res["reject"]], ["ns", "sig"], default="pointwise"
    )

    # Sanity flag: significant after correction but CI covers 0 (would be shown as grey).
    clash = res.loc[res["reject"] & ~res["excludes_zero"]]
    if not clash.empty:
        print(f"WARNING: {sig_label} but CI includes 0 (plotted grey):")
        print(clash[["coef", "rat_id", "p_raw", "p_adj", "ci_low", "ci_high"]])

    # -- file names
    tag = "" if panel_coefs == list(COEF_KEYS.values()) else (
        "_" + "-".join(k for k in ORDER if COEF_KEYS[k] in panel_coefs)
    )
    if correction != "bh":
        tag += f"_{correction}"
    res.to_csv(OUTPUT_DIR / f"glm_coefficient_forest_data{tag}.csv", index=False)

    # -- plot
    plt.rcParams.update({
        "font.size": 10.5, "axes.edgecolor": "#c9c8c0",
        "axes.linewidth": 0.9, "svg.fonttype": "none",
    })
    n_panels, n_rats = len(panel_coefs), len(rats)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(3.05 * n_panels, 0.52 * n_rats + 1.6),
        sharey=True, squeeze=False,
    )
    axes = axes[0]
    y_pos = {rat: i for i, rat in enumerate(rats)}

    for ax, coef in zip(axes, panel_coefs):
        color = PANEL_COLOR[coef]
        ax.axvline(0, color="#c9c8c0", linewidth=1.1, linestyle=(0, (4, 3)), zorder=1)

        for _, row in res.loc[res["coef"] == coef].iterrows():
            y = y_pos[row["rat_id"]]
            if row["tier"] == "ns":
                mec, mfc, bar, mew = GREY, GREY, GREY, 1.2
            elif row["tier"] == "pointwise":
                mec, mfc, bar, mew = color, "white", color, 1.6
            else:
                mec, mfc, bar, mew = color, color, color, 1.6

            ax.plot([row["ci_low"], row["ci_high"]], [y, y], color=bar,
                    linewidth=2.0, solid_capstyle="round", zorder=2)
            ax.plot(row["value"], y, marker="o", markersize=6.4,
                    markerfacecolor=mfc, markeredgecolor=mec,
                    markeredgewidth=mew, zorder=3)
            ax.annotate(f"{row['value']:+.2f}", xy=(row["ci_high"], y),
                        xytext=(4, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=7.4,
                        color=TEXT_SECONDARY, zorder=4)

        ax.set_title(PANEL_LABELS[coef], fontsize=10.2, color=TEXT_PRIMARY, pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(ax is axes[0])
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", colors=TEXT_SECONDARY, labelsize=8.6)
        ax.margins(x=0.32)
        ax.set_xlabel("log-odds", fontsize=8.6, color=TEXT_SECONDARY, labelpad=6)

    axes[0].set_yticks([y_pos[r] for r in rats])
    axes[0].set_yticklabels([f"Rat {r}" for r in rats], fontsize=9.4, color=TEXT_PRIMARY)
    axes[0].set_ylim(-0.8, n_rats - 0.2)
    axes[0].invert_yaxis()

    fig.suptitle(
        f"Pooled GLM log-odds coefficients, all {n_rats} rats ({FIT} fit)",
        fontsize=12.6, color=TEXT_PRIMARY, y=1.015, x=0.02, ha="left",
    )
    fam_names = ", ".join(PANEL_LABELS[c].split(" (current")[0] for c in family_coefs)
    fig.text(
        0.02, 0.965,
        f"95% session-level bootstrap CI (n={n_boot:,}). {corr_name} correction across "
        f"{len(family_coefs)} coefficients x {n_rats} rats ({n_tests} tests, "
        f"alpha={ALPHA:g}): {fam_names}.",
        fontsize=8.6, color=TEXT_SECONDARY, ha="left",
    )

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREY,
               markeredgecolor=GREY, markersize=7, label="CI includes 0"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=TEXT_SECONDARY, markersize=7, markeredgewidth=1.6,
               label=f"CI excludes 0 (not {sig_label})"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TEXT_SECONDARY,
               markeredgecolor=TEXT_SECONDARY, markersize=7, markeredgewidth=1.6,
               label=f"CI excludes 0, {sig_label}"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
               fontsize=8.8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.93])

    for ext in ("svg", "pdf"):
        fig.savefig(OUTPUT_DIR / f"glm_coefficient_forest{tag}.{ext}", bbox_inches="tight")
    plt.close(fig)

    print(f"Saved to {OUTPUT_DIR} (suffix: {tag or 'none'})")
    print(f"{corr_name} family: {family_coefs} -> {n_tests} tests")
    print(res.loc[res["coef"].isin(panel_coefs)]
          .groupby(["coef", "tier"]).size())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--show", nargs="+", default=DEFAULT_SHOW, metavar="KEY",
                        help=f"Panels to show. Keys: {ORDER}")
    parser.add_argument("--drop", nargs="+", default=[], metavar="KEY",
                        help="Panels to remove from --show")
    parser.add_argument("--family", nargs="+", default=DEFAULT_FAMILY,
                        metavar="KEY",
                        help="Coefficients forming the correction family (default: the "
                             "4 history terms, independent of what is shown)")
    parser.add_argument("--correction", choices=list(CORRECTIONS),
                        default=DEFAULT_CORRECTION,
                        help="Multiple-comparison correction (default: bh = Benjamini-Hochberg FDR)")
    args = parser.parse_args()

    show = [k for k in args.show if k not in args.drop]
    main(show, args.family, args.correction)