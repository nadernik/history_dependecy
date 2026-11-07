"""
Forest plot of pooled-GLM log-odds coefficients for all 12 rats.

Panels (left to right): beta_angle (current-trial slope; reference only),
beta_angle_n-1, beta_reward_n-1, beta_success_n-1, beta_failure_n-1.

Holm correction is applied jointly across the four tested history
coefficients x 12 rats (48 tests). beta_angle is displayed only as a scale
reference and is excluded from that correction family.

Outputs: CSV data, SVG and PDF in OUTPUT_DIR.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# ---- Statistical helpers ---------------------------------------------------

def holm_correction(pvals, alpha=0.05):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n, dtype=float)
    running_max = 0.0

    for rank, idx in enumerate(order):
        multiplier = n - rank
        candidate = pvals[idx] * multiplier
        running_max = max(running_max, candidate)
        adj[idx] = min(running_max, 1.0)

    reject = adj < alpha
    return adj, reject


def boot_p_value(draws, n_boot=None):
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return np.nan

    n = len(draws) if n_boot is None else n_boot
    frac_pos = np.mean(draws > 0)
    frac_neg = np.mean(draws < 0)
    p = 2 * min(frac_pos, frac_neg)
    floor = 2.0 / (n + 1)
    return min(max(p, floor), 1.0)


# ---- Paths -----------------------------------------------------------------

CANDIDATE_DATA_DIRS = [
    r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_4_GLM\glm_results_thesis_filtered",
    str(Path(__file__).resolve().parent),
]


def _find(filename_substr):
    """Find one matching file in the first candidate directory containing it."""
    for directory in CANDIDATE_DATA_DIRS:
        if not os.path.isdir(directory):
            continue

        matches = sorted(
            filename
            for filename in os.listdir(directory)
            if filename_substr in filename
        )
        if matches:
            return os.path.join(directory, matches[0])

    raise FileNotFoundError(
        f"No file containing {filename_substr!r} was found in: "
        f"{CANDIDATE_DATA_DIRS}"
    )


OUTPUT_DIR = Path(
    r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes"
    r"\ale_thesis_fig_tab\images_thesis\section_4"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---- Load data -------------------------------------------------------------

summary = pd.read_csv(_find("glm_summary_long.csv"))
draws = pd.read_parquet(_find("glm_bootstrap_draws.parquet"))
with open(_find("glm_metadata.json"), encoding="utf-8") as file:
    meta = json.load(file)

ALPHA = 0.05
FIT = "unpenalized"
SUBSET = "pooled"

REFERENCE_COEF = "beta_angle"
TESTED_COEFS = [
    "beta_angle_n-1",
    "beta_reward_n-1",
    "beta_success_n-1",
    "beta_failure_n-1",
]
PANEL_COEFS = [REFERENCE_COEF] + TESTED_COEFS

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

required_summary = {"subset", "C", "quantity", "rat_id", "value", "ci_low", "ci_high"}
required_draws = {"subset", "C", "quantity", "rat_id", "value"}

missing_summary = required_summary.difference(summary.columns)
missing_draws = required_draws.difference(draws.columns)
if missing_summary:
    raise KeyError(f"Missing summary columns: {sorted(missing_summary)}")
if missing_draws:
    raise KeyError(f"Missing bootstrap-draw columns: {sorted(missing_draws)}")

RATS = sorted(
    summary.loc[summary["subset"] == SUBSET, "rat_id"].dropna().unique().tolist()
)
N_BOOT = int(meta["config"]["n_boot"])

if not RATS:
    raise ValueError(f"No rats found for subset={SUBSET!r}.")

sub = summary.loc[
    (summary["subset"] == SUBSET)
    & (summary["C"] == FIT)
    & (summary["quantity"].isin(PANEL_COEFS))
].copy()

# Each rat/coefficient combination must occur exactly once in the summary.
duplicates = sub.duplicated(["quantity", "rat_id"], keep=False)
if duplicates.any():
    duplicate_keys = (
        sub.loc[duplicates, ["quantity", "rat_id"]]
        .drop_duplicates()
        .to_dict("records")
    )
    raise ValueError(
        "Duplicate summary rows found for coefficient/rat combinations: "
        f"{duplicate_keys}"
    )


# ---- Assemble plotting and testing table ----------------------------------

rows = []
for coef in PANEL_COEFS:
    for rat in RATS:
        matched_summary = sub.loc[
            (sub["quantity"] == coef) & (sub["rat_id"] == rat)
        ]
        if matched_summary.empty:
            continue

        summary_row = matched_summary.iloc[0]
        bootstrap_values = draws.loc[
            (draws["rat_id"] == rat)
            & (draws["subset"] == SUBSET)
            & (draws["C"] == FIT)
            & (draws["quantity"] == coef),
            "value",
        ].to_numpy(dtype=float)

        if bootstrap_values.size == 0:
            raise ValueError(
                f"No bootstrap draws found for rat={rat}, coefficient={coef!r}."
            )

        rows.append(
            {
                "coef": coef,
                "rat_id": rat,
                "value": summary_row["value"],
                "ci_low": summary_row["ci_low"],
                "ci_high": summary_row["ci_high"],
                "p_raw": boot_p_value(bootstrap_values, n_boot=N_BOOT),
            }
        )

res = pd.DataFrame.from_records(rows)
if res.empty:
    raise ValueError("No coefficient rows were assembled; check FIT, SUBSET and quantities.")

for column in ["value", "ci_low", "ci_high", "p_raw"]:
    res[column] = pd.to_numeric(res[column], errors="raise")

# Holm family: four tested coefficients x all rats.
family = res.loc[res["coef"].isin(TESTED_COEFS)].copy()
expected_tests = len(TESTED_COEFS) * len(RATS)
if len(family) != expected_tests:
    observed = family.groupby("coef")["rat_id"].nunique().to_dict()
    raise ValueError(
        f"Expected {expected_tests} Holm-family tests, got {len(family)}. "
        f"Observed unique rats by coefficient: {observed}"
    )

if family["p_raw"].isna().any():
    bad = family.loc[family["p_raw"].isna(), ["coef", "rat_id"]].to_dict("records")
    raise ValueError(f"Cannot apply Holm correction: missing p-values for {bad}")

p_adj, reject = holm_correction(family["p_raw"].to_numpy(), alpha=ALPHA)
family["p_holm"] = p_adj
family["holm_reject"] = reject

res = res.merge(
    family[["coef", "rat_id", "p_holm", "holm_reject"]],
    on=["coef", "rat_id"],
    how="left",
    validate="one_to_one",
)

# beta_angle legitimately has no Holm result. The explicit nullable-Boolean
# conversion avoids pandas' deprecated silent object downcasting in fillna().
res["holm_reject"] = (
    res["holm_reject"].astype("boolean").fillna(False).astype(bool)
)

res["excludes_zero"] = (res["ci_low"] > 0) | (res["ci_high"] < 0)

# Vectorized classification avoids the slow/problematic row-wise DataFrame.apply.
res["tier"] = np.select(
    [
        ~res["excludes_zero"],
        res["holm_reject"],
    ],
    [
        "ns",
        "holm",
    ],
    default="pointwise",
)
res["tier"] = pd.Categorical(
    res["tier"],
    categories=["ns", "pointwise", "holm"],
    ordered=True,
)

res.to_csv(OUTPUT_DIR / "glm_coefficient_forest_data.csv", index=False)


# ---- Plot ------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.size": 10.5,
        "axes.edgecolor": "#c9c8c0",
        "axes.linewidth": 0.9,
        "svg.fonttype": "none",
    }
)

n_panels = len(PANEL_COEFS)
n_rats = len(RATS)
fig, axes = plt.subplots(
    1,
    n_panels,
    figsize=(3.05 * n_panels, 0.52 * n_rats + 1.6),
    sharey=True,
)

y_pos = {rat: index for index, rat in enumerate(RATS)}

for ax, coef in zip(axes, PANEL_COEFS):
    color = PANEL_COLOR[coef]
    panel_data = res.loc[res["coef"] == coef].sort_values("rat_id")

    ax.axvline(
        0,
        color="#c9c8c0",
        linewidth=1.1,
        linestyle=(0, (4, 3)),
        zorder=1,
    )

    for _, row in panel_data.iterrows():
        y = y_pos[row["rat_id"]]
        tier_value = row["tier"]

        if tier_value == "ns":
            marker_edge = GREY
            marker_face = GREY
            bar_color = GREY
            marker_edge_width = 1.2
        elif tier_value == "pointwise":
            marker_edge = color
            marker_face = "white"
            bar_color = color
            marker_edge_width = 1.6
        else:
            marker_edge = color
            marker_face = color
            bar_color = color
            marker_edge_width = 1.6

        ax.plot(
            [row["ci_low"], row["ci_high"]],
            [y, y],
            color=bar_color,
            linewidth=2.0,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            row["value"],
            y,
            marker="o",
            markersize=6.4,
            markerfacecolor=marker_face,
            markeredgecolor=marker_edge,
            markeredgewidth=marker_edge_width,
            zorder=3,
        )
        ax.annotate(
            f"{row['value']:+.2f}",
            xy=(row["ci_high"], y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=7.4,
            color=TEXT_SECONDARY,
            zorder=4,
        )

    ax.set_title(PANEL_LABELS[coef], fontsize=10.2, color=TEXT_PRIMARY, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(coef == PANEL_COEFS[0])
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=TEXT_SECONDARY, labelsize=8.6)
    ax.margins(x=0.32)
    ax.set_xlabel("log-odds", fontsize=8.6, color=TEXT_SECONDARY, labelpad=6)

axes[0].set_yticks([y_pos[rat] for rat in RATS])
axes[0].set_yticklabels(
    [f"Rat {rat}" for rat in RATS],
    fontsize=9.4,
    color=TEXT_PRIMARY,
)
axes[0].set_ylim(-0.8, n_rats - 0.2)
axes[0].invert_yaxis()

fig.suptitle(
    "Pooled GLM log-odds coefficients, all 12 rats (unpenalized fit)",
    fontsize=12.6,
    color=TEXT_PRIMARY,
    y=1.015,
    x=0.02,
    ha="left",
)
fig.text(
    0.02,
    0.965,
    f"95% session-level bootstrap CI (n={N_BOOT:,}). Holm correction across "
    "the 4 tested history coefficients x 12 rats (48 tests, alpha=.05); "
    r"$\beta_{angle}$ shown for scale reference, excluded from that family.",
    fontsize=8.6,
    color=TEXT_SECONDARY,
    ha="left",
)

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=GREY,
        markeredgecolor=GREY,
        markersize=7,
        label="CI includes 0",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="white",
        markeredgecolor="#52514e",
        markersize=7,
        markeredgewidth=1.6,
        label="CI excludes 0 (not Holm-significant)",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="#52514e",
        markeredgecolor="#52514e",
        markersize=7,
        markeredgewidth=1.6,
        label="CI excludes 0, Holm-significant",
    ),
]
fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=3,
    frameon=False,
    fontsize=8.8,
    bbox_to_anchor=(0.5, -0.02),
)

fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.93])

svg_path = OUTPUT_DIR / "glm_coefficient_forest.svg"
pdf_path = OUTPUT_DIR / "glm_coefficient_forest.pdf"
fig.savefig(svg_path, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

print(f"Saved data and figures to: {OUTPUT_DIR}")
print(res.groupby(["coef", "tier"], observed=True).size())


"""
Forest plot of the pooled-GLM log-odds coefficients, all 12 rats.

Panels (left to right): beta_angle (current-trial slope, REFERENCE only -
not part of the correction family), beta_angle_n-1, beta_reward_n-1,
beta_success_n-1, beta_failure_n-1. Intercept dropped per Alessandro.

Fit level: unpenalized (MLE), subset = pooled -- matches the convention
already used for every other Section 4 pooled-GLM table/figure.

Significance tiers (per Alessandro's spec):
  - grey                         : 95% bootstrap CI includes 0
  - open marker, colourful bar   : CI excludes 0, but NOT significant after
                                    Holm correction
  - filled marker, colourful bar : CI excludes 0 AND significant after Holm
                                    correction

Holm correction (holm_correction() from Alessandro's stats_helpers.py,
percentile-bootstrap p-values via boot_p_value() from the same file) is
applied within the family of the 4 TESTED history coefficients x 12 rats
= 48 tests (beta_angle_n-1, beta_reward_n-1, beta_success_n-1,
beta_failure_n-1). This mirrors the existing "joint 48-test" convention
used elsewhere in this project for the degree-scale contrasts. beta_angle
is shown for scale reference only and is excluded from the correction
family (it can reach the "open" tier via its own CI, but never "filled").

Outputs (working directory): SVG + PDF, vector, no PNG -- matching every
other pooled-GLM figure in this project.
"""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---- Alessandro's stats helpers (pasted verbatim) --------------------------

def holm_correction(pvals, alpha=0.05):
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        m = n - rank
        candidate = pvals[idx] * m
        running_max = max(running_max, candidate)
        adj[idx] = min(running_max, 1.0)
    reject = adj < alpha
    return adj, reject


def boot_p_value(draws, n_boot=None):
    draws = np.asarray(draws, dtype=float)
    n = len(draws) if n_boot is None else n_boot
    frac_pos = np.mean(draws > 0)
    frac_neg = np.mean(draws < 0)
    p = 2 * min(frac_pos, frac_neg)
    floor = 2.0 / (n + 1)
    return max(p, floor)


# ---- Load data ---------------------------------------------------------

# Real local folder (Alessandro's machine) first, then this script's own
# folder (in case the 3 data files sit next to it instead), then the
# uploaded-files dir from the Claude session that built this figure -- same
# fallback pattern used by the Section 3 scripts in this project.
CANDIDATE_DATA_DIRS = [
    r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_4_GLM\glm_results_thesis_filtered"]

def _find(filename_substr):
    """Locate the one file whose name contains filename_substr in the first
    candidate directory that has it, so this works whether files keep their
    original names or an uploaded hash-prefixed name (e.g.
    'a8411f52-glm_summary_long.csv')."""
    for d in CANDIDATE_DATA_DIRS:
        if not os.path.isdir(d):
            continue
        matches = [f for f in os.listdir(d) if filename_substr in f]
        if matches:
            return os.path.join(d, sorted(matches)[0])
    raise FileNotFoundError(
        f"No file containing '{filename_substr}' in any of: {CANDIDATE_DATA_DIRS}"
    )


OUTPUT_DIR = Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\ale_thesis_fig_tab\images_thesis\section_4")
os.makedirs(OUTPUT_DIR, exist_ok=True)


summary = pd.read_csv(_find("glm_summary_long.csv"))
draws = pd.read_parquet(_find("glm_bootstrap_draws.parquet"))
with open(_find("glm_metadata.json")) as f:
    meta = json.load(f)

ALPHA = 0.05
FIT = "unpenalized"
SUBSET = "pooled"

REFERENCE_COEF = "beta_angle"
TESTED_COEFS = ["beta_angle_n-1", "beta_reward_n-1", "beta_success_n-1", "beta_failure_n-1"]
PANEL_COEFS = [REFERENCE_COEF] + TESTED_COEFS

PANEL_LABELS = {
    "beta_angle": r"$\beta_{angle}$ (current trial, reference)",
    "beta_angle_n-1": r"$\beta_{angle}$ (n-1)",
    "beta_reward_n-1": r"$\beta_{reward}$ (n-1)",
    "beta_success_n-1": r"$\beta_{success}$ (n-1)",
    "beta_failure_n-1": r"$\beta_{failure}$ (n-1)",
}

# Fixed categorical order (validated: node scripts/validate_palette.js
# "#2a78d6,#eb6834,#1baf7a,#eda100,#e87ba4" --mode light -> ALL CHECKS PASS)
PANEL_COLOR = {
    "beta_angle": "#e87ba4",          # magenta -- reference panel, kept visually distinct
    "beta_angle_n-1": "#2a78d6",       # blue
    "beta_reward_n-1": "#eb6834",      # orange
    "beta_success_n-1": "#1baf7a",     # aqua
    "beta_failure_n-1": "#eda100",     # yellow
}
GREY = "#9a9892"
TEXT_SECONDARY = "#52514e"
TEXT_PRIMARY = "#0b0b0b"

RATS = sorted(summary.loc[summary["subset"] == SUBSET, "rat_id"].unique().tolist())
N_BOOT = meta["config"]["n_boot"]

sub = summary[(summary["subset"] == SUBSET) & (summary["C"] == FIT) &
              (summary["quantity"].isin(PANEL_COEFS))].copy()

rows = []
for coef in PANEL_COEFS:
    for rat in RATS:
        r = sub[(sub["quantity"] == coef) & (sub["rat_id"] == rat)]
        if r.empty:
            continue
        r = r.iloc[0]
        d = draws[(draws["rat_id"] == rat) & (draws["subset"] == SUBSET) &
                   (draws["C"] == FIT) & (draws["quantity"] == coef)]["value"].values
        p_raw = boot_p_value(d, n_boot=N_BOOT)
        rows.append({
            "coef": coef, "rat_id": rat,
            "value": r["value"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
            "p_raw": p_raw,
        })

res = pd.DataFrame(rows)

# Holm correction, family = 4 tested coefficients x 12 rats = 48 tests
family = res[res["coef"].isin(TESTED_COEFS)].copy()
p_adj, reject = holm_correction(family["p_raw"].values, alpha=ALPHA)
family["p_holm"] = p_adj
family["holm_reject"] = reject
res = res.merge(family[["coef", "rat_id", "p_holm", "holm_reject"]],
                 on=["coef", "rat_id"], how="left")
res["holm_reject"] = res["holm_reject"].fillna(False)

res["excludes_zero"] = (res["ci_low"] > 0) | (res["ci_high"] < 0)


def tier(row):
    if not row["excludes_zero"]:
        return "ns"
    if row["holm_reject"]:
        return "holm"
    return "pointwise"


res["tier"] = res.apply(tier, axis=1)

assert len(family) == 4 * len(RATS), f"expected 48 tests, got {len(family)}"

res.to_csv(os.path.join(OUTPUT_DIR, "glm_coefficient_forest_data.csv"), index=False)

# ---- Plot ---------------------------------------------------------------

plt.rcParams.update({
    "font.size": 10.5,
    "axes.edgecolor": "#c9c8c0",
    "axes.linewidth": 0.9,
    "svg.fonttype": "none",
})

n_panels = len(PANEL_COEFS)
n_rats = len(RATS)
fig, axes = plt.subplots(1, n_panels, figsize=(3.05 * n_panels, 0.52 * n_rats + 1.6),
                          sharey=True)

y_pos = {rat: i for i, rat in enumerate(RATS)}  # rat 2 at bottom -> top? use ascending top->bottom

for ax, coef in zip(axes, PANEL_COEFS):
    color = PANEL_COLOR[coef]
    d = res[res["coef"] == coef].sort_values("rat_id")
    ax.axvline(0, color="#c9c8c0", lw=1.1, linestyle=(0, (4, 3)), zorder=1)

    for _, row in d.iterrows():
        y = y_pos[row["rat_id"]]
        t = row["tier"]
        if t == "ns":
            mec, mfc, barcol, mew = GREY, GREY, GREY, 1.2
        elif t == "pointwise":
            mec, mfc, barcol, mew = color, "white", color, 1.6
        else:  # holm
            mec, mfc, barcol, mew = color, color, color, 1.6

        ax.plot([row["ci_low"], row["ci_high"]], [y, y], color=barcol, lw=2.0,
                 solid_capstyle="round", zorder=2)
        ax.plot(row["value"], y, marker="o", markersize=6.4,
                 markerfacecolor=mfc, markeredgecolor=mec, markeredgewidth=mew,
                 zorder=3)
        # direct value label (relief for low-contrast hues; also just useful)
        ax.text(ax.get_xlim()[1] if False else row["ci_high"], y,
                 f"  {row['value']:+.2f}", va="center", ha="left",
                 fontsize=7.4, color=TEXT_SECONDARY, zorder=4)

    ax.set_title(PANEL_LABELS[coef], fontsize=10.2, color=TEXT_PRIMARY, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(coef == PANEL_COEFS[0])
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", colors=TEXT_SECONDARY, labelsize=8.6)
    ax.margins(x=0.32)
    ax.set_xlabel("log-odds", fontsize=8.6, color=TEXT_SECONDARY, labelpad=6)
    for lbl in ax.get_xticklabels():
        lbl.set_color(TEXT_SECONDARY)

axes[0].set_yticks([y_pos[r] for r in RATS])
axes[0].set_yticklabels([f"Rat {r}" for r in RATS], fontsize=9.4, color=TEXT_PRIMARY)
axes[0].set_ylim(-0.8, n_rats - 0.2)
axes[0].invert_yaxis()

fig.suptitle("Pooled GLM log-odds coefficients, all 12 rats (unpenalized fit)",
             fontsize=12.6, color=TEXT_PRIMARY, y=1.015, x=0.02, ha="left")
fig.text(0.02, 0.965,
         "95% session-level bootstrap CI (n=5,000). Holm correction across the "
         "4 tested history coefficients x 12 rats (48 tests, alpha=.05); "
         r"$\beta_{angle}$ shown for scale reference, excluded from that family.",
         fontsize=8.6, color=TEXT_SECONDARY, ha="left")

legend_elems = [
    Line2D([0], [0], marker="o", color="none", markerfacecolor=GREY,
           markeredgecolor=GREY, markersize=7, label="CI includes 0"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
           markeredgecolor="#52514e", markersize=7, markeredgewidth=1.6,
           label="CI excludes 0 (not Holm-significant)"),
    Line2D([0], [0], marker="o", color="none", markerfacecolor="#52514e",
           markeredgecolor="#52514e", markersize=7, markeredgewidth=1.6,
           label="CI excludes 0, Holm-significant"),
]
fig.legend(handles=legend_elems, loc="lower center", ncol=3, frameon=False,
           fontsize=8.8, bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.93])

fig.savefig(os.path.join(OUTPUT_DIR, "glm_coefficient_forest.svg"), bbox_inches="tight")
fig.savefig(os.path.join(OUTPUT_DIR, "glm_coefficient_forest.pdf"), bbox_inches="tight")
print(f"saved to {OUTPUT_DIR}")
print(res.groupby(["coef", "tier"]).size())