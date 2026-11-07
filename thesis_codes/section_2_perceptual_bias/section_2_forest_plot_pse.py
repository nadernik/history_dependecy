import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

#D = Path("/home/claude/section2_perceptual_bias/data")
OUT = Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\ale_thesis_fig_tab\images_thesis\section_2")

summary = pd.read_csv(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_2_perceptual_bias\2.0_tables_perceptual_bias_revised\summary_perceptual_bias.csv")
bins = pd.read_csv(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_2_perceptual_bias\2.0_tables_perceptual_bias_revised\bin_fits_perceptual_bias.csv")

MOD_LABEL = {1: "Tactile", 2: "Visual", 3: "Visuotactile"}
MOD_ORDER = [1, 2, 3]
RATS = sorted(summary["rat"].unique())  # 2,3,4,5,6,7,9,10,11,12,13,16
y_pos = {r: i for i, r in enumerate(RATS)}

# boundary-hit flag, same definition as the table script
bins["at_bound"] = (
    bins["sigma_hat_at_lower_bound"]
    | bins["sigma_hat_at_upper_bound"]
    | (bins["gamma_hat"] >= 0.399)
    | (bins["lapse_hat"] >= 0.399)
)
extreme_bins = bins[bins["bin"].isin([1, 3])]
boundary_flag = extreme_bins.groupby(["rat", "modality"])["at_bound"].any()

# colors keyed to modality (per Alessandro): tactile=green, visual=blue,
# visuotactile=red. Significance is now encoded by fill state, not hue:
# grey+open = not significant; modality-color+open = pointwise significant
# (uncorrected); modality-color+filled = FDR-significant.
COL_GREY = "#999999"
MOD_COLOR = {1: "#2E8B57", 2: "#3B78C2", 3: "#C23B3B"}  # tactile, visual, visuotactile

def status_style(row, mod_color):
    if row["fdr_reject"]:
        return mod_color, True  # colored, filled
    if row["pointwise_ci_excludes_0"]:
        return mod_color, False  # colored, open
    return COL_GREY, False  # grey, open

fig, axes = plt.subplots(1, 3, figsize=(11, 6.5), sharey=True)

for ax, mod in zip(axes, MOD_ORDER):
    sub = summary[summary["modality"] == mod].set_index("rat")
    mod_color = MOD_COLOR[mod]
    ax.axvline(0, color="#B0B0B0", lw=1, ls="--", zorder=1)
    for rat in RATS:
        row = sub.loc[rat]
        y = y_pos[rat]
        color, filled = status_style(row, mod_color)
        lo, hi = row["ci_lo"], row["ci_hi"]
        est = row["delta_pse_hat"]
        ax.plot([lo, hi], [y, y], color=color, lw=1.6, zorder=2, solid_capstyle="round")
        ax.scatter(
            [est], [y],
            s=46,
            facecolor=color if filled else "white",
            edgecolor=color,
            linewidth=1.4,
            zorder=3,
        )
        if boundary_flag.get((rat, mod), False):
            ax.annotate(
                "†", (hi, y), xytext=(4, -3), textcoords="offset points",
                fontsize=9, color="#555555",
            )
    ax.set_title(MOD_LABEL[mod], fontsize=12, fontweight="bold", color=mod_color)
    ax.set_xlabel("ΔPSE (°)\n[hard → easy]", fontsize=9.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(mod == MOD_ORDER[0])
    ax.tick_params(axis="y", left=(mod == MOD_ORDER[0]))
    ax.grid(axis="x", color="#EDEDED", lw=0.8, zorder=0)

axes[0].set_yticks([y_pos[r] for r in RATS])
axes[0].set_yticklabels([f"Rat {r}" for r in RATS], fontsize=10)
axes[0].set_ylim(-0.7, len(RATS) - 0.3)
axes[0].set_ylabel("")

legend_elems = [
    Line2D([0], [0], marker="o", color="#666666", markerfacecolor="#666666",
           markeredgecolor="#666666", lw=1.6, markersize=7,
           label="FDR-significant (q<.05): filled, modality color"),
    Line2D([0], [0], marker="o", color="#666666", markerfacecolor="white",
           markeredgecolor="#666666", lw=1.6, markersize=7,
           label="Pointwise sig. (uncorrected), not FDR: open, modality color"),
    Line2D([0], [0], marker="o", color=COL_GREY, markerfacecolor="white",
           markeredgecolor=COL_GREY, lw=1.6, markersize=7, label="Not significant: open, grey"),
    Line2D([0], [0], marker="$†$", color="#555555", lw=0, markersize=9,
           label="Hard/easy bin fit at parameter bound"),
]
fig.legend(
    handles=legend_elems, loc="lower center", ncol=2, frameon=False,
    fontsize=9.5, bbox_to_anchor=(0.5, -0.06),
)

fig.suptitle(
    "Primary perceptual-history contrast ($\\Delta$PSE, hard→easy) by rat and modality",
    fontsize=13, y=1.02,
)
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(OUT / "forest_pse_by_modality.svg", dpi=220, bbox_inches="tight", facecolor="white")
print("saved", OUT / "forest_pse_by_modality.svg")