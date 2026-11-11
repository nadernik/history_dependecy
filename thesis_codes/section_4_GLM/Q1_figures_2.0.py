"""Section 4 Q1: psychometric-versus-GLM convergence figures.

This is a rendering script, not an inference script. All sign-agreement,
Spearman, Holm-adjusted p-value, and geometric-agreement statistics are read
from the canonical outputs of q1_cross_method_comparison_updated.py.

Outputs
-------
figA_q1_sign_rank_scatter.svg/.pdf
figB_q1_geometric_plane.svg/.pdf
table_q1_convergence.tex
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import chi2


# =============================================================================
# CONFIG: EDIT THESE PATHS
# =============================================================================

PROJECT_ROOT = Path(
    r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes"
)

Q1_DIR = PROJECT_ROOT / "section_4_GLM" / "Q1_results_2.0"
PSYCH_TABLE_DIR = (
    PROJECT_ROOT
    / "section_3_choice_reward"
    / "choice_reward_joint_results"
    / "filtered"
    / "pooled"
    / "tables"
)
GLM_DIR = PROJECT_ROOT / "section_4_GLM" / "fit_results" / "pooled_filtered"
OUTPUT_DIR = PROJECT_ROOT / "ale_thesis_fig_tab" / "section_4"

PATHS = {
    "layer1_summary": Q1_DIR / "layer1_sign_agreement.csv",
    "layer1_per_rat": Q1_DIR / "layer1_per_rat_signs.csv",
    "layer2": Q1_DIR / "layer2_spearman.csv",
    "layer3_per_rat": Q1_DIR / "layer3_per_rat_geometry.csv",
    "layer3_cohort": Q1_DIR / "layer3_cohort_summary.csv",
    "psych_covariance": PSYCH_TABLE_DIR / "contrast_covariance_long.csv",
    "glm_draws": GLM_DIR / "glm_bootstrap_draws.parquet",
}

GLM_C = "unpenalized"
GLM_SUBSET = "pooled"
CI_LEVEL = 95

QUANTITIES = ["delta_mu_rew", "delta_mu_unrew", "D", "P"]
LABELS = {
    "delta_mu_rew": r"$\Delta\mu_R$",
    "delta_mu_unrew": r"$\Delta\mu_U$",
    "D": r"$D$",
    "P": r"$P$",
}


# =============================================================================
# LOAD AND VALIDATE CANONICAL RESULTS
# =============================================================================

def load_inputs():
    missing = [str(path) for path in PATHS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n  " + "\n  ".join(missing))

    data = {
        "l1_summary": pd.read_csv(PATHS["layer1_summary"]),
        "l1_per_rat": pd.read_csv(PATHS["layer1_per_rat"]),
        "l2": pd.read_csv(PATHS["layer2"]),
        "l3_per_rat": pd.read_csv(PATHS["layer3_per_rat"]),
        "l3_cohort": pd.read_csv(PATHS["layer3_cohort"]).iloc[0],
        "psych_cov": pd.read_csv(PATHS["psych_covariance"]),
    }

    rats = sorted(data["l1_per_rat"]["rat_id"].unique().tolist())
    if len(rats) != 12:
        raise ValueError(f"Expected 12 rats, found {len(rats)}: {rats}")
    if len(data["l3_per_rat"]) != len(rats):
        raise ValueError(
            f"Layer 3 has {len(data['l3_per_rat'])} rows; expected {len(rats)}."
        )
    if int(data["l3_cohort"]["n_rats"]) != len(rats):
        raise ValueError("Layer 3 cohort count does not match the per-rat table.")

    expected_l2 = {
        "formal_primary_spearman",
        "p_holm_beta_aligned_primary4",
        "reject_holm_05_beta_aligned_primary4",
    }
    missing_l2 = expected_l2.difference(data["l2"].columns)
    if missing_l2:
        raise ValueError(
            "Layer 2 is not from the corrected Q1 script; missing columns: "
            f"{sorted(missing_l2)}"
        )

    return data, rats


# =============================================================================
# STYLE
# =============================================================================

ACCENT = "#2a78d6"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
})


def save_figure(fig, stem):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# FIGURE A: SIGN AND RANK AGREEMENT
# =============================================================================

def build_figure_a(data, rats):
    per_rat = data["l1_per_rat"]
    l1 = data["l1_summary"].set_index("quantity")
    l2 = data["l2"].set_index("quantity")
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.6))

    for ax, quantity in zip(axes, QUANTITIES):
        rows = per_rat[per_rat["quantity"] == quantity].set_index("rat_id").loc[rats]
        xs = rows["psych_value"].to_numpy(dtype=float)
        ys = rows["glm_value_beta_aligned"].to_numpy(dtype=float)
        qualifies = rows["qualifies_option3"].astype(bool).to_numpy()

        ax.axhline(0, color=MUTED, lw=0.8, zorder=0)
        ax.axvline(0, color=MUTED, lw=0.8, zorder=0)

        for rat, x, y, selected in zip(rats, xs, ys, qualifies):
            ax.plot(
                x, y, marker="o", linestyle="none", ms=7, zorder=3,
                mfc=ACCENT if selected else "white",
                mec="white" if selected else ACCENT,
                mew=0.7 if selected else 1.3,
            )
            ax.annotate(
                str(rat), (x, y), textcoords="offset points", xytext=(5, 4),
                fontsize=6.5, color=MUTED, zorder=4,
            )

        r1 = l1.loc[quantity]
        r2 = l2.loc[quantity]
        q_holm = float(r2["p_holm_beta_aligned_primary4"])
        ax.text(
            0.04, 0.96,
            f"ρ = {r2['rho_beta_aligned']:.2f}\n"
            f"pHolm = {q_holm:.3f}\n"
            f"sign agreement: {int(r1['opt1_match'])}/{int(r1['opt1_denominator'])}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
            bbox=dict(facecolor="white", edgecolor=GRID, lw=0.7, pad=3),
        )

        ax.set_title(LABELS[quantity], fontsize=11, fontweight="bold")
        ax.set_xlabel("Psychometric estimate (°)", fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=3, labelsize=8)
        ax.grid(color=GRID, lw=0.6, zorder=-1)
        xpad = 0.12 * max(np.ptp(xs), 1.0)
        ypad = 0.12 * max(np.ptp(ys), 0.05)
        ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
        ax.set_ylim(ys.min() - ypad, ys.max() + ypad)

    axes[0].set_ylabel("GLM, sign-aligned (log-odds)", fontsize=8.5)
    legend = [
        Line2D([0], [0], marker="o", color="none", mfc=ACCENT,
               mec="white", ms=7, label="CI-gated sensitivity set"),
        Line2D([0], [0], marker="o", color="none", mfc="white",
               mec=ACCENT, ms=7, label="set aside in CI-gated sensitivity"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=8.5)
    fig.suptitle(
        "Psychometric–GLM sign and rank agreement (N = 12 rats)",
        fontsize=11.5, y=1.04,
    )
    fig.tight_layout()
    save_figure(fig, "figA_q1_sign_rank_scatter")


# =============================================================================
# FIGURE B: CHOICE–OUTCOME PLANES
# =============================================================================

def covariance_entry(frame, rat, row_name, column_name):
    row = frame[
        (frame["rat"] == rat)
        & (frame["contrast_row"] == row_name)
        & (frame["contrast_column"] == column_name)
    ]
    if len(row) != 1:
        raise ValueError(
            f"Expected one covariance entry for rat {rat}, "
            f"{row_name} x {column_name}; found {len(row)}."
        )
    return float(row.iloc[0]["covariance"])


def psych_covariance(frame, rat):
    r, u = "delta_mu_rew", "delta_mu_unrew"
    return np.array([
        [covariance_entry(frame, rat, r, r), covariance_entry(frame, rat, r, u)],
        [covariance_entry(frame, rat, u, r), covariance_entry(frame, rat, u, u)],
    ])


def load_glm_covariances(rats):
    draws = pd.read_parquet(PATHS["glm_draws"])
    draws = draws[
        (draws["C"] == GLM_C)
        & (draws["subset"] == GLM_SUBSET)
        & draws["quantity"].isin(["beta_success_n-1", "beta_failure_n-1"])
    ]
    wide = draws.pivot_table(
        index=["rat_id", "boot_idx"], columns="quantity", values="value"
    ).reset_index()
    output = {}
    for rat in rats:
        subset = wide[wide["rat_id"] == rat]
        if subset.empty:
            raise ValueError(f"No GLM bootstrap draws found for rat {rat}.")
        # Cov(-success, -failure) = Cov(success, failure).
        output[rat] = np.cov(
            subset["beta_success_n-1"].to_numpy(),
            subset["beta_failure_n-1"].to_numpy(),
        )
    return output


def draw_ellipse(ax, mean, covariance, color):
    values, vectors = np.linalg.eigh(covariance)
    values = np.clip(values, 0.0, None)
    order = values.argsort()[::-1]
    values, vectors = values[order], vectors[:, order]
    scale = np.sqrt(chi2.ppf(CI_LEVEL / 100.0, df=2))
    width, height = 2.0 * scale * np.sqrt(values)
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    ax.add_patch(mpatches.Ellipse(
        mean, width, height, angle=angle, facecolor=color,
        edgecolor=color, lw=1.0, alpha=0.14, zorder=1,
    ))


def ellipse_reach(mean, covariance):
    largest = max(float(np.linalg.eigvalsh(covariance).max()), 0.0)
    radius = np.sqrt(chi2.ppf(CI_LEVEL / 100.0, df=2) * largest)
    return max(abs(mean[0]) + radius, abs(mean[1]) + radius)


def draw_plane(ax, means, covariances, rats, colors, xlabel, ylabel, unit):
    limit = 1.10 * max(ellipse_reach(means[r], covariances[r]) for r in rats)
    for rat in rats:
        mean, covariance, color = means[rat], covariances[rat], colors[rat]
        draw_ellipse(ax, mean, covariance, color)
        ax.plot(*mean, marker="o", linestyle="none", mfc=color, mec="white",
                mew=0.7, ms=7, zorder=3)
        ax.annotate(str(rat), mean, textcoords="offset points", xytext=(6, 5),
                    fontsize=7.5, fontweight="bold", color=color, zorder=4)

    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvline(0, color=MUTED, lw=0.8)
    diagonal = np.array([-1.4 * limit, 1.4 * limit])
    ax.plot(diagonal, diagonal, color=GRID, lw=1.0, ls=(0, (4, 3)))
    ax.plot(diagonal, -diagonal, color=GRID, lw=1.0, ls=(0, (4, 3)))
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"{xlabel} ({unit})")
    ax.set_ylabel(f"{ylabel} ({unit})")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, lw=0.6, zorder=-2)


def build_figure_b(data, rats):
    geometry = data["l3_per_rat"].set_index("rat_id").loc[rats]
    psych_means = {
        r: geometry.loc[r, ["psych_R", "psych_U"]].to_numpy(dtype=float)
        for r in rats
    }
    glm_means = {
        r: geometry.loc[r, ["glm_R_beta_aligned", "glm_U_beta_aligned"]]
        .to_numpy(dtype=float)
        for r in rats
    }
    psych_covariances = {r: psych_covariance(data["psych_cov"], r) for r in rats}
    glm_covariances = load_glm_covariances(rats)

    palette = [
        "#009ab1", "#c46183", "#429c5a", "#9372c8", "#a58100", "#248fcc",
        "#c86556", "#009f89", "#b167ab", "#7c9128", "#6880d4", "#be7125",
    ]
    colors = dict(zip(rats, palette))
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 5.6))

    draw_plane(left, psych_means, psych_covariances, rats, colors,
               r"$\Delta\mu_R$", r"$\Delta\mu_U$", "°")
    left.set_title("Psychometric", fontsize=11, fontweight="bold")
    draw_plane(right, glm_means, glm_covariances, rats, colors,
               r"$-\beta_{success,n-1}$", r"$-\beta_{failure,n-1}$", "log-odds")
    right.set_title("GLM (sign-aligned)", fontsize=11, fontweight="bold")

    handles = [
        Line2D([0], [0], marker="o", color="none", mfc=colors[r],
               mec=colors[r], ms=7, label=f"Rat {r}")
        for r in rats
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.06), fontsize=8)
    fig.suptitle(
        f"Choice–outcome geometry ({CI_LEVEL}% covariance ellipses)",
        fontsize=11.5, y=1.02,
    )
    fig.text(
        0.5, -0.105,
        "Panels use different units; compare rat-specific direction and sector, not absolute distance.",
        ha="center", fontsize=7.5, color=MUTED,
    )
    fig.tight_layout()
    save_figure(fig, "figB_q1_geometric_plane")


# =============================================================================
# COMPACT LATEX SUMMARY TABLE FROM CANONICAL OUTPUTS
# =============================================================================

def build_table(data):
    l1 = data["l1_summary"].set_index("quantity")
    l2 = data["l2"].set_index("quantity")
    l3 = data["l3_cohort"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Cross-method agreement between psychometric and pooled-GLM estimates.}",
        r"\label{tab:q1-convergence}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Quantity & Sign agreement & Spearman $\rho$ & 95\% BCa CI & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
    ]
    for quantity in QUANTITIES:
        a = int(l1.loc[quantity, "opt1_match"])
        n = int(l1.loc[quantity, "opt1_denominator"])
        rho = float(l2.loc[quantity, "rho_beta_aligned"])
        lo = float(l2.loc[quantity, "bca_lo_beta_aligned"])
        hi = float(l2.loc[quantity, "bca_hi_beta_aligned"])
        p = float(l2.loc[quantity, "p_holm_beta_aligned_primary4"])
        lines.append(
            f"{LABELS[quantity]} & {a}/{n} & {rho:.2f} & "
            f"[{lo:.2f}, {hi:.2f}] & {p:.3f} \\\\"
        )
    lines += [
        r"\midrule",
        rf"\multicolumn{{4}}{{l}}{{Exact quadrant agreement}} & {int(l3['q4_exact_match_beta'])}/12 \\",
        rf"\multicolumn{{4}}{{l}}{{Exact octant agreement}} & {int(l3['o8_exact_match_beta'])}/12 \\",
        rf"\multicolumn{{4}}{{l}}{{Within one octant}} & {int(l3['o8_within1_beta'])}/12 \\",
        rf"\multicolumn{{4}}{{l}}{{Median angular disagreement}} & {float(l3['angular_disagree_median_beta_deg']):.1f}$^\circ$ \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "table_q1_convergence.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main():
    data, rats = load_inputs()
    build_figure_a(data, rats)
    build_figure_b(data, rats)
    build_table(data)
    print(f"Outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
