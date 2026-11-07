"""
Create Section 3 SVG figures from SAVED analysis outputs only.

This script does NOT load trial data, fit psychometric functions, run a
bootstrap, or change multiplicity corrections. It reads the CSV/NPZ files
already written by Section_3_choice_reward_joint_bootstrap.py.

Outputs
-------
1. rat_<id>_contrast_summary.svg
   The same three-panel per-rat summary produced by the core analysis.
2. forest_core_contrasts.svg
   The same five-contrast cohort forest plot produced by the core analysis.
3. section3_choice_plane_<selection>_<style>.svg
   A stand-alone Delta-mu_R / Delta-mu_U plane, with the S and D axes and a
   tunable rat selection.

Required saved files
--------------------
RESULTS_DIR/
    tables/contrast_summary_long.csv
    tables/contrast_covariance_long.csv
    bootstraps/rat_<id>_joint_bootstrap.npz

Run
---
    python section3_svg_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import chi2


# =============================================================================
# USER CONFIGURATION -- normally this is the only section to edit
# =============================================================================

# Directory containing tables/ and bootstraps/ for ONE completed Section 3 run.
# Change "filtered" to "unfiltered" if that is the run you want to plot.
RESULTS_DIR = (
    Path(__file__).resolve().parent
    / "choice_reward_joint_results"
    / "filtered"
    / "pooled"
)

OUTPUT_DIR = Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\ale_thesis_fig_tab\images_thesis\section_3")

# Which figures to make.
MAKE_PER_RAT_SUMMARIES = True
MAKE_FOREST = True
MAKE_CHOICE_PLANE = True

# None means all available rats. Otherwise use, for example, [9, 11, 12, 13].
# This flag controls the stand-alone plane only.
RATS_TO_PLOT = [9, 11, 12, 13]

# "ellipse": bivariate-normal CI ellipse from saved bootstrap covariance.
# "raw": a subsample of the actual aligned bootstrap draws from each NPZ.
CLOUD_STYLE = "ellipse"
CI_LEVEL = 95
N_SCATTER = 400

# True: size the plane using the full cohort, so different rat selections have
# an identical frame. False: crop the frame to the selected rats.
PLANE_FIXED = True

# Keep labels as editable text in the SVG rather than converting them to paths.
SVG_TEXT_AS_TEXT = True
RANDOM_SEED = 0


# =============================================================================
# CONSTANTS AND STYLE
# =============================================================================

CELL_ORDER = ("h_rew", "v_rew", "h_unrew", "v_unrew")
CELL_LABELS = {
    "h_rew": "prev-H, rewarded",
    "v_rew": "prev-V, rewarded",
    "h_unrew": "prev-H, unrewarded",
    "v_unrew": "prev-V, unrewarded",
}
PARAMETER_ORDER = ("mu", "sigma", "gamma", "lapse")
CONTRAST_ORDER = (
    "delta_mu_rew",
    "delta_mu_unrew",
    "S",
    "D",
    "P",
    "choice_main_half",
    "choice_by_outcome_half",
    "reward_effect_after_h",
    "reward_effect_after_v",
)
CORE_CONTRASTS = ("delta_mu_rew", "delta_mu_unrew", "S", "D", "P")

PALETTE_8 = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

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
    "svg.fonttype": "none" if SVG_TEXT_AS_TEXT else "path",
})

RNG = np.random.default_rng(RANDOM_SEED)


# =============================================================================
# INPUT AND VALIDATION
# =============================================================================

TABLE_DIR = RESULTS_DIR / "tables"
BOOTSTRAP_DIR = RESULTS_DIR / "bootstraps"
CONTRAST_PATH = TABLE_DIR / "contrast_summary_long.csv"
COVARIANCE_PATH = TABLE_DIR / "contrast_covariance_long.csv"


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required saved output not found: {path}")


def load_inputs():
    require_file(CONTRAST_PATH)
    require_file(COVARIANCE_PATH)
    contrast = pd.read_csv(CONTRAST_PATH)
    covariance = pd.read_csv(COVARIANCE_PATH)

    required_contrast = {
        "rat", "contrast", "estimate_full_data", "ci_low", "ci_high",
        "ci_excludes_zero", "declared_fdr_reject",
    }
    required_covariance = {
        "rat", "contrast_row", "contrast_column", "covariance",
    }
    missing = required_contrast - set(contrast.columns)
    if missing:
        raise KeyError(f"Missing columns in {CONTRAST_PATH.name}: {sorted(missing)}")
    missing = required_covariance - set(covariance.columns)
    if missing:
        raise KeyError(f"Missing columns in {COVARIANCE_PATH.name}: {sorted(missing)}")

    contrast["rat"] = contrast["rat"].astype(int)
    covariance["rat"] = covariance["rat"].astype(int)
    return contrast, covariance


def load_rat_npz(rat: int) -> dict[str, np.ndarray]:
    path = BOOTSTRAP_DIR / f"rat_{rat}_joint_bootstrap.npz"
    require_file(path)
    with np.load(path, allow_pickle=False) as archive:
        needed = {
            "parameter_bootstrap",
            "contrast_bootstrap",
            "full_cell_parameters",
            "full_contrast_estimates",
        }
        missing = needed - set(archive.files)
        if missing:
            raise KeyError(f"Missing arrays in {path.name}: {sorted(missing)}")
        return {name: np.asarray(archive[name]) for name in needed}


def save_svg(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def as_bool(value) -> bool:
    """Handle real booleans and booleans read from CSV robustly."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


# =============================================================================
# SAME CONTRAST FIGURES AS THE CORE ANALYSIS, NOW SAVED AS SVG
# =============================================================================

def plot_rat_summary_svg(
    rat: int,
    contrast_table: pd.DataFrame,
    output_path: Path,
) -> None:
    arrays = load_rat_npz(rat)
    parameter_bootstrap = arrays["parameter_bootstrap"]
    contrast_bootstrap = arrays["contrast_bootstrap"]
    full_parameters = arrays["full_cell_parameters"]
    full_contrasts = arrays["full_contrast_estimates"]

    if len(parameter_bootstrap) == 0:
        print(f"Skipped Rat {rat}: no valid saved bootstrap replicates")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)

    # Panel A: four PSE cell estimates and percentile intervals.
    mu_index = PARAMETER_ORDER.index("mu")
    for index, cell_name in enumerate(CELL_ORDER):
        estimate = full_parameters[index, mu_index]
        samples = parameter_bootstrap[:, index, mu_index]
        low, high = np.percentile(samples, [2.5, 97.5])
        axes[0].vlines(index, low, high, linewidth=2)
        axes[0].hlines([low, high], index - 0.08, index + 0.08, linewidth=2)
        axes[0].scatter(index, estimate, s=55, zorder=3)
    axes[0].axhline(45, color="gray", linestyle=":")
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(
        [CELL_LABELS[name] for name in CELL_ORDER], rotation=20, ha="right"
    )
    axes[0].set_ylabel("PSE, μ (°)")
    axes[0].set_title("Four joint cell fits")

    # Panel B: direct-effect plane from actual aligned bootstrap samples.
    x_index = CONTRAST_ORDER.index("delta_mu_rew")
    y_index = CONTRAST_ORDER.index("delta_mu_unrew")
    x_boot = contrast_bootstrap[:, x_index]
    y_boot = contrast_bootstrap[:, y_index]
    axes[1].scatter(x_boot, y_boot, s=5, alpha=0.12, color="#444444")
    x_hat, y_hat = full_contrasts[x_index], full_contrasts[y_index]
    axes[1].scatter([x_hat], [y_hat], s=75, color="crimson", zorder=4)
    limit = max(2.0, 1.05 * np.nanmax(np.abs(np.r_[x_boot, y_boot, x_hat, y_hat])))
    axes[1].plot([-limit, limit], [-limit, limit], ":", label="D = 0 (S axis)")
    axes[1].plot([-limit, limit], [limit, -limit], ":", label="S = 0 (D axis)")
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].axvline(0, color="black", linewidth=0.6)
    axes[1].set(xlim=(-limit, limit), ylim=(-limit, limit))
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("Post-success shift, Δμ$_R$ (°)")
    axes[1].set_ylabel("Post-failure shift, Δμ$_U$ (°)")
    axes[1].set_title("Direct-effect geometry")
    axes[1].legend(fontsize=8)

    # Panel C: five core contrasts using the already-saved inference table.
    rat_rows = contrast_table[
        (contrast_table["rat"] == rat)
        & contrast_table["contrast"].isin(CORE_CONTRASTS)
    ].set_index("contrast")
    for index, name in enumerate(CORE_CONTRASTS):
        if name not in rat_rows.index:
            continue
        row = rat_rows.loc[name]
        estimate, low, high = row["estimate_full_data"], row["ci_low"], row["ci_high"]
        declared = as_bool(row["declared_fdr_reject"])
        pointwise = as_bool(row["ci_excludes_zero"])
        color = "crimson" if declared else ("steelblue" if pointwise else "gray")
        axes[2].vlines(index, low, high, linewidth=2, color=color)
        axes[2].hlines([low, high], index - 0.08, index + 0.08, linewidth=2, color=color)
        axes[2].scatter(index, estimate, s=55, color=color, zorder=3)
    axes[2].axhline(0, color="gray", linestyle="--")
    axes[2].set_xticks(range(len(CORE_CONTRASTS)))
    axes[2].set_xticklabels(CORE_CONTRASTS, rotation=25, ha="right")
    axes[2].set_ylabel("Contrast (°)")
    axes[2].set_title("Gray: unresolved; blue: pointwise; red: declared FDR")

    fig.suptitle(f"Rat {rat} | {RESULTS_DIR.parent.name} | {RESULTS_DIR.name}")
    save_svg(fig, output_path)


def plot_forest_svg(contrast_table: pd.DataFrame, output_path: Path) -> None:
    core = contrast_table[contrast_table["contrast"].isin(CORE_CONTRASTS)].copy()
    if core.empty:
        print("Skipped forest plot: no core-contrast rows")
        return

    rats = sorted(core["rat"].unique())
    fig, axes = plt.subplots(
        1, len(CORE_CONTRASTS), figsize=(20, 8), sharey=True,
        constrained_layout=True,
    )
    y_positions = {rat: i for i, rat in enumerate(rats)}

    for axis, contrast_name in zip(axes, CORE_CONTRASTS):
        rows = core[core["contrast"] == contrast_name].set_index("rat")
        for rat in rats:
            if rat not in rows.index:
                continue
            row = rows.loc[rat]
            estimate, low, high = row["estimate_full_data"], row["ci_low"], row["ci_high"]
            if not np.all(np.isfinite([estimate, low, high])):
                continue
            declared = as_bool(row["declared_fdr_reject"])
            pointwise = as_bool(row["ci_excludes_zero"])
            color = "crimson" if declared else ("steelblue" if pointwise else "gray")
            y = y_positions[rat]
            axis.hlines(y, low, high, color=color, linewidth=2)
            axis.vlines([low, high], y - 0.08, y + 0.08, color=color, linewidth=1.5)
            axis.scatter(estimate, y, color=color, s=32, zorder=3)
        axis.axvline(0, color="black", linestyle="--", linewidth=0.7)
        axis.set_title(contrast_name)
        axis.set_xlabel("Degrees")

    axes[0].set_yticks(range(len(rats)))
    axes[0].set_yticklabels([f"Rat {rat}" for rat in rats])
    save_svg(fig, output_path)


# =============================================================================
# STAND-ALONE SELECTABLE-RAT CHOICE PLANE WITH S AND D AXES
# =============================================================================

def contrast_mean(contrast_table: pd.DataFrame, rat: int) -> np.ndarray:
    rows = contrast_table[contrast_table["rat"] == rat].set_index("contrast")
    try:
        return np.array([
            rows.loc["delta_mu_rew", "estimate_full_data"],
            rows.loc["delta_mu_unrew", "estimate_full_data"],
        ], dtype=float)
    except KeyError as exc:
        raise KeyError(f"Missing direct-effect estimate for Rat {rat}") from exc


def covariance_matrix(covariance_table: pd.DataFrame, rat: int) -> np.ndarray:
    names = ("delta_mu_rew", "delta_mu_unrew")
    matrix = np.empty((2, 2), dtype=float)
    for i, row_name in enumerate(names):
        for j, column_name in enumerate(names):
            row = covariance_table[
                (covariance_table["rat"] == rat)
                & (covariance_table["contrast_row"] == row_name)
                & (covariance_table["contrast_column"] == column_name)
            ]
            if len(row) != 1:
                raise ValueError(
                    f"Expected one covariance entry for Rat {rat}, "
                    f"{row_name} × {column_name}; found {len(row)}"
                )
            matrix[i, j] = row.iloc[0]["covariance"]
    return matrix


def direct_effect_draws(rat: int) -> tuple[np.ndarray, np.ndarray]:
    arrays = load_rat_npz(rat)
    draws = arrays["contrast_bootstrap"]
    return (
        draws[:, CONTRAST_ORDER.index("delta_mu_rew")],
        draws[:, CONTRAST_ORDER.index("delta_mu_unrew")],
    )


def ellipse_reach(mean: np.ndarray, covariance: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.clip(eigenvalues, 0, None)
    radius = np.sqrt(chi2.ppf(CI_LEVEL / 100, df=2)) * np.sqrt(eigenvalues.max())
    return float(max(abs(mean[0]) + radius, abs(mean[1]) + radius))


def draw_ellipse(ax, mean, covariance, color) -> None:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0, None)
    eigenvectors = eigenvectors[:, order]
    scale = np.sqrt(chi2.ppf(CI_LEVEL / 100, df=2))
    width, height = 2 * scale * np.sqrt(eigenvalues)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    ax.add_patch(mpatches.Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        linewidth=1.2,
        alpha=0.16,
        zorder=1,
    ))


def plot_choice_plane_svg(
    contrast_table: pd.DataFrame,
    covariance_table: pd.DataFrame,
    output_path: Path,
) -> None:
    if CLOUD_STYLE not in {"ellipse", "raw"}:
        raise ValueError('CLOUD_STYLE must be "ellipse" or "raw"')
    if not 0 < CI_LEVEL < 100:
        raise ValueError("CI_LEVEL must be between 0 and 100")

    all_rats = sorted(contrast_table["rat"].unique().astype(int).tolist())
    selected = all_rats if RATS_TO_PLOT is None else [int(r) for r in RATS_TO_PLOT]
    unknown = sorted(set(selected) - set(all_rats))
    if unknown:
        raise ValueError(f"RATS_TO_PLOT contains unavailable rats: {unknown}")
    if not selected:
        raise ValueError("RATS_TO_PLOT selects no rats")

    rat_color = {rat: PALETTE_8[i % len(PALETTE_8)] for i, rat in enumerate(all_rats)}
    frame_rats = all_rats if PLANE_FIXED else selected
    max_reach = max(
        ellipse_reach(
            contrast_mean(contrast_table, rat),
            covariance_matrix(covariance_table, rat),
        )
        for rat in frame_rats
    )
    limit = max(2.0, 1.08 * max_reach)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for rat in selected:
        color = rat_color[rat]
        mean = contrast_mean(contrast_table, rat)
        covariance = covariance_matrix(covariance_table, rat)

        if CLOUD_STYLE == "ellipse":
            draw_ellipse(ax, mean, covariance, color)
        else:
            x_draws, y_draws = direct_effect_draws(rat)
            if len(x_draws) == 0:
                print(f"Rat {rat}: no bootstrap draws; plotting point only")
            else:
                indices = RNG.choice(
                    len(x_draws), size=min(N_SCATTER, len(x_draws)), replace=False
                )
                ax.scatter(
                    x_draws[indices], y_draws[indices], s=5, color=color,
                    alpha=0.10, linewidths=0, zorder=1,
                )

        ax.plot(*mean, marker="o", mfc=color, mec="white", mew=0.8, ms=8, zorder=4)
        ax.annotate(
            str(rat), mean, textcoords="offset points", xytext=(7, 6),
            fontsize=9, fontweight="bold", color=color, zorder=5,
        )

    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=0)
    ax.axvline(0, color=MUTED, linewidth=0.8, zorder=0)
    diagonal = np.array([-1.4 * limit, 1.4 * limit])
    ax.plot(diagonal, diagonal, color=GRID, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.plot(diagonal, -diagonal, color=GRID, linewidth=1.0, linestyle=(0, (4, 3)))

    label_position = 0.58 * limit
    label_box = dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1)
    ax.text(
        label_position, label_position, "S axis\n(D = 0)",
        fontsize=7.5, color=MUTED, ha="center", va="center", rotation=45,
        clip_on=True, bbox=label_box,
    )
    ax.text(
        label_position, -label_position, "D axis\n(S = 0)",
        fontsize=7.5, color=MUTED, ha="center", va="center", rotation=-45,
        clip_on=True, bbox=label_box,
    )

    ax.set(xlim=(-limit, limit), ylim=(-limit, limit))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Post-success shift, Δμ$_R$ (°)")
    ax.set_ylabel("Post-failure shift, Δμ$_U$ (°)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=GRID, linewidth=0.6, zorder=-2)

    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="none", mfc=rat_color[rat],
            mec=rat_color[rat], ms=7, label=f"Rat {rat}",
        )
        for rat in selected
    ]
    legend_columns = 1 if len(selected) <= 6 else 2
    ax.legend(
        handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
        frameon=False, fontsize=8.5, ncol=legend_columns,
    )

    style_label = (
        f"{CI_LEVEL}% bootstrap-covariance ellipse"
        if CLOUD_STYLE == "ellipse"
        else f"{N_SCATTER} raw bootstrap draws per rat"
    )
    ax.set_title("Section 3 direct-effect plane", fontsize=11, fontweight="bold", pad=10)
    frame_label = "full-cohort frame" if PLANE_FIXED else "selection-fitted frame"
    fig.text(
        0.5, 0.01, f"{style_label}; {frame_label}",
        ha="center", fontsize=8, color=MUTED,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    save_svg(fig, output_path)


# =============================================================================
# RUN
# =============================================================================

def selection_label(rats: list[int] | None, all_rats: list[int]) -> str:
    if rats is None or sorted(rats) == sorted(all_rats):
        return "full_cohort"
    return "rats_" + "_".join(str(int(rat)) for rat in rats)


def main() -> None:
    contrast_table, covariance_table = load_inputs()
    all_rats = sorted(contrast_table["rat"].unique().astype(int).tolist())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if MAKE_PER_RAT_SUMMARIES:
        for rat in all_rats:
            plot_rat_summary_svg(
                rat,
                contrast_table,
                OUTPUT_DIR / f"rat_{rat}_contrast_summary.svg",
            )

    if MAKE_FOREST:
        plot_forest_svg(
            contrast_table,
            OUTPUT_DIR / "forest_core_contrasts.svg",
        )

    if MAKE_CHOICE_PLANE:
        label = selection_label(RATS_TO_PLOT, all_rats)
        plot_choice_plane_svg(
            contrast_table,
            covariance_table,
            OUTPUT_DIR / f"section3_choice_plane_{label}_{CLOUD_STYLE}.svg",
        )

    print("Done. No model fitting or bootstrap computation was performed.")


if __name__ == "__main__":
    main()
