"""
Plotting for section 4 GLM analysis.

Reads the three artifacts written by the fitting script and produces the
thesis figures. Log-odds is the primary scale throughout; Figure 2 is the
one probability-scale exception, using the exact sigmoid rather than the
0.25 boundary approximation.

Figures:
  1. Coefficient forest (log-odds) for the history predictors.
     Filled markers = CI excludes zero; hollow markers = CI includes zero.
  2. Per-cell probability shift from intercept, per rat.
     Cells in order H_rew, V_rew, H_unrew, V_unrew; three modality points
     per cell (dodged) with 95% CIs computed via the exact sigmoid on
     bootstrap draws.
  3. (beta_failure, beta_success) plane, log-odds.
     Top row: three group panels, one per modality, all rats + covariance
     ellipses.
     Bottom: 12 per-rat panels, three modality points + ellipses.
     Ellipse coverage set by ELLIPSE_QUANTILE (default 68%).
  4. S, D, P forest in log-odds. S = beta_success + beta_failure,
     D = beta_success - beta_failure, P = beta_reward. CIs derived from
     bootstrap draws to preserve the beta_success / beta_failure correlation.

Diagnostics:
  - Intercept sanity check per (rat, modality). Flags fits whose |beta_0|
    exceeds INTERCEPT_ABS_THRESHOLD.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_4_GLM\fit_results\per_modality_filtered")


FIGURES_DIR = Path(
    r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy"
    r"\thesis_codes\section_4_GLM\figures"
)

DEFAULT_C = "unpenalized"

MODALITY_COLORS = {
    "tactile":      "#2ca02c",  # green
    "visual":       "#1f77b4",  # blue
    "visuotactile": "#d62728",  # red
}
MODALITY_ORDER = ["tactile", "visual", "visuotactile"]

INTERCEPT_ABS_THRESHOLD = 1.0
ELLIPSE_QUANTILE        = 0.68   # for Fig 3 covariance ellipses

DPI          = 300
SAVE_FORMATS = ["pdf", "png"]

FIG1_COEFFS: List[Tuple[str, str]] = [
    ("beta_success_n-1", r"$\beta_{\mathrm{success}}$"),
    ("beta_failure_n-1", r"$\beta_{\mathrm{failure}}$"),
    ("beta_reward_n-1",  r"$\beta_{\mathrm{reward}}$"),
    ("beta_angle_n-1",   r"$\beta_{\mathrm{prev\ angle}}$"),
]

# Figure 2 cell definitions in display order (H_rew, V_rew, H_unrew, V_unrew).
# Each entry: (label, coefficients-of-Delta-logit-for-that-cell). The Delta
# logit is a linear combination of beta_reward, beta_success, beta_failure
# matching the coding used in the fit:
#   V rewarded:   +0.5 * beta_reward + beta_success
#   H rewarded:   +0.5 * beta_reward - beta_success
#   V unrewarded: -0.5 * beta_reward + beta_failure
#   H unrewarded: -0.5 * beta_reward - beta_failure
FIG2_CELLS: List[Tuple[str, Dict[str, float]]] = [
    ("H rew",   {"beta_reward_n-1":  0.5, "beta_success_n-1": -1.0}),
    ("V rew",   {"beta_reward_n-1":  0.5, "beta_success_n-1": +1.0}),
    ("H unrew", {"beta_reward_n-1": -0.5, "beta_failure_n-1": -1.0}),
    ("V unrew", {"beta_reward_n-1": -0.5, "beta_failure_n-1": +1.0}),
]


# ============================================================
# LOADERS
# ============================================================

def load_data(input_dir: Path = INPUT_DIR):
    input_dir    = Path(input_dir)
    summary_path = input_dir / "glm_summary_long.csv"
    boot_parquet = input_dir / "glm_bootstrap_draws.parquet"
    boot_csv     = input_dir / "glm_bootstrap_draws.csv.gz"
    meta_path    = input_dir / "glm_metadata.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_path}")
    summary = pd.read_csv(summary_path)

    if boot_parquet.exists():
        boot = pd.read_parquet(boot_parquet)
    elif boot_csv.exists():
        boot = pd.read_csv(boot_csv)
    else:
        raise FileNotFoundError(
            f"Missing bootstrap file (looked for {boot_parquet} and {boot_csv})"
        )

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata JSON: {meta_path}")
    with open(meta_path) as f:
        metadata = json.load(f)

    return summary, boot, metadata


def get_rat_order(summary: pd.DataFrame) -> List:
    return sorted(summary["rat_id"].unique().tolist())


# ============================================================
# INTERCEPT DIAGNOSTIC
# ============================================================

def check_intercepts(
    summary: pd.DataFrame,
    C: str = DEFAULT_C,
    threshold: float = INTERCEPT_ABS_THRESHOLD,
    verbose: bool = True,
) -> pd.DataFrame:
    df = summary[
        (summary["quantity"] == "intercept") &
        (summary["C"] == C)
    ].copy()

    if df.empty:
        raise ValueError(
            f"No intercept rows in summary at C = {C!r}. "
            f"Available: {sorted(summary['C'].unique())}"
        )

    df["p_at_intercept"] = 1.0 / (1.0 + np.exp(-df["value"]))
    df["dp_dlogit"]      = df["p_at_intercept"] * (1.0 - df["p_at_intercept"])
    df["flagged"]        = df["value"].abs() > threshold

    if verbose:
        _print_intercept_table(df, C=C, threshold=threshold)

    return (
        df[["rat_id", "subset", "value", "ci_low", "ci_high",
            "p_at_intercept", "dp_dlogit", "flagged"]]
        .rename(columns={"value": "intercept"})
        .sort_values(["subset", "rat_id"])
        .reset_index(drop=True)
    )


def _print_intercept_table(df: pd.DataFrame, C: str, threshold: float) -> None:
    header = f"Intercept diagnostic  (C = {C};  flag |beta_0| > {threshold:.2f})"
    col_hdr = (
        f"{'rat':>4}  {'modality':<13}  {'beta_0':>8}  "
        f"{'95% CI':>20}  {'P(V) at bdry':>12}  {'dp/dlogit':>10}  flag"
    )
    bar = "=" * max(len(header), len(col_hdr))
    print()
    print(bar)
    print(header)
    print(bar)
    print(col_hdr)
    print("-" * len(col_hdr))
    for _, row in df.sort_values(["subset", "rat_id"]).iterrows():
        flag = "!!" if row["flagged"] else ""
        ci   = f"[{row['ci_low']:+.2f}, {row['ci_high']:+.2f}]"
        print(
            f"{int(row['rat_id']):>4}  {row['subset']:<13}  "
            f"{row['value']:+8.3f}  {ci:>20}  "
            f"{row['p_at_intercept']:>12.3f}  {row['dp_dlogit']:>10.3f}  {flag}"
        )
    print("-" * len(col_hdr))
    n_flag = int(df["flagged"].sum())
    print(f"Flagged {n_flag}/{len(df)} fits.  "
          f"Fig 2 uses exact sigmoid so numbers are correct; "
          f"but for flagged rats the 0.5*beta shortcut no longer applies.")
    print()


# ============================================================
# HELPERS
# ============================================================

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _ci_excludes_zero(lo, hi) -> bool:
    return bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))


def _percentile_ci(values: np.ndarray, ci_level: float = 95) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (np.nan, np.nan)
    alpha = (100.0 - ci_level) / 2.0
    return tuple(float(v) for v in np.percentile(values, [alpha, 100.0 - alpha]))


def _pivot_boot_wide(
    boot: pd.DataFrame,
    C: str,
    quantities: List[str],
) -> pd.DataFrame:
    sub = boot[(boot["C"] == C) & (boot["quantity"].isin(quantities))]
    wide = sub.pivot_table(
        index=["rat_id", "subset", "boot_idx"],
        columns="quantity",
        values="value",
    ).reset_index()
    return wide


def _cov_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    quantile: float = ELLIPSE_QUANTILE,
    **kwargs,
) -> Ellipse:
    x = np.asarray(x); y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if x.size < 2:
        return Ellipse((np.nan, np.nan), 0.0, 0.0, angle=0.0, **kwargs)

    cov = np.cov(x, y)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    chi2_val = chi2.ppf(quantile, df=2)
    width  = 2.0 * np.sqrt(chi2_val * max(eigvals[0], 0.0))
    height = 2.0 * np.sqrt(chi2_val * max(eigvals[1], 0.0))
    angle  = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    return Ellipse(
        (float(x.mean()), float(y.mean())),
        width, height, angle=angle, **kwargs,
    )


def _save_figure(fig: plt.Figure, name: str, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in SAVE_FORMATS:
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        print(f"  Saved {path}")


# ============================================================
# FIGURE 1: COEFFICIENT FOREST (log-odds)
# ============================================================

def figure_1_coefficient_forest(
    summary: pd.DataFrame,
    C: str = DEFAULT_C,
    out_dir: Path = FIGURES_DIR,
    save: bool = True,
) -> plt.Figure:
    """Log-odds forest for history predictors. Filled = CI excludes zero."""
    rats = get_rat_order(summary)
    n_rats   = len(rats)
    n_panels = len(FIG1_COEFFS)

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(2.9 * n_panels, 0.34 * n_rats + 1.8),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    dodge = {mod: (i - 1) * 0.22 for i, mod in enumerate(MODALITY_ORDER)}

    for ax, (quantity, label) in zip(axes, FIG1_COEFFS):
        sub = summary[
            (summary["quantity"] == quantity) &
            (summary["C"] == C)
        ]
        if sub.empty:
            ax.text(0.5, 0.5, f"no data\nfor {quantity}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray")
            ax.set_title(label, fontsize=11)
            continue

        for mod in MODALITY_ORDER:
            mod_sub = sub[sub["subset"] == mod]
            if mod_sub.empty:
                continue

            color = MODALITY_COLORS[mod]
            sig_ys, sig_xs, sig_lo, sig_hi = [], [], [], []
            ns_ys,  ns_xs,  ns_lo,  ns_hi  = [], [], [], []
            for i, rat in enumerate(rats):
                row = mod_sub[mod_sub["rat_id"] == rat]
                if row.empty:
                    continue
                r = row.iloc[0]
                if not np.isfinite(r["value"]):
                    continue
                y_pos = (n_rats - 1 - i) + dodge[mod]
                sig  = _ci_excludes_zero(r["ci_low"], r["ci_high"])
                tgt  = (sig_ys, sig_xs, sig_lo, sig_hi) if sig \
                       else (ns_ys, ns_xs, ns_lo, ns_hi)
                tgt[0].append(y_pos)
                tgt[1].append(r["value"])
                tgt[2].append(r["value"] - r["ci_low"])
                tgt[3].append(r["ci_high"] - r["value"])

            if ns_xs:
                ax.errorbar(
                    ns_xs, ns_ys, xerr=[ns_lo, ns_hi],
                    fmt="o", markersize=3.5,
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=1.0, ecolor=color,
                    elinewidth=0.8, capsize=0, alpha=0.75,
                )
            if sig_xs:
                ax.errorbar(
                    sig_xs, sig_ys, xerr=[sig_lo, sig_hi],
                    fmt="o", markersize=4.5,
                    markerfacecolor=color, markeredgecolor=color,
                    markeredgewidth=1.0, ecolor=color,
                    elinewidth=1.4, capsize=0,
                )

        ax.axvline(0, color="k", linewidth=0.8, alpha=0.4)
        ax.set_title(label, fontsize=11)
        ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_yticks(range(n_rats))
    axes[0].set_yticklabels([f"rat {r}" for r in rats[::-1]])
    axes[0].set_ylabel("")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color=MODALITY_COLORS[m],
                   markersize=6, linestyle="none", label=m)
        for m in MODALITY_ORDER
    ]
    legend_handles += [
        plt.Line2D([0], [0], marker="o", color="k", markerfacecolor="k",
                   markersize=6, linestyle="none", label="CI excludes 0"),
        plt.Line2D([0], [0], marker="o", color="k", markerfacecolor="white",
                   markersize=6, linestyle="none", label="CI includes 0"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper center",
        ncol=len(legend_handles), frameon=False,
        bbox_to_anchor=(0.5, 1.005), fontsize=9,
    )
    fig.supxlabel("log-odds", fontsize=11, y=-0.01)
    fig.suptitle(f"History predictors, log-odds  (C = {C})",
                 fontsize=12, y=1.06)
    fig.tight_layout()

    if save:
        _save_figure(fig, "fig1_coefficient_forest", out_dir)
    return fig


# ============================================================
# FIGURE 2: PER-CELL PROBABILITY SHIFT FROM INTERCEPT, PER RAT
# ============================================================

def _compute_cell_shifts_boot(boot: pd.DataFrame, C: str) -> pd.DataFrame:
    quantities = ["intercept", "beta_reward_n-1",
                  "beta_success_n-1", "beta_failure_n-1"]
    wide = _pivot_boot_wide(boot, C=C, quantities=quantities)
    b0 = wide["intercept"].to_numpy()
    p0 = _sigmoid(b0)

    rows = []
    for cell_label, coefs in FIG2_CELLS:
        dlogit = np.zeros_like(b0)
        for q, w in coefs.items():
            dlogit = dlogit + w * wide[q].to_numpy()
        shift = _sigmoid(b0 + dlogit) - p0
        df_cell = wide[["rat_id", "subset", "boot_idx"]].copy()
        df_cell["cell"]  = cell_label
        df_cell["shift"] = shift
        rows.append(df_cell)
    return pd.concat(rows, ignore_index=True)


def _compute_cell_shifts_point(summary: pd.DataFrame, C: str) -> pd.DataFrame:
    wide = (
        summary[(summary["C"] == C) &
                (summary["quantity"].isin([
                    "intercept", "beta_reward_n-1",
                    "beta_success_n-1", "beta_failure_n-1",
                ]))]
        .pivot_table(index=["rat_id", "subset"],
                     columns="quantity", values="value")
        .reset_index()
    )
    b0 = wide["intercept"].to_numpy()
    p0 = _sigmoid(b0)

    rows = []
    for cell_label, coefs in FIG2_CELLS:
        dlogit = np.zeros_like(b0)
        for q, w in coefs.items():
            dlogit = dlogit + w * wide[q].to_numpy()
        shift = _sigmoid(b0 + dlogit) - p0
        df_cell = wide[["rat_id", "subset"]].copy()
        df_cell["cell"]  = cell_label
        df_cell["shift"] = shift
        rows.append(df_cell)
    return pd.concat(rows, ignore_index=True)


def figure_2_cell_shifts(
    summary: pd.DataFrame,
    boot: pd.DataFrame,
    C: str = DEFAULT_C,
    out_dir: Path = FIGURES_DIR,
    save: bool = True,
    flagged_rats_by_mod: Dict[str, List] = None,
) -> plt.Figure:
    rats   = get_rat_order(summary)
    n_rats = len(rats)
    n_cols = 4
    n_rows = int(np.ceil(n_rats / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 2.2 * n_rows),
        sharex=True, sharey=True,
    )
    axes = np.atleast_2d(axes)

    boot_shifts  = _compute_cell_shifts_boot(boot, C=C)
    point_shifts = _compute_cell_shifts_point(summary, C=C)

    cell_labels = [c[0] for c in FIG2_CELLS]
    x_positions = np.arange(len(cell_labels))
    dodge = {mod: (i - 1) * 0.22 for i, mod in enumerate(MODALITY_ORDER)}
    flagged_rats_by_mod = flagged_rats_by_mod or {}

    for idx, rat in enumerate(rats):
        r, c = divmod(idx, n_cols)
        ax = axes[r, c]
        for mod in MODALITY_ORDER:
            color = MODALITY_COLORS[mod]
            for xi, cell in enumerate(cell_labels):
                pt_row = point_shifts[
                    (point_shifts["rat_id"] == rat) &
                    (point_shifts["subset"] == mod) &
                    (point_shifts["cell"] == cell)
                ]
                boot_row = boot_shifts[
                    (boot_shifts["rat_id"] == rat) &
                    (boot_shifts["subset"] == mod) &
                    (boot_shifts["cell"] == cell)
                ]
                if pt_row.empty or boot_row.empty:
                    continue
                val = float(pt_row["shift"].iloc[0])
                lo, hi = _percentile_ci(boot_row["shift"].to_numpy(),
                                        ci_level=95)
                sig  = _ci_excludes_zero(lo, hi)
                xpos = xi + dodge[mod]

                if sig:
                    ax.errorbar(
                        [xpos], [val],
                        yerr=[[val - lo], [hi - val]],
                        fmt="o", markersize=4.5,
                        markerfacecolor=color, markeredgecolor=color,
                        markeredgewidth=1.0, ecolor=color,
                        elinewidth=1.4, capsize=0,
                    )
                else:
                    ax.errorbar(
                        [xpos], [val],
                        yerr=[[val - lo], [hi - val]],
                        fmt="o", markersize=3.5,
                        markerfacecolor="white", markeredgecolor=color,
                        markeredgewidth=1.0, ecolor=color,
                        elinewidth=0.8, capsize=0, alpha=0.75,
                    )

        flagged_here = any(rat in flagged_rats_by_mod.get(m, [])
                           for m in MODALITY_ORDER)
        title = f"rat {rat}" + ("  (!)" if flagged_here else "")
        ax.set_title(title, fontsize=10)

        ax.axhline(0, color="k", linewidth=0.8, alpha=0.4)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(cell_labels, fontsize=9)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for idx in range(len(rats), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r, c].axis("off")

    for r in range(n_rows):
        axes[r, 0].set_ylabel(r"$\Delta P(V)$ from intercept", fontsize=9)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color=MODALITY_COLORS[m],
                   markersize=6, linestyle="none", label=m)
        for m in MODALITY_ORDER
    ]
    legend_handles += [
        plt.Line2D([0], [0], marker="o", color="k", markerfacecolor="k",
                   markersize=6, linestyle="none", label="CI excludes 0"),
        plt.Line2D([0], [0], marker="o", color="k", markerfacecolor="white",
                   markersize=6, linestyle="none", label="CI includes 0"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper center",
        ncol=len(legend_handles), frameon=False,
        bbox_to_anchor=(0.5, 1.005), fontsize=9,
    )
    fig.suptitle(
        f"Probability shift from intercept, per rat  (C = {C})  "
        r"$\;$(!) = intercept flagged",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()

    if save:
        _save_figure(fig, "fig2_cell_shifts", out_dir)
    return fig


# ============================================================
# FIGURE 3: (beta_failure, beta_success) PLANE, LOG-ODDS
# ============================================================

def _draw_sd_plane_axes(ax, lim: float, label_diagonals: bool = False) -> None:
    ax.axhline(0, color="k", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="k", linewidth=0.8, alpha=0.5)
    ax.plot([-lim, lim], [-lim, lim],
            linestyle="--", color="#7F77DD", linewidth=0.8, alpha=0.7)
    ax.plot([-lim, lim], [lim, -lim],
            linestyle="--", color="#7F77DD", linewidth=0.8, alpha=0.7)
    if label_diagonals:
        ax.text( lim * 0.95,  lim * 0.95, "+S", color="#3C3489",
                fontsize=8, ha="right", va="top")
        ax.text(-lim * 0.95,  lim * 0.95, "+D", color="#3C3489",
                fontsize=8, ha="left",  va="top")
        ax.text(-lim * 0.95, -lim * 0.95, "-S", color="#3C3489",
                fontsize=8, ha="left",  va="bottom")
        ax.text( lim * 0.95, -lim * 0.95, "-D", color="#3C3489",
                fontsize=8, ha="right", va="bottom")


def _compute_sd_plane_limit(
    boot_wide: pd.DataFrame,
    point_wide: pd.DataFrame,
    quantile: float,
    margin: float = 1.15,
) -> float:
    chi2_val = chi2.ppf(quantile, df=2)
    max_extent = 0.0
    for (_, _), grp in boot_wide.groupby(["rat_id", "subset"]):
        x = grp["beta_failure_n-1"].to_numpy()
        y = grp["beta_success_n-1"].to_numpy()
        if x.size < 2:
            continue
        cov = np.cov(x, y)
        eig = np.linalg.eigvalsh(cov)
        rad = np.sqrt(chi2_val * max(eig.max(), 0.0))
        mx = max(abs(x.mean()) + rad, abs(y.mean()) + rad)
        if mx > max_extent:
            max_extent = mx
    max_point = point_wide[["beta_failure_n-1", "beta_success_n-1"]].abs().max().max()
    return float(max(max_extent, max_point) * margin)


def figure_3_sd_plane(
    summary: pd.DataFrame,
    boot: pd.DataFrame,
    C: str = DEFAULT_C,
    quantile: float = ELLIPSE_QUANTILE,
    out_dir: Path = FIGURES_DIR,
    save: bool = True,
) -> plt.Figure:
    rats = get_rat_order(summary)

    boot_wide = _pivot_boot_wide(
        boot, C=C,
        quantities=["beta_failure_n-1", "beta_success_n-1"],
    )
    point_wide = (
        summary[(summary["C"] == C) &
                (summary["quantity"].isin(
                    ["beta_failure_n-1", "beta_success_n-1"]))]
        .pivot_table(index=["rat_id", "subset"],
                     columns="quantity", values="value")
        .reset_index()
    )

    lim = _compute_sd_plane_limit(boot_wide, point_wide, quantile=quantile)

    fig = plt.figure(figsize=(11.5, 15.5))
    gs = fig.add_gridspec(
        5, 3,
        height_ratios=[1.7, 1, 1, 1, 1],
        hspace=0.35, wspace=0.25,
    )
    group_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    rat_axes   = [fig.add_subplot(gs[1 + i // 3, i % 3]) for i in range(len(rats))]

    # Group panels (one per modality).
    for ax, mod in zip(group_axes, MODALITY_ORDER):
        color = MODALITY_COLORS[mod]
        _draw_sd_plane_axes(ax, lim, label_diagonals=True)
        for rat in rats:
            pt = point_wide[
                (point_wide["rat_id"] == rat) &
                (point_wide["subset"] == mod)
            ]
            bt = boot_wide[
                (boot_wide["rat_id"] == rat) &
                (boot_wide["subset"] == mod)
            ]
            if pt.empty or bt.empty:
                continue
            x0 = float(pt["beta_failure_n-1"].iloc[0])
            y0 = float(pt["beta_success_n-1"].iloc[0])

            ell = _cov_ellipse(
                bt["beta_failure_n-1"].to_numpy(),
                bt["beta_success_n-1"].to_numpy(),
                quantile=quantile,
                edgecolor=color, facecolor=color, alpha=0.12,
                linewidth=0.8,
            )
            ax.add_patch(ell)
            ax.plot(x0, y0, "o", color=color, markersize=4,
                    markeredgecolor="white", markeredgewidth=0.5)
            ax.text(x0 + lim * 0.02, y0 + lim * 0.02, str(int(rat)),
                    fontsize=8, color=color)

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(mod, fontsize=11, color=color)
        ax.set_xlabel(r"$\beta_{\mathrm{failure}}$", fontsize=10)
        ax.set_ylabel(r"$\beta_{\mathrm{success}}$", fontsize=10)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    # Per-rat panels.
    for ax, rat in zip(rat_axes, rats):
        _draw_sd_plane_axes(ax, lim, label_diagonals=False)
        for mod in MODALITY_ORDER:
            color = MODALITY_COLORS[mod]
            pt = point_wide[
                (point_wide["rat_id"] == rat) &
                (point_wide["subset"] == mod)
            ]
            bt = boot_wide[
                (boot_wide["rat_id"] == rat) &
                (boot_wide["subset"] == mod)
            ]
            if pt.empty or bt.empty:
                continue
            x0 = float(pt["beta_failure_n-1"].iloc[0])
            y0 = float(pt["beta_success_n-1"].iloc[0])

            ell = _cov_ellipse(
                bt["beta_failure_n-1"].to_numpy(),
                bt["beta_success_n-1"].to_numpy(),
                quantile=quantile,
                edgecolor=color, facecolor=color, alpha=0.18,
                linewidth=0.8,
            )
            ax.add_patch(ell)
            ax.plot(x0, y0, "o", color=color, markersize=4,
                    markeredgecolor="white", markeredgewidth=0.5)

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_title(f"rat {rat}", fontsize=9)
        ax.tick_params(labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for ax in rat_axes[-3:]:
        ax.set_xlabel(r"$\beta_{\mathrm{failure}}$", fontsize=9)
    for ax in [rat_axes[i] for i in range(0, len(rat_axes), 3)]:
        ax.set_ylabel(r"$\beta_{\mathrm{success}}$", fontsize=9)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color=MODALITY_COLORS[m],
                   markersize=6, linestyle="none", label=m)
        for m in MODALITY_ORDER
    ]
    fig.legend(
        handles=legend_handles, loc="upper center",
        ncol=len(MODALITY_ORDER), frameon=False,
        bbox_to_anchor=(0.5, 1.00), fontsize=10,
    )
    fig.suptitle(
        f"(beta_failure, beta_success) plane  "
        f"(C = {C}, ellipses = {int(quantile * 100)}% coverage)",
        fontsize=12, y=1.02,
    )

    if save:
        _save_figure(fig, "fig3_sd_plane", out_dir)
    return fig


# ============================================================
# FIGURE 4: S, D, P FOREST (log-odds)
# ============================================================

def _compute_sdp(
    summary: pd.DataFrame,
    boot: pd.DataFrame,
    C: str,
) -> pd.DataFrame:
    quantities = ["beta_success_n-1", "beta_failure_n-1", "beta_reward_n-1"]
    boot_wide = _pivot_boot_wide(boot, C=C, quantities=quantities)
    boot_wide["S"] = boot_wide["beta_success_n-1"] + boot_wide["beta_failure_n-1"]
    boot_wide["D"] = boot_wide["beta_success_n-1"] - boot_wide["beta_failure_n-1"]
    boot_wide["P"] = boot_wide["beta_reward_n-1"]

    point_wide = (
        summary[(summary["C"] == C) &
                (summary["quantity"].isin(quantities))]
        .pivot_table(index=["rat_id", "subset"],
                     columns="quantity", values="value")
        .reset_index()
    )
    point_wide["S"] = point_wide["beta_success_n-1"] + point_wide["beta_failure_n-1"]
    point_wide["D"] = point_wide["beta_success_n-1"] - point_wide["beta_failure_n-1"]
    point_wide["P"] = point_wide["beta_reward_n-1"]

    rows = []
    for (rat, sub), grp in boot_wide.groupby(["rat_id", "subset"]):
        pt = point_wide[(point_wide["rat_id"] == rat) &
                        (point_wide["subset"] == sub)]
        if pt.empty:
            continue
        for comp in ("S", "D", "P"):
            lo, hi = _percentile_ci(grp[comp].to_numpy(), ci_level=95)
            rows.append({
                "rat_id":   rat,
                "subset":   sub,
                "component": comp,
                "value":    float(pt[comp].iloc[0]),
                "ci_low":   lo,
                "ci_high":  hi,
            })
    return pd.DataFrame(rows)


def figure_4_sdp_forest(
    summary: pd.DataFrame,
    boot: pd.DataFrame,
    C: str = DEFAULT_C,
    out_dir: Path = FIGURES_DIR,
    save: bool = True,
) -> plt.Figure:
    rats   = get_rat_order(summary)
    n_rats = len(rats)
    sdp    = _compute_sdp(summary, boot, C=C)

    components = [
        ("S", "S  =  " + r"$\beta_{\mathrm{success}} + \beta_{\mathrm{failure}}$"),
        ("D", "D  =  " + r"$\beta_{\mathrm{success}} - \beta_{\mathrm{failure}}$"),
        ("P", "P  =  " + r"$\beta_{\mathrm{reward}}$"),
    ]
    n_panels = len(components)

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(3.2 * n_panels, 0.34 * n_rats + 1.8),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    dodge = {mod: (i - 1) * 0.22 for i, mod in enumerate(MODALITY_ORDER)}

    for ax, (comp, title) in zip(axes, components):
        sub = sdp[sdp["component"] == comp]
        for mod in MODALITY_ORDER:
            color = MODALITY_COLORS[mod]
            mod_sub = sub[sub["subset"] == mod]
            sig_ys, sig_xs, sig_lo, sig_hi = [], [], [], []
            ns_ys,  ns_xs,  ns_lo,  ns_hi  = [], [], [], []
            for i, rat in enumerate(rats):
                row = mod_sub[mod_sub["rat_id"] == rat]
                if row.empty:
                    continue
                r = row.iloc[0]
                if not np.isfinite(r["value"]):
                    continue
                y_pos = (n_rats - 1 - i) + dodge[mod]
                sig  = _ci_excludes_zero(r["ci_low"], r["ci_high"])
                tgt  = (sig_ys, sig_xs, sig_lo, sig_hi) if sig \
                       else (ns_ys, ns_xs, ns_lo, ns_hi)
                tgt[0].append(y_pos)
                tgt[1].append(r["value"])
                tgt[2].append(r["value"] - r["ci_low"])
                tgt[3].append(r["ci_high"] - r["value"])

            if ns_xs:
                ax.errorbar(
                    ns_xs, ns_ys, xerr=[ns_lo, ns_hi],
                    fmt="o", markersize=3.5,
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=1.0, ecolor=color,
                    elinewidth=0.8, capsize=0, alpha=0.75,
                )
            if sig_xs:
                ax.errorbar(
                    sig_xs, sig_ys, xerr=[sig_lo, sig_hi],
                    fmt="o", markersize=4.5,
                    markerfacecolor=color, markeredgecolor=color,
                    markeredgewidth=1.0, ecolor=color,
                    elinewidth=1.4, capsize=0,
                )

        ax.axvline(0, color="k", linewidth=0.8, alpha=0.4)
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_yticks(range(n_rats))
    axes[0].set_yticklabels([f"rat {r}" for r in rats[::-1]])

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color=MODALITY_COLORS[m],
                   markersize=6, linestyle="none", label=m)
        for m in MODALITY_ORDER
    ]
    legend_handles += [
        plt.Line2D([0], [0], marker="o", color="k", markerfacecolor="k",
                   markersize=6, linestyle="none", label="CI excludes 0"),
        plt.Line2D([0], [0], marker="o", color="k", markerfacecolor="white",
                   markersize=6, linestyle="none", label="CI includes 0"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper center",
        ncol=len(legend_handles), frameon=False,
        bbox_to_anchor=(0.5, 1.005), fontsize=9,
    )
    fig.supxlabel("log-odds", fontsize=11, y=-0.01)
    fig.suptitle(f"S, D, P components in log-odds  (C = {C})",
                 fontsize=12, y=1.06)
    fig.tight_layout()

    if save:
        _save_figure(fig, "fig4_sdp_forest", out_dir)
    return fig


# ============================================================
# MAIN
# ============================================================

def _flagged_rats_by_mod(intercept_df: pd.DataFrame) -> Dict[str, List]:
    out = {}
    for mod, grp in intercept_df.groupby("subset"):
        out[mod] = grp.loc[grp["flagged"], "rat_id"].tolist()
    return out


if __name__ == "__main__":
    print("Loading data...")
    summary, boot, metadata = load_data()
    print(f"  {len(summary):>7} summary rows")
    print(f"  {len(boot):>7,} bootstrap rows")
    print(f"  C labels: {sorted(summary['C'].unique())}")
    print(f"  Modalities: {sorted(summary['subset'].unique())}")

    print("\nRunning intercept diagnostic...")
    intercept_df = check_intercepts(summary, C=DEFAULT_C)
    flagged     = _flagged_rats_by_mod(intercept_df)

    print("Generating Figure 1: coefficient forest...")
    figure_1_coefficient_forest(summary, C=DEFAULT_C)

    print("Generating Figure 2: cell probability shifts per rat...")
    figure_2_cell_shifts(summary, boot, C=DEFAULT_C,
                         flagged_rats_by_mod=flagged)

    print("Generating Figure 3: SD plane (beta_failure, beta_success)...")
    figure_3_sd_plane(summary, boot, C=DEFAULT_C,
                      quantile=ELLIPSE_QUANTILE)

    print("Generating Figure 4: S, D, P forest in log-odds...")
    figure_4_sdp_forest(summary, boot, C=DEFAULT_C)

    plt.show()