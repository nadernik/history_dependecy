"""
psychometric_2x2_figure.py
==========================
One presentation figure per rat (default: rat 11) that explains the
conditional psychometric analysis:

  TOP     a 2 x 2 grid of coloured boxes, one per conditioning cell
          (n-1 choice H / V  x  n-1 outcome rewarded / unrewarded),
          each labelled mu_{choice,outcome} and its trial count.
  BOTTOM  two psychometric panels, paired exactly as in
          psychometric_4analyses_all_rats.py:
              left  = Analysis 3, n-1 rewarded    (h_rew   vs v_rew)
              right = Analysis 4, n-1 unrewarded  (h_unrew vs v_unrew)
          Each curve has the colour of its box. Unrewarded curves are
          dashed (second cue besides colour).

Data, parameters and curve shape are read exactly as in
psychometric_4analyses_all_rats.py (same bootstrap_qc helpers, same npz
files, same cumulative Gaussian, same sigma-degenerate exclusion for the
bands). Nothing is re-fitted.

Usage
-----
    python psychometric_2x2_figure.py                 # rat 11
    python psychometric_2x2_figure.py --rat 9         # another rat
    python psychometric_2x2_figure.py --rat 11 --layout rows_outcome
    python psychometric_2x2_figure.py --no-mu         # hide the mu markers

--layout
    columns_match_plots (default): matrix columns = outcome, so each matrix
        column sits directly above the panel that contains its two curves.
    rows_outcome: matrix rows = outcome, columns = choice (same orientation
        as the boxes drawn on slide 9).

Output: OUT_DIR/rat_<rat>_psychometric_2x2.svg (text kept as editable text).
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from scipy.stats import norm

import bootstrap_qc as qc

# ----------------------------------------------------------------------------
# CONFIG -- same paths as psychometric_4analyses_all_rats.py
# ----------------------------------------------------------------------------
DATA_DIR = r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_3_choice_reward\choice_reward_joint_results\filtered\pooled\tables"
BOOTSTRAP_DIR = r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_3_choice_reward\choice_reward_joint_results\filtered\pooled\bootstraps"
OUT_DIR = r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\ale_thesis_fig_tab\images_thesis\section_3"

CELL_COMP_FILE = "cell_composition.csv"
FULL_FITS_FILE = "full_data_cell_fits.csv"
NPZ_PATTERN = "rat_{rat}_joint_bootstrap.npz"
DEFAULT_RAT = 11

X_PLOT = np.linspace(0, 90, 200)

# ----------------------------------------------------------------------------
# STYLE -- colours match the 2 x 2 boxes on slide 9 of the presentation
# ----------------------------------------------------------------------------
CELL_COLOR = {
    "h_rew": "#C85200",    # H, rewarded
    "v_rew": "#5E3C99",    # V, rewarded
    "h_unrew": "#F0A04B",  # H, unrewarded
    "v_unrew": "#9C83CF",  # V, unrewarded
}
CELL_TEXT_COLOR = {"h_rew": "white", "v_rew": "white", "h_unrew": "#262B33", "v_unrew": "#262B33"}
CELL_LINESTYLE = {"h_rew": "-", "v_rew": "-", "h_unrew": "--", "v_unrew": "--"}
CELL_CHOICE = {"h_rew": "H", "v_rew": "V", "h_unrew": "H", "v_unrew": "V"}
CELL_OUTCOME = {"h_rew": "R", "v_rew": "R", "h_unrew": "U", "v_unrew": "U"}

INK = "#262B33"
MUTED = "#6B7280"
REF = "#B0B7C3"
BAND_ALPHA = 0.18

FONT_FAMILY = ["Calibri", "Carlito", "DejaVu Sans"]  # first one installed is used
plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 12,
    "svg.fonttype": "none",          # keep text as text in the SVG
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#9AA3AF",
    "xtick.color": "#4B5563",
    "ytick.color": "#4B5563",
    "axes.labelcolor": INK,
})

# Panels, paired as in psychometric_4analyses_all_rats.py
PANELS = [
    {"cells": ("h_rew", "v_rew"), "title": "After a rewarded trial", "outcome": "R"},
    {"cells": ("h_unrew", "v_unrew"), "title": "After an unrewarded trial", "outcome": "U"},
]


# ----------------------------------------------------------------------------
# DATA -- identical readout to psychometric_4analyses_all_rats.py
# ----------------------------------------------------------------------------
def psychometric(x, mu, sigma, gamma, lapse):
    """4-parameter cumulative Gaussian, as in the fitting script."""
    return gamma + (1.0 - gamma - lapse) * norm.cdf((x - mu) / sigma)


def load_tables():
    cell_comp = pd.read_csv(os.path.join(DATA_DIR, CELL_COMP_FILE))
    cell_comp = cell_comp[cell_comp["n_trials"] > 0]
    full_fits = pd.read_csv(os.path.join(DATA_DIR, FULL_FITS_FILE))
    return cell_comp, full_fits


def n_trials_for(cell_comp, rat, cell):
    row = cell_comp[(cell_comp["rat"] == rat) & (cell_comp["cell"] == cell)]
    return int(row["n_trials"].sum()) if len(row) else 0


def cell_curves(data, cell):
    """Median curve, 95% band and median mu for one cell (sigma-degenerate
    bootstrap iterations excluded from the band, as in the original)."""
    mask = qc.sigma_degenerate_mask(data, cell)
    keep = ~mask
    p = {k: qc.cell_param(data, cell, k) for k in ("mu", "sigma", "gamma", "lapse")}
    curves = psychometric(X_PLOT[None, :], p["mu"][keep, None], p["sigma"][keep, None],
                          p["gamma"][keep, None], p["lapse"][keep, None])
    return {
        "median": np.median(curves, axis=0),
        "lo": np.percentile(curves, 2.5, axis=0),
        "hi": np.percentile(curves, 97.5, axis=0),
        "mu": float(np.median(p["mu"][keep])),
        "n_excluded": int(mask.sum()),
    }


# ----------------------------------------------------------------------------
# DRAWING
# ----------------------------------------------------------------------------
def draw_matrix(ax, ns, layout):
    """2 x 2 coloured boxes in axes coordinates [0,1] x [0,1]."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if layout == "columns_match_plots":
        col_keys = ["R", "U"]            # columns = outcome (above each panel)
        row_keys = ["H", "V"]            # rows = previous choice
        col_label = {"R": "n−1 rewarded", "U": "n−1 unrewarded"}
        row_label = {"H": "n−1 choice\nHorizontal", "V": "n−1 choice\nVertical"}
        cell_at = lambda r, c: {("H", "R"): "h_rew", ("V", "R"): "v_rew",
                                ("H", "U"): "h_unrew", ("V", "U"): "v_unrew"}[(r, c)]
    else:  # rows_outcome
        col_keys = ["H", "V"]
        row_keys = ["R", "U"]
        col_label = {"H": "n−1 choice: Horizontal", "V": "n−1 choice: Vertical"}
        row_label = {"R": "n−1\nrewarded", "U": "n−1\nunrewarded"}
        cell_at = lambda r, c: {("R", "H"): "h_rew", ("R", "V"): "v_rew",
                                ("U", "H"): "h_unrew", ("U", "V"): "v_unrew"}[(r, c)]

    left, right, top, bottom = 0.14, 0.98, 0.80, 0.02
    gap = 0.03
    cw = (right - left - gap) / 2
    rh = (top - bottom - gap) / 2
    for j, ck in enumerate(col_keys):
        cx = left + j * (cw + gap)
        ax.text(cx + cw / 2, top + 0.06, col_label[ck], ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=INK)
        for i, rk in enumerate(row_keys):
            cy = top - (i + 1) * rh - i * gap
            cell = cell_at(rk, ck)
            ax.add_patch(FancyBboxPatch((cx, cy), cw, rh, boxstyle="round,pad=0,rounding_size=0.03",
                                        facecolor=CELL_COLOR[cell], edgecolor="none",
                                        transform=ax.transAxes, mutation_aspect=0.35))
            ch, oc = CELL_CHOICE[cell], CELL_OUTCOME[cell]
            ax.text(cx + cw / 2, cy + rh * 0.58, rf"$\mu_{{{ch},{oc}}}$", ha="center", va="center",
                    fontsize=20, color=CELL_TEXT_COLOR[cell])
            ax.text(cx + cw / 2, cy + rh * 0.22, f"n = {ns[cell]:,}", ha="center", va="center",
                    fontsize=10.5, color=CELL_TEXT_COLOR[cell])
    for i, rk in enumerate(row_keys):
        cy = top - (i + 1) * rh - i * gap
        ax.text(left - 0.015, cy + rh / 2, row_label[rk], ha="right", va="center",
                fontsize=11.5, fontweight="bold", color=INK, linespacing=1.1)


def draw_panel(ax, panel, curves, ns, show_mu):
    for cell in panel["cells"]:
        c = curves[cell]
        col = CELL_COLOR[cell]
        ax.fill_between(X_PLOT, c["lo"], c["hi"], color=col, alpha=BAND_ALPHA, lw=0)
        ax.plot(X_PLOT, c["median"], color=col, lw=2.4, ls=CELL_LINESTYLE[cell])
        if show_mu:
            # mu is the curve's midpoint between its asymptotes, so mark it ON
            # the median curve (not at y = 0.5, which differs when gamma != lapse)
            y_mu = float(np.interp(c["mu"], X_PLOT, c["median"]))
            ax.plot([c["mu"], c["mu"]], [-0.03, y_mu], color=col, lw=1.4, ls=":")
            ax.plot(c["mu"], y_mu, "o", ms=7, color=col, mec="white", mew=1.2, zorder=5)
    ax.axvline(45, color=REF, lw=1, ls=":", zorder=0)
    ax.axhline(0.5, color=REF, lw=1, ls=":", zorder=0)
    ax.set_xlim(0, 90)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xlabel("Current stimulus orientation (°)")
    ax.set_title(panel["title"], fontsize=13, color=INK, fontweight="bold", loc="left")

    handles = []
    for cell in panel["cells"]:
        ch, oc = CELL_CHOICE[cell], CELL_OUTCOME[cell]
        name = "previous choice H" if ch == "H" else "previous choice V"
        mu_txt = f", μ = {curves[cell]['mu']:.1f}°" if show_mu else ""
        handles.append(Line2D([0], [0], color=CELL_COLOR[cell], lw=2.4, ls=CELL_LINESTYLE[cell],
                              label=f"{name}{mu_txt}"))
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=10.5)


def build(rat=DEFAULT_RAT, layout="columns_match_plots", show_mu=True, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    cell_comp, full_fits = load_tables()
    data = qc.load_bootstrap(rat, os.path.join(BOOTSTRAP_DIR, NPZ_PATTERN.format(rat=rat)))

    cells = [c for p in PANELS for c in p["cells"]]
    curves = {c: cell_curves(data, c) for c in cells}
    ns = {c: n_trials_for(cell_comp, rat, c) for c in cells}
    not_identified = {c: bool(qc.full_data_boundary_flags(full_fits, rat, c)) for c in cells}

    fig = plt.figure(figsize=(10, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.85, 2.3], hspace=0.24, wspace=0.12,
                          left=0.09, right=0.98, top=0.97, bottom=0.09)
    ax_m = fig.add_subplot(gs[0, :])
    draw_matrix(ax_m, ns, layout)

    ax_l = fig.add_subplot(gs[1, 0])
    ax_r = fig.add_subplot(gs[1, 1], sharey=ax_l)
    for ax, panel in zip((ax_l, ax_r), PANELS):
        draw_panel(ax, panel, curves, ns, show_mu)
    ax_l.set_ylabel("P(vertical)")
    plt.setp(ax_r.get_yticklabels(), visible=False)

    flagged = [c for c, f in not_identified.items() if f]
    if flagged:
        fig.text(0.09, 0.005, "γ/λ at a fitting bound for " + ", ".join(flagged) + " (μ unaffected)",
                 fontsize=9, color=MUTED, ha="left", va="bottom")

    path = os.path.join(out_dir, f"rat_{rat}_psychometric_2x2.svg")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    for c in cells:
        print(f"  {c:8s} n={ns[c]:>6}  mu={curves[c]['mu']:6.2f}  "
              f"sigma-degenerate iterations excluded from band: {curves[c]['n_excluded']}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="2 x 2 conditioning boxes + paired psychometric curves")
    ap.add_argument("--rat", type=int, default=DEFAULT_RAT)
    ap.add_argument("--layout", choices=["columns_match_plots", "rows_outcome"], default="columns_match_plots")
    ap.add_argument("--no-mu", action="store_true", help="hide the mu markers and the mu values in the legends")
    args = ap.parse_args()
    build(rat=args.rat, layout=args.layout, show_mu=not args.no_mu)