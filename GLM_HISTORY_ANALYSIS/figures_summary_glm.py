"""
Figure generation for history-bias GLM analysis.

Workflow:
    1. If history_bias_merged.csv exists, load it.
       Else merge history_bias_summary_per_modality.csv + hit_n1_test_summary.csv.
    2. Recompute choice-pattern classification using the revised rule (see doc).
    3. Save merged CSV with recomputed classification.
    4. Produce two figures:
         - population_choice_geometry.pdf: scatter (beta_success vs beta_failure)
           + beta_hit forest
         - per_rat_cards.pdf: grid of per-rat coefficient cards

Run from a folder containing the two source CSVs. Edit paths in the CONFIG
section if needed.
"""

from __future__ import annotations

import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent

CHOICE_CSV = SCRIPT_DIR / "history_bias_summary_per_modality.csv"
HIT_CSV = SCRIPT_DIR / "hit_n1_test_summary.csv"
MERGED_CSV = SCRIPT_DIR / "history_bias_merged.csv"

OUT_POP = SCRIPT_DIR / "population_choice_geometry.png"
OUT_CARDS = SCRIPT_DIR / "per_rat_cards.png"

MOD_ORDER = ["tactile", "visual", "visuotactile"]
MOD_COLORS = {
    "tactile": "#1baf7a",
    "visual": "#eb6834",
    "visuotactile": "#2a78d6",
}
MOD_SHORT = {"tactile": "T", "visual": "V", "visuotactile": "VT"}

CLASS_LABEL = {
    "reward_x_choice": "reward \u00d7 choice",
    "pure_choice": "pure choice",
    "ambiguous": "ambiguous",
}

# Card grid dimensions (n_rows * n_cols must be >= number of rats)
CARD_ROWS = 4
CARD_COLS = 3

# Shared x-axis range across all cards (coefficient log-odds)
CARD_XLIM = (-0.5, 0.5)


# ============================================================
# LOAD / MERGE
# ============================================================

def load_or_merge() -> pd.DataFrame:
    """Load merged CSV if present, else merge the two source CSVs."""
    if os.path.exists(MERGED_CSV):
        print(f"Loading existing {MERGED_CSV}")
        return pd.read_csv(MERGED_CSV)

    print(f"Merging {CHOICE_CSV} + {HIT_CSV}")
    if not os.path.exists(CHOICE_CSV):
        raise FileNotFoundError(f"Missing input: {CHOICE_CSV}")
    if not os.path.exists(HIT_CSV):
        raise FileNotFoundError(f"Missing input: {HIT_CSV}")

    choice_df = pd.read_csv(CHOICE_CSV).rename(
        columns={"classification": "choice_pattern"}
    )
    hit_df = pd.read_csv(HIT_CSV).rename(
        columns={"classification": "pure_reward"}
    )

    merged = choice_df.merge(
        hit_df, on=["rat_id", "subset"], how="outer", suffixes=("", "_hit")
    )

    if "n_trials_hit" in merged.columns:
        mismatches = merged[merged["n_trials"] != merged["n_trials_hit"]]
        if len(mismatches):
            print(f"WARNING: {len(mismatches)} rows with mismatched n_trials "
                  "between the two scripts")
            print(mismatches[["rat_id", "subset", "n_trials", "n_trials_hit"]])
        else:
            print("n_trials matches across both CSVs")
        merged = merged.drop(columns=["n_trials_hit"])

    return merged


# ============================================================
# CLASSIFICATION (revised rule from doc)
# ============================================================

def classify_choice_pattern(row: pd.Series) -> str:
    """Revised choice-pattern classification.

    reward_x_choice: diff CI excludes 0, unless both beta_success and
                     beta_failure CIs exclude 0 with the same sign (that
                     would be pure choice with different magnitudes).
    pure_choice:     diff CI includes 0 AND at least one coefficient CI
                     excludes 0 AND beta_success, beta_failure have same sign.
    ambiguous:       otherwise.
    """
    d_lo = row["diff_ci_low"]
    d_hi = row["diff_ci_high"]
    s_lo = row["ci_low_success_n-1"]
    s_hi = row["ci_high_success_n-1"]
    f_lo = row["ci_low_failure_n-1"]
    f_hi = row["ci_high_failure_n-1"]
    s = row["beta_success_n-1"]
    f = row["beta_failure_n-1"]

    diff_excl = not (d_lo <= 0 <= d_hi)
    s_excl = not (s_lo <= 0 <= s_hi)
    f_excl = not (f_lo <= 0 <= f_hi)
    same_sign = (s * f) > 0

    if diff_excl:
        if s_excl and f_excl and same_sign:
            return "pure_choice"
        return "reward_x_choice"
    if (s_excl or f_excl) and same_sign:
        return "pure_choice"
    return "ambiguous"


def add_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute choice_pattern column using the revised rule."""
    df = df.copy()
    df["choice_pattern"] = df.apply(classify_choice_pattern, axis=1)
    return df


# ============================================================
# VERDICT + SORT
# ============================================================

def _rat_modality_map(rat_rows: pd.DataFrame, col: str) -> dict:
    """Return {modality: value} for one rat's rows on a given column."""
    return {
        m: rat_rows[rat_rows["subset"] == m][col].iloc[0]
        for m in MOD_ORDER
        if len(rat_rows[rat_rows["subset"] == m]) > 0
    }


def rat_verdict(rat_rows: pd.DataFrame) -> str:
    """Verdict string for a rat's card header.

    Examples:
        'reward x choice \u00b7 consistent'
        'pure choice \u00b7 consistent'
        'reward x choice (T) \u00b7 pure choice (V, VT)'
    """
    classes = _rat_modality_map(rat_rows, "choice_pattern")
    if not classes:
        return "no data"
    unique = set(classes.values())
    if len(unique) == 1:
        c = next(iter(unique))
        return f"{CLASS_LABEL.get(c, c)} \u00b7 consistent"

    parts = []
    for c in ["reward_x_choice", "pure_choice", "ambiguous"]:
        mods_with_c = [MOD_SHORT[m] for m in MOD_ORDER if classes.get(m) == c]
        if mods_with_c:
            parts.append(f"{CLASS_LABEL[c]} ({', '.join(mods_with_c)})")
    return " \u00b7 ".join(parts)


def rat_sort_key(rat_rows: pd.DataFrame) -> tuple:
    """Sort key: (group_order, -max|beta_s - beta_f|).

    Group order: reward x choice -> ambiguous -> pure choice.
    Majority classification across modalities, tie-broken by tactile.
    Within group, largest max|beta_success - beta_failure| first.
    """
    classes = _rat_modality_map(rat_rows, "choice_pattern")
    if not classes:
        return (99, 0)

    counts = Counter(classes.values())
    max_count = counts.most_common(1)[0][1]
    tied = [c for c, cnt in counts.items() if cnt == max_count]

    if len(tied) == 1:
        majority = tied[0]
    else:
        tactile_class = classes.get("tactile")
        majority = tactile_class if tactile_class in tied else tied[0]

    group_order = {"reward_x_choice": 0, "ambiguous": 1, "pure_choice": 2}
    group = group_order.get(majority, 3)

    max_diff = rat_rows.apply(
        lambda r: abs(r["beta_success_n-1"] - r["beta_failure_n-1"]), axis=1
    ).max()

    return (group, -max_diff)


def sorted_rat_ids(df: pd.DataFrame) -> list:
    rat_ids = sorted(df["rat_id"].unique())
    rat_data = {rid: df[df["rat_id"] == rid] for rid in rat_ids}
    return sorted(rat_ids, key=lambda rid: rat_sort_key(rat_data[rid]))


# ============================================================
# FIGURE 1: POPULATION (scatter + beta_hit forest)
# ============================================================

def _compute_scatter_ranges(df: pd.DataFrame) -> tuple:
    """Data-driven ranges for scatter, ensuring 0 is visible."""
    s = df["beta_success_n-1"].dropna().values
    f = df["beta_failure_n-1"].dropna().values

    s_lo, s_hi = np.percentile(s, [2, 98])
    f_lo, f_hi = np.percentile(f, [2, 98])

    s_lo = min(s_lo, -0.05)
    s_hi = max(s_hi, 0.05)
    f_lo = min(f_lo, -0.05)
    f_hi = max(f_hi, 0.05)

    s_pad = (s_hi - s_lo) * 0.05
    f_pad = (f_hi - f_lo) * 0.05
    return s_lo - s_pad, s_hi + s_pad, f_lo - f_pad, f_hi + f_pad


def make_scatter(ax, df: pd.DataFrame) -> None:
    x_lo, x_hi, y_lo, y_hi = _compute_scatter_ranges(df)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal", adjustable="box")

    ax.axhline(0, color="#888", linestyle=(0, (2, 3)), linewidth=0.8, alpha=0.6)
    ax.axvline(0, color="#888", linestyle=(0, (2, 3)), linewidth=0.8, alpha=0.6)

    diag_min = max(x_lo, y_lo)
    diag_max = min(x_hi, y_hi)
    ax.plot([diag_min, diag_max], [diag_min, diag_max],
            color="#5F5E5A", linestyle=(0, (6, 4)), linewidth=1.1, zorder=1)

    anti_min = max(x_lo, -y_hi)
    anti_max = min(x_hi, -y_lo)
    ax.plot([anti_min, anti_max], [-anti_min, -anti_max],
            color="#5F5E5A", linestyle=(0, (6, 4)), linewidth=1.1, zorder=1)

    t = 0.7
    pc_pos = diag_min + t * (diag_max - diag_min)
    ax.annotate("pure choice", xy=(pc_pos, pc_pos),
                xytext=(6, -4), textcoords="offset points",
                rotation=45, fontsize=10, style="italic", color="#5F5E5A")

    rc_x = anti_min + t * (anti_max - anti_min)
    ax.annotate("reward \u00d7 choice", xy=(rc_x, -rc_x),
                xytext=(-6, 6), textcoords="offset points",
                rotation=-45, fontsize=10, style="italic",
                color="#5F5E5A", ha="right")

    for _, row in df.iterrows():
        sv = row["beta_success_n-1"]
        fv = row["beta_failure_n-1"]
        if pd.isna(sv) or pd.isna(fv):
            continue
        color = MOD_COLORS.get(row["subset"])
        if color is None:
            continue

        sv_clip = float(np.clip(sv, x_lo, x_hi))
        fv_clip = float(np.clip(fv, y_lo, y_hi))
        clipped = (sv != sv_clip) or (fv != fv_clip)

        if clipped:
            ax.plot(sv_clip, fv_clip, marker="s", markersize=13,
                    color=color, markeredgecolor="white", markeredgewidth=1.2,
                    zorder=5)
            ax.annotate(f"{int(row['rat_id'])}", (sv_clip, fv_clip),
                        ha="center", va="center", fontsize=8, color="white",
                        fontweight="bold", zorder=6)
            ax.annotate(
                f"rat {int(row['rat_id'])} {MOD_SHORT[row['subset']]}: "
                f"({sv:.2f}, {fv:.2f})",
                xy=(sv_clip, fv_clip),
                xytext=(-8, -14), textcoords="offset points",
                fontsize=8, color=color, style="italic", ha="right",
            )
        else:
            ax.scatter(sv_clip, fv_clip, s=190, c=color,
                       edgecolors="white", linewidths=1.2,
                       alpha=0.88, zorder=5)
            ax.annotate(f"{int(row['rat_id'])}", (sv_clip, fv_clip),
                        ha="center", va="center", fontsize=9, color="white",
                        fontweight="bold", zorder=6)

    ax.set_xlabel(r"$\beta_{\mathrm{success}}$  "
                  r"(effect of previous choice | previous correct)",
                  fontsize=10)
    ax.set_ylabel(r"$\beta_{\mathrm{failure}}$  "
                  r"(effect of previous choice | previous wrong)",
                  fontsize=10)
    ax.set_title("Choice-pattern geometry", fontsize=13, loc="left", pad=8)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=MOD_COLORS[m],
               markersize=9, label=m)
        for m in MOD_ORDER
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              frameon=False, fontsize=9)


def make_hit_forest(ax, df: pd.DataFrame) -> None:
    """beta_hit forest plot, one row per rat, sorted by choice-pattern group."""
    rat_ids = sorted_rat_ids(df)
    rat_data = {rid: df[df["rat_id"] == rid] for rid in rat_ids}
    n_rats = len(rat_ids)
    y_pos = np.arange(n_rats)[::-1]

    mod_offsets = {"tactile": 0.22, "visual": 0.0, "visuotactile": -0.22}

    lo_vals = df["ci_low_hit_n-1"].dropna().values
    hi_vals = df["ci_high_hit_n-1"].dropna().values
    x_lo = min(lo_vals.min(), -0.10)
    x_hi = max(hi_vals.max(), 0.30)
    x_lo = np.floor(x_lo * 20) / 20
    x_hi = np.ceil(x_hi * 20) / 20

    ax.axvline(0, color="#888", linestyle=(0, (2, 3)), linewidth=0.8, alpha=0.6)

    for i, rid in enumerate(rat_ids):
        y = y_pos[i]
        rat_rows = rat_data[rid]
        for m in MOD_ORDER:
            r = rat_rows[rat_rows["subset"] == m]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            beta = r["beta_hit_n-1"]
            lo = r["ci_low_hit_n-1"]
            hi = r["ci_high_hit_n-1"]
            if pd.isna(beta):
                continue
            color = MOD_COLORS[m]
            y_off = y + mod_offsets[m]
            ax.hlines(y_off, max(lo, x_lo), min(hi, x_hi),
                      color=color, linewidth=1.3, alpha=0.75)

            is_pure = r.get("pure_reward") == "pure_reward_present"
            edge = "black" if is_pure else color
            ew = 1.3 if is_pure else 0
            ms = 6 if is_pure else 5
            ax.plot(beta, y_off, marker="o", markersize=ms,
                    color=color, markeredgecolor=edge, markeredgewidth=ew,
                    zorder=5)

    groups = [rat_sort_key(rat_data[rid])[0] for rid in rat_ids]
    for i in range(len(groups) - 1):
        if groups[i] != groups[i + 1]:
            y_div = (y_pos[i] + y_pos[i + 1]) / 2
            ax.axhline(y_div, color="#888", linestyle=(0, (4, 4)),
                       linewidth=0.8, alpha=0.55)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"rat {rid}" for rid in rat_ids], fontsize=9)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.6, n_rats - 0.4)
    ax.set_xlabel(r"$\beta_{\mathrm{hit}}$  (log-odds; pure reward main effect)",
                  fontsize=10)
    ax.set_title("Pure reward main effect", fontsize=13, loc="left", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_population_figure(df: pd.DataFrame, out_path: str) -> None:
    n_rats = df["rat_id"].nunique()
    fig = plt.figure(figsize=(9, 7 + 0.3 * n_rats))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1], hspace=0.32)

    make_scatter(fig.add_subplot(gs[0]), df)
    make_hit_forest(fig.add_subplot(gs[1]), df)

    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ============================================================
# FIGURE 2: PER-RAT CARDS
# ============================================================

COEF_SPEC = [
    ("beta_success_n-1", "ci_low_success_n-1", "ci_high_success_n-1",
     r"$\beta_{\mathrm{success}}$"),
    ("beta_failure_n-1", "ci_low_failure_n-1", "ci_high_failure_n-1",
     r"$\beta_{\mathrm{failure}}$"),
    ("beta_hit_n-1", "ci_low_hit_n-1", "ci_high_hit_n-1",
     r"$\beta_{\mathrm{hit}}$"),
    ("beta_angle_n-1", "ci_low_angle_n-1", "ci_high_angle_n-1",
     r"$\beta_{\mathrm{angle},\,n-1}$"),
]


def draw_card(ax, rid: int, rat_rows: pd.DataFrame,
              x_lo: float, x_hi: float) -> None:
    verdict = rat_verdict(rat_rows)
    ax.set_title(f"rat {rid}", fontsize=10, loc="left", pad=16, fontweight="bold")
    ax.text(0, 1.03, verdict, transform=ax.transAxes, fontsize=8,
            style="italic", color="#666", va="bottom")

    y_pos = np.arange(len(COEF_SPEC))[::-1]
    mod_offsets = {"tactile": 0.22, "visual": 0.0, "visuotactile": -0.22}
    ax.axvline(0, color="#888", linestyle=(0, (2, 3)), linewidth=0.8, alpha=0.6)

    for ci, (beta_col, lo_col, hi_col, _) in enumerate(COEF_SPEC):
        y = y_pos[ci]
        for m in MOD_ORDER:
            r = rat_rows[rat_rows["subset"] == m]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            beta = r[beta_col]
            if pd.isna(beta):
                continue
            lo = r[lo_col]
            hi = r[hi_col]
            color = MOD_COLORS[m]
            y_off = y + mod_offsets[m]

            beta_c = float(np.clip(beta, x_lo, x_hi))
            lo_c = float(np.clip(lo, x_lo, x_hi))
            hi_c = float(np.clip(hi, x_lo, x_hi))
            ax.hlines(y_off, lo_c, hi_c, color=color, linewidth=1.1, alpha=0.75)

            if beta > x_hi:
                ax.plot(x_hi, y_off, marker=">", markersize=6, color=color)
            elif beta < x_lo:
                ax.plot(x_lo, y_off, marker="<", markersize=6, color=color)
            else:
                if (beta_col == "beta_hit_n-1"
                        and r.get("pure_reward") == "pure_reward_present"):
                    ax.plot(beta_c, y_off, marker="o", markersize=5.5,
                            color=color, markeredgecolor="black",
                            markeredgewidth=1.2)
                else:
                    ax.plot(beta_c, y_off, marker="o", markersize=4, color=color)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([spec[3] for spec in COEF_SPEC], fontsize=9)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(-0.55, len(COEF_SPEC) - 0.45)
    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.tick_params(axis="both", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_cards_figure(df: pd.DataFrame, out_path: str) -> None:
    rat_ids = sorted_rat_ids(df)
    rat_data = {rid: df[df["rat_id"] == rid] for rid in rat_ids}

    n_rats = len(rat_ids)
    if CARD_ROWS * CARD_COLS < n_rats:
        raise ValueError(
            f"Grid ({CARD_ROWS}x{CARD_COLS}) too small for {n_rats} rats; "
            "increase CARD_ROWS/CARD_COLS."
        )

    fig, axes = plt.subplots(CARD_ROWS, CARD_COLS,
                              figsize=(3.4 * CARD_COLS, 2.8 * CARD_ROWS))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    for idx, ax in enumerate(axes_flat):
        if idx >= n_rats:
            ax.axis("off")
            continue
        rid = rat_ids[idx]
        draw_card(ax, int(rid), rat_data[rid], *CARD_XLIM)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=MOD_COLORS[m],
               markersize=8, label=m)
        for m in MOD_ORDER
    ]
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
               markeredgecolor="black", markeredgewidth=1.3,
               markersize=8, label="pure reward present (beta_hit only)")
    )
    fig.legend(handles=legend_elements, loc="lower center",
               bbox_to_anchor=(0.5, -0.005), frameon=False,
               ncol=4, fontsize=9)

    fig.suptitle("Per-rat coefficient cards", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


# ============================================================
# MAIN
# ============================================================

def print_summary(df: pd.DataFrame) -> None:
    print("\nChoice-pattern classification counts by modality:")
    print(df.groupby("subset")["choice_pattern"].value_counts()
            .unstack(fill_value=0))
    if "pure_reward" in df.columns:
        print("\nPure-reward classification counts by modality:")
        print(df.groupby("subset")["pure_reward"].value_counts()
                .unstack(fill_value=0))


if __name__ == "__main__":
    df = load_or_merge()
    df = add_classification(df)
    df.to_csv(MERGED_CSV, index=False)
    print(f"Saved {MERGED_CSV} with recomputed choice_pattern")

    print_summary(df)

    make_population_figure(df, OUT_POP)
    make_cards_figure(df, OUT_CARDS)

    print("\nDone.")