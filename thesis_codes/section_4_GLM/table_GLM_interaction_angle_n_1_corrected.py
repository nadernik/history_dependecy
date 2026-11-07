"""Recompute bootstrap p values, apply global BH-FDR, and make a LaTeX table.

Input
-----
angle_n1_boot_summary.csv, with one row per rat x modality and columns:
rat_id, modality, C, n_boot, mean_draw, frac_positive, frac_negative.

Outputs
-------
angle_n1_bh_results.csv
angle_n1_bh_table.tex

The two-sided bootstrap p value uses the finite-sample plus-one convention

    p = 2 * (min(n(theta* < 0), n(theta* > 0)) + 1) / (B + 1),

clipped to 1.  BH-FDR is then applied once across all 36 rat x modality tests.
"""



from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODALITY_ORDER = ["tactile", "visual", "visuotactile"]
MODALITY_LABEL = {
    "tactile": "Tactile",
    "visual": "Visual",
    "visuotactile": "Visuotactile",
}


def bootstrap_p_plus_one(frac_positive: float, frac_negative: float, n_boot: int) -> float:
    """Two-sided sign/bootstrap p value with a plus-one finite-sample correction."""
    # Fractions in the summary were computed from integer bootstrap counts.
    # rint safely reconstructs those counts despite decimal serialization.
    n_pos = int(np.rint(frac_positive * n_boot))
    n_neg = int(np.rint(frac_negative * n_boot))
    if not (0 <= n_pos <= n_boot and 0 <= n_neg <= n_boot):
        raise ValueError("Invalid bootstrap sign fractions")
    return min(1.0, 2.0 * (min(n_pos, n_neg) + 1) / (n_boot + 1))


def bh_fdr(p_values) -> np.ndarray:
    """Benjamini-Hochberg adjusted p values, returned in original order."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p values must be a finite one-dimensional array in [0, 1]")
    m = p.size
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    raw_adjusted = ranked * m / np.arange(1, m + 1)
    adjusted_sorted = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = np.clip(adjusted_sorted, 0.0, 1.0)
    return adjusted


def stars_for_q(q: float) -> str:
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def latex_number(x: float, decimals: int, pad_sign: bool = False) -> str:
    sign = "-" if x < 0 else (r"\hphantom{-}" if pad_sign else "")
    return f"${sign}{abs(x):.{decimals}f}$"


def latex_q(q: float) -> str:
    stars = stars_for_q(q)
    number = f"{q:.4f}"
    if stars:
        return rf"${number}^{{{stars}}}$"
    return f"${number}$"


def build_latex_table(df: pd.DataFrame) -> str:
    indexed = df.set_index(["rat_id", "modality"])
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Previous-stimulus angle coefficients by rat and modality.}",
        r"\label{tab:angle-n1-bh}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lccccccccc}",
        r"\toprule",
        r"Rat & \multicolumn{3}{c}{Tactile} & \multicolumn{3}{c}{Visual} & \multicolumn{3}{c}{Visuotactile} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
        r" & $\beta_{\mathrm{angle}_{n-1}}$ & $p_{\mathrm{boot}}$ & $q_{\mathrm{BH}}$"
        r" & $\beta_{\mathrm{angle}_{n-1}}$ & $p_{\mathrm{boot}}$ & $q_{\mathrm{BH}}$"
        r" & $\beta_{\mathrm{angle}_{n-1}}$ & $p_{\mathrm{boot}}$ & $q_{\mathrm{BH}}$ \\",
        r"\midrule",
    ]
    for rat in sorted(df["rat_id"].unique()):
        cells = [str(int(rat))]
        for modality in MODALITY_ORDER:
            row = indexed.loc[(rat, modality)]
            cells.extend(
                [
                    latex_number(float(row["beta_angle_n1"]), 3, pad_sign=True),
                    f"${float(row['p_boot_plus1']):.4f}$",
                    latex_q(float(row["q_bh_36"])),
                ]
            )
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\vspace{4pt}",
            r"\par\noindent",
            r"\begin{minipage}{\linewidth}\small",
            r"Values are unpenalized modality-specific GLM coefficients. Two-sided bootstrap p values use "
            r"$2(k+1)/(B+1)$, where $k$ is the smaller number of draws strictly above or below zero "
            r"and $B=5000$. The 36 p values were corrected together using Benjamini--Hochberg FDR. "
            r"Significance markers appear only on q values: *** $q<.001$, ** $q<.01$, * $q<.05$.",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    required = {
        "rat_id", "modality", "quantity", "C", "n_boot", "mean_draw",
        "frac_positive", "frac_negative",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(df) != 36 or df[["rat_id", "modality"]].duplicated().any():
        raise ValueError("Expected exactly 36 unique rat x modality rows")
    if set(df["modality"]) != set(MODALITY_ORDER):
        raise ValueError(f"Unexpected modalities: {sorted(set(df['modality']))}")
    if not (df["quantity"] == "beta_angle_n-1").all():
        raise ValueError("Input contains a quantity other than beta_angle_n-1")
    if not (df["C"] == "unpenalized").all():
        raise ValueError("Input must contain only the unpenalized reference fits")

    df = df.copy()
    df["beta_angle_n1"] = df["mean_draw"]
    df["p_boot_plus1"] = [
        bootstrap_p_plus_one(pos, neg, int(b))
        for pos, neg, b in zip(df["frac_positive"], df["frac_negative"], df["n_boot"])
    ]
    df["q_bh_36"] = bh_fdr(df["p_boot_plus1"])
    df["significant_bh_05"] = df["q_bh_36"] < 0.05

    df["modality"] = pd.Categorical(df["modality"], MODALITY_ORDER, ordered=True)
    df = df.sort_values(["rat_id", "modality"]).reset_index(drop=True)

    out = df[
        [
            "rat_id", "modality", "beta_angle_n1", "n_boot",
            "frac_positive", "frac_negative", "p_boot_plus1",
            "q_bh_36", "significant_bh_05",
        ]
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "angle_n1_bh_results.csv"
    tex_path = args.out_dir / "angle_n1_bh_table.tex"
    out.to_csv(csv_path, index=False)
    tex_path.write_text(build_latex_table(out), encoding="utf-8")

    significant = out.loc[out["significant_bh_05"], ["rat_id", "modality", "beta_angle_n1", "p_boot_plus1", "q_bh_36"]]
    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")
    print(f"BH-FDR significant cells: {len(significant)}/36")
    print(significant.to_string(index=False))


if __name__ == "__main__":
    main()
