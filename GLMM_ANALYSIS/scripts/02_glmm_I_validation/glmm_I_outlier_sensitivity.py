"""glmm_I_outlier_sensitivity.py

Sensitivity of the GLMM I random-effect covariance structure to omission
of individual rats.

Motivation
----------
The primary GLMM I fit reports a between-rat random-slope correlation of
ρ_sf ≈ −0.85 and a variance anisotropy τ_D_β / τ_S_β ≈ 2.4. The equivalent
quantities computed on Section 4's unshrunk per-rat GLM estimates are
weaker (Pearson r ≈ −0.56, SD(D)/SD(S) ≈ 1.79). Leave-one-out analysis
shows both findings are substantially driven by one WSLS-outlier rat in
the pooled cohort. This script quantifies that sensitivity across three
complementary estimators so the reported GLMM numbers can be interpreted
correctly in the Results section.

The three estimators compared:
    1. Section 4 raw per-rat GLM       — no pooling, one β per rat
    2. Section 4 trial-count-weighted  — same estimates, weighted by trials
    3. GLMM BLUPs (b_S, b_F)           — partial-pooling estimates

plus the parametric ρ_sf estimated from the fitted covariance matrix.

Inputs
------
    --pkl  Path to glmm_I_full.pkl produced by 01_glmm_I/GLMM_I.py
           (e.g. outputs/01_glmm_I/legacy_ref-visuotactile/glmm_I_full.pkl)
    --subset / --C  passed through to common.load_subset (defaults: pooled, unpenalized)

Outputs (into GLMM_ANALYSIS/outputs/02_glmm_I_validation/outlier_sensitivity/<subset>/, or --out-dir)
------
    outlier_sensitivity_<subset>.csv  long-format table (source, quantity, rat_dropped, value, Δ)
    outlier_sensitivity_<subset>.txt  formatted report
"""
from __future__ import annotations

import argparse
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Section 4 quantity names and the pooled, training-filtered Section 4 GLM summary
PROJECT_ROOT = Path(__file__).resolve().parents[3]   # history_dependecy/
Q_BS = "beta_success_n-1"
Q_BF = "beta_failure_n-1"


def glm_summary_path() -> Path:
    return PROJECT_ROOT / "thesis_codes" / "section_4_GLM" / "glm_results_thesis_filtered" / "glm_summary_long.csv"


# ============================================================
# Loaders
# ============================================================
def load_glmm_result(pkl_path: Path) -> dict:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if data.get("used_fallback", False):
        print("[warn] this pickle used the FALLBACK random structure "
              "(independent success/failure slopes). Under the fallback, "
              "ρ_sf is 0 by construction and τ_S_β = τ_D_β. The parametric "
              "correlation and anisotropy comparisons below therefore lose "
              "their diagnostic value.")
    return data["final_result"]


def load_section4_per_rat(subset: str = "pooled",
                          C_label: str = "unpenalized") -> pd.DataFrame:
    """Per-rat β_s, β_f, and trial count from Section 4's GLM summary CSV."""
    summary = pd.read_csv(glm_summary_path())
    m = (summary["subset"] == subset) & (summary["C"] == C_label)
    df = summary.loc[m].copy()
    if df.empty:
        raise ValueError(
            f"No Section 4 rows with subset={subset!r}, C={C_label!r}. "
            f"Available: subsets={sorted(summary['subset'].unique())}, "
            f"C={sorted(summary['C'].unique())}"
        )
    wide = df.pivot(index="rat_id", columns="quantity", values="value")
    n_trials = df.drop_duplicates("rat_id").set_index("rat_id")["n_trials"]
    return pd.DataFrame({
        "rat_id":   wide.index.astype(int),
        "beta_s":   wide[Q_BS].to_numpy(dtype=float),
        "beta_f":   wide[Q_BF].to_numpy(dtype=float),
        "n_trials": n_trials.reindex(wide.index).to_numpy(dtype=int),
    }).reset_index(drop=True)


def extract_glmm_blups(final_result: dict) -> pd.DataFrame:
    prc = final_result["per_rat_coords"].copy()
    return pd.DataFrame({
        "rat_id":      prc["rat_id"].astype(int),
        "b_S":         prc["b_S"].to_numpy(dtype=float),
        "b_F":         prc["b_F"].to_numpy(dtype=float),
        "beta_s_full": prc["beta_S_plus_b"].to_numpy(dtype=float),
        "beta_f_full": prc["beta_F_plus_b"].to_numpy(dtype=float),
    })


def extract_glmm_parametric(final_result: dict) -> dict:
    het = final_result["heterogeneity"].set_index("quantity")["value"].to_dict()
    return {
        "tau_S":      float(het.get("tau_S", np.nan)),
        "tau_F":      float(het.get("tau_F", np.nan)),
        "tau_SF_cov": float(het.get("tau_SF_cov", np.nan)),
        "cor_SF":     float(het.get("cor_SF", np.nan)),
        "tau_S_beta": float(het.get("tau_S_beta", np.nan)),
        "tau_D_beta": float(het.get("tau_D_beta", np.nan)),
    }


# ============================================================
# Statistics
# ============================================================
def weighted_pearson(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """Weighted Pearson correlation."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    W  = w.sum()
    mx = np.sum(w * x) / W
    my = np.sum(w * y) / W
    cov = np.sum(w * (x - mx) * (y - my)) / W
    vx  = np.sum(w * (x - mx) ** 2) / W
    vy  = np.sum(w * (y - my) ** 2) / W
    return float(cov / np.sqrt(vx * vy))


def sd_ratio_D_over_S(x: np.ndarray, y: np.ndarray) -> float:
    """SD(D)/SD(S) where S = x+y, D = x-y  (ddof=1)."""
    S = np.asarray(x) + np.asarray(y)
    D = np.asarray(x) - np.asarray(y)
    return float(np.std(D, ddof=1) / np.std(S, ddof=1))


def loo_correlation(df: pd.DataFrame, x_col: str, y_col: str,
                    method: str = "pearson") -> pd.DataFrame:
    """Leave-one-out correlation with delta from full-sample value."""
    fn = stats.pearsonr if method == "pearson" else stats.spearmanr
    r_full, p_full = fn(df[x_col].to_numpy(dtype=float),
                        df[y_col].to_numpy(dtype=float))
    rows = [{"rat_dropped": "none", "r": float(r_full),
             "p": float(p_full), "delta_from_full": 0.0}]
    for rat in df["rat_id"]:
        sub = df.loc[df["rat_id"] != rat]
        r_i, p_i = fn(sub[x_col].to_numpy(dtype=float),
                      sub[y_col].to_numpy(dtype=float))
        rows.append({
            "rat_dropped":     str(int(rat)),
            "r":               float(r_i),
            "p":               float(p_i),
            "delta_from_full": float(r_i) - float(r_full),
        })
    return pd.DataFrame(rows)


def loo_sd_ratio(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    ratio_full = sd_ratio_D_over_S(df[x_col], df[y_col])
    rows = [{"rat_dropped": "none", "sd_ratio": ratio_full,
             "delta_from_full": 0.0}]
    for rat in df["rat_id"]:
        sub = df.loc[df["rat_id"] != rat]
        r_i = sd_ratio_D_over_S(sub[x_col], sub[y_col])
        rows.append({
            "rat_dropped":     str(int(rat)),
            "sd_ratio":        float(r_i),
            "delta_from_full": float(r_i) - float(ratio_full),
        })
    return pd.DataFrame(rows)


# ============================================================
# Assembly
# ============================================================
def build_summary_table(sec4: pd.DataFrame, blups: pd.DataFrame,
                        params: dict) -> pd.DataFrame:
    """Headline cross-source table for the report and CSV."""
    rows = []

    # --- Section 4 raw ---
    r, _   = stats.pearsonr(sec4["beta_s"], sec4["beta_f"])
    rho, _ = stats.spearmanr(sec4["beta_s"], sec4["beta_f"])
    rows += [
        {"source": "section4_raw", "quantity": "pearson_r",    "value": float(r)},
        {"source": "section4_raw", "quantity": "spearman_rho", "value": float(rho)},
        {"source": "section4_raw", "quantity": "sd_ratio_D_S",
         "value": sd_ratio_D_over_S(sec4["beta_s"], sec4["beta_f"])},
    ]

    # --- Section 4 trial-count-weighted ---
    rows.append({
        "source":   "section4_weighted",
        "quantity": "pearson_r",
        "value":    weighted_pearson(sec4["beta_s"], sec4["beta_f"], sec4["n_trials"]),
    })

    # --- GLMM BLUPs (empirical on shrunk per-rat deviations) ---
    r, _   = stats.pearsonr(blups["b_S"], blups["b_F"])
    rho, _ = stats.spearmanr(blups["b_S"], blups["b_F"])
    rows += [
        {"source": "glmm_blup", "quantity": "pearson_r",    "value": float(r)},
        {"source": "glmm_blup", "quantity": "spearman_rho", "value": float(rho)},
    ]

    # --- GLMM parametric (from covariance matrix) ---
    tau_ratio = (params["tau_D_beta"] / params["tau_S_beta"]
                 if params["tau_S_beta"] and params["tau_S_beta"] > 0 else np.nan)
    rows += [
        {"source": "glmm_parametric", "quantity": "cor_SF",       "value": params["cor_SF"]},
        {"source": "glmm_parametric", "quantity": "tau_S",        "value": params["tau_S"]},
        {"source": "glmm_parametric", "quantity": "tau_F",        "value": params["tau_F"]},
        {"source": "glmm_parametric", "quantity": "tau_S_beta",   "value": params["tau_S_beta"]},
        {"source": "glmm_parametric", "quantity": "tau_D_beta",   "value": params["tau_D_beta"]},
        {"source": "glmm_parametric", "quantity": "tau_D_over_S", "value": tau_ratio},
    ]

    return pd.DataFrame(rows)


def format_report(summary: pd.DataFrame,
                  loo_sec4_p: pd.DataFrame, loo_sec4_s: pd.DataFrame,
                  loo_glmm_p: pd.DataFrame, loo_glmm_s: pd.DataFrame,
                  loo_sd: pd.DataFrame,
                  meta: dict) -> str:
    """Assemble the human-readable text report."""
    lines = [
        "Outlier sensitivity analysis — GLMM I random-effect covariance",
        "=" * 64,
        f"GLMM fit:     {meta['pkl_path']}",
        f"Section 4:    subset={meta['subset']}, C={meta['C_label']}",
        f"N rats:       {meta['n_rats']}",
        "",
        "1. Headline correlations and variance anisotropy across estimators",
        "-" * 66,
    ]
    disp = summary.copy()
    disp["value"] = disp["value"].map(lambda v: f"{v:+.3f}" if np.isfinite(v) else "n/a")
    lines.append(disp.to_string(index=False))

    def loo_block(title: str, df: pd.DataFrame, col: str) -> str:
        block = [f"\n{title}", "-" * len(title)]
        for _, row in df.iterrows():
            flag = "   <-- driver (|Δ| > 0.10)" if abs(row["delta_from_full"]) > 0.10 else ""
            block.append(
                f"  drop rat {row['rat_dropped']:>4}: "
                f"{col} = {row[col]:+.3f}   (Δ = {row['delta_from_full']:+.3f}){flag}"
            )
        return "\n".join(block)

    lines.append(loo_block(
        "2a. LOO — Section 4 raw Pearson(β_s, β_f)   (headline in P1)",
        loo_sec4_p, "r",
    ))
    lines.append(loo_block(
        "2b. LOO — Section 4 raw Spearman(β_s, β_f)  (rank version, less outlier-sensitive)",
        loo_sec4_s, "r",
    ))
    lines.append(loo_block(
        "3a. LOO — GLMM BLUP Pearson(b_S, b_F)       (empirical on shrunk deviations)",
        loo_glmm_p, "r",
    ))
    lines.append(loo_block(
        "3b. LOO — GLMM BLUP Spearman(b_S, b_F)",
        loo_glmm_s, "r",
    ))
    lines.append(loo_block(
        "4.  LOO — Section 4 SD(D) / SD(S) ratio     (variance anisotropy in raw data)",
        loo_sd, "sd_ratio",
    ))

    lines.append("\nInterpretation")
    lines.append("-" * 14)
    lines.append(
        "Any rat whose omission changes the reported quantity by more than 0.10\n"
        "(the '<-- driver' marker) is meaningfully driving that estimate. Count\n"
        "the drivers across the five blocks to judge which of the population-\n"
        "level claims about the between-rat covariance structure are supported\n"
        "by the broad cohort vs by one or two individuals.\n"
    )
    return "\n".join(lines)


# ============================================================
# CLI
# ============================================================
def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles (cp1252) cannot print beta, rho, tau
    except Exception:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkl", type=Path, required=True,
                        help="Path to glmm_I_full.pkl")
    parser.add_argument("--subset", default="pooled",
                        help="Section 4 subset name (default: pooled)")
    parser.add_argument("--C", default="unpenalized",
                        help="Section 4 regularisation label (default: unpenalized)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: GLMM_ANALYSIS/outputs/02_glmm_I_validation/outlier_sensitivity/<subset>/)")
    args = parser.parse_args()

    out_dir = args.out_dir or (
        Path(__file__).resolve().parents[2] / "outputs" / "02_glmm_I_validation"
        / "outlier_sensitivity" / args.subset
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading GLMM I result from: {args.pkl}")
    glmm = load_glmm_result(args.pkl)
    print(f"Loading Section 4 per-rat GLM (subset={args.subset}, C={args.C})")
    sec4  = load_section4_per_rat(subset=args.subset, C_label=args.C)
    blups = extract_glmm_blups(glmm)
    params = extract_glmm_parametric(glmm)

    # Align rat sets
    common = sorted(set(sec4["rat_id"]) & set(blups["rat_id"]))
    only_sec4 = sorted(set(sec4["rat_id"]) - set(blups["rat_id"]))
    only_glmm = sorted(set(blups["rat_id"]) - set(sec4["rat_id"]))
    if only_sec4 or only_glmm:
        print(f"[warn] rat sets differ: only in Section 4 = {only_sec4}; "
              f"only in GLMM = {only_glmm}. Using intersection ({len(common)} rats).")
    sec4  = sec4[sec4["rat_id"].isin(common)].sort_values("rat_id").reset_index(drop=True)
    blups = blups[blups["rat_id"].isin(common)].sort_values("rat_id").reset_index(drop=True)

    # Compute
    summary    = build_summary_table(sec4, blups, params)
    loo_sec4_p = loo_correlation(sec4,  "beta_s", "beta_f", method="pearson")
    loo_sec4_s = loo_correlation(sec4,  "beta_s", "beta_f", method="spearman")
    loo_glmm_p = loo_correlation(blups, "b_S",    "b_F",    method="pearson")
    loo_glmm_s = loo_correlation(blups, "b_S",    "b_F",    method="spearman")
    loo_sd     = loo_sd_ratio(sec4, "beta_s", "beta_f")

    # Long-format CSV
    all_rows = summary.assign(rat_dropped="none", delta_from_full=0.0).to_dict("records")
    for src, block, val_col in [
        ("section4_raw_loo_pearson",  loo_sec4_p, "r"),
        ("section4_raw_loo_spearman", loo_sec4_s, "r"),
        ("glmm_blup_loo_pearson",     loo_glmm_p, "r"),
        ("glmm_blup_loo_spearman",    loo_glmm_s, "r"),
        ("section4_sd_ratio_loo",     loo_sd,     "sd_ratio"),
    ]:
        for _, row in block.iterrows():
            all_rows.append({
                "source":          src,
                "quantity":        val_col,
                "value":           float(row[val_col]),
                "rat_dropped":     row["rat_dropped"],
                "delta_from_full": float(row["delta_from_full"]),
            })
    long_df = pd.DataFrame(all_rows)
    csv_path = out_dir / f"outlier_sensitivity_{args.subset}.csv"
    long_df.to_csv(csv_path, index=False)

    # Formatted text
    report = format_report(
        summary=summary,
        loo_sec4_p=loo_sec4_p, loo_sec4_s=loo_sec4_s,
        loo_glmm_p=loo_glmm_p, loo_glmm_s=loo_glmm_s,
        loo_sd=loo_sd,
        meta={
            "pkl_path": str(args.pkl),
            "subset":   args.subset,
            "C_label":  args.C,
            "n_rats":   len(common),
        },
    )
    txt_path = out_dir / f"outlier_sensitivity_{args.subset}.txt"
    txt_path.write_text(report, encoding="utf-8")

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {txt_path}\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())