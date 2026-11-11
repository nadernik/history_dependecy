


"""
Q1 cross-method comparison: psychometric (Section 3) vs pooled GLM.

Runs Layers 1, 2 and 3 as agreed:
  Layer 1 - Directional (sign) agreement, option 1 primary + option 3 sensitivity.
  Layer 2 - Spearman rank agreement with BCa & percentile CIs, permutation & asymptotic p, leave-one-out range.
  Layer 3 - Geometric agreement in the (Delta mu_R, Delta mu_U) plane:
              4-quadrant classification, 8-octant classification (axes {R, U, S, D}),
              cyclic distance, angular disagreement, and probabilistic soft-match
              from bootstrap draws.

Rerun on the filtered or unfiltered psych data by pointing PSYCH_DIR and
PSYCH_BOOTSTRAP_DIR at the corresponding folders, and OUTPUT_DIR wherever you
want the results to land.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

# ==============================================================================
# CONFIG - EDIT THESE PATHS
# ==============================================================================
PSYCH_DIR            = Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_3_choice_reward\choice_reward_joint_results\filtered\pooled\tables")
    # must contain: contrast_summary_long.csv, contrast_covariance_long.csv
PSYCH_BOOTSTRAP_DIR  = Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_3_choice_reward\choice_reward_joint_results\filtered\pooled\bootstraps")
    # must contain: rat_{ID}_joint_bootstrap.npz files (one per rat)
GLM_SUMMARY_CSV      = Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_4_GLM\fit_results\pooled_filtered\glm_summary_long.csv")
GLM_BOOTSTRAP_PARQUET = Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_4_GLM\fit_results\pooled_filtered\glm_bootstrap_draws.parquet")
OUTPUT_DIR           = Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_4_GLM\Q1_results_2.0")

# ---- Analysis parameters (leave defaults unless you have a reason) ----
GLM_C_VALUE   = "unpenalized"    # which regularization level to pull from GLM outputs
GLM_SUBSET    = "pooled"         # which subset column in GLM outputs
N_BOOT_SPEARMAN = 5000           # bootstrap draws for Layer 2 CIs
N_PERM_SPEARMAN = 10000          # permutations for Layer 2 p-values
CI_ALPHA      = 0.05             # 1 - confidence level
RNG_SEED      = 42
MIN_VALID_BOOT_FRACTION = 0.80   # minimum usable Spearman bootstrap fraction
MIN_VALID_BOOT_COUNT = 100

# Core quantities we compare
QUANTITIES = ["delta_mu_rew", "delta_mu_unrew", "S", "D", "P"]
# Formal cross-method family. S remains descriptive, as in Section 3.
PRIMARY_SPEARMAN_QUANTITIES = ["delta_mu_rew", "delta_mu_unrew", "D", "P"]
# Mapping to GLM column names in glm_summary_long.csv and glm_bootstrap_draws.parquet
GLM_QUANTITY_NAMES = {
    "delta_mu_rew":    ("beta_success_n-1",  "delta_mu_rewarded",   -1.0),  # (beta_name, degrees_name, sign_flip_for_beta)
    "delta_mu_unrew":  ("beta_failure_n-1",  "delta_mu_unrewarded", -1.0),
    "S":               (None,                "S_glm",               -1.0),  # S needs beta_success + beta_failure
    "D":               (None,                "D_glm",               -1.0),  # D needs beta_success - beta_failure
    "P":               ("beta_reward_n-1",   "P_glm",               +1.0),
}

# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_psych_summary(psych_dir: Path) -> pd.DataFrame:
    """Return long-format psych contrasts with CIs, one row per rat x quantity."""
    df = pd.read_csv(psych_dir / "contrast_summary_long.csv")
    df = df[df.contrast.isin(QUANTITIES)].copy()
    df = df.rename(columns={"rat": "rat_id", "estimate_full_data": "value"})
    return df[["rat_id", "contrast", "value", "ci_low", "ci_high"]]


def load_psych_covariance(psych_dir: Path) -> dict:
    """Return dict {rat_id: 2x2 covariance of (delta_mu_rew, delta_mu_unrew)}."""
    df = pd.read_csv(psych_dir / "contrast_covariance_long.csv")
    df = df[df.contrast_row.isin(["delta_mu_rew", "delta_mu_unrew"]) &
            df.contrast_column.isin(["delta_mu_rew", "delta_mu_unrew"])]
    out = {}
    for rat, g in df.groupby("rat"):
        Sigma = (g.pivot(index="contrast_row", columns="contrast_column", values="covariance")
                  .reindex(index=["delta_mu_rew","delta_mu_unrew"],
                           columns=["delta_mu_rew","delta_mu_unrew"])
                  .to_numpy())
        out[int(rat)] = Sigma
    return out


def load_psych_bootstrap(bootstrap_dir: Path, rat_id: int) -> dict:
    """Return {quantity: (n_draws,) array} plus 'theta_xy' as (n_draws, 2) for (R, U)."""
    fp = bootstrap_dir / f"rat_{rat_id}_joint_bootstrap.npz"
    if not fp.exists():
        return None
    z = np.load(fp, allow_pickle=True)
    order = list(z["contrast_order"])
    cb = z["contrast_bootstrap"]  # (n_draws, n_contrasts)
    out = {q: cb[:, order.index(q)] for q in QUANTITIES if q in order}
    out["theta_xy"] = z["theta_bootstrap_x_y"]  # (n_draws, 2)
    return out


def load_glm_summary(csv_path: Path, C_value: str, subset: str) -> pd.DataFrame:
    """Return long-format GLM contrasts (degrees scale) with CIs, one row per rat x quantity.

    Also computes sign-aligned beta-scale versions in wide form."""
    df = pd.read_csv(csv_path)
    df = df[(df.C == C_value) & (df.subset == subset)].copy()

    # For each of the 5 quantities, pull both a degrees-scale reading and a
    # sign-aligned beta-scale reading.
    rows = []
    for rat, g in df.groupby("rat_id"):
        gd = g.set_index("quantity")
        # Sign-aligned beta values
        b_succ = gd.loc["beta_success_n-1", "value"] if "beta_success_n-1" in gd.index else np.nan
        b_fail = gd.loc["beta_failure_n-1", "value"] if "beta_failure_n-1" in gd.index else np.nan
        b_rew  = gd.loc["beta_reward_n-1",  "value"] if "beta_reward_n-1"  in gd.index else np.nan
        aligned_beta = {
            "delta_mu_rew":   -b_succ,
            "delta_mu_unrew": -b_fail,
            "S":              -(b_succ + b_fail),
            "D":              -(b_succ - b_fail),
            "P":              +b_rew,
        }
        for q in QUANTITIES:
            deg_name = GLM_QUANTITY_NAMES[q][1]
            row = {"rat_id": int(rat), "contrast": q,
                   "value_beta_aligned": aligned_beta[q],
                   "value_degrees":      gd.loc[deg_name, "value"]    if deg_name in gd.index else np.nan,
                   "degrees_ci_low":     gd.loc[deg_name, "ci_low"]   if deg_name in gd.index else np.nan,
                   "degrees_ci_high":    gd.loc[deg_name, "ci_high"]  if deg_name in gd.index else np.nan}
            rows.append(row)
    return pd.DataFrame(rows)


def load_glm_bootstrap(parquet_path: Path, C_value: str, subset: str) -> dict:
    """Return dict {rat_id: {'theta_xy': (n_draws, 2), 'quantity_draws': DataFrame}}.

    theta_xy uses GLM's own degrees-scale delta_mu_rewarded / delta_mu_unrewarded per draw."""
    df = pd.read_parquet(parquet_path)
    df = df[(df.C == C_value) & (df.subset == subset)]
    out = {}
    for rat, g in df.groupby("rat_id"):
        wide = g.pivot_table(index="boot_idx", columns="quantity", values="value")
        # Sign-aligned beta-scale draws
        wide["_aligned_delta_mu_rew"]   = -wide["beta_success_n-1"]
        wide["_aligned_delta_mu_unrew"] = -wide["beta_failure_n-1"]
        wide["_aligned_S"]              = -(wide["beta_success_n-1"] + wide["beta_failure_n-1"])
        wide["_aligned_D"]              = -(wide["beta_success_n-1"] - wide["beta_failure_n-1"])
        wide["_aligned_P"]              =  wide["beta_reward_n-1"]
        # theta_xy uses degrees-scale (already sign-aligned by construction)
        theta_xy = wide[["delta_mu_rewarded", "delta_mu_unrewarded"]].to_numpy()
        out[int(rat)] = {"wide": wide, "theta_xy": theta_xy}
    return out


# ==============================================================================
# LAYER 1 - DIRECTIONAL AGREEMENT
# ==============================================================================

def layer1_sign_agreement(psych: pd.DataFrame, glm: pd.DataFrame,
                          rats: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Option 1 (all rats) primary, option 3 (CI-gated) sensitivity."""
    per_rat_rows = []
    summary_rows = []
    for q in QUANTITIES:
        psy_q = psych[psych.contrast == q].set_index("rat_id")
        glm_q = glm[glm.contrast == q].set_index("rat_id")

        per_q = []
        for r in rats:
            pv, pl, ph = psy_q.loc[r, ["value","ci_low","ci_high"]]
            gv         = glm_q.loc[r, "value_beta_aligned"]
            gd, gdl, gdh = glm_q.loc[r, ["value_degrees","degrees_ci_low","degrees_ci_high"]]
            # Sign-only match (option 1)
            match_opt1 = (np.sign(pv) == np.sign(gv))
            # CI-gated: at least one CI excludes zero on either side (use degrees-scale
            # CIs on GLM side because beta-scale CIs are not in the summary CSV)
            psy_conf = (pl > 0) or (ph < 0)
            glm_conf = (gdl > 0) or (gdh < 0)
            qualifies = psy_conf or glm_conf
            per_q.append({
                "rat_id": r, "quantity": q,
                "psych_value": pv, "psych_ci_low": pl, "psych_ci_high": ph,
                "glm_value_beta_aligned": gv,
                "glm_value_degrees": gd, "glm_ci_low_degrees": gdl, "glm_ci_high_degrees": gdh,
                "sign_match": bool(match_opt1),
                "psych_ci_excludes_zero": bool(psy_conf),
                "glm_ci_excludes_zero": bool(glm_conf),
                "qualifies_option3": bool(qualifies),
            })
        per_rat_rows.extend(per_q)

        opt1_match = sum(1 for r in per_q if r["sign_match"])
        qual = [r for r in per_q if r["qualifies_option3"]]
        opt3_match = sum(1 for r in qual if r["sign_match"])
        summary_rows.append({
            "quantity": q,
            "n_rats": len(rats),
            "opt1_match": opt1_match,
            "opt1_denominator": len(rats),
            "opt1_match_rate": opt1_match / len(rats),
            "opt3_match": opt3_match,
            "opt3_denominator": len(qual),
            "opt3_match_rate": opt3_match / len(qual) if qual else np.nan,
            "opt3_set_aside": len(rats) - len(qual),
            "n_psych_confident": sum(1 for r in per_q if r["psych_ci_excludes_zero"]),
            "n_glm_confident":   sum(1 for r in per_q if r["glm_ci_excludes_zero"]),
            "n_both_confident":  sum(1 for r in per_q if r["psych_ci_excludes_zero"] and r["glm_ci_excludes_zero"]),
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(per_rat_rows)


# ==============================================================================
# LAYER 2 - SPEARMAN RANK AGREEMENT
# ==============================================================================

def spearman_bca(x: np.ndarray, y: np.ndarray, n_boot: int, alpha: float, seed: int):
    """Spearman rho + BCa CI + percentile CI. Returns dict."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x); y = np.asarray(y); n = len(x)
    rho_hat, _ = stats.spearmanr(x, y)
    # Bootstrap distribution (paired resample -> preserves the psych-GLM pairing)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        rho_b = stats.spearmanr(x[idx], y[idx]).correlation
        if np.isfinite(rho_b):
            boot.append(float(rho_b))
    boot = np.asarray(boot, dtype=float)
    min_valid = max(
        MIN_VALID_BOOT_COUNT,
        int(np.ceil(MIN_VALID_BOOT_FRACTION * n_boot)),
    )
    if len(boot) < min_valid:
        raise RuntimeError(
            f"Too few valid Spearman bootstrap draws: {len(boot)}/{n_boot}; "
            f"required at least {min_valid}."
        )
    # z0 from proportion of boot values less than the observed rho
    p0 = np.clip((boot < rho_hat).mean(), 1e-10, 1 - 1e-10)
    z0 = norm.ppf(p0)
    # Acceleration via jackknife
    jack = np.array([
        stats.spearmanr(np.delete(x, i), np.delete(y, i)).correlation
        for i in range(n)
    ], dtype=float)
    if not np.all(np.isfinite(jack)):
        raise RuntimeError("A leave-one-out Spearman correlation was undefined.")
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6.0 * ((jm - jack) ** 2).sum() ** 1.5
    a = num / den if den != 0 else 0.0
    z_alpha = norm.ppf(alpha / 2)
    z_1_alpha = norm.ppf(1 - alpha / 2)
    a1 = norm.cdf(z0 + (z0 + z_alpha)   / (1 - a * (z0 + z_alpha)))
    a2 = norm.cdf(z0 + (z0 + z_1_alpha) / (1 - a * (z0 + z_1_alpha)))
    a1, a2 = np.clip([a1, a2], 0.0, 1.0)
    bca_lo = np.percentile(boot, 100 * a1)
    bca_hi = np.percentile(boot, 100 * a2)
    perc_lo = np.percentile(boot, 100 * alpha / 2)
    perc_hi = np.percentile(boot, 100 * (1 - alpha / 2))
    return {"rho": rho_hat,
            "bca_lo": bca_lo, "bca_hi": bca_hi,
            "perc_lo": perc_lo, "perc_hi": perc_hi,
            "z0": z0, "acceleration": a,
            "n_boot_requested": int(n_boot),
            "n_boot_valid": int(len(boot))}


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm-adjusted p-values, preserving the original order."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or np.any(~np.isfinite(p)):
        raise ValueError("Holm adjustment requires finite one-dimensional p-values.")
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    m = len(ranked)
    adjusted_ranked = (m - np.arange(m)) * ranked
    adjusted_ranked = np.maximum.accumulate(adjusted_ranked)
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_ranked
    return adjusted


def spearman_permutation_p(x: np.ndarray, y: np.ndarray, n_perm: int, seed: int):
    rng = np.random.default_rng(seed)
    rho_obs, p_asymp = stats.spearmanr(x, y)
    count = 0
    for _ in range(n_perm):
        y_perm = rng.permutation(y)
        r, _ = stats.spearmanr(x, y_perm)
        if abs(r) >= abs(rho_obs):
            count += 1
    return {"rho": rho_obs, "p_permutation": (count + 1) / (n_perm + 1),
            "p_asymptotic": p_asymp}


def spearman_leave_one_out(x: np.ndarray, y: np.ndarray):
    rhos = [stats.spearmanr(np.delete(x, i), np.delete(y, i)).correlation
            for i in range(len(x))]
    return {"loo_rho_min": float(np.min(rhos)),
            "loo_rho_max": float(np.max(rhos)),
            "loo_rho_range": float(np.max(rhos) - np.min(rhos))}


def layer2_spearman(psych: pd.DataFrame, glm: pd.DataFrame,
                    rats: list, n_boot: int, n_perm: int,
                    alpha: float, seed: int) -> pd.DataFrame:
    rows = []
    for q in QUANTITIES:
        psy = psych[psych.contrast == q].set_index("rat_id").loc[rats, "value"].to_numpy()
        # Primary: beta-aligned. Sensitivity: degrees.
        glm_b = glm[glm.contrast == q].set_index("rat_id").loc[rats, "value_beta_aligned"].to_numpy()
        glm_d = glm[glm.contrast == q].set_index("rat_id").loc[rats, "value_degrees"].to_numpy()

        r = {"quantity": q}
        # Primary
        bca_b = spearman_bca(psy, glm_b, n_boot, alpha, seed)
        perm_b = spearman_permutation_p(psy, glm_b, n_perm, seed)
        loo_b = spearman_leave_one_out(psy, glm_b)
        r.update({"rho_beta_aligned": bca_b["rho"],
                  "bca_lo_beta_aligned": bca_b["bca_lo"], "bca_hi_beta_aligned": bca_b["bca_hi"],
                  "perc_lo_beta_aligned": bca_b["perc_lo"], "perc_hi_beta_aligned": bca_b["perc_hi"],
                  "p_permutation_beta_aligned": perm_b["p_permutation"],
                  "p_asymptotic_beta_aligned":  perm_b["p_asymptotic"],
                  "loo_rho_min_beta_aligned": loo_b["loo_rho_min"],
                  "loo_rho_max_beta_aligned": loo_b["loo_rho_max"],
                  "loo_rho_range_beta_aligned": loo_b["loo_rho_range"],
                  "n_boot_requested_beta_aligned": bca_b["n_boot_requested"],
                  "n_boot_valid_beta_aligned": bca_b["n_boot_valid"]})
        # Sensitivity (degrees)
        bca_d = spearman_bca(psy, glm_d, n_boot, alpha, seed)
        perm_d = spearman_permutation_p(psy, glm_d, n_perm, seed)
        loo_d = spearman_leave_one_out(psy, glm_d)
        r.update({"rho_degrees": bca_d["rho"],
                  "bca_lo_degrees": bca_d["bca_lo"], "bca_hi_degrees": bca_d["bca_hi"],
                  "perc_lo_degrees": bca_d["perc_lo"], "perc_hi_degrees": bca_d["perc_hi"],
                  "p_permutation_degrees": perm_d["p_permutation"],
                  "p_asymptotic_degrees":  perm_d["p_asymptotic"],
                  "loo_rho_min_degrees": loo_d["loo_rho_min"],
                  "loo_rho_max_degrees": loo_d["loo_rho_max"],
                  "loo_rho_range_degrees": loo_d["loo_rho_range"],
                  "n_boot_requested_degrees": bca_d["n_boot_requested"],
                  "n_boot_valid_degrees": bca_d["n_boot_valid"]})
        rows.append(r)
    result = pd.DataFrame(rows)

    # One formal family: four coefficient-scale permutation tests. S and all
    # degree-scale tests remain descriptive/sensitivity analyses.
    result["formal_primary_spearman"] = result["quantity"].isin(
        PRIMARY_SPEARMAN_QUANTITIES
    )
    result["p_holm_beta_aligned_primary4"] = np.nan
    result["reject_holm_05_beta_aligned_primary4"] = False
    primary = result["formal_primary_spearman"]
    adjusted = holm_adjust(
        result.loc[primary, "p_permutation_beta_aligned"].to_numpy()
    )
    result.loc[primary, "p_holm_beta_aligned_primary4"] = adjusted
    result.loc[primary, "reject_holm_05_beta_aligned_primary4"] = adjusted < 0.05
    return result


# ==============================================================================
# LAYER 3 - GEOMETRIC AGREEMENT
# ==============================================================================

def octant8(r, u):
    """Nearest of 8 axes centred at multiples of 45 degrees; 0 at origin."""
    if np.isclose(r, 0.0) and np.isclose(u, 0.0):
        return 0
    ang = (np.degrees(np.arctan2(u, r))) % 360
    return int(((ang + 22.5) % 360) // 45) + 1


def octant_vec(R, U):
    """Vectorized nearest-axis classification; exact origins are unresolved (0)."""
    R = np.asarray(R); U = np.asarray(U)
    ang = np.degrees(np.arctan2(U, R)) % 360
    k = (((ang + 22.5) % 360) // 45).astype(int) + 1
    k[np.isclose(R, 0.0) & np.isclose(U, 0.0)] = 0
    return k


def quadrant4(r, u):
    if r > 0 and u > 0: return "WSw+LSw"
    if r < 0 and u > 0: return "WSt+LSw"
    if r < 0 and u < 0: return "WSt+LSt"
    if r > 0 and u < 0: return "WSw+LSt"
    return "on-axis"


def cyclic_dist_octant(k1, k2, K=8):
    d = abs(k1 - k2) % K
    return min(d, K - d)


def angular_disagreement(r1, u1, r2, u2):
    a1 = np.degrees(np.arctan2(u1, r1)) % 360
    a2 = np.degrees(np.arctan2(u2, r2)) % 360
    diff = abs(a1 - a2)
    return min(diff, 360 - diff)


def origin_in_95_ellipse(mu: np.ndarray, Sigma: np.ndarray) -> bool:
    """True if the origin lies inside the joint 95% CI ellipse of a 2D estimate."""
    if Sigma is None or np.any(np.isnan(Sigma)):
        return True
    try:
        Sinv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        return True
    m = np.asarray(mu)
    md2 = float(m @ Sinv @ m)  # Mahalanobis distance squared to origin
    return bool(md2 < stats.chi2.ppf(0.95, df=2))


def layer3_geometric(psych: pd.DataFrame, glm: pd.DataFrame,
                     psych_cov: dict, glm_boot: dict,
                     psych_bootstrap_dir: Path,
                     rats: list) -> dict:
    """Return dict of DataFrames: per_rat, confusion_o8, cohort, probabilistic."""
    per_rat_rows = []
    prob_rows = []  # long: rat, octant, P_psych, P_glm

    for r in rats:
        pr = psych[(psych.contrast == "delta_mu_rew") & (psych.rat_id == r)]["value"].iloc[0]
        pu = psych[(psych.contrast == "delta_mu_unrew") & (psych.rat_id == r)]["value"].iloc[0]
        gr_b = glm[(glm.contrast == "delta_mu_rew") & (glm.rat_id == r)]["value_beta_aligned"].iloc[0]
        gu_b = glm[(glm.contrast == "delta_mu_unrew") & (glm.rat_id == r)]["value_beta_aligned"].iloc[0]
        gr_d = glm[(glm.contrast == "delta_mu_rew") & (glm.rat_id == r)]["value_degrees"].iloc[0]
        gu_d = glm[(glm.contrast == "delta_mu_unrew") & (glm.rat_id == r)]["value_degrees"].iloc[0]

        # 4-quadrant and 8-octant classifications (point-estimate)
        q4p, q4g_b, q4g_d = quadrant4(pr, pu), quadrant4(gr_b, gu_b), quadrant4(gr_d, gu_d)
        o8p, o8g_b, o8g_d = octant8(pr, pu),  octant8(gr_b, gu_b),  octant8(gr_d, gu_d)

        # Cyclic distance and angular disagreement (primary vs beta-aligned)
        cd_b = cyclic_dist_octant(o8p, o8g_b)
        ad_b = angular_disagreement(pr, pu, gr_b, gu_b)
        cd_d = cyclic_dist_octant(o8p, o8g_d)
        ad_d = angular_disagreement(pr, pu, gr_d, gu_d)

        # Probabilistic classification from bootstrap draws
        psych_boot = load_psych_bootstrap(psych_bootstrap_dir, r)
        glm_boot_r = glm_boot.get(r)

        P_psych_k = np.full(8, np.nan)
        P_glm_k   = np.full(8, np.nan)
        psych_map = np.nan; psych_map_prob = np.nan
        glm_map   = np.nan; glm_map_prob   = np.nan
        soft_match = np.nan
        n_psych_draws = 0
        n_glm_draws = 0

        if psych_boot is not None and glm_boot_r is not None:
            psy_xy = psych_boot["theta_xy"]        # (n_draws, 2)
            glm_xy = glm_boot_r["theta_xy"]        # (n_draws, 2) degrees-scale
            k_psy = octant_vec(psy_xy[:, 0], psy_xy[:, 1])
            k_glm = octant_vec(glm_xy[:, 0], glm_xy[:, 1])
            n_psych_draws = len(k_psy); n_glm_draws = len(k_glm)
            for k in range(1, 9):
                P_psych_k[k - 1] = (k_psy == k).mean()
                P_glm_k[k - 1]   = (k_glm == k).mean()
            psych_map = int(np.argmax(P_psych_k) + 1); psych_map_prob = float(P_psych_k.max())
            glm_map   = int(np.argmax(P_glm_k)   + 1); glm_map_prob   = float(P_glm_k.max())
            soft_match = float((P_psych_k * P_glm_k).sum())
            for k in range(1, 9):
                prob_rows.append({"rat_id": r, "octant": k,
                                  "P_psych": P_psych_k[k - 1],
                                  "P_glm":   P_glm_k[k - 1]})

        # Option 3 gate (Layer 3 flavour): origin inside joint 95% ellipse on BOTH sides
        psy_origin_in = origin_in_95_ellipse(np.array([pr, pu]), psych_cov.get(r))
        # For GLM: use covariance from the bootstrap theta_xy
        if glm_boot_r is not None:
            Sigma_glm = np.cov(glm_boot_r["theta_xy"].T)
            glm_origin_in = origin_in_95_ellipse(np.array([gr_d, gu_d]), Sigma_glm)
        else:
            glm_origin_in = None

        per_rat_rows.append({
            "rat_id": r,
            "psych_R": pr, "psych_U": pu,
            "glm_R_beta_aligned": gr_b, "glm_U_beta_aligned": gu_b,
            "glm_R_degrees":      gr_d, "glm_U_degrees":      gu_d,
            "psych_quadrant4": q4p, "glm_quadrant4_beta": q4g_b, "glm_quadrant4_degrees": q4g_d,
            "q4_match_beta":    q4p == q4g_b, "q4_match_degrees": q4p == q4g_d,
            "psych_octant8": o8p, "glm_octant8_beta": o8g_b, "glm_octant8_degrees": o8g_d,
            "o8_match_beta":    bool(o8p != 0 and o8g_b != 0 and o8p == o8g_b),
            "o8_match_degrees": bool(o8p != 0 and o8g_d != 0 and o8p == o8g_d),
            "cyclic_dist_beta":    cd_b, "cyclic_dist_degrees":    cd_d,
            "angular_disagree_beta_deg": ad_b, "angular_disagree_degrees_deg": ad_d,
            "psych_MAP_octant": psych_map, "psych_MAP_confidence": psych_map_prob,
            "glm_MAP_octant":   glm_map,   "glm_MAP_confidence":   glm_map_prob,
            "octant_distribution_overlap": soft_match,
            "psych_n_bootstrap_draws": n_psych_draws,
            "glm_n_bootstrap_draws":   n_glm_draws,
            "psych_origin_in_95ellipse": bool(psy_origin_in),
            "glm_origin_in_95ellipse":   glm_origin_in,
            "opt3_set_aside": (bool(psy_origin_in) and bool(glm_origin_in))
                              if glm_origin_in is not None else None,
        })

    per_rat = pd.DataFrame(per_rat_rows)
    prob_df = pd.DataFrame(prob_rows)

    expected_rats = [int(r) for r in rats]
    observed_rats = sorted(per_rat["rat_id"].astype(int).unique().tolist())
    if len(per_rat) != len(expected_rats) or observed_rats != sorted(expected_rats):
        raise RuntimeError(
            "Layer 3 did not produce exactly one geometry row per input rat. "
            f"Expected {len(expected_rats)} rats {sorted(expected_rats)}, "
            f"but obtained {len(per_rat)} rows for rats {observed_rats}."
        )

    # 8x8 confusion matrix (beta-aligned)
    conf = pd.DataFrame(0, index=list(range(1, 9)), columns=list(range(1, 9)))
    for _, row in per_rat.iterrows():
        if row["psych_octant8"] != 0 and row["glm_octant8_beta"] != 0:
            conf.loc[row["psych_octant8"], row["glm_octant8_beta"]] += 1

    # Cohort summary
    def counts(match_col):
        return int(per_rat[match_col].sum())

    cohort = pd.DataFrame([{
        "n_rats": len(per_rat),
        "o8_unresolved_exact_origin_beta": int(
            ((per_rat["psych_octant8"] == 0)
             | (per_rat["glm_octant8_beta"] == 0)).sum()
        ),
        # 4-quadrant
        "q4_exact_match_beta":    counts("q4_match_beta"),
        "q4_exact_match_degrees": counts("q4_match_degrees"),
        # 8-octant
        "o8_exact_match_beta":    counts("o8_match_beta"),
        "o8_exact_match_degrees": counts("o8_match_degrees"),
        "o8_within1_beta":    int((per_rat["cyclic_dist_beta"]    <= 1).sum()),
        "o8_within2_beta":    int((per_rat["cyclic_dist_beta"]    <= 2).sum()),
        "o8_within1_degrees": int((per_rat["cyclic_dist_degrees"] <= 1).sum()),
        "o8_within2_degrees": int((per_rat["cyclic_dist_degrees"] <= 2).sum()),
        # Angular disagreement
        "angular_disagree_median_beta_deg":    float(per_rat["angular_disagree_beta_deg"].median()),
        "angular_disagree_max_beta_deg":       float(per_rat["angular_disagree_beta_deg"].max()),
        "angular_disagree_median_degrees_deg": float(per_rat["angular_disagree_degrees_deg"].median()),
        "angular_disagree_max_degrees_deg":    float(per_rat["angular_disagree_degrees_deg"].max()),
        # Soft-match probability
        "octant_distribution_overlap_median": float(
            per_rat["octant_distribution_overlap"].median(skipna=True)
        ),
        "octant_distribution_overlap_mean": float(
            per_rat["octant_distribution_overlap"].mean(skipna=True)
        ),
        # Option-3 set-aside
        "opt3_set_aside_count": int(per_rat["opt3_set_aside"].fillna(False).sum()),
    }])
    return {"per_rat": per_rat, "confusion_octant": conf, "cohort": cohort, "probabilistic": prob_df}


# ==============================================================================
# ORCHESTRATION
# ==============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading psych summary from {PSYCH_DIR}")
    psych = load_psych_summary(PSYCH_DIR)
    psych_cov = load_psych_covariance(PSYCH_DIR)

    print(f"Loading GLM summary from {GLM_SUMMARY_CSV}")
    glm = load_glm_summary(GLM_SUMMARY_CSV, GLM_C_VALUE, GLM_SUBSET)

    # Rats present in BOTH sides
    rats_psych = set(psych.rat_id.unique())
    rats_glm   = set(glm.rat_id.unique())
    rats = sorted(rats_psych & rats_glm)
    only_psych = rats_psych - rats_glm
    only_glm   = rats_glm - rats_psych
    if only_psych: print(f"  Rats in psych but not GLM (excluded): {sorted(only_psych)}")
    if only_glm:   print(f"  Rats in GLM but not psych (excluded): {sorted(only_glm)}")
    print(f"  Analysing {len(rats)} rats: {rats}")

    print(f"Loading GLM bootstrap from {GLM_BOOTSTRAP_PARQUET}")
    glm_boot = load_glm_bootstrap(GLM_BOOTSTRAP_PARQUET, GLM_C_VALUE, GLM_SUBSET)

    # Check which rats have psych bootstrap files available
    missing_psych_bootstrap = [r for r in rats
                               if not (PSYCH_BOOTSTRAP_DIR / f"rat_{r}_joint_bootstrap.npz").exists()]
    if missing_psych_bootstrap:
        print(f"WARNING: psych bootstrap NPZ missing for rats {missing_psych_bootstrap}; "
              f"probabilistic classification (Layer 3) will be skipped for these.")

    # ---- Layer 1 ----
    print("Running Layer 1 (directional agreement)...")
    l1_summary, l1_per_rat = layer1_sign_agreement(psych, glm, rats)
    l1_summary.to_csv(OUTPUT_DIR / "layer1_sign_agreement.csv", index=False)
    l1_per_rat.to_csv(OUTPUT_DIR / "layer1_per_rat_signs.csv", index=False)

    # ---- Layer 2 ----
    print(f"Running Layer 2 (Spearman, BCa n_boot={N_BOOT_SPEARMAN}, permutation n_perm={N_PERM_SPEARMAN})...")
    l2 = layer2_spearman(psych, glm, rats, N_BOOT_SPEARMAN, N_PERM_SPEARMAN, CI_ALPHA, RNG_SEED)
    l2.to_csv(OUTPUT_DIR / "layer2_spearman.csv", index=False)

    # ---- Layer 3 ----
    print("Running Layer 3 (geometric agreement, probabilistic classification)...")
    l3 = layer3_geometric(psych, glm, psych_cov, glm_boot, PSYCH_BOOTSTRAP_DIR, rats)
    print(f"  Layer 3 completed for {len(l3['per_rat'])}/{len(rats)} rats")
    l3["per_rat"].to_csv(OUTPUT_DIR / "layer3_per_rat_geometry.csv", index=False)
    l3["confusion_octant"].to_csv(OUTPUT_DIR / "layer3_octant_confusion.csv")
    l3["cohort"].to_csv(OUTPUT_DIR / "layer3_cohort_summary.csv", index=False)
    l3["probabilistic"].to_csv(OUTPUT_DIR / "layer3_probabilistic_classification.csv", index=False)

    print(f"\nAll outputs written to {OUTPUT_DIR}")
    print("Files:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
