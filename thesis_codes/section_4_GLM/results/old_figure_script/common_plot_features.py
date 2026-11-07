"""Shared data loading, style constants, and per-rat significance / profile
computation for the Section 4 GLM figures (P1, P2, P3, P4).

All four scripts import from here so that adding subsets ('pooled',
'tactile', 'visual', 'visuotactile') means changing one CLI flag rather
than four independent copies of the loader.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from Portable_path_resolution_for_Section_4_GLM_figures import (
    glm_bootstrap_path,
    glm_summary_path,
    psychometric_contrasts_path,
)

# --------------------------------------------------------------------------
# Subsets and quantity name constants
# --------------------------------------------------------------------------
KNOWN_SUBSETS: tuple[str, ...] = ("pooled", "tactile", "visual", "visuotactile")

Q_INTERCEPT = "intercept"
Q_ANGLE     = "beta_angle"
Q_ANGLE_1   = "beta_angle_n-1"
Q_BR        = "beta_reward_n-1"
Q_BS        = "beta_success_n-1"
Q_BF        = "beta_failure_n-1"

# Ordered for the 6-panel forest (top row: task; bottom row: history)
COEF_ORDER = (Q_INTERCEPT, Q_ANGLE, Q_ANGLE_1, Q_BR, Q_BS, Q_BF)

COEF_LABELS = {
    Q_INTERCEPT: r"$\beta_0$  (intercept)",
    Q_ANGLE:     r"$\beta_\theta$  (current angle)",
    Q_ANGLE_1:   r"$\beta_{\theta,\,n-1}$  (previous angle)",
    Q_BR:        r"$\beta_{r,\,n-1}$  (previous reward)",
    Q_BS:        r"$\beta_{s,\,n-1}$  (previous success)",
    Q_BF:        r"$\beta_{f,\,n-1}$  (previous failure)",
}

COEF_COLOURS = {
    Q_INTERCEPT: "#7f8c8d",
    Q_ANGLE:     "#34495e",
    Q_ANGLE_1:   "#95a5a6",
    Q_BR:        "#2ecc71",
    Q_BS:        "#3498db",
    Q_BF:        "#c0392b",
}

# S/D/P panel colours (match Section-3 figure_B_forest exactly)
CONTRAST_COLOURS = {
    "S_logit": "#E68722",
    "D_logit": "#2922E6",
    "P_logit": "#2ecc71",
}

# Psychometric analog column names
PSY_S_COL, PSY_D_COL, PSY_P_COL = "S", "D", "P"
PSY_S_LO, PSY_S_HI, PSY_S_EX = "S_ci_lo", "S_ci_hi", "S_excludes_0"
PSY_D_LO, PSY_D_HI, PSY_D_EX = "D_ci_lo", "D_ci_hi", "D_excludes_0"
PSY_P_LO, PSY_P_HI, PSY_P_EX = "P_ci_lo", "P_ci_hi", "P_excludes_0"

# axis_pattern palette — matches Section_3_group_figures.PATTERN_COLOURS
PSY_COL_PATTERN = "axis_pattern"
PATTERN_COLOURS = {
    "null":            "#95a5a6",
    "rew-only":        "#2ecc71",
    "unrew-only":      "#d946ef",
    "S (choice)":      "#E68722",
    "D (interaction)": "#2922E6",
}
PATTERN_ORDER = ["null", "rew-only", "unrew-only", "S (choice)", "D (interaction)"]


# --------------------------------------------------------------------------
# Data container
# --------------------------------------------------------------------------
@dataclass
class SubsetData:
    subset: str
    rats: list[int]
    glm_point: pd.DataFrame
    glm_ci_lo: pd.DataFrame
    glm_ci_hi: pd.DataFrame
    glm_derived_point: pd.DataFrame
    glm_derived_ci_lo: pd.DataFrame
    glm_derived_ci_hi: pd.DataFrame
    psy: pd.DataFrame
    C_label: str
    ci_level: float


def _pivot_summary(long: pd.DataFrame):
    p  = long.pivot(index="rat_id", columns="quantity", values="value").sort_index()
    lo = long.pivot(index="rat_id", columns="quantity", values="ci_low").sort_index()
    hi = long.pivot(index="rat_id", columns="quantity", values="ci_high").sort_index()
    return p, lo, hi


def _compute_derived_ci(boot: pd.DataFrame, ci_level: float):
    wide = boot.pivot_table(
        index=["rat_id", "boot_idx"], columns="quantity", values="value"
    ).reset_index()
    for req in (Q_BS, Q_BF, Q_BR):
        if req not in wide.columns:
            raise ValueError(f"Bootstrap missing {req!r}. Found {sorted(wide.columns)}")
    wide["S_logit"] = wide[Q_BS] + wide[Q_BF]
    wide["D_logit"] = wide[Q_BS] - wide[Q_BF]
    wide["P_logit"] = wide[Q_BR]
    alpha = (100.0 - ci_level) / 200.0
    cols = ["S_logit", "D_logit", "P_logit"]
    grp = wide.groupby("rat_id")[cols]
    return (
        grp.median().sort_index(),
        grp.quantile(alpha).sort_index(),
        grp.quantile(1 - alpha).sort_index(),
    )


def load_subset(
    subset: str,
    C_label: str = "unpenalized",
    ci_level: float = 95.0,
    summary_path: Path | None = None,
    bootstrap_path: Path | None = None,
    psy_path: Path | None = None,
) -> SubsetData:
    """Load GLM + psychometric outputs for ONE subset, aligned on the common
    rat set.

    Arguments
    ---------
    subset       : 'pooled', 'tactile', 'visual', 'visuotactile', or any
                   other value present as ``subset`` in the GLM CSV and as a
                   directory component in the psychometric CSV path.
    C_label      : GLM regularisation label (default 'unpenalized').
    ci_level     : CI level for derived S/D/P contrasts (default 95).
    *_path       : Explicit overrides for the three files.
    """
    summary_path   = summary_path   or glm_summary_path()
    bootstrap_path = bootstrap_path or glm_bootstrap_path()
    psy_path       = psy_path       or psychometric_contrasts_path(subset)

    summary = pd.read_csv(summary_path)
    mask = (summary["subset"] == subset) & (summary["C"] == C_label)
    if not mask.any():
        available_subsets = sorted(summary["subset"].unique())
        available_c = sorted(summary["C"].unique())
        raise ValueError(
            f"No rows in {summary_path} with subset={subset!r}, C={C_label!r}. "
            f"Available subsets: {available_subsets}. "
            f"Available C values: {available_c}."
        )
    glm_point, glm_ci_lo, glm_ci_hi = _pivot_summary(summary.loc[mask])

    boot = pd.read_parquet(bootstrap_path)
    bmask = (
        (boot["subset"] == subset)
        & (boot["C"] == C_label)
        & (boot["quantity"].isin([Q_BS, Q_BF, Q_BR]))
    )
    if not bmask.any():
        raise ValueError(
            f"No bootstrap rows for subset={subset!r}, C={C_label!r}."
        )
    d_point, d_lo, d_hi = _compute_derived_ci(boot.loc[bmask], ci_level=ci_level)

    psy = pd.read_csv(psy_path, keep_default_na=False, na_values=[""])
    for req in (PSY_S_COL, PSY_D_COL, PSY_P_COL, "rat",
                "delta_mu_rew", "delta_mu_unrew"):
        if req not in psy.columns:
            raise ValueError(
                f"Psychometric CSV {psy_path} missing column {req!r}. "
                f"Found: {sorted(psy.columns)}"
            )
    psy = psy.set_index("rat").sort_index()

    rats = sorted(set(glm_point.index) & set(psy.index) & set(d_point.index))
    if not rats:
        raise ValueError(
            f"No rats in common between GLM and psychometric outputs "
            f"for subset={subset!r}."
        )

    return SubsetData(
        subset=subset,
        rats=rats,
        glm_point=glm_point.loc[rats],
        glm_ci_lo=glm_ci_lo.loc[rats],
        glm_ci_hi=glm_ci_hi.loc[rats],
        glm_derived_point=d_point.loc[rats],
        glm_derived_ci_lo=d_lo.loc[rats],
        glm_derived_ci_hi=d_hi.loc[rats],
        psy=psy.loc[rats],
        C_label=C_label,
        ci_level=ci_level,
    )


# --------------------------------------------------------------------------
# CI-exclusion helpers
# --------------------------------------------------------------------------
def excludes_zero(lo, hi):
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    return (lo > 0) | (hi < 0)


def glm_contrast_excludes(data: SubsetData, contrast: str) -> pd.Series:
    lo = data.glm_derived_ci_lo[contrast]
    hi = data.glm_derived_ci_hi[contrast]
    return pd.Series(excludes_zero(lo.to_numpy(), hi.to_numpy()),
                     index=lo.index, name=contrast)


def glm_coef_excludes(data: SubsetData, quantity: str) -> pd.Series:
    lo = data.glm_ci_lo[quantity]
    hi = data.glm_ci_hi[quantity]
    return pd.Series(excludes_zero(lo.to_numpy(), hi.to_numpy()),
                     index=lo.index, name=quantity)


def any_history_excludes(data: SubsetData) -> pd.Series:
    """True per rat if any of {β_s, β_f, β_r, S_logit, D_logit} excludes 0."""
    return (
        glm_coef_excludes(data, Q_BS)
        | glm_coef_excludes(data, Q_BF)
        | glm_coef_excludes(data, Q_BR)
        | glm_contrast_excludes(data, "S_logit")
        | glm_contrast_excludes(data, "D_logit")
    )


# --------------------------------------------------------------------------
# Profile classification
# --------------------------------------------------------------------------
def profile_from_flags(has_S: bool, has_D: bool, has_P: bool) -> str:
    parts = []
    if has_S: parts.append("choice")
    if has_D: parts.append("interaction")
    if has_P: parts.append("reward")
    return "null" if not parts else "+".join(parts)


def glm_profile_series(data: SubsetData) -> pd.Series:
    S = glm_contrast_excludes(data, "S_logit")
    D = glm_contrast_excludes(data, "D_logit")
    P = glm_contrast_excludes(data, "P_logit")
    return pd.Series(
        [profile_from_flags(bool(S.loc[r]), bool(D.loc[r]), bool(P.loc[r]))
         for r in data.rats],
        index=data.rats, name="profile_glm",
    )


def psy_profile_series(data: SubsetData) -> pd.Series:
    S = excludes_zero(data.psy[PSY_S_LO].to_numpy(dtype=float),
                      data.psy[PSY_S_HI].to_numpy(dtype=float))
    D = excludes_zero(data.psy[PSY_D_LO].to_numpy(dtype=float),
                      data.psy[PSY_D_HI].to_numpy(dtype=float))
    P = excludes_zero(data.psy[PSY_P_LO].to_numpy(dtype=float),
                      data.psy[PSY_P_HI].to_numpy(dtype=float))
    return pd.Series(
        [profile_from_flags(bool(s), bool(d), bool(p))
         for s, d, p in zip(S, D, P)],
        index=data.rats, name="profile_psy",
    )


def profile_agreement(psy: str, glm: str) -> str:
    def _parts(s: str) -> set[str]:
        return set() if s == "null" else set(s.split("+"))
    a, b = _parts(psy), _parts(glm)
    if a == b:
        return "exact"
    if a & b:
        return "partial"
    return "none"


# --------------------------------------------------------------------------
# CLI helpers
# --------------------------------------------------------------------------
def add_subset_arg(parser) -> None:
    """Add a common ``--subset`` argument.

    Not restricted (any value works) but the help text lists standard values.
    """
    parser.add_argument(
        "--subset", default="pooled",
        help=(
            "Analysis subset to plot. Standard values: "
            + ", ".join(KNOWN_SUBSETS)
            + " (default: pooled). Must match a value present in the "
              "GLM CSV's 'subset' column and a directory component of "
              "the psychometric CSV path."
        ),
    )


def resolve_paths_from_args(args) -> tuple:
    """Return (summary_path, bootstrap_path, psy_path) — or (None, None, None)
    if args.root is not given (in which case the loader resolves them)."""
    if not getattr(args, "root", None):
        return None, None, None
    return (
        glm_summary_path(args.root),
        glm_bootstrap_path(args.root),
        psychometric_contrasts_path(args.subset, root=args.root),
    )


# Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\REWARD_and_CHOICE_HISTORY_PSYCHOMETRICS\tables_choice_effects\orthogonal_contrasts.csv" )
# 