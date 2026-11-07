"""Helpers for reading and quality-checking Section 3 bootstrap archives.

This module is intentionally small.  It supplies the five functions used by
``psychometric_4analyses_all_rats.py`` and follows the array manifest written
by ``Section_3_choice_reward_joint_bootstrap.py``.

The fitting pipeline constrains sigma to [1, 50] degrees.  The downstream QC
figures and tables use the pre-specified practical criterion sigma < 3 degrees
to identify collapsed curve shapes.  The contrast bootstrap is deliberately
left untouched unless a caller explicitly applies the returned mask.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


SIGMA_BOUNDS = (1.0, 50.0)
BOUND_TOLERANCE = 1e-3
SIGMA_DEGENERATE_CUTOFF = 3.0
GAMMA_LAPSE_NEAR_CEILING = 0.29

OUTCOME_CELLS = {
    "rew": ("h_rew", "v_rew"),
    "unrew": ("h_unrew", "v_unrew"),
}
OUTCOME_CONTRAST = {
    "rew": "delta_mu_rew",
    "unrew": "delta_mu_unrew",
}

_REQUIRED_ARRAYS = {
    "parameter_bootstrap",
    "contrast_bootstrap",
    "cell_order",
    "parameter_order",
    "contrast_order",
}


def _names(values: Any) -> list[str]:
    """Convert an NPZ string array to ordinary Python strings."""
    return [str(value) for value in np.asarray(values).reshape(-1)]


def load_bootstrap(rat: int, path: str | Path) -> dict[str, Any]:
    """Load one rat's compressed bootstrap archive and validate its layout.

    Arrays are copied before the NPZ file is closed, so the returned mapping is
    independent of the file handle.  ``n_boot`` is added as a convenience.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Bootstrap archive not found: {path}")

    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(_REQUIRED_ARRAYS.difference(archive.files))
        if missing:
            raise KeyError(f"{path.name} is missing required arrays: {missing}")
        data = {name: archive[name].copy() for name in archive.files}

    stored_rat = int(np.asarray(data["rat"]).item()) if "rat" in data else None
    if stored_rat is not None and stored_rat != int(rat):
        raise ValueError(
            f"Requested rat {rat}, but {path.name} contains rat {stored_rat}."
        )

    parameters = np.asarray(data["parameter_bootstrap"])
    contrasts = np.asarray(data["contrast_bootstrap"])
    expected_parameter_shape = (
        len(_names(data["cell_order"])),
        len(_names(data["parameter_order"])),
    )
    if parameters.ndim != 3 or parameters.shape[1:] != expected_parameter_shape:
        raise ValueError(
            "parameter_bootstrap has shape "
            f"{parameters.shape}; expected (n_boot, {expected_parameter_shape[0]}, "
            f"{expected_parameter_shape[1]})."
        )
    if contrasts.ndim != 2 or contrasts.shape[1] != len(_names(data["contrast_order"])):
        raise ValueError("contrast_bootstrap does not match contrast_order.")
    if contrasts.shape[0] != parameters.shape[0]:
        raise ValueError("Parameter and contrast arrays have different bootstrap counts.")

    data["n_boot"] = int(parameters.shape[0])
    return data


def cell_param(data: Mapping[str, Any], cell: str, parameter: str) -> np.ndarray:
    """Return all bootstrap draws for one cell and one fitted parameter."""
    cells = _names(data["cell_order"])
    parameters = _names(data["parameter_order"])
    if cell not in cells:
        raise KeyError(f"Unknown cell {cell!r}; available cells: {cells}")
    if parameter not in parameters:
        raise KeyError(
            f"Unknown parameter {parameter!r}; available parameters: {parameters}"
        )
    return np.asarray(data["parameter_bootstrap"])[:, cells.index(cell), parameters.index(parameter)]


def cell_contrast(data: Mapping[str, Any], contrast: str) -> np.ndarray:
    """Return the complete, unfiltered bootstrap distribution of a contrast."""
    contrasts = _names(data["contrast_order"])
    if contrast not in contrasts:
        raise KeyError(f"Unknown contrast {contrast!r}; available contrasts: {contrasts}")
    return np.asarray(data["contrast_bootstrap"])[:, contrasts.index(contrast)]


def sigma_degenerate_mask(
    data: Mapping[str, Any],
    cell: str,
    *,
    cutoff: float = SIGMA_DEGENERATE_CUTOFF,
) -> np.ndarray:
    """Flag bootstrap draws whose sigma is invalid or below 3 degrees.

    This mask is suitable for excluding unstable curve shapes from a plotted
    confidence band.  It must not be applied to the separately stored contrast
    draws, because doing so would change the bootstrap estimand.
    """
    sigma = np.asarray(cell_param(data, cell, "sigma"), dtype=float)
    return ~np.isfinite(sigma) | (sigma < float(cutoff))


def full_data_boundary_flags(full_fits: pd.DataFrame, rat: int, cell: str) -> bool:
    """Report whether gamma or lapse is unidentifiable in a full-data fit.

    In this figure, the warning text specifically concerns gamma/lapse.  A fit
    is flagged if it is missing, duplicated, unsuccessful, non-finite, or if
    gamma/lapse touches either box constraint.
    """
    required = {"rat", "cell"}
    missing = sorted(required.difference(full_fits.columns))
    if missing:
        raise KeyError(f"Full-fit table is missing columns: {missing}")

    rows = full_fits.loc[
        (pd.to_numeric(full_fits["rat"], errors="coerce") == int(rat))
        & (full_fits["cell"].astype(str) == str(cell))
    ]
    if len(rows) != 1:
        return True

    row = rows.iloc[0]
    if "success" in rows.columns and not bool(row["success"]):
        return True

    for parameter in ("gamma", "lapse"):
        if parameter in rows.columns and not np.isfinite(pd.to_numeric(row[parameter], errors="coerce")):
            return True
        for side in ("lower", "upper"):
            column = f"{parameter}_at_{side}"
            if column in rows.columns and bool(row[column]):
                return True
    return False


def _find_companion_table(bootstrap_dir: Path, filename: str) -> Path:
    """Locate a table beside a standard ``.../bootstraps`` directory."""
    candidates = (
        bootstrap_dir.parent / "tables" / filename,
        bootstrap_dir / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {filename}; tried: {tried}")


def build_qc_report_all(
    npz_files: Mapping[int, str | Path],
    bootstrap_dir: str | Path,
) -> pd.DataFrame:
    """Build the per-rat/per-cell census required by ``section3_tables.py``.

    The standard output layout is expected: ``bootstraps`` and ``tables`` are
    sibling directories.  Trial counts come from ``cell_composition.csv`` and
    full-data boundary markers from ``full_data_cell_fits.csv``.
    """
    bootstrap_dir = Path(bootstrap_dir)
    composition_path = _find_companion_table(bootstrap_dir, "cell_composition.csv")
    fits_path = _find_companion_table(bootstrap_dir, "full_data_cell_fits.csv")
    composition = pd.read_csv(composition_path)
    full_fits = pd.read_csv(fits_path)

    rows: list[dict[str, Any]] = []
    for rat in sorted(npz_files, key=int):
        archive_path = Path(npz_files[rat])
        if not archive_path.is_absolute():
            archive_path = bootstrap_dir / archive_path
        data = load_bootstrap(int(rat), archive_path)

        for cell in _names(data["cell_order"]):
            sigma_bad = sigma_degenerate_mask(data, cell)
            gamma = np.asarray(cell_param(data, cell, "gamma"), dtype=float)
            lapse = np.asarray(cell_param(data, cell, "lapse"), dtype=float)
            gl_bad = (
                ~np.isfinite(gamma)
                | ~np.isfinite(lapse)
                | (gamma >= GAMMA_LAPSE_NEAR_CEILING)
                | (lapse >= GAMMA_LAPSE_NEAR_CEILING)
            )

            comp_rows = composition.loc[
                (pd.to_numeric(composition["rat"], errors="coerce") == int(rat))
                & (composition["cell"].astype(str) == cell)
            ]
            n_trials = int(pd.to_numeric(comp_rows["n_trials"], errors="coerce").fillna(0).sum())

            rows.append(
                {
                    "rat": int(rat),
                    "cell": cell,
                    "n_trials": n_trials,
                    "n_boot": int(data["n_boot"]),
                    "n_sigma_degenerate": int(sigma_bad.sum()),
                    "pct_sigma_degenerate": float(100.0 * sigma_bad.mean()),
                    "n_gamma_lapse_near_ceiling": int(gl_bad.sum()),
                    "pct_gamma_lapse_near_ceiling": float(100.0 * gl_bad.mean()),
                    "full_data_boundary_params": full_data_boundary_flags(
                        full_fits, int(rat), cell
                    ),
                }
            )

    return pd.DataFrame(rows)


__all__ = [
    "load_bootstrap",
    "cell_param",
    "cell_contrast",
    "sigma_degenerate_mask",
    "full_data_boundary_flags",
    "build_qc_report_all",
    "OUTCOME_CELLS",
    "OUTCOME_CONTRAST",
]
