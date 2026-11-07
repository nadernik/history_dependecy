"""
Statistics helpers for the Q2/Q3 GLM tables (Holm step-down correction,
percentile-bootstrap p-values). Deliberately kept OUT of table_utils.py:
that file's own docstring scopes it to rendering ("build a table once as
plain data, render it to LaTeX/docx") -- Holm correction and bootstrap
p-values are analysis, not rendering, so they don't belong there. Import
this alongside table_utils in any script that needs both.
"""
import numpy as np


def holm_correction(pvals, alpha=0.05):
    """Standard Holm step-down. Returns (adjusted_p, reject) as np.arrays,
    in the ORIGINAL order of pvals. adjusted_p is monotonic non-decreasing
    along the sorted order and clipped to [0, 1]."""
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n)
    running_max = 0.0
    for rank, idx in enumerate(order):
        m = n - rank
        candidate = pvals[idx] * m
        running_max = max(running_max, candidate)
        adj[idx] = min(running_max, 1.0)
    reject = adj <= alpha
    return adj, reject


def boot_p_value(draws):
    """Two-sided equal-tail bootstrap p-value with plus-one correction."""
    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]

    if len(draws) == 0:
        return np.nan

    n_neg = np.count_nonzero(draws < 0)
    n_pos = np.count_nonzero(draws > 0)
    B = len(draws)

    lower = (n_neg + 1.0) / (B + 1.0)
    upper = (n_pos + 1.0) / (B + 1.0)

    return float(min(1.0, 2.0 * min(lower, upper)))


def compute_global_int_digits(values):
    """Max integer-part digit count across a flat list of numeric values
    (ignoring sign), for table_utils.latex_num's int_digits parameter."""
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return 1
    return max(max(1, len(str(int(abs(v))))) for v in vals)