"""
Quick descriptive checks on the trial sequence (not for the thesis, only to guide Discussion wording).

Run it on the SAME filtered dataframe used for the GLM (after filter_training_trials),
so that lags already respect session boundaries and warm-up removal.

It answers two questions per rat:
  1. Is the category sequence anti-correlated?  Nikbakht et al. (2018) sampled orientations
     "in a semi-random fashion (sampling without replacement)". Without replacement, a vertical
     trial makes the next trial slightly more likely to be horizontal. If so, avoiding the
     previously rewarded category (rats 2 and 12) would partly exploit the sequence.
  2. Were there correction trials after errors?  If the same orientation is repeated far more
     often after an error than after a correct trial, errors were followed by repeats.

Also prints the lag-1..5 category autocorrelation (sampling without replacement should give
small negative values at short lags).
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from serial_dependence_analysis import SerialDependenceAnalyzer
from thesis_codes.section_2_perceptual_bias.functions_Ale import filter_training_trials

# ---------------- edit to match your dataframe ----------------
RAT, SESS = "rat", "date"
ANGLE, REW = "angle", "hitmiss"                 # REW: 1 = rewarded, 0 = unrewarded
ANGLE_PREV, REW_PREV = "angle_n-1", "hitmiss_n-1"  # set both to None to compute lags here
BOUNDARY = 45.0
# ---------------------------------------------------------------


def _category(a):
    """+1 vertical, -1 horizontal, 0 boundary."""
    return np.sign(np.asarray(a, dtype=float) - BOUNDARY)


def _add_lags(df):
    d = df.copy()
    if ANGLE_PREV is not None and ANGLE_PREV in d.columns:
        d["_a_prev"], d["_r_prev"] = d[ANGLE_PREV], d[REW_PREV]
    else:
        # assumes rows are already in trial order within each session
        g = d.groupby([RAT, SESS], sort=False)
        d["_a_prev"], d["_r_prev"] = g[ANGLE].shift(1), g[REW].shift(1)
    d = d.dropna(subset=["_a_prev", "_r_prev"])
    d["cat"], d["cat_prev"] = _category(d[ANGLE]), _category(d["_a_prev"])
    return d


def sequence_checks(df):
    d = _add_lags(df)
    rows = []
    for rat, g in d.groupby(RAT):
        # --- 1. category repetition vs independence ---
        valid = (g["cat"] != 0) & (g["cat_prev"] != 0)
        gg = g[valid]
        same = (gg["cat"] == gg["cat_prev"])
        p_v = (g["cat"] == 1).sum() / (g["cat"] != 0).sum()
        p_exp = p_v**2 + (1 - p_v) ** 2          # expected P(same) if trials were independent
        test = binomtest(int(same.sum()), int(len(same)), p_exp)

        # --- 2. correction-trial check ---
        after_err = gg["_r_prev"] == 0
        same_angle = g[ANGLE] == g["_a_prev"]
        err_all = g["_r_prev"] == 0

        rows.append(dict(
            rat=rat,
            n_pairs=len(gg),
            p_same_cat=round(same.mean(), 4),
            p_same_expected=round(p_exp, 4),
            diff=round(same.mean() - p_exp, 4),
            p_binom=test.pvalue,
            p_same_cat_after_correct=round(same[~after_err].mean(), 4),
            p_same_cat_after_error=round(same[after_err].mean(), 4),
            p_same_angle_after_correct=round(same_angle[~err_all].mean(), 4),
            p_same_angle_after_error=round(same_angle[err_all].mean(), 4),
        ))
    return pd.DataFrame(rows)


def category_autocorr(df, max_lag=5):
    """Mean within-session correlation between cat(n) and cat(n-k), boundary trials excluded."""
    d = df.copy()
    d["cat"] = _category(d[ANGLE])
    out = []
    for rat, g in d.groupby(RAT):
        row = {"rat": rat}
        for k in range(1, max_lag + 1):
            prev = g.groupby(SESS, sort=False)["cat"].shift(k)
            m = prev.notna() & (g["cat"] != 0) & (prev != 0)
            row[f"lag{k}"] = round(np.corrcoef(g.loc[m, "cat"], prev[m])[0, 1], 4)
        out.append(row)
    return pd.DataFrame(out)


if __name__ == "__main__":
    an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
    an.load_and_preprocess_data(fold=False)
    df = an.create_lagged_features()
    df = df[(df['angle'] >= 0) & (df['angle'] <= 90)].copy()
    df = df[(df['angle_n-1'] >= 0) & (df['angle_n-1'] <= 90)].copy()
    df = df[(df['difficulty_n-1'] >= -46) & (df['difficulty_n-1'] <= 45)].copy()
    df = filter_training_trials(df, method='criterion', criterion=0.80, min_consec=2)
    print(sequence_checks(df).to_string(index=False))
    print(category_autocorr(df).to_string(index=False))
    