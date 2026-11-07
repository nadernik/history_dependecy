"""
Diagnostic: rat 9 visual, File 2 vs File 3 preprocessing.
Prints trial counts at each stage, then bootstraps to compare fit quality.
"""
import sys
sys.path.append(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy")

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

from serial_dependence_analysis import SerialDependenceAnalyzer
from functions_Ale import filter_training_trials

# --- config ---
RAT             = 9
VISUAL_CODE     = 2            # verify against the printout below; change if needed
PSE_FLANKS      = (40, 50)
N_PROBE_ITERS   = 100
SEED            = 0

# match the OLD fit bounds that generated both Image 1 (clean) and Image 2 (noisy),
# so the only thing varying is the preprocessing.
SIGMA_UPPER     = 50           # File 1/3 bound. Set to 90 to match File 2 harmonised.
LAPSE_UPPER     = 0.1          # File 1/3 bound. Set to 0.4 to match File 2 harmonised.

FILTER_KWARGS   = dict(method='criterion', criterion=0.80, min_consec=2)

COL_STIM        = 'angle'
COL_STIM_NM1    = 'angle_n-1'
COL_CHOICE      = 'action'
COL_CHOICE_NM1  = 'action_n-1'
COL_REWARD      = 'hitmiss_n-1'
COL_RAT         = 'rat'
COL_MOD         = 'mod'


# --- load ---
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data()
df_raw = an.create_lagged_features()

print(f"\nRaw df: {len(df_raw)} trials")
print(f"Modality codes and counts:")
print(df_raw[COL_MOD].value_counts().to_string())
assert VISUAL_CODE in df_raw[COL_MOD].unique(), \
    f"VISUAL_CODE={VISUAL_CODE} not present in mod column — adjust the constant above."


def apply_preprocessing(df, use_training_filter, use_angle_filter):
    d = df.copy()
    if use_training_filter:
        d = filter_training_trials(d, **FILTER_KWARGS)
    if use_angle_filter:
        d = d[(d[COL_STIM]     >= 0) & (d[COL_STIM]     <= 90)].copy()
        d = d[(d[COL_STIM_NM1] >= 0) & (d[COL_STIM_NM1] <= 90)].copy()
    return d


def report_counts(df, label):
    print(f"\n{'='*72}\n{label}\n{'='*72}")
    d = df[(df[COL_RAT] == RAT) & (df[COL_MOD] == VISUAL_CODE)]
    print(f"Rat {RAT} visual total trials: {len(d)}")
    d45 = d[d[COL_STIM_NM1].isin(PSE_FLANKS)]
    print(f"Fake-45 pool (angle_n-1 in {PSE_FLANKS}): {len(d45)}   "
          f"[{PSE_FLANKS[0]}°: {(d[COL_STIM_NM1]==PSE_FLANKS[0]).sum()}, "
          f"{PSE_FLANKS[1]}°: {(d[COL_STIM_NM1]==PSE_FLANKS[1]).sum()}]")

    for rew_val, rew_label in [(1, 'REW=1  (analysis_3)'), (0, 'REW=0  (analysis_4)')]:
        pool = d45[d45[COL_REWARD] == rew_val]
        print(f"\n  {rew_label}   pool n={len(pool)}")
        if len(pool) == 0:
            print("    (empty)")
            continue
        cell = (pool.groupby([COL_STIM_NM1, COL_CHOICE_NM1])
                    .size().unstack(fill_value=0))
        cell.columns = [f'prev-{"H" if c==0 else "V"}' for c in cell.columns]
        print(cell.to_string().replace('\n', '\n    '))

        # what prepare_trials actually hands to fit_psychometric
        low_pool  = pool[pool[COL_STIM_NM1] == PSE_FLANKS[0]]
        high_pool = pool[pool[COL_STIM_NM1] == PSE_FLANKS[1]]
        n1 = min(len(low_pool), len(high_pool))
        if n1 == 0:
            print("    -> one flank empty; bootstrap will always skip")
            continue
        bal45 = pd.concat([low_pool.head(n1), high_pool.head(n1)])
        n_h = (bal45[COL_CHOICE_NM1] == 0).sum()
        n_v = (bal45[COL_CHOICE_NM1] == 1).sum()
        eff = min(n_h, n_v)
        n_unique = bal45[COL_STIM].nunique()
        print(f"    -> after flank balance (n={2*n1}):  prev-H={n_h}, prev-V={n_v}")
        print(f"    -> per-group sample the fit actually sees: {eff}")
        print(f"    -> unique current-trial angles in balanced pool: {n_unique}")


regimes = {
    'A: no filters                       (matches File 3 / older script)': (False, False),
    'B: angle-range filter only':                                          (False, True),
    'C: training filter only':                                             (True,  False),
    'D: both filters                     (matches File 2 harmonised)':     (True,  True),
}
preprocessed = {label: apply_preprocessing(df_raw, uf, ua)
                for label, (uf, ua) in regimes.items()}
for label, d in preprocessed.items():
    report_counts(d, label)


# =============================================================================
# Bootstrap probe on the two extremes
# =============================================================================

def neg_log_likelihood(params, x, c):
    mu, sigma, gamma, lapse = params
    p = gamma + (1 - gamma - lapse) * norm.cdf((x - mu) / sigma)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -np.sum(c * np.log(p) + (1 - c) * np.log(1 - p))


def fit_psy(df, sigma_upper, lapse_upper, min_per_angle=5):
    counts = df.groupby(COL_STIM)[COL_CHOICE].count()
    df = df[df[COL_STIM].isin(counts[counts >= min_per_angle].index)]
    if df[COL_STIM].nunique() < 3 or len(df) < 10:
        return None
    try:
        r = minimize(neg_log_likelihood, x0=[45, 10, 0.05, 0.05],
                     args=(df[COL_STIM].values, df[COL_CHOICE].values),
                     method='L-BFGS-B',
                     bounds=[(0, 90), (1, sigma_upper),
                             (0, lapse_upper), (0, lapse_upper)])
    except Exception:
        return None
    return {'mu': r.x[0], 'sigma': r.x[1], 'gamma': r.x[2],
            'lapse': r.x[3], 'success': r.success}


def prepare_trials(d_rat, reward_val, rng):
    low  = d_rat[d_rat[COL_STIM_NM1] == PSE_FLANKS[0]]
    high = d_rat[d_rat[COL_STIM_NM1] == PSE_FLANKS[1]]
    n1 = min(len(low), len(high))
    if n1 == 0:
        return None, None
    df45 = pd.concat([
        low.sample(n=n1,  replace=False, random_state=rng.integers(0, 2**31)),
        high.sample(n=n1, replace=False, random_state=rng.integers(0, 2**31)),
    ])
    df45 = df45[df45[COL_REWARD] == reward_val]
    g0 = df45[df45[COL_CHOICE_NM1] == 0]
    g1 = df45[df45[COL_CHOICE_NM1] == 1]
    n2 = min(len(g0), len(g1))
    if n2 == 0:
        return None, None
    return (g0.sample(n=n2, replace=True, random_state=rng.integers(0, 2**31)),
            g1.sample(n=n2, replace=True, random_state=rng.integers(0, 2**31)))


def probe(df, label, reward_val):
    print(f"\n{'-'*72}\nProbe: {label}   reward={reward_val}   "
          f"σ_upper={SIGMA_UPPER}, λ_upper={LAPSE_UPPER}\n{'-'*72}")
    d = df[(df[COL_RAT] == RAT) & (df[COL_MOD] == VISUAL_CODE)]
    rng = np.random.default_rng(SEED)
    fits_0, fits_1 = [], []
    n_prep_skip = n_fit_skip = n_nonconv = 0
    for _ in range(N_PROBE_ITERS):
        g0, g1 = prepare_trials(d, reward_val, rng)
        if g0 is None:
            n_prep_skip += 1; continue
        f0 = fit_psy(g0, SIGMA_UPPER, LAPSE_UPPER)
        f1 = fit_psy(g1, SIGMA_UPPER, LAPSE_UPPER)
        if f0 is None or f1 is None:
            n_fit_skip += 1; continue
        if not (f0['success'] and f1['success']):
            n_nonconv += 1
        fits_0.append(f0); fits_1.append(f1)
    print(f"  prep-skipped: {n_prep_skip}   fit-skipped: {n_fit_skip}   "
          f"non-converged (still kept): {n_nonconv}   valid: {len(fits_0)}")
    if not fits_0:
        return
    df0, df1 = pd.DataFrame(fits_0), pd.DataFrame(fits_1)
    for name, dd in [('group_0 prev-H', df0), ('group_1 prev-V', df1)]:
        sig_bound = (dd['sigma'] > SIGMA_UPPER * 0.98).mean()
        lap_bound = ((dd['lapse'] > LAPSE_UPPER * 0.98) |
                     (dd['gamma'] > LAPSE_UPPER * 0.98)).mean()
        print(f"  {name}: "
              f"μ med={dd['mu'].median():5.1f} IQR=[{dd['mu'].quantile(.25):.1f},{dd['mu'].quantile(.75):.1f}]  "
              f"σ med={dd['sigma'].median():5.1f} IQR=[{dd['sigma'].quantile(.25):.1f},{dd['sigma'].quantile(.75):.1f}]  "
              f"σ@bound:{100*sig_bound:3.0f}%  γ/λ@bound:{100*lap_bound:3.0f}%")
    delta = df1['mu'].values - df0['mu'].values
    print(f"  Δμ: median={np.median(delta):+.2f}°  "
          f"95% CI [{np.percentile(delta, 2.5):+.2f}, {np.percentile(delta, 97.5):+.2f}]  "
          f"IQR width={np.percentile(delta, 75) - np.percentile(delta, 25):.1f}°")


for label in list(regimes)[::3]:                              # A and D only
    for reward_val in (1, 0):
        probe(preprocessed[label], label, reward_val)