"""
Modality Transition History Effect Analysis
============================================

Question: does the effect of the previous trial's outcome (beta_success_n-1,
beta_failure_n-1, from a per-rat GLM) change with the modality transition
(T/V/VT on trial n-1 -> trial n)? Each rat's modalities are ranked by
sensitivity (hierarchy model), the betas are fitted separately in each of the 9
transition cells, and plotted ordered by that hierarchy (downgrade / same /
upgrade transitions).

SIBLING SCRIPT: the two scripts share the same core (hierarchy fit, per-cell
GLMs, beta plot) and differ only in what they do with the betas afterwards.
  THIS script (descriptive_correlations): betas normalized by beta_angle
    (NORMALIZE_BY_ANGLE = True); correlates the betas with the modality
    hierarchy scores (compute_descriptive_correlations); saves per-cell
    results to per_cell_glm_results/ for comparison with the mixed models.
  modality_transition_glm_additive_decomposition.py: raw log-odds betas
    (NORMALIZE_BY_ANGLE = False); decomposes the transition effect into
    additive components (compute_additive_decomposition); saves figures only.

Details of the pipeline:
For each rat:
    1. Fit a hierarchy model (features_2) on full data to extract per-rat
       modality slopes and rank V / T / VT strength
    2. Diagnostic: count trials per transition cell, flag sparse cells
    3. Loop over 9 transition cells, fit history model on filtered df
    4. Collect beta_success and beta_failure across cells
    5. Plot: x-axis ordered by rat-specific hierarchy, colored by
       downgrade / same / upgrade

Plugs directly into your existing pipeline:
    - analyze_individual_rats_ale_N(features_keys, df_processed, ...)
    - flatten_model_results(result)
    - df already has: mod_transition_n-1, action_n-1 (±1), success_n-1,
      failure_n-1, hit_n-1, angle, angle_n-1, mod (1=T,2=V,3=VT), rat col
"""
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy")
from functions_Ale import filter_training_trials
from serial_dependence_analysis import SerialDependenceAnalyzer

# ── CONFIG ────────────────────────────────────────────────────────────────────
RAT_COL            = "rat"
MOD_COL            = "mod"                  # 1=T, 2=V, 3=VT
TRANSITION_COL     = "mod_transition_n-1"   # already in your df
MIN_TRIALS_PER_CELL = 200                   # flag cells below this
HISTORY_DEPTH = 1
MOD_LABELS = {1: "T", 2: "V", 3: "VT"}

# Model used only to extract per-rat modality hierarchy
FEATURES_HIERARCHY = [
    'angle', 'tactile', 'visuotactile',
    'angle_tactile', 'angle_visuotactile'
]

# Model fitted in each transition cell — covariates + tracked betas
FEATURES_CELL = ['angle', 'angle_n-1', 'hit_n-1', 'success_n-1', 'failure_n-1']

# Betas to track across cells (others are covariates, sanity-checked only)
BETAS_TO_TRACK = ['success_n-1', 'failure_n-1']

# mod_transition_n-1 is already a string in 'V->T' format — use directly
def transition_label(row):
    """Direct passthrough — column is already a readable string like 'VT->V'."""
    return row[TRANSITION_COL]

ALL_TRANSITIONS = [
    f"{a}->{b}"
    for a in ["T", "V", "VT"] for b in ["T", "V", "VT"]
]  # 9 cells: T->T, T->V, T->VT, V->T, V->V, V->VT, VT->T, VT->V, VT->VT


# ── STEP 1: PER-RAT HIERARCHY FROM FEATURES_2 ────────────────────────────────

def extract_rat_hierarchy(an, df, features_hierarchy=FEATURES_HIERARCHY):
    """
    Fit hierarchy model on full df (all rats at once via analyze_individual_rats_ale_N),
    then for each rat compute effective psychometric slope per modality:
        slope_V  = beta_angle                        (V is reference)
        slope_T  = beta_angle + beta_angle_tactile
        slope_VT = beta_angle + beta_angle_visuotactile

    Returns:
        hierarchy_df : DataFrame with columns
            rat, slope_V, slope_T, slope_VT,
            stronger_unimodal ('V' or 'T'),
            modality_rank (dict: modality_label -> rank 1=weakest)
    """
    print("Step 1: Fitting hierarchy model to extract per-rat modality slopes...")
    hier_result = an.analyze_individual_rats_ale_N(
        features_keys=features_hierarchy,
        df_processed=df,
        k=None,
        scaling='cont_only'
    )

    rows = []
    for rat, d in hier_result.items():
        # coefficients is a numpy array — zip with feature_names to get mapping
        coef_map  = dict(zip(d['feature_names'], d['coefficients']))
        b_angle   = coef_map.get('angle', np.nan)
        b_tactile = coef_map.get('angle_tactile', np.nan)
        b_vt      = coef_map.get('angle_visuotactile', np.nan)

        slope_V  = b_angle
        slope_T  = b_angle + b_tactile
        slope_VT = b_angle + b_vt

        slopes = {'V': slope_V, 'T': slope_T, 'VT': slope_VT}

        # Rank modalities: higher slope = stronger sensory evidence
        ranked = sorted(slopes, key=lambda k: slopes[k])  # weakest first
        modality_rank = {m: i+1 for i, m in enumerate(ranked)}  # 1=weakest

        stronger_unimodal = 'V' if slope_V >= slope_T else 'T'

        rows.append({
            RAT_COL:              rat,
            'slope_V':            slope_V,
            'slope_T':            slope_T,
            'slope_VT':           slope_VT,
            'stronger_unimodal':  stronger_unimodal,
            'modality_rank':      modality_rank,  # e.g. {'T':1, 'V':2, 'VT':3}
        })
        print(f"  Rat {rat}: slope_V={slope_V:.3f}, slope_T={slope_T:.3f}, "
              f"slope_VT={slope_VT:.3f}  → stronger unimodal: {stronger_unimodal}")

    return pd.DataFrame(rows)


# ── STEP 2: DIAGNOSTIC ───────────────────────────────────────────────────────

def transition_diagnostic(df, hierarchy_df, min_trials=MIN_TRIALS_PER_CELL):
    """
    Print trial counts per rat × transition cell.
    Flag cells below min_trials threshold.
    Returns a pivot table (rats × transitions) for inspection.
    """
    print(f"\nStep 2: Trial count diagnostic (flagging cells < {min_trials} trials)")

    # Column is already a readable string — copy directly
    df = df.copy()
    df['_transition_label'] = df[TRANSITION_COL]

    counts = (
        df.groupby([RAT_COL, '_transition_label'])
        .size()
        .reset_index(name='n_trials')
    )
    pivot = counts.pivot(index=RAT_COL, columns='_transition_label', values='n_trials').fillna(0).astype(int)

    # Reorder columns to ALL_TRANSITIONS order (only those present)
    col_order = [t for t in ALL_TRANSITIONS if t in pivot.columns]
    pivot = pivot[col_order]

    print("\nTrial counts per rat per transition cell:")
    print(pivot.to_string())

    # Flag sparse cells
    sparse = counts[counts['n_trials'] < min_trials]
    if len(sparse) > 0:
        print(f"\n⚠ Sparse cells (< {min_trials} trials):")
        print(sparse.to_string(index=False))
    else:
        print(f"\n✓ All cells have ≥ {min_trials} trials")

    return pivot, df  # return df with _transition_label added


# ── STEP 3 & 4: FIT CELL GLMS + COLLECT BETAS ───────────────────────────────

def fit_cell_glms(an, df, hierarchy_df,
                  features_cell=FEATURES_CELL,
                  betas_to_track=BETAS_TO_TRACK,
                  min_trials=MIN_TRIALS_PER_CELL):
    """
    For each rat × transition cell:
        - Filter df to that rat + that transition cell
        - Call analyze_individual_rats_ale_N on filtered single-rat df
        - Store betas for betas_to_track

    Returns:
        beta_records : list of dicts with keys
            rat, transition, beta_name, beta, ci_lower, ci_upper
    """
    print(f"\nStep 3 & 4: Fitting per-cell GLMs ({len(hierarchy_df)} rats × 9 cells)...")

    rats = hierarchy_df[RAT_COL].tolist()
    beta_records = []

    for rat in rats:
        rat_df = df[df[RAT_COL] == rat].copy()
        transitions_present = rat_df['_transition_label'].unique()

        for trans in ALL_TRANSITIONS:
            cell_df = rat_df[rat_df['_transition_label'] == trans].copy()

            if len(cell_df) < min_trials:
                print(f"  Skipping rat {rat} | {trans}: only {len(cell_df)} trials")
                continue

            try:
                result = an.analyze_individual_rats_ale_N(
                    features_keys=features_cell,
                    df_processed=cell_df,
                    k=None,
                    scaling='cont_only',
                    cv_split=False,
                )
                # result has one key = rat (since we passed single-rat df)
                rat_key  = list(result.keys())[0]
                d        = result[rat_key]
                # coefficients is a numpy array — build name→value map
                coef_map = dict(zip(d['feature_names'], d['coefficients']))

                # Sanity check covariates — print if suspiciously large
                for cov in ['angle', 'angle_n-1', 'hit_n-1']:
                    val = coef_map.get(cov, np.nan)
                    if not np.isnan(val) and abs(val) > 10:
                        print(f"  ⚠ Rat {rat} | {trans} | {cov} = {val:.2f} (suspicious)")

                # Store tracked betas (point estimates only)
                for beta_name in betas_to_track:
                    beta_records.append({
                        RAT_COL:      rat,
                        'transition': trans,
                        'beta_name':  beta_name,
                        'beta':       coef_map.get(beta_name, np.nan),
                    })

            except Exception as e:
                print(f"  ✗ Rat {rat} | {trans} failed: {e}")

    print(f"  Done. {len(beta_records)} beta records collected.")
    return pd.DataFrame(beta_records)


# ── STEP 5: PLOT ─────────────────────────────────────────────────────────────

def get_transition_color(transition, modality_rank):
    """
    Color by upgrade / same / downgrade based on rat-specific modality rank.
    modality_rank: dict e.g. {'T': 1, 'V': 2, 'VT': 3}  (1 = weakest)
    """
    parts = transition.split('->')
    if len(parts) != 2:
        return '#7f7f7f'
    prev, curr = parts
    rank_prev = modality_rank.get(prev, 2)
    rank_curr = modality_rank.get(curr, 2)
    if rank_curr < rank_prev:
        return '#d62728'   # downgrade → red (stronger history predicted)
    elif rank_curr > rank_prev:
        return '#1f77b4'   # upgrade   → blue (weaker history predicted)
    else:
        return '#7f7f7f'   # same      → gray


def order_transitions_for_rat(modality_rank):
    """
    Order transitions from biggest downgrade to biggest upgrade.
    modality_rank: dict e.g. {'T': 1, 'V': 2, 'VT': 3}
    """
    def score(t):
        parts = t.split('->')
        if len(parts) != 2:
            return 0
        prev, curr = parts
        return modality_rank.get(prev, 2) - modality_rank.get(curr, 2)  # positive = downgrade

    return sorted(ALL_TRANSITIONS, key=score, reverse=True)


def plot_transition_betas(beta_df, hierarchy_df,
                           betas_to_track=BETAS_TO_TRACK,
                           save_path="transition_history_betas.png"):
    """
    One subplot per rat. Two traces per subplot (success_n-1, failure_n-1).
    X-axis: 9 transitions ordered by rat-specific hierarchy.
    Color: red=downgrade, gray=same, blue=upgrade.
    """
    rats = hierarchy_df[RAT_COL].tolist()
    n_rats = len(rats)
    ncols = 3
    nrows = int(np.ceil(n_rats / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 4 * nrows),
        constrained_layout=True
    )
    axes = axes.flatten()

    hier_dict = hierarchy_df.set_index(RAT_COL)['modality_rank'].to_dict()
    stronger_dict = hierarchy_df.set_index(RAT_COL)['stronger_unimodal'].to_dict()

    beta_styles = {
        'success_n-1': dict(marker='o', linestyle='-',  label='β success (win-stay)'),
        'failure_n-1': dict(marker='s', linestyle='--', label='β failure (lose-shift)'),
    }

    for ax, rat in zip(axes, rats):
        modality_rank = hier_dict.get(rat, {'T': 1, 'V': 2, 'VT': 3})
        stronger      = stronger_dict.get(rat, '?')
        ordered_trans = order_transitions_for_rat(modality_rank)

        # Background color bands for downgrade / same / upgrade regions
        scores = [modality_rank.get(t.split('->')[0], 2) -
                  modality_rank.get(t.split('->')[1], 2)
                  for t in ordered_trans]
        x = np.arange(len(ordered_trans))

        # Shade regions
        for xi, (trans, score) in enumerate(zip(ordered_trans, scores)):
            color = get_transition_color(trans, modality_rank)
            ax.axvspan(xi - 0.5, xi + 0.5, alpha=0.08, color=color, zorder=0)

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)

        # Plot each tracked beta as a separate trace
        rdf = beta_df[beta_df[RAT_COL] == rat]
        for beta_name, style in beta_styles.items():
            bdf = rdf[rdf['beta_name'] == beta_name].set_index('transition')
            betas = [bdf.loc[t, 'beta'] if t in bdf.index else np.nan
                     for t in ordered_trans]
            ax.plot(x, betas, color='black', alpha=0.15, linewidth=1, zorder=1)
            ax.scatter(x, betas,
                       c=[get_transition_color(t, modality_rank) for t in ordered_trans],
                       marker=style['marker'], s=60, zorder=3,
                       label=style['label'])
            ax.plot(x, betas,
                    color='gray', linewidth=1, linestyle=style['linestyle'],
                    alpha=0.5, zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels(ordered_trans, rotation=45, ha='right', fontsize=7)
        ax.set_title(f"Rat {rat}  (stronger: {stronger})", fontsize=9)
        ax.set_ylabel("Beta", fontsize=8)
        ax.set_xlabel("Transition n-1 → n", fontsize=8)

        # Annotate modality rank
        rank_str = " < ".join(
            sorted(modality_rank, key=modality_rank.get)
        )
        ax.text(0.02, 0.97, f"Hierarchy: {rank_str}",
                transform=ax.transAxes, fontsize=6.5,
                verticalalignment='top', color='dimgray')

    # Hide unused subplots
    for ax in axes[n_rats:]:
        ax.set_visible(False)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#d62728', alpha=0.6, label='Downgrade (weaker n)'),
        mpatches.Patch(facecolor='#7f7f7f', alpha=0.6, label='Same modality'),
        mpatches.Patch(facecolor='#1f77b4', alpha=0.6, label='Upgrade (stronger n)'),
        plt.Line2D([0], [0], marker='o', color='gray', label='β success_n-1', linestyle='-'),
        plt.Line2D([0], [0], marker='s', color='gray', label='β failure_n-1', linestyle='--'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        "History effect (success/failure) across modality transitions\n"
        "X-axis ordered by rat-specific sensory hierarchy  |  "
        "Color: red=downgrade, gray=same, blue=upgrade",
        fontsize=11
    )

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {save_path}")
    return fig


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def run_transition_analysis(an, df, save_path="transition_history_betas.png"):
    """
    Full pipeline. Pass your SerialDependenceAnalyzer instance and
    preprocessed df (with lagged features already created).
    """
    # Step 1: hierarchy
    hierarchy_df = extract_rat_hierarchy(an, df)

    # Step 2: diagnostic + add _transition_label to df
    count_pivot, df = transition_diagnostic(df, hierarchy_df)

    # Steps 3 & 4: fit cell GLMs
    beta_df = fit_cell_glms(an, df, hierarchy_df)

    # Step 5: plot
    fig = plot_transition_betas(beta_df, hierarchy_df, save_path=save_path)

    return {
        'hierarchy':   hierarchy_df,
        'counts':      count_pivot,
        'betas':       beta_df,
        'figure':      fig,
    }


# ── USAGE ─────────────────────────────────────────────────────────────────────
# Paste this at the bottom of your existing script, after df is built:
# =============================
# LOAD + PREPROCESS
# =============================
an = SerialDependenceAnalyzer(history_depth=HISTORY_DEPTH, save_figures=False)
an.load_and_preprocess_data()
df = an.create_lagged_features()
df = filter_training_trials(df, method='criterion', criterion=0.80, min_consec=2)


df['action_n-1'] = df['action_n-1'] * 2 - 1
df['success_n-1'] = df['hit_n-1'] * df['action_n-1']
df['failure_n-1'] = (1 - df['hit_n-1']) * df['action_n-1']
df['angle']       = df['angle'] - 45
df['angle_n-1']   = df['angle_n-1'] - 45
df['tactile'] = df['mod'] == 1
# df['visual'] = df['mod'] == 2
df['visuotactile'] = df['mod'] == 3
df['angle_tactile'] = df['angle']*df['tactile']
df['angle_visuotactile'] = df['angle']*df['visuotactile']
#
# from transition_history_analysis import run_transition_analysis
#out = run_transition_analysis(an, df, save_path="transition_history_betas.png")
#
# or inline:
out = run_transition_analysis(an, df)
hierarchy_df = out['hierarchy']
beta_df      = out['betas']



'''

"""
Modality Transition History Effect Analysis
============================================
For each rat:
    1. Fit a hierarchy model (features_2) on full data to extract per-rat
       modality slopes and rank V / T / VT strength
    2. Diagnostic: count trials per transition cell, flag sparse cells
    3. Loop over 9 transition cells, fit history model on filtered df
    4. Collect beta_success and beta_failure across cells
    5. Plot: x-axis ordered by rat-specific hierarchy, colored by
       downgrade / same / upgrade

Plugs directly into your existing pipeline:
    - analyze_individual_rats_ale_N(features_keys, df_processed, ...)
    - flatten_model_results(result)
    - df already has: mod_transition_n-1, action_n-1 (±1), success_n-1,
      failure_n-1, hit_n-1, angle, angle_n-1, mod (1=T,2=V,3=VT), rat col
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy")
from functions_Ale import filter_training_trials
from serial_dependence_analysis import SerialDependenceAnalyzer

# ── CONFIG ────────────────────────────────────────────────────────────────────
RAT_COL            = "rat"
MOD_COL            = "mod"                  # 1=T, 2=V, 3=VT
TRANSITION_COL     = "mod_transition_n-1"   # already in your df
MIN_TRIALS_PER_CELL = 200                   # flag cells below this

MOD_LABELS = {1: "T", 2: "V", 3: "VT"}
HISTORY_DEPTH = 1
# Model used only to extract per-rat modality hierarchy
FEATURES_HIERARCHY = [
    'angle', 'tactile', 'visuotactile',
    'angle_tactile', 'angle_visuotactile'
]

# Model fitted in each transition cell — covariates + tracked betas
FEATURES_CELL = ['angle', 'angle_n-1', 'hit_n-1', 'success_n-1', 'failure_n-1']

# Betas to track across cells (others are covariates, sanity-checked only)
BETAS_TO_TRACK = ['success_n-1', 'failure_n-1']

# ── NORMALIZATION FLAG ────────────────────────────────────────────────────────
# If True, divide each rat's history betas by its beta_angle from the same
# cell model. This expresses history effects in units of 'stimulus equivalents'
# and removes the cross-rat comparability problem from logistic regression's
# fixed residual variance assumption (Mood 2010, Allison 1999).
# If False, raw logistic coefficients are used (fine for within-rat comparisons).
NORMALIZE_BY_ANGLE = True

# mod_transition_n-1 is already a string in 'V->T' format — use directly
def transition_label(row):
    """Direct passthrough — column is already a readable string like 'VT->V'."""
    return row[TRANSITION_COL]

ALL_TRANSITIONS = [
    f"{a}->{b}"
    for a in ["T", "V", "VT"] for b in ["T", "V", "VT"]
]  # 9 cells: T->T, T->V, T->VT, V->T, V->V, V->VT, VT->T, VT->V, VT->VT


# ── STEP 1: PER-RAT HIERARCHY FROM FEATURES_2 ────────────────────────────────

def extract_rat_hierarchy(an, df, features_hierarchy=FEATURES_HIERARCHY):
    """
    Fit hierarchy model on full df (all rats at once via analyze_individual_rats_ale_N),
    then for each rat compute effective psychometric slope per modality:
        slope_V  = beta_angle                        (V is reference)
        slope_T  = beta_angle + beta_angle_tactile
        slope_VT = beta_angle + beta_angle_visuotactile

    Returns:
        hierarchy_df : DataFrame with columns
            rat, slope_V, slope_T, slope_VT,
            stronger_unimodal ('V' or 'T'),
            modality_rank (dict: modality_label -> rank 1=weakest)
    """
    print("Step 1: Fitting hierarchy model to extract per-rat modality slopes...")
    hier_result = an.analyze_individual_rats_ale_N(
        features_keys=features_hierarchy,
        df_processed=df,
        k=None,
        scaling='cont_only'
    )

    rows = []
    for rat, d in hier_result.items():
        # coefficients is a numpy array — zip with feature_names to get mapping
        coef_map  = dict(zip(d['feature_names'], d['coefficients']))
        b_angle   = coef_map.get('angle', np.nan)
        b_tactile = coef_map.get('angle_tactile', np.nan)
        b_vt      = coef_map.get('angle_visuotactile', np.nan)

        slope_V  = b_angle
        slope_T  = b_angle + b_tactile
        slope_VT = b_angle + b_vt

        slopes = {'V': slope_V, 'T': slope_T, 'VT': slope_VT}

        # Rank modalities: higher slope = stronger sensory evidence
        ranked = sorted(slopes, key=lambda k: slopes[k])  # weakest first
        modality_rank = {m: i+1 for i, m in enumerate(ranked)}  # 1=weakest

        stronger_unimodal = 'V' if slope_V >= slope_T else 'T'

        rows.append({
            RAT_COL:              rat,
            'slope_V':            slope_V,
            'slope_T':            slope_T,
            'slope_VT':           slope_VT,
            'stronger_unimodal':  stronger_unimodal,
            'modality_rank':      modality_rank,  # e.g. {'T':1, 'V':2, 'VT':3}
        })
        print(f"  Rat {rat}: slope_V={slope_V:.3f}, slope_T={slope_T:.3f}, "
              f"slope_VT={slope_VT:.3f}  → stronger unimodal: {stronger_unimodal}")

    return pd.DataFrame(rows)


# ── STEP 2: DIAGNOSTIC ───────────────────────────────────────────────────────

def transition_diagnostic(df, hierarchy_df, min_trials=MIN_TRIALS_PER_CELL):
    """
    Print trial counts per rat × transition cell.
    Flag cells below min_trials threshold.
    Returns a pivot table (rats × transitions) for inspection.
    """
    print(f"\nStep 2: Trial count diagnostic (flagging cells < {min_trials} trials)")

    # Column is already a readable string — copy directly
    df = df.copy()
    df['_transition_label'] = df[TRANSITION_COL]

    counts = (
        df.groupby([RAT_COL, '_transition_label'])
        .size()
        .reset_index(name='n_trials')
    )
    pivot = counts.pivot(index=RAT_COL, columns='_transition_label', values='n_trials').fillna(0).astype(int)

    # Reorder columns to ALL_TRANSITIONS order (only those present)
    col_order = [t for t in ALL_TRANSITIONS if t in pivot.columns]
    pivot = pivot[col_order]

    print("\nTrial counts per rat per transition cell:")
    print(pivot.to_string())

    # Flag sparse cells
    sparse = counts[counts['n_trials'] < min_trials]
    if len(sparse) > 0:
        print(f"\n⚠ Sparse cells (< {min_trials} trials):")
        print(sparse.to_string(index=False))
    else:
        print(f"\n✓ All cells have ≥ {min_trials} trials")

    return pivot, df  # return df with _transition_label added


# ── STEP 3 & 4: FIT CELL GLMS + COLLECT BETAS ───────────────────────────────

def fit_cell_glms(an, df, hierarchy_df,
                  features_cell=FEATURES_CELL,
                  betas_to_track=BETAS_TO_TRACK,
                  min_trials=MIN_TRIALS_PER_CELL):
    """
    For each rat × transition cell:
        - Filter df to that rat + that transition cell
        - Call analyze_individual_rats_ale_N on filtered single-rat df
        - Store betas for betas_to_track

    Returns:
        beta_records : list of dicts with keys
            rat, transition, beta_name, beta, ci_lower, ci_upper
    """
    print(f"\nStep 3 & 4: Fitting per-cell GLMs ({len(hierarchy_df)} rats × 9 cells)...")

    rats = hierarchy_df[RAT_COL].tolist()
    beta_records = []

    for rat in rats:
        rat_df = df[df[RAT_COL] == rat].copy()
        transitions_present = rat_df['_transition_label'].unique()

        for trans in ALL_TRANSITIONS:
            cell_df = rat_df[rat_df['_transition_label'] == trans].copy()

            if len(cell_df) < min_trials:
                print(f"  Skipping rat {rat} | {trans}: only {len(cell_df)} trials")
                continue

            try:
                result = an.analyze_individual_rats_ale_N(
                    features_keys=features_cell,
                    df_processed=cell_df,
                    k=None,
                    scaling='cont_only',
                    cv_split=False,
                )
                # result has one key = rat (since we passed single-rat df)
                rat_key  = list(result.keys())[0]
                d        = result[rat_key]
                # coefficients is a numpy array — build name→value map
                coef_map = dict(zip(d['feature_names'], d['coefficients']))

                # Sanity check covariates — print if suspiciously large
                for cov in ['angle', 'angle_n-1', 'hit_n-1']:
                    val = coef_map.get(cov, np.nan)
                    if not np.isnan(val) and abs(val) > 10:
                        print(f"  ⚠ Rat {rat} | {trans} | {cov} = {val:.2f} (suspicious)")

                # Store tracked betas + beta_angle (needed for normalization)
                for beta_name in betas_to_track + ['angle']:
                    beta_records.append({
                        RAT_COL:      rat,
                        'transition': trans,
                        'beta_name':  beta_name,
                        'beta':       coef_map.get(beta_name, np.nan),
                    })

            except Exception as e:
                print(f"  ✗ Rat {rat} | {trans} failed: {e}")

    print(f"  Done. {len(beta_records)} beta records collected.")
    return pd.DataFrame(beta_records)


# ── STEP 5: PLOT ─────────────────────────────────────────────────────────────

def get_transition_color(transition, modality_rank):
    """
    Color by upgrade / same / downgrade based on rat-specific modality rank.
    modality_rank: dict e.g. {'T': 1, 'V': 2, 'VT': 3}  (1 = weakest)
    """
    parts = transition.split('->')
    if len(parts) != 2:
        return '#7f7f7f'
    prev, curr = parts
    rank_prev = modality_rank.get(prev, 2)
    rank_curr = modality_rank.get(curr, 2)
    if rank_curr < rank_prev:
        return '#d62728'   # downgrade → red (stronger history predicted)
    elif rank_curr > rank_prev:
        return '#1f77b4'   # upgrade   → blue (weaker history predicted)
    else:
        return '#7f7f7f'   # same      → gray


def order_transitions_for_rat(modality_rank):
    """
    Order transitions from biggest downgrade to biggest upgrade.
    modality_rank: dict e.g. {'T': 1, 'V': 2, 'VT': 3}
    """
    def score(t):
        parts = t.split('->')
        if len(parts) != 2:
            return 0
        prev, curr = parts
        return modality_rank.get(prev, 2) - modality_rank.get(curr, 2)  # positive = downgrade

    return sorted(ALL_TRANSITIONS, key=score, reverse=True)


def plot_transition_betas(beta_df, hierarchy_df,
                           betas_to_track=BETAS_TO_TRACK,
                           save_path="transition_history_betas.png"):
    """
    One subplot per rat. Two traces per subplot (success_n-1, failure_n-1).
    X-axis: 9 transitions ordered by rat-specific hierarchy.
    Color: red=downgrade, gray=same, blue=upgrade.
    """
    rats = hierarchy_df[RAT_COL].tolist()
    n_rats = len(rats)
    ncols = 3
    nrows = int(np.ceil(n_rats / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 4 * nrows),
        constrained_layout=True
    )
    axes = axes.flatten()

    hier_dict = hierarchy_df.set_index(RAT_COL)['modality_rank'].to_dict()
    stronger_dict = hierarchy_df.set_index(RAT_COL)['stronger_unimodal'].to_dict()

    beta_styles = {
        'success_n-1': dict(marker='o', linestyle='-',  label='β success (win-stay)'),
        'failure_n-1': dict(marker='s', linestyle='--', label='β failure (lose-shift)'),
    }

    for ax, rat in zip(axes, rats):
        modality_rank = hier_dict.get(rat, {'T': 1, 'V': 2, 'VT': 3})
        stronger      = stronger_dict.get(rat, '?')
        ordered_trans = order_transitions_for_rat(modality_rank)

        # Background color bands for downgrade / same / upgrade regions
        scores = [modality_rank.get(t.split('->')[0], 2) -
                  modality_rank.get(t.split('->')[1], 2)
                  for t in ordered_trans]
        x = np.arange(len(ordered_trans))

        # Shade regions
        for xi, (trans, score) in enumerate(zip(ordered_trans, scores)):
            color = get_transition_color(trans, modality_rank)
            ax.axvspan(xi - 0.5, xi + 0.5, alpha=0.08, color=color, zorder=0)

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.4)

        # Plot each tracked beta as a separate trace
        rdf = beta_df[beta_df[RAT_COL] == rat]
        for beta_name, style in beta_styles.items():
            bdf = rdf[rdf['beta_name'] == beta_name].set_index('transition')
            betas = [bdf.loc[t, 'beta'] if t in bdf.index else np.nan
                     for t in ordered_trans]
            ax.plot(x, betas, color='black', alpha=0.15, linewidth=1, zorder=1)
            ax.scatter(x, betas,
                       c=[get_transition_color(t, modality_rank) for t in ordered_trans],
                       marker=style['marker'], s=60, zorder=3,
                       label=style['label'])
            ax.plot(x, betas,
                    color='gray', linewidth=1, linestyle=style['linestyle'],
                    alpha=0.5, zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels(ordered_trans, rotation=45, ha='right', fontsize=7)
        ax.set_title(f"Rat {rat}  (stronger: {stronger})", fontsize=9)
        ax.set_ylabel("Beta", fontsize=8)
        ax.set_xlabel("Transition n-1 → n", fontsize=8)

        # Annotate modality rank
        rank_str = " < ".join(
            sorted(modality_rank, key=modality_rank.get)
        )
        ax.text(0.02, 0.97, f"Hierarchy: {rank_str}",
                transform=ax.transAxes, fontsize=6.5,
                verticalalignment='top', color='dimgray')

    # Hide unused subplots
    for ax in axes[n_rats:]:
        ax.set_visible(False)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#d62728', alpha=0.6, label='Downgrade (weaker n)'),
        mpatches.Patch(facecolor='#7f7f7f', alpha=0.6, label='Same modality'),
        mpatches.Patch(facecolor='#1f77b4', alpha=0.6, label='Upgrade (stronger n)'),
        plt.Line2D([0], [0], marker='o', color='gray', label='β success_n-1', linestyle='-'),
        plt.Line2D([0], [0], marker='s', color='gray', label='β failure_n-1', linestyle='--'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        "History effect (success/failure) across modality transitions\n"
        "X-axis ordered by rat-specific sensory hierarchy  |  "
        "Color: red=downgrade, gray=same, blue=upgrade",
        fontsize=11
    )

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {save_path}")
    return fig


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

def normalize_betas_by_angle(beta_df, betas_to_track=BETAS_TO_TRACK):
    """
    Divide each history beta by beta_angle from the same rat x cell model.
    This puts history effects in units of 'stimulus equivalents', making
    cross-rat comparisons valid despite logistic regression's fixed residual
    variance assumption (Mood 2010, Allison 1999).

    Rows where beta_angle == 0 or is NaN are set to NaN (can't normalize).
    The 'angle' rows are dropped from the output since they're always 1.0
    after normalization and carry no further information.
    """
    wide = beta_df.pivot_table(
        index=[RAT_COL, 'transition'],
        columns='beta_name',
        values='beta'
    ).reset_index()

    angle_vals = wide['angle'].replace(0, np.nan)  # avoid division by zero

    for beta_name in betas_to_track:
        if beta_name in wide.columns:
            wide[beta_name] = wide[beta_name] / angle_vals

    long = wide.drop(columns='angle').melt(
        id_vars=[RAT_COL, 'transition'],
        var_name='beta_name',
        value_name='beta'
    )
    return long


def run_transition_analysis(an, df, save_path="transition_history_betas.png"):
    """
    Full pipeline. Pass your SerialDependenceAnalyzer instance and
    preprocessed df (with lagged features already created).
    """
    # Step 1: hierarchy
    hierarchy_df = extract_rat_hierarchy(an, df)

    # Step 2: diagnostic + add _transition_label to df
    count_pivot, df = transition_diagnostic(df, hierarchy_df)

    # Steps 3 & 4: fit cell GLMs
    beta_df = fit_cell_glms(an, df, hierarchy_df)

    # Optional normalization (controlled by NORMALIZE_BY_ANGLE flag)
    if NORMALIZE_BY_ANGLE:
        print("\nNormalizing betas by beta_angle (stimulus equivalents)...")
        plot_df = normalize_betas_by_angle(beta_df)
    else:
        # Drop angle rows — only tracked betas needed for plot
        plot_df = beta_df[beta_df['beta_name'].isin(BETAS_TO_TRACK)].copy()

    # Step 5: plot
    fig = plot_transition_betas(plot_df, hierarchy_df, save_path=save_path)

    return {
        'hierarchy':  hierarchy_df,
        'counts':     count_pivot,
        'betas_raw':  beta_df,    # always contains raw betas + angle
        'betas_plot': plot_df,    # what was actually plotted
        'figure':     fig,
    }

# ── SAVE RESULTS FOR DOWNSTREAM COMPARISON ────────────────────────────────────

import os, json
from datetime import datetime


def save_per_cell_results(out, save_dir, df_used=None,
                          normalize_flag=NORMALIZE_BY_ANGLE):
    """
    Save outputs of run_transition_analysis to disk for use in the
    comparison script (mixed model vs per-cell GLM vs raw empirical).

    Files written:
      - per_cell_betas.csv         long format, raw (unnormalized) betas
                                   columns: rat, transition, beta_name, beta
      - rat_hierarchy.csv          per-rat modality slopes and ranks
                                   columns: rat, slope_V, slope_T, slope_VT,
                                            stronger_unimodal, rank_T, rank_V,
                                            rank_VT
      - transition_dir_mapping.csv each (rat × transition) tagged as
                                   down / same / up using rat-specific rank.
                                   columns: rat, transition, rank_score,
                                            transition_dir
      - trial_counts.csv           per (rat × transition) cell trial counts
                                   columns: rat, transition, n_trials
      - meta.json                  metadata: normalize flag, timestamp, script
                                   path, etc.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n→ Saving per-cell GLM results to: {save_dir}")

    # --- 1. per-cell betas (raw, unnormalized) ---
    beta_df = out['betas_raw'].copy()
    beta_df.to_csv(os.path.join(save_dir, 'per_cell_betas.csv'), index=False)
    print(f"  ✓ per_cell_betas.csv         ({len(beta_df)} rows)")

    # --- 2. rat hierarchy (flatten the modality_rank dict into columns) ---
    hier = out['hierarchy'].copy()
    for mod in ['T', 'V', 'VT']:
        hier[f'rank_{mod}'] = hier['modality_rank'].apply(lambda d: d.get(mod, np.nan))
    hier = hier.drop(columns=['modality_rank'])  # drop the dict column
    hier.to_csv(os.path.join(save_dir, 'rat_hierarchy.csv'), index=False)
    print(f"  ✓ rat_hierarchy.csv          ({len(hier)} rats)")

    # --- 3. transition direction mapping per rat × cell ---
    scores = compute_transition_scores(out['hierarchy'], score_type='both')
    scores['transition_dir'] = scores['rank_score'].apply(
        lambda s: 'same' if s == 0 else ('down' if s > 0 else 'up')
    )
    scores.to_csv(os.path.join(save_dir, 'transition_dir_mapping.csv'),
                  index=False)
    print(f"  ✓ transition_dir_mapping.csv ({len(scores)} rows)")

    # --- 4. trial counts (from count_pivot) ---
    counts_long = (
        out['counts']
        .reset_index()
        .melt(id_vars=RAT_COL, var_name='transition', value_name='n_trials')
    )
    counts_long.to_csv(os.path.join(save_dir, 'trial_counts.csv'), index=False)
    print(f"  ✓ trial_counts.csv           ({len(counts_long)} rows)")

    # --- 5. metadata ---
    meta = {
        'timestamp':           datetime.now().isoformat(),
        'normalize_by_angle':  bool(normalize_flag),
        'min_trials_per_cell': MIN_TRIALS_PER_CELL,
        'features_cell':       FEATURES_CELL,
        'features_hierarchy':  FEATURES_HIERARCHY,
        'n_rats':              int(len(hier)),
        'rats':                hier[RAT_COL].tolist(),
        'all_transitions':     ALL_TRANSITIONS,
        'n_trials_used':       int(len(df_used)) if df_used is not None else None,
    }
    with open(os.path.join(save_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  ✓ meta.json")
    print(f"→ Done.")

# ── DESCRIPTIVE SUMMARY STEP ──────────────────────────────────────────────────
 
def compute_transition_scores(hierarchy_df, score_type='both'):
    """
    For each rat, compute a score for each of the 9 transition cells
    reflecting the direction and magnitude of the sensory transition.
 
    Two versions:
        'rank'       : score = rank(n-1) - rank(n)
                       Integer in {-2, -1, 0, 1, 2}. Equal spacing assumed.
        'continuous' : score = slope(n-1) - slope(n)
                       Real-valued. Uses actual psychometric slopes from
                       hierarchy_df, preserving quantitative distances.
        'both'       : compute both (default)
 
    Returns a DataFrame with columns:
        rat, transition, rank_score, continuous_score (depending on score_type)
    """
    rows = []
    for _, row in hierarchy_df.iterrows():
        rat          = row[RAT_COL]
        mod_rank     = row['modality_rank']      # e.g. {'T':1, 'V':2, 'VT':3}
        slope        = {                          # actual psychometric slopes
            'T':  row['slope_T'],
            'V':  row['slope_V'],
            'VT': row['slope_VT'],
        }
 
        for trans in ALL_TRANSITIONS:
            prev, curr = trans.split('->')
            entry = {RAT_COL: rat, 'transition': trans}
 
            if score_type in ('rank', 'both'):
                entry['rank_score'] = (
                    mod_rank.get(prev, np.nan) - mod_rank.get(curr, np.nan)
                )
            if score_type in ('continuous', 'both'):
                entry['continuous_score'] = (
                    slope.get(prev, np.nan) - slope.get(curr, np.nan)
                )
            rows.append(entry)
 
    return pd.DataFrame(rows)
 
 
def compute_descriptive_correlations(beta_df, hierarchy_df,
                                      betas_to_track=BETAS_TO_TRACK):
    """
    For each rat x beta_name, compute correlations between transition
    hierarchy score and beta value across the 9 cells.
 
    Four correlations per rat per beta:
        signed   Spearman(rank_score,       beta)
        signed   Pearson( continuous_score, beta)
        absolute Spearman(rank_score,       |beta|)
        absolute Pearson( continuous_score, |beta|)
 
    Also computes per-rat overall beta sign (mean beta > 0 = stay, < 0 = switch)
    for stratified reporting.
 
    Returns:
        corr_df : DataFrame with columns
            rat, beta_name, overall_sign,
            spearman_r,    spearman_p,
            pearson_r,     pearson_p,
            spearman_r_abs, spearman_p_abs,
            pearson_r_abs,  pearson_p_abs,
            n_cells
    """
    from scipy.stats import spearmanr, pearsonr
 
    scores_df = compute_transition_scores(hierarchy_df, score_type='both')
 
    plot_betas = beta_df[beta_df['beta_name'].isin(betas_to_track)].copy()
    merged = plot_betas.merge(scores_df, on=[RAT_COL, 'transition'], how='inner')
 
    rows = []
    for (rat, beta_name), gdf in merged.groupby([RAT_COL, 'beta_name']):
        gdf = gdf.dropna(subset=['rank_score', 'continuous_score', 'beta'])
 
        if len(gdf) < 5:
            print(f"  Warning: Rat {rat} | {beta_name}: only {len(gdf)} valid cells, skipping")
            continue
 
        # Overall sign: is this rat a stay-er or switcher for this beta?
        mean_beta    = gdf['beta'].mean()
        overall_sign = 'stay' if mean_beta > 0 else 'switch'
 
        # Signed correlations
        sp_r,     sp_p     = spearmanr(gdf['rank_score'],       gdf['beta'])
        pe_r,     pe_p     = pearsonr( gdf['continuous_score'], gdf['beta'])
 
        # Absolute correlations — tests magnitude modulation regardless of direction
        sp_r_abs, sp_p_abs = spearmanr(gdf['rank_score'],       gdf['beta'].abs())
        pe_r_abs, pe_p_abs = pearsonr( gdf['continuous_score'], gdf['beta'].abs())
 
        rows.append({
            RAT_COL:           rat,
            'beta_name':       beta_name,
            'overall_sign':    overall_sign,
            'mean_beta':       mean_beta,
            'spearman_r':      sp_r,
            'spearman_p':      sp_p,
            'pearson_r':       pe_r,
            'pearson_p':       pe_p,
            'spearman_r_abs':  sp_r_abs,
            'spearman_p_abs':  sp_p_abs,
            'pearson_r_abs':   pe_r_abs,
            'pearson_p_abs':   pe_p_abs,
            'n_cells':         len(gdf),
        })
 
    corr_df = pd.DataFrame(rows)
    _print_correlation_summary(corr_df)
    return corr_df
 
 
def _print_correlation_summary(corr_df):
    """Print a structured summary of correlations, stratified by overall sign."""
    from scipy.stats import ttest_1samp
 
    print("\n=== DESCRIPTIVE CORRELATIONS ===")
    print("Positive = history effect magnitude larger in downgrade transitions\n")
 
    for beta_name, bdf in corr_df.groupby('beta_name'):
        print(f"  {beta_name}  (n={len(bdf)} rats)")
 
        for sign_group, label in [(None, 'ALL RATS'), ('stay', 'STAY rats'), ('switch', 'SWITCH rats')]:
            gdf = bdf if sign_group is None else bdf[bdf['overall_sign'] == sign_group]
            if len(gdf) == 0:
                continue
            n = len(gdf)
            print(f"    [{label}  n={n}]")
 
            for r_col, p_col, method in [
                ('spearman_r',     'spearman_p',     'Signed   Spearman'),
                ('pearson_r',      'pearson_p',      'Signed   Pearson '),
                ('spearman_r_abs', 'spearman_p_abs', 'Absolute Spearman'),
                ('pearson_r_abs',  'pearson_p_abs',  'Absolute Pearson '),
            ]:
                vals = gdf[r_col].values
                t, p = ttest_1samp(vals, 0) if n > 1 else (np.nan, np.nan)
                p_str = f"t-test p={p:.3f}" if p >= 0.001 else "t-test p<0.001"
                print(f"      {method}: mean={vals.mean():.3f}  "
                      f"median={np.median(vals):.3f}  "
                      f"n_pos={( vals > 0).sum()}/{n}  {p_str}")
        print()
    print("================================\n")
 
 
def plot_descriptive_correlations(corr_df, hierarchy_df,
                                   save_path="descriptive_correlations.png"):
    """
    Two-panel figure per beta (success, failure):
 
    Left panel  — dot plot of Spearman r per rat (rank-based)
    Right panel — dot plot of Pearson  r per rat (continuous slope-based)
 
    Dots colored by rat's stronger unimodal modality (V or T).
    Vertical dashed line at 0. Population mean ± SEM shown as horizontal band.
    One-sample t-test against 0 reported on each panel.
    """
    from scipy.stats import ttest_1samp
 
    stronger_dict = hierarchy_df.set_index(RAT_COL)['stronger_unimodal'].to_dict()
    beta_names    = corr_df['beta_name'].unique()
 
    n_betas = len(beta_names)
    fig, axes = plt.subplots(
        n_betas, 2,
        figsize=(10, 4 * n_betas),
        constrained_layout=True
    )
    if n_betas == 1:
        axes = axes[np.newaxis, :]
 
    color_map = {'V': '#2196F3', 'T': '#FF5722', 'equal': '#9E9E9E', 'unknown': '#9E9E9E'}
 
    for row_idx, beta_name in enumerate(beta_names):
        bdf = corr_df[corr_df['beta_name'] == beta_name].copy()
        bdf['stronger'] = bdf[RAT_COL].map(stronger_dict)
 
        for col_idx, (r_col, label, method_label) in enumerate([
            ('spearman_r', 'Spearman r\n(rank-based score)',     'Rank'),
            ('pearson_r',  'Pearson r\n(continuous slope score)', 'Continuous'),
        ]):
            ax = axes[row_idx, col_idx]
 
            vals   = bdf[r_col].values
            colors = [color_map.get(bdf.iloc[i]['stronger'], '#9E9E9E')
                      for i in range(len(bdf))]
 
            # Population mean ± SEM band
            mean_r = np.mean(vals)
            sem_r  = np.std(vals, ddof=1) / np.sqrt(len(vals))
            ax.axhspan(mean_r - sem_r, mean_r + sem_r,
                       color='gold', alpha=0.3, zorder=0, label='Mean ± SEM')
            ax.axhline(mean_r, color='goldenrod', linewidth=1.5,
                       linestyle='-', zorder=1)
            ax.axhline(0, color='black', linewidth=0.8,
                       linestyle='--', alpha=0.5, zorder=0)
 
            # Individual rat dots
            x_jitter = np.random.uniform(-0.05, 0.05, size=len(vals))
            for xi, (r_val, color) in enumerate(zip(vals, colors)):
                ax.scatter(x_jitter[xi], r_val, color=color,
                           s=80, zorder=3, alpha=0.85)
                ax.text(x_jitter[xi] + 0.07, r_val,
                        str(bdf.iloc[xi][RAT_COL]),
                        fontsize=7, va='center', color='dimgray')
 
            # One-sample t-test vs 0
            t_stat, p_val = ttest_1samp(vals, 0)
            p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
            ax.set_title(
                f"{beta_name}  |  {method_label} score\n"
                f"t({len(vals)-1})={t_stat:.2f}, {p_str}",
                fontsize=9
            )
            ax.set_ylabel(label, fontsize=8)
            ax.set_xlim(-0.3, 0.5)
            ax.set_xticks([])
            ax.set_ylim(-1.1, 1.1)
            ax.axhline( 1, color='gray', linewidth=0.4, linestyle=':')
            ax.axhline(-1, color='gray', linewidth=0.4, linestyle=':')
 
    # Legend for rat color coding
    legend_elements = [
        mpatches.Patch(color='#2196F3', label='Stronger: V'),
        mpatches.Patch(color='#FF5722', label='Stronger: T'),
        mpatches.Patch(color='#9E9E9E', label='Equal / unknown'),
        mpatches.Patch(color='gold',    alpha=0.5, label='Mean ± SEM'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.03))
 
    fig.suptitle(
        "Per-rat correlation: transition hierarchy score vs history beta\n"
        "Positive = stronger history effect in downgrade transitions",
        fontsize=11
    )
 
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Descriptive correlation figure saved: {save_path}")
    return fig
 
 
def run_descriptive_summary(out, save_path="descriptive_correlations.png"):
    """
    Convenience wrapper — pass the output dict from run_transition_analysis.
 
    Usage:
        out      = run_transition_analysis(an, df)
        corr_df  = run_descriptive_summary(out)
    """
    corr_df = compute_descriptive_correlations(
        beta_df      = out['betas_raw'],
        hierarchy_df = out['hierarchy'],
    )
    fig = plot_descriptive_correlations(
        corr_df      = corr_df,
        hierarchy_df = out['hierarchy'],
        save_path    = save_path,
    )
    return corr_df
# ── USAGE ─────────────────────────────────────────────────────────────────────
# Paste this at the bottom of your existing script, after df is built:
#
# =============================
# LOAD + PREPROCESS
# =============================
an = SerialDependenceAnalyzer(history_depth=HISTORY_DEPTH, save_figures=False)
an.load_and_preprocess_data()
df = an.create_lagged_features()
df = filter_training_trials(df, method='criterion', criterion=0.80, min_consec=2)


df['action_n-1'] = df['action_n-1'] * 2 - 1
df['success_n-1'] = df['hit_n-1'] * df['action_n-1']
df['failure_n-1'] = (1 - df['hit_n-1']) * df['action_n-1']
df['angle']       = df['angle'] - 45
df['angle_n-1']   = df['angle_n-1'] - 45
df['tactile'] = df['mod'] == 1
# df['visual'] = df['mod'] == 2
df['visuotactile'] = df['mod'] == 3
df['angle_tactile'] = df['angle']*df['tactile']
df['angle_visuotactile'] = df['angle']*df['visuotactile']
# from transition_history_analysis import run_transition_analysis
out = run_transition_analysis(an, df, save_path="transition_history_betas.png")
corr_df = run_descriptive_summary(out, save_path="descriptive_correlations.png")

# Save outputs for downstream comparison with mixed-model results
SAVE_DIR = r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\per_cell_glm_results"
save_per_cell_results(out, SAVE_DIR, df_used=df)
# or inline:
# out = run_transition_analysis(an, df)
hierarchy_df = out['hierarchy']
beta_df      = out['betas']