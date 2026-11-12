"""
Modality Transition History Effect Analysis -- ADDITIVE DECOMPOSITION variant
==============================================================================

Question: does the effect of the previous trial's outcome (beta_success_n-1,
beta_failure_n-1, from a per-rat GLM) change with the modality transition
(T/V/VT on trial n-1 -> trial n)? Each rat's modalities are ranked by
sensitivity (hierarchy model), the betas are fitted separately in each of the 9
transition cells, and plotted ordered by that hierarchy (downgrade / same /
upgrade transitions).

SIBLING SCRIPT: the two scripts share the same core (hierarchy fit, per-cell
GLMs, beta plot) and differ only in what they do with the betas afterwards.

  THIS script (additive_decomposition): raw log-odds betas
    (NORMALIZE_BY_ANGLE = False); decomposes the transition effect into
    additive components (compute_additive_decomposition); saves figures only.
  modality_transition_glm_descriptive_correlations.py: betas normalized by
    beta_angle (NORMALIZE_BY_ANGLE = True); correlates the betas with the
    modality hierarchy scores; saves per-cell results to per_cell_glm_results/.
"""
'''
SAFELY CODING: 

START session: 

- check branch --> on the bottom left or run 'git branch' in terminal
-UPDATE branch --> 
git checkout main (it goes to main branch)
git pull (it updates my version of main branch to the current one if anybody changed it)
- UPDATE your working branch -->
git checkout feature/exploratory_psych_curves --> goes to your working branch
git merge main --> merges main into your working branch, so that on my branch I have the latest updates from main

START CODING


END session:
(optional) check changes made: git status
git add exploratory_psych_curves.py --> stages the changes made to this file
git commit -m "progress" --> makes a local checkpoint in your branch.
(only the first time) git push -u origin feature/exploratory_psych_curves --> pushes the changes to my remote working branch
git push --> from the second time onwards
'''

# =============================
# SESSION LENGTH DISTRIBUTION PER RAT
# =============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy")
from functions_Ale import filter_training_trials
from serial_dependence_analysis import SerialDependenceAnalyzer

# =============================
# LOAD + PREPROCESS (same pipeline as your main script)
# =============================
an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
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

# ── CONFIG ────────────────────────────────────────────────────────────────────
RAT_COL            = "rat"
MOD_COL            = "mod"                  # 1=T, 2=V, 3=VT
TRANSITION_COL     = "mod_transition_n-1"   # already in your df
MIN_TRIALS_PER_CELL = 200                   # flag cells below this

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

# ── NORMALIZATION FLAG ────────────────────────────────────────────────────────
# If True, divide each rat's history betas by its beta_angle from the same
# cell model. This expresses history effects in units of 'stimulus equivalents'
# and removes the cross-rat comparability problem from logistic regression's
# fixed residual variance assumption (Mood 2010, Allison 1999).
# If False, raw logistic coefficients are used (fine for within-rat comparisons).
NORMALIZE_BY_ANGLE = False

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


# ── USAGE ─────────────────────────────────────────────────────────────────────
# Paste this at the bottom of your existing script, after df is built:
#
# from transition_history_analysis import run_transition_analysis
# out = run_transition_analysis(an, df, save_path="transition_history_betas.png")
#
# or inline:
# out = run_transition_analysis(an, df)
# hierarchy_df = out['hierarchy']
# beta_df      = out['betas']


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
    4-column figure per beta (success, failure):
        Col 1 - Signed   Spearman r  (rank score)
        Col 2 - Signed   Pearson  r  (continuous score)
        Col 3 - Absolute Spearman r  (rank score)
        Col 4 - Absolute Pearson  r  (continuous score)
    Dots: color = stay (green) vs switch (red); shape = stronger modality.
    """
    from scipy.stats import ttest_1samp

    stronger_dict = hierarchy_df.set_index(RAT_COL)["stronger_unimodal"].to_dict()
    beta_names    = corr_df["beta_name"].unique()
    n_betas       = len(beta_names)

    col_specs = [
        ("spearman_r",     "Signed Spearman r (rank score)"),
        ("pearson_r",      "Signed Pearson r (continuous score)"),
        ("spearman_r_abs", "Absolute Spearman r (rank score)"),
        ("pearson_r_abs",  "Absolute Pearson r (continuous score)"),
    ]

    fig, axes = plt.subplots(
        n_betas, 4,
        figsize=(14, 4.5 * n_betas),
        constrained_layout=True
    )
    if n_betas == 1:
        axes = axes[np.newaxis, :]

    sign_color = {"stay": "#2ecc71", "switch": "#e74c3c"}
    mod_marker = {"V": "o", "T": "s", "equal": "^", "unknown": "^"}
    np.random.seed(42)

    for row_idx, beta_name in enumerate(beta_names):
        bdf = corr_df[corr_df["beta_name"] == beta_name].copy()
        bdf["stronger"] = bdf[RAT_COL].map(stronger_dict)

        for col_idx, (r_col, ylabel) in enumerate(col_specs):
            ax     = axes[row_idx, col_idx]
            vals   = bdf[r_col].values
            is_abs = "_abs" in r_col

            mean_r = np.mean(vals)
            sem_r  = np.std(vals, ddof=1) / np.sqrt(len(vals))
            ax.axhspan(mean_r - sem_r, mean_r + sem_r,
                       color="gold", alpha=0.35, zorder=0)
            ax.axhline(mean_r, color="goldenrod", linewidth=1.5, zorder=1)
            ax.axhline(0, color="black", linewidth=0.8,
                       linestyle="--", alpha=0.5)

            x_jitter = np.random.uniform(-0.08, 0.08, size=len(bdf))
            for xi, row in enumerate(bdf.itertuples()):
                color  = sign_color.get(row.overall_sign, "gray")
                marker = mod_marker.get(row.stronger, "o")
                r_val  = getattr(row, r_col)
                ax.scatter(x_jitter[xi], r_val,
                           color=color, marker=marker,
                           s=90, zorder=3, alpha=0.9,
                           edgecolors="white", linewidths=0.5)
                ax.text(x_jitter[xi] + 0.1, r_val,
                        str(getattr(row, RAT_COL)),
                        fontsize=6.5, va="center", color="dimgray")

            t_stat, p_val = ttest_1samp(vals, 0)
            p_str      = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
            title_beta = "|beta|" if is_abs else "beta"
            ax.set_title(
                f"{beta_name} - {title_beta}\nt({len(vals)-1})={t_stat:.2f}, {p_str}",
                fontsize=8
            )
            ax.set_ylabel(ylabel, fontsize=7.5)
            ax.set_xlim(-0.4, 0.6)
            ax.set_xticks([])
            ax.set_ylim(-1.15, 1.15)
            for hl in [-1, 1]:
                ax.axhline(hl, color="gray", linewidth=0.4, linestyle=":")
            if is_abs:
                ax.set_facecolor("#f7f7f7")

    legend_elements = [
        mpatches.Patch(color="#2ecc71", label="Stay (mean beta > 0)"),
        mpatches.Patch(color="#e74c3c", label="Switch (mean beta < 0)"),
        plt.Line2D([0],[0], marker="o", color="gray", linestyle="None",
                   markersize=7, label="Stronger: V"),
        plt.Line2D([0],[0], marker="s", color="gray", linestyle="None",
                   markersize=7, label="Stronger: T"),
        mpatches.Patch(color="gold", alpha=0.5, label="Mean +/- SEM"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        "Per-rat correlation: transition hierarchy score vs history beta\n"
        "Cols 1-2: signed beta  |  Cols 3-4: |beta| magnitude\n"
        "Green=stay  Red=switch  Circle=stronger V  Square=stronger T",
        fontsize=10
    )

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
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


# ── ADDITIVE DECOMPOSITION: strength(n) vs strength(n-1) ─────────────────────

def compute_additive_decomposition(beta_df, hierarchy_df,
                                    betas_to_track=BETAS_TO_TRACK):
    """
    For each rat x beta_name, fit a weighted OLS regression:

        beta ~ strength(n) + strength(n-1)

    across the 9 transition cells, using actual psychometric slopes as
    strength values (from hierarchy_df: slope_V, slope_T, slope_VT).

    Weights = 1 / n_trials_in_cell as a proxy for beta precision
    (proper weighting would use 1/SE^2 but SE requires bootstrap).

    This is EXPLORATORY — see methodological caveats in comments.
    The formal inferential test belongs in the mixed model on raw data.

    Returns:
        decomp_df : DataFrame with columns
            rat, beta_name, overall_sign,
            coef_strength_n,    coef_strength_n1,
            coef_strength_n_se, coef_strength_n1_se,
            r_squared, n_cells
    """
    from sklearn.linear_model import LinearRegression

    # Build slope lookup per rat
    slope_lookup = {}
    for _, row in hierarchy_df.iterrows():
        slope_lookup[row[RAT_COL]] = {
            'T':  row['slope_T'],
            'V':  row['slope_V'],
            'VT': row['slope_VT'],
        }

    plot_betas = beta_df[beta_df['beta_name'].isin(betas_to_track)].copy()

    # Add strength(n) and strength(n-1) to each row
    def get_strengths(row):
        rat    = row[RAT_COL]
        trans  = row['transition']
        slopes = slope_lookup.get(rat, {})
        parts  = trans.split('->')
        if len(parts) != 2:
            return pd.Series({'strength_n1': np.nan, 'strength_n': np.nan})
        prev, curr = parts
        return pd.Series({
            'strength_n1': slopes.get(prev, np.nan),
            'strength_n':  slopes.get(curr, np.nan),
        })

    plot_betas = plot_betas.join(
        plot_betas.apply(get_strengths, axis=1)
    )

    rows = []
    for (rat, beta_name), gdf in plot_betas.groupby([RAT_COL, 'beta_name']):
        gdf = gdf.dropna(subset=['strength_n', 'strength_n1', 'beta'])

        if len(gdf) < 5:
            print(f"  Warning: Rat {rat} | {beta_name}: only {len(gdf)} cells, skipping")
            continue

        X = gdf[['strength_n', 'strength_n1']].values
        y = gdf['beta'].values

        # OLS via numpy for coefficient SE access
        # Add intercept manually
        X_int = np.column_stack([np.ones(len(X)), X])
        try:
            # OLS: (X'X)^-1 X'y
            coeffs, residuals, rank, sv = np.linalg.lstsq(X_int, y, rcond=None)
            y_hat    = X_int @ coeffs
            ss_res   = np.sum((y - y_hat) ** 2)
            ss_tot   = np.sum((y - y.mean()) ** 2)
            r_sq     = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            df_resid = len(y) - X_int.shape[1]  # n - p

            if df_resid > 0:
                sigma_sq = ss_res / df_resid
                cov      = sigma_sq * np.linalg.inv(X_int.T @ X_int)
                ses      = np.sqrt(np.diag(cov))
            else:
                ses = np.full(len(coeffs), np.nan)

            mean_beta    = gdf['beta'].mean()
            overall_sign = 'stay' if mean_beta > 0 else 'switch'

            rows.append({
                RAT_COL:              rat,
                'beta_name':          beta_name,
                'overall_sign':       overall_sign,
                'mean_beta':          mean_beta,
                'coef_intercept':     coeffs[0],
                'coef_strength_n':    coeffs[1],   # Hypothesis A
                'coef_strength_n1':   coeffs[2],   # Hypothesis B
                'se_strength_n':      ses[1],
                'se_strength_n1':     ses[2],
                'r_squared':          r_sq,
                'n_cells':            len(gdf),
            })
        except np.linalg.LinAlgError as e:
            print(f"  Rat {rat} | {beta_name}: regression failed ({e})")

    decomp_df = pd.DataFrame(rows)
    _print_decomposition_summary(decomp_df)
    return decomp_df


def _print_decomposition_summary(decomp_df):
    """Print summary of additive decomposition coefficients."""
    from scipy.stats import ttest_1samp

    print("\n=== ADDITIVE DECOMPOSITION: strength(n) vs strength(n-1) ===")
    print("Negative coef_strength_n  -> history stronger when current is WEAK  (Hyp A)")
    print("Positive coef_strength_n1 -> history stronger when previous is STRONG (Hyp B)\n")

    for beta_name, bdf in decomp_df.groupby('beta_name'):
        print(f"  {beta_name}  (n={len(bdf)} rats, mean R2={bdf['r_squared'].mean():.3f})")

        for sign_group, label in [
            (None,     'ALL RATS'),
            ('stay',   'STAY rats'),
            ('switch', 'SWITCH rats'),
        ]:
            gdf = bdf if sign_group is None else bdf[bdf['overall_sign'] == sign_group]
            if len(gdf) == 0:
                continue
            n = len(gdf)

            for col, hyp_label in [
                ('coef_strength_n',  'strength(n)  [Hyp A]'),
                ('coef_strength_n1', 'strength(n-1)[Hyp B]'),
            ]:
                vals = gdf[col].values
                t, p = ttest_1samp(vals, 0) if n > 1 else (np.nan, np.nan)
                p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
                print(f"    [{label} n={n}] {hyp_label}: "
                      f"mean={vals.mean():.3f}  "
                      f"median={np.median(vals):.3f}  "
                      f"n_pos={(vals > 0).sum()}/{n}  {p_str}")
        print()
    print("Note: exploratory only — formal test belongs in mixed model on raw data")
    print("============================================================\n")


def plot_additive_decomposition(decomp_df, hierarchy_df,
                                 save_path="additive_decomposition.png"):
    """
    2-column x n_betas figure.

    Left col  — coef_strength_n   (current trial modality effect)
    Right col — coef_strength_n1  (previous trial modality effect)

    One dot per rat. Color = stay (green) / switch (red).
    Shape = stronger modality (circle=V, square=T).
    Gold band = mean +/- SEM. t-test vs 0 in title.
    Dashed vertical line at 0.
    """
    from scipy.stats import ttest_1samp

    stronger_dict = hierarchy_df.set_index(RAT_COL)['stronger_unimodal'].to_dict()
    beta_names    = decomp_df['beta_name'].unique()
    n_betas       = len(beta_names)

    fig, axes = plt.subplots(
        n_betas, 2,
        figsize=(9, 4.5 * n_betas),
        constrained_layout=True
    )
    if n_betas == 1:
        axes = axes[np.newaxis, :]

    sign_color = {'stay': '#2ecc71', 'switch': '#e74c3c'}
    mod_marker = {'V': 'o', 'T': 's', 'equal': '^', 'unknown': '^'}
    np.random.seed(42)

    col_specs = [
        ('coef_strength_n',  'se_strength_n',
         'coef strength(n)\n[current trial]',
         'Negative = history stronger\nwhen current is WEAK'),
        ('coef_strength_n1', 'se_strength_n1',
         'coef strength(n-1)\n[previous trial]',
         'Positive = history stronger\nwhen previous is STRONG'),
    ]

    for row_idx, beta_name in enumerate(beta_names):
        bdf = decomp_df[decomp_df['beta_name'] == beta_name].copy()
        bdf['stronger'] = bdf[RAT_COL].map(stronger_dict)

        for col_idx, (coef_col, se_col, xlabel, direction_note) in enumerate(col_specs):
            ax   = axes[row_idx, col_idx]
            vals = bdf[coef_col].values
            ses  = bdf[se_col].values

            # Mean +/- SEM band across rats
            mean_v = np.mean(vals)
            sem_v  = np.std(vals, ddof=1) / np.sqrt(len(vals))
            ax.axvspan(mean_v - sem_v, mean_v + sem_v,
                       color='gold', alpha=0.35, zorder=0)
            ax.axvline(mean_v, color='goldenrod', linewidth=1.5, zorder=1)
            ax.axvline(0, color='black', linewidth=0.8,
                       linestyle='--', alpha=0.5)

            # Individual rats — horizontal forest-plot style
            y_jitter = np.random.uniform(-0.08, 0.08, size=len(bdf))
            for xi, row in enumerate(bdf.itertuples()):
                color  = sign_color.get(row.overall_sign, 'gray')
                marker = mod_marker.get(row.stronger, 'o')
                coef   = getattr(row, coef_col)
                se     = getattr(row, se_col)
                y_pos  = xi + y_jitter[xi]

                # Error bar (1 SE)
                ax.errorbar(coef, y_pos, xerr=se,
                            fmt='none', color=color, alpha=0.4,
                            capsize=2, linewidth=1, zorder=2)
                ax.scatter(coef, y_pos,
                           color=color, marker=marker,
                           s=80, zorder=3, alpha=0.9,
                           edgecolors='white', linewidths=0.5)
                ax.text(coef + 0.01, y_pos,
                        str(getattr(row, RAT_COL)),
                        fontsize=6.5, va='center', color='dimgray')

            t_stat, p_val = ttest_1samp(vals, 0)
            p_str = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"

            ax.set_title(
                f"{beta_name}\n{coef_col}\n"
                f"t({len(vals)-1})={t_stat:.2f}, {p_str}",
                fontsize=8
            )
            ax.set_xlabel(direction_note, fontsize=7.5)
            ax.set_yticks([])
            ax.set_ylim(-1, len(bdf))

    legend_elements = [
        mpatches.Patch(color='#2ecc71', label='Stay (mean beta > 0)'),
        mpatches.Patch(color='#e74c3c', label='Switch (mean beta < 0)'),
        plt.Line2D([0],[0], marker='o', color='gray', linestyle='None',
                   markersize=7, label='Stronger: V'),
        plt.Line2D([0],[0], marker='s', color='gray', linestyle='None',
                   markersize=7, label='Stronger: T'),
        mpatches.Patch(color='gold', alpha=0.5, label='Mean +/- SEM'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        "Additive decomposition: what drives history modulation?\n"
        "Left = current trial modality  |  Right = previous trial modality\n"
        "Error bars = 1 SE (within-rat OLS)  |  EXPLORATORY",
        fontsize=10
    )

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Additive decomposition figure saved: {save_path}")
    return fig


def run_additive_decomposition(out, save_path="additive_decomposition.png"):
    """
    Convenience wrapper — pass the output dict from run_transition_analysis.

    Usage:
        out       = run_transition_analysis(an, df)
        decomp_df = run_additive_decomposition(out)
    """
    decomp_df = compute_additive_decomposition(
        beta_df      = out['betas_raw'],
        hierarchy_df = out['hierarchy'],
    )
    fig = plot_additive_decomposition(
        decomp_df    = decomp_df,
        hierarchy_df = out['hierarchy'],
        save_path    = save_path,
    )
    return decomp_df


# ── USAGE ─────────────────────────────────────────────────────────────────────
# Paste this at the bottom of your existing script, after df is built:
#
#from transition_history_analysis import run_transition_analysis
out = run_transition_analysis(an, df, save_path="transition_history_betas.png")
#corr_df   = run_descriptive_summary(out)       # correlation analysis (existing)
decomp_df = run_additive_decomposition(out)    # new decomposition
#
# or inline:
# out = run_transition_analysis(an, df)
# hierarchy_df = out['hierarchy']
# beta_df      = out['betas']


