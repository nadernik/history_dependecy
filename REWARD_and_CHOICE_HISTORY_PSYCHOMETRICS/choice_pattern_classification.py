"""
choice_pattern_classification.py
=================================
Add-on module for choice_effects_psychometric.py.

Adds the difference-of-differences test (D) alongside the existing
sum test (S), plus a formal per-rat classification of choice pattern
that mirrors the GLM β_success / β_failure rule.

Rationale
---------
The existing symmetry test computes  S = Δμ_rew + Δμ_unrew:
    S ≈ 0   → consistent with reward × choice (opposite signs cancel)
    S ≠ 0   → both effects point the same way (pure choice)

But  S ≈ 0  is also consistent with *no effect at all*
(both Δμ close to zero). To disambiguate we add the difference

    D = Δμ_rew − Δμ_unrew:
        D ≠ 0   → effect changes with reward       → NOT pure choice
        D ≈ 0   → effect is the same regardless    → pure choice or no effect

Combining S, D, and the individual Δμ CIs reproduces the GLM
classification rule at the psychometric level.

Integration
-----------
1. Place this file next to choice_effects_psychometric.py and add:

       from choice_pattern_classification import (
           compute_S_and_D,
           classify_choice_pattern,
           plot_choice_pattern_classification,
           summarise_classifications,
       )

2. In run_analysis(), replace the block that runs the current
   symmetry test with the block shown at the bottom of this file
   (see MAIN-LOOP INTEGRATION).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
FIGURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "figures_choice_effects")
TABLE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "tables_choice_effects")
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(TABLE_DIR,  exist_ok=True)


# =============================================================================
# TUNABLE PARAMETERS
# =============================================================================

N_INDEPENDENT_SAMPLES = 10_000     # paired resamples for S and D
SAME_SIGN_THRESHOLD   = 0.9        # fraction of paired resamples with
                                    # sign(Δμ_rew) == sign(Δμ_unrew) needed
                                    # to count as "same-sign majority"


# =============================================================================
# S and D distributions
# =============================================================================

def compute_S_and_D(delta_mu_rew, delta_mu_unrew,
                     n_samples=N_INDEPENDENT_SAMPLES, seed=None):
    """
    Build the sum (S) and difference (D) distributions via *independent*
    resampling of the two bootstrap distributions. Independence is the
    right assumption because the reward=1 and reward=0 fits use disjoint
    trials and independent bootstrap iterations — no pairing is justified.

    Returns
    -------
    dict with keys:
        S              : Δμ_rew + Δμ_unrew            (n_samples,)
        D              : Δμ_rew − Δμ_unrew            (n_samples,)
        rew_paired     : paired resample of Δμ_rew    (n_samples,)
        unrew_paired   : paired resample of Δμ_unrew  (n_samples,)
        p_same_sign    : fraction with sign(rew) == sign(unrew)
    """
    rng          = np.random.default_rng(seed)
    rew_paired   = rng.choice(delta_mu_rew,   size=n_samples, replace=True)
    unrew_paired = rng.choice(delta_mu_unrew, size=n_samples, replace=True)

    S = rew_paired + unrew_paired
    D = rew_paired - unrew_paired

    p_same_sign = float(np.mean(np.sign(rew_paired) == np.sign(unrew_paired)))

    return {
        'S':            S,
        'D':            D,
        'rew_paired':   rew_paired,
        'unrew_paired': unrew_paired,
        'p_same_sign':  p_same_sign,
    }


# =============================================================================
# Per-rat classification
# =============================================================================

def classify_choice_pattern(delta_mu_rew, delta_mu_unrew, joint,
                             same_sign_threshold=SAME_SIGN_THRESHOLD):
    """
    Classify choice pattern from bootstrap CIs on Δμ_rew, Δμ_unrew, S, D.

    Rule (psychometric analog of the GLM β_success / β_failure rule):
        reward_x_choice : D CI excludes 0, UNLESS both individual Δμ CIs
                          exclude 0 with same-sign majority (that would
                          be pure choice with different magnitudes).
        pure_choice     : D CI includes 0 AND at least one individual Δμ
                          CI excludes 0 AND same-sign majority.
        ambiguous       : otherwise.

    Returns a dict with the label and every intermediate quantity used.
    """
    rew_lo,   rew_hi   = np.percentile(delta_mu_rew,   [2.5, 97.5])
    unrew_lo, unrew_hi = np.percentile(delta_mu_unrew, [2.5, 97.5])
    S_lo,     S_hi     = np.percentile(joint['S'],     [2.5, 97.5])
    D_lo,     D_hi     = np.percentile(joint['D'],     [2.5, 97.5])

    rew_excludes_0   = not (rew_lo   <= 0 <= rew_hi)
    unrew_excludes_0 = not (unrew_lo <= 0 <= unrew_hi)
    S_excludes_0     = not (S_lo     <= 0 <= S_hi)
    D_excludes_0     = not (D_lo     <= 0 <= D_hi)

    same_sign_majority = joint['p_same_sign'] >= same_sign_threshold

    if D_excludes_0:
        # D excludes 0 → effect differs across reward conditions.
        # Only override to pure_choice if BOTH individual effects are real
        # AND same sign (asymmetric-magnitude pure choice edge case).
        if rew_excludes_0 and unrew_excludes_0 and same_sign_majority:
            label = 'pure_choice'
        else:
            label = 'reward_x_choice'
    else:
        # D includes 0 → same effect regardless of reward.
        # pure_choice requires at least one real effect AND same sign.
        if (rew_excludes_0 or unrew_excludes_0) and same_sign_majority:
            label = 'pure_choice'
        else:
            label = 'ambiguous'

    return {
        'classification':      label,
        'rew_ci':              (float(rew_lo),   float(rew_hi)),
        'unrew_ci':            (float(unrew_lo), float(unrew_hi)),
        'S_ci':                (float(S_lo),     float(S_hi)),
        'D_ci':                (float(D_lo),     float(D_hi)),
        'rew_excludes_0':      bool(rew_excludes_0),
        'unrew_excludes_0':    bool(unrew_excludes_0),
        'S_excludes_0':        bool(S_excludes_0),
        'D_excludes_0':        bool(D_excludes_0),
        'p_same_sign':         float(joint['p_same_sign']),
        'same_sign_majority':  bool(same_sign_majority),
        'same_sign_threshold': float(same_sign_threshold),
    }


# =============================================================================
# Plotting
# =============================================================================

_CLASSIFICATION_COLORS = {
    'reward_x_choice': '#F32128',
    'pure_choice':     '#5C36F4',
    'ambiguous':       '#888888',
}


def plot_choice_pattern_classification(delta_mu_rew, delta_mu_unrew,
                                        joint, classification,
                                        rat_id,
                                        n_bootstrap_rew, n_bootstrap_unrew,
                                        savefig=True):
    """
    2×2 figure summarising the classification test.

        (0,0) Δμ per reward condition with 95% bootstrap CI.
        (0,1) Joint scatter of paired (independent) resamples in
              (Δμ_rew, Δμ_unrew) space, with the two reference diagonals:
                * y = −x  (perfect reward × choice: opposite signs, equal mag.)
                * y =  x  (perfect pure choice: same signs, equal magnitude)
              Cloud in quadrants II or IV → opposite signs → reward × choice.
              Cloud along y=x           → same signs      → pure choice.
        (1,0) S = Δμ_rew + Δμ_unrew  distribution with 95% CI and null line.
              CI excluding 0 → same-direction effect (pure choice signature).
        (1,1) D = Δμ_rew − Δμ_unrew  distribution with 95% CI and null line.
              CI excluding 0 → reward-dependent effect (rules out pure choice).
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    label = classification['classification']
    label_color = _CLASSIFICATION_COLORS[label]

    # ------------------------------------------------------------------
    # (0,0) Δμ per condition
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    for i, (side, data, color) in enumerate([
        ('prev rewarded',   delta_mu_rew,   '#21B04A'),
        ('prev unrewarded', delta_mu_unrew, '#B026A0'),
    ]):
        med   = np.median(data)
        ci_lo = np.percentile(data,  2.5)
        ci_hi = np.percentile(data, 97.5)
        ax.errorbar(
            x=i, y=med,
            yerr=[[med - ci_lo], [ci_hi - med]],
            fmt='o', color=color, capsize=8, markersize=10, lw=2,
        )
        ax.text(
            i, ci_hi + 0.02 * (abs(ci_hi) + 1),
            f'{med:.2f}°\n[{ci_lo:.2f}, {ci_hi:.2f}]',
            ha='center', va='bottom', fontsize=9,
        )
    ax.axhline(0, color='gray', linestyle='--', alpha=0.6)
    ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['prev rewarded', 'prev unrewarded'], fontsize=11)
    ax.set_ylabel('Δμ  (n-1 vertical − n-1 horizontal) [°]', fontsize=12)
    ax.set_title('Δμ per reward condition (95% bootstrap CI)', fontsize=11)

    # ------------------------------------------------------------------
    # (0,1) Joint scatter of paired independent resamples
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    ax.scatter(joint['rew_paired'], joint['unrew_paired'],
               s=4, alpha=0.15, color=label_color, rasterized=True)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.6)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.6)

    lim = max(
        float(np.max(np.abs(np.percentile(joint['rew_paired'],   [1, 99])))),
        float(np.max(np.abs(np.percentile(joint['unrew_paired'], [1, 99])))),
        1.0,
    )
    ax.plot([-lim, lim], [ lim, -lim], color='red',  linestyle=':', alpha=0.7,
            label='y = −x  (pure reward × choice)')
    ax.plot([-lim, lim], [-lim,  lim], color='blue', linestyle=':', alpha=0.7,
            label='y =  x  (pure choice)')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('Δμ  (prev rewarded) [°]',   fontsize=12)
    ax.set_ylabel('Δμ  (prev unrewarded) [°]', fontsize=12)
    ax.set_title(
        f'Joint (independent) resample cloud\n'
        f'p(same sign) = {joint["p_same_sign"]:.2f}   '
        f'threshold = {classification["same_sign_threshold"]:.2f}',
        fontsize=10,
    )
    ax.legend(fontsize=8, loc='upper left')
    ax.set_aspect('equal')

    # ------------------------------------------------------------------
    # (1,0) S histogram — pure_choice signature
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    S = joint['S']
    med   = float(np.median(S))
    ci_lo = classification['S_ci'][0]
    ci_hi = classification['S_ci'][1]
    ax.hist(S, bins=80, color='steelblue', alpha=0.7, density=True)
    ax.axvline(0,     color='red',   linestyle='--', lw=2, label='null (S=0)')
    ax.axvline(med,   color='black', linestyle='-',  lw=2,
               label=f'median = {med:.2f}°')
    ax.axvline(ci_lo, color='black', linestyle=':',  lw=1.5, alpha=0.7)
    ax.axvline(ci_hi, color='black', linestyle=':',  lw=1.5, alpha=0.7,
               label=f'95% CI [{ci_lo:.2f}, {ci_hi:.2f}]')
    verdict = ('CI includes 0 → consistent with R×C or no effect'
               if not classification['S_excludes_0']
               else 'CI excludes 0 → same-direction effect (pure choice)')
    ax.set_xlabel('S = Δμ_rew + Δμ_unrew [°]', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Sum test — pure_choice signature\n{verdict}', fontsize=10)
    ax.legend(fontsize=9)

    # ------------------------------------------------------------------
    # (1,1) D histogram — reward × choice signature
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    D = joint['D']
    med   = float(np.median(D))
    ci_lo = classification['D_ci'][0]
    ci_hi = classification['D_ci'][1]
    ax.hist(D, bins=80, color='indianred', alpha=0.7, density=True)
    ax.axvline(0,     color='red',   linestyle='--', lw=2, label='null (D=0)')
    ax.axvline(med,   color='black', linestyle='-',  lw=2,
               label=f'median = {med:.2f}°')
    ax.axvline(ci_lo, color='black', linestyle=':',  lw=1.5, alpha=0.7)
    ax.axvline(ci_hi, color='black', linestyle=':',  lw=1.5, alpha=0.7,
               label=f'95% CI [{ci_lo:.2f}, {ci_hi:.2f}]')
    verdict = ('CI includes 0 → same effect regardless of reward'
               if not classification['D_excludes_0']
               else 'CI excludes 0 → effect changes with reward (R×C signature)')
    ax.set_xlabel('D = Δμ_rew − Δμ_unrew [°]', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Difference test — reward × choice signature\n{verdict}',
                 fontsize=10)
    ax.legend(fontsize=9)

    # ------------------------------------------------------------------
    # Suptitle with the classification result
    # ------------------------------------------------------------------
    plt.suptitle(
        f'Rat {rat_id}  |  choice pattern classification  →  '
        f'{label.upper().replace("_", " ")}\n'
        f'rew: {n_bootstrap_rew} boot iters,   '
        f'unrew: {n_bootstrap_unrew} boot iters,   '
        f'{N_INDEPENDENT_SAMPLES:,} paired resamples',
        fontsize=13, color=label_color, y=1.00,
    )
    plt.tight_layout()

    if savefig:
        stem = f'rat_{rat_id}_choice_pattern_classification'
    for ext in ('png', 'svg'):
        path = os.path.join(FIGURE_DIR, f'{stem}.{ext}')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Figure saved: {path}")
    plt.close(fig)


# =============================================================================
# Group-level summary
# =============================================================================

def summarise_classifications(all_results, key='choice_pattern_classification'):
    """
    Print a compact per-rat table and the group-level count of
    classifications. Call once at the end of run_analysis.
    """
    rows = []
    for rat, res in all_results.items():
        if key not in res:
            continue
        r = res[key]
        rows.append({
            'rat':           rat,
            'class':         r['classification'],
            'Δμ_rew':        r['delta_mu_rew_median'],
            'Δμ_unrew':      r['delta_mu_unrew_median'],
            'S':             r['S_median'],
            'S_CI':          f"[{r['S_ci_low']:.2f}, {r['S_ci_high']:.2f}]",
            'D':             r['D_median'],
            'D_CI':          f"[{r['D_ci_low']:.2f}, {r['D_ci_high']:.2f}]",
            'p(same sign)':  r['p_same_sign'],
        })

    if not rows:
        print("  (no classifications to summarise)")
        return

    import pandas as pd
    df = pd.DataFrame(rows).set_index('rat')
    print("\n" + "=" * 70)
    print("Per-rat choice pattern classification")
    print("=" * 70)
    print(df.to_string(float_format='{:.2f}'.format))

    print("\nGroup-level counts:")
    counts = df['class'].value_counts()
    total = len(df)
    for label in ('reward_x_choice', 'pure_choice', 'ambiguous'):
        n = int(counts.get(label, 0))
        print(f"  {label:20s}: {n:2d}/{total}  ({100 * n / total:.0f}%)")
    print("=" * 70)


# =============================================================================
# MAIN-LOOP INTEGRATION
# =============================================================================
"""
In run_analysis() inside choice_effects_psychometric.py, replace the
existing block that runs the symmetry test with this one.

The trigger is the same: run once, after analysis_4 completes, provided
both reward conditions produced valid bootstraps.

----------------------------------------------------------------------
    elif analysis['label'] == 'analysis_4_fake45_reward_controlled_opposite':
        delta_mus['unrew']  = delta_mu
        boot_valid['unrew'] = n_valid

        if 'rew' in delta_mus and 'unrew' in delta_mus:
            print(f"\\n  --- Choice pattern classification (S + D) ---")

            # 1. build paired S and D distributions (independent resampling)
            joint = compute_S_and_D(delta_mus['rew'], delta_mus['unrew'])

            # 2. classify per-rat (mirrors GLM β_success/β_failure rule)
            classification = classify_choice_pattern(
                delta_mus['rew'], delta_mus['unrew'], joint,
            )

            # 3. plot 2×2 summary
            plot_choice_pattern_classification(
                delta_mus['rew'], delta_mus['unrew'],
                joint, classification,
                rat_id=rat,
                n_bootstrap_rew=boot_valid.get('rew',   0),
                n_bootstrap_unrew=boot_valid.get('unrew', 0),
            )

            # 4. store
            all_results[rat]['choice_pattern_classification'] = {
                'classification':        classification['classification'],
                'delta_mu_rew_median':   float(np.median(delta_mus['rew'])),
                'delta_mu_unrew_median': float(np.median(delta_mus['unrew'])),
                'S_median':              float(np.median(joint['S'])),
                'S_ci_low':              classification['S_ci'][0],
                'S_ci_high':             classification['S_ci'][1],
                'D_median':              float(np.median(joint['D'])),
                'D_ci_low':              classification['D_ci'][0],
                'D_ci_high':             classification['D_ci'][1],
                'p_same_sign':           classification['p_same_sign'],
                'D_excludes_0':          classification['D_excludes_0'],
                'S_excludes_0':          classification['S_excludes_0'],
                'rew_excludes_0':        classification['rew_excludes_0'],
                'unrew_excludes_0':      classification['unrew_excludes_0'],
            }
            S_ci = classification['S_ci']
            D_ci = classification['D_ci']
            print(f"  S = {np.median(joint['S']):+.2f}°  "
                  f"[{S_ci[0]:+.2f}, {S_ci[1]:+.2f}]  "
                  f"({'excludes' if classification['S_excludes_0'] else 'includes'} 0)")
            print(f"  D = {np.median(joint['D']):+.2f}°  "
                  f"[{D_ci[0]:+.2f}, {D_ci[1]:+.2f}]  "
                  f"({'excludes' if classification['D_excludes_0'] else 'includes'} 0)")
            print(f"  p(same sign) = {classification['p_same_sign']:.2f}")
            print(f"  → {classification['classification']}\\n")
        else:
            print(f"  ⚠ Classification skipped: one or both reward "
                  f"conditions missing\\n")

# ... and at the very end of run_analysis(), just before `return all_results`:
    summarise_classifications(all_results)
----------------------------------------------------------------------
"""