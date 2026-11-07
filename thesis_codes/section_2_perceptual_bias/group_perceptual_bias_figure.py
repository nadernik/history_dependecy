"""
group_perceptual_bias_figure.py
================================
Cohort-level forest plot of primary ΔPSE (μ_easy − μ_hard) with 95%
bootstrap CIs, one panel per modality, all rats on the vertical axis.
Reads details_perceptual_bias.csv produced by the per-cell script.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- config ---------------------------------------------------------------
CSV_PATH   = r'C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\section_2_perceptual_bias\results\tables_perceptual_bias\details_perceptual_bias.csv'    # from the per-cell script
OUT_STEM   = 'group_perceptual_bias_forest'
OUT_DIR    = r'C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_codes\ale_thesis_fig_tab\images_thesis\section_2'
MOD_NAMES  = {1: 'Modality 1', 2: 'Modality 2', 3: 'Modality 3'}  # rename as needed
DETECTED_C = 'crimson'
NULL_C     = '#4a7ba8'    # muted steel blue
ZERO_C     = 'black'

# ---- load + reduce to primary contrast ------------------------------------
df   = pd.read_csv(CSV_PATH)
prim = df[df.is_primary].copy()

rats       = sorted(prim['rat'].unique())
modalities = sorted(prim['modality'].unique())
y_pos      = {r: i for i, r in enumerate(rats)}

# per-modality x-limits — mod 1 has genuinely much wider CIs than mod 2/3,
# and sharing x would squash the informative range of the tighter modalities.
def _xlim(sub):
    lo = min(sub['delta_pse_ci_lo'].min(), 0)
    hi = max(sub['delta_pse_ci_hi'].max(), 0)
    span = max(abs(lo), abs(hi)) * 1.08
    return (-span, span)

# ---- figure ---------------------------------------------------------------
fig, axes = plt.subplots(
    1, len(modalities),
    figsize=(4.6 * len(modalities), 0.42 * len(rats) + 1.6),
    sharey=True,
    constrained_layout=False,   # manual layout to keep legend clear
)

for ax, mod in zip(axes, modalities):
    sub = prim[prim.modality == mod].set_index('rat')
    for rat in rats:
        if rat not in sub.index:
            continue
        row      = sub.loc[rat]
        detected = bool(row['delta_pse_excl_0'])
        color    = DETECTED_C if detected else NULL_C
        lw       = 2.0 if detected else 1.2
        ms       = 7 if detected else 5

        ax.errorbar(
            row['delta_pse'], y_pos[rat],
            xerr=[[row['delta_pse'] - row['delta_pse_ci_lo']],
                  [row['delta_pse_ci_hi'] - row['delta_pse']]],
            fmt='o', color=color, ecolor=color,
            markersize=ms, capsize=3, lw=lw,
            markeredgecolor='white' if detected else color,
            markeredgewidth=0.6,
            zorder=3 if detected else 2,
        )

    ax.axvline(0, color=ZERO_C, ls='--', alpha=0.6, lw=1)
    ax.set_xlim(*_xlim(sub))
    ax.set_ylim(-0.7, len(rats) - 0.3)
    ax.set_title(MOD_NAMES.get(mod, f'Modality {mod}'), fontsize=11, pad=8)
    ax.set_xlabel('primary ΔPSE (°)  =  μ$_{easy}$ − μ$_{hard}$')
    ax.grid(axis='y', alpha=0.2, ls=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_yticks(list(y_pos.values()))
axes[0].set_yticklabels([f'Rat {r}' for r in rats])
axes[0].invert_yaxis()   # rat 2 at the top, rat 16 at the bottom
axes[0].set_ylabel('')

# legend
from matplotlib.lines import Line2D
legend_h = [
    Line2D([], [], color=NULL_C, marker='o', lw=1.2, markersize=5,
           label='CI contains 0 (null)'),
    Line2D([], [], color=DETECTED_C, marker='o', lw=2.0, markersize=7,
           markeredgecolor='white', markeredgewidth=0.6,
           label='CI excludes 0 (bias detected)'),
]
fig.legend(handles=legend_h, loc='upper center',
           bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False,
           fontsize=10)

# leave top room for the legend, side/bottom room for labels
fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.10, wspace=0.15)

# ---- save -----------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
for ext in ('png', 'svg', 'pdf'):
    path = OUT_DIR #os.path.join(OUT_DIR, f'{OUT_STEM}.{ext}')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f'✓ saved {path}')
plt.close(fig)