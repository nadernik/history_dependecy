import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load BLUPs (you already have them in the CSV above)
blups = pd.read_csv(r'C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\fits\formula1_uncorrelated_try_more_random\blups_rat.csv')

# Fixed effects from the latest model
fixed = {
    'failure_n1_c':         0.0689,
    'success_n1_c':        -0.0109,
    'failure_n1_c:td_down': -0.2555,
    'failure_n1_c:td_up':   -0.0250,
    'success_n1_c:td_down':  0.1022,
    'success_n1_c:td_up':   -0.0847,
}

# Per-rat estimates = fixed effect + BLUP
for term, fe in fixed.items():
    blups[f'{term}_per_rat'] = fe + blups[term]

# Group assignment from the BLUP scatter
TYPE_B = {11, 13, 16}
blups['strategy'] = blups['id'].apply(lambda x: 'B' if int(x) in TYPE_B else 'A')

# Quick summary
print(blups[['id', 'strategy', 
             'failure_n1_c:td_down_per_rat',
             'success_n1_c:td_down_per_rat',
             'failure_n1_c:td_up_per_rat',
             'success_n1_c:td_up_per_rat']].round(3))

# Scatter: heterogeneity in failure × td_down vs success × td_down
fig, ax = plt.subplots(figsize=(7, 6))
for strat, color in zip(['A', 'B'], ['steelblue', 'crimson']):
    sub = blups[blups['strategy'] == strat]
    ax.scatter(sub['failure_n1_c:td_down_per_rat'],
               sub['success_n1_c:td_down_per_rat'],
               c=color, s=80, label=f'Type {strat}')
    for _, row in sub.iterrows():
        ax.annotate(int(row['id']),
                    (row['failure_n1_c:td_down_per_rat'],
                     row['success_n1_c:td_down_per_rat']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
ax.axhline(0, color='gray', lw=0.5)
ax.axvline(0, color='gray', lw=0.5)
ax.set_xlabel('Per-rat failure × td_down (logit)')
ax.set_ylabel('Per-rat success × td_down (logit)')
ax.set_title('Per-rat downgrade interaction effects')
ax.legend()
plt.tight_layout()
plt.show()