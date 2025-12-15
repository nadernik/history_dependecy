
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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from serial_dependence_analysis import SerialDependenceAnalyzer
from serial_dependence_analysis import fit_psychometric_curve, cumulative_gaussian_lapse
import functions_Ale as fa

BIN_COL_N_1     = 'bin_angle_n-1'     # previous-trial angle bin label
BIN_COL_N       = 'bin_angle_n'       # current-trial angle bin label
ANGLE_COL       = 'angle'             # current-trial raw angle          
HIT_COL         = 'hitmiss_n-1'       # filter on previous-trial hits (True/1) if desired
BIN_COL_N_MID   = 'bin_n_midpoint'    # numeric midpoint for current-trial bin (x-axis)
RAT_COL         = 'rat'               # subject/animal ID  
ACTION_N_1      = 'action_n-1'        # previous-trial response (0/1)
ACTION          = 'action'            # current-trial response (0/1)
TRANS_COL  = 'mod_transition_n-1'

an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data()
df_processed = an.create_lagged_features()
transitions = sorted(df_processed[TRANS_COL].dropna().unique(), key=str)
n_trans = len(transitions)
# set up subplots
ncols = int(np.ceil(np.sqrt(n_trans))) 
nrows = int(np.ceil(n_trans / ncols))  

fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 6.5*nrows),
                        sharex=True, sharey=True, constrained_layout=True)
axes = np.atleast_1d(axes).reshape(-1)
# analyze rats for each transition type
for ax, tr in zip(axes, transitions):
    rat_results = an.analyze_individual_rats(minimal_glm=True, mod_tr=tr)  # modality: 1=touch, 2=vision, 3=vt

    # estract betas per rat
    rows = []
    for rat_id, res in rat_results.items():
        names = res['feature_names']
        coefs = res['coefficients']
        if 'angle_n-1' not in names:
            continue
        beta = coefs[names.index('angle_n-1')]
        rows.append((rat_id, beta))
        
    # store betas in dataframe
    beta_df = pd.DataFrame(rows, columns=['rat', 'beta_prev_angle']).sort_values('rat')

    # colori per segno
    colors = np.where(beta_df['beta_prev_angle'] > 0, 'red',
            np.where(beta_df['beta_prev_angle'] < 0, 'blue', 'gray'))

    rats = beta_df['rat'].tolist()
    x = np.arange(len(rats))
    y = beta_df['beta_prev_angle'].values
    ax.axhline(0, color='k', ls='--', alpha=0.4)
   
    ax.bar(
        x,
        y,
        color=colors,
        width=0.6
    )


    ylim = ax.get_ylim()
    yrange = ylim[1] - ylim[0]
    offset = 0.04 * yrange
    # add text labels
    for xi, yi in zip(x, y):
        if yi > 0:
            ax.text(
                xi,
                yi + offset,   # slightly above bar
                f"{yi:.3f}",
                ha='center',
                va='bottom',
                fontsize=8,
                clip_on = False
            )
        elif yi < 0:
            ax.text(
                xi,
                yi - offset,   # slightly below bar
                f"{yi:.3f}",
                ha='center',
                va='top',
                fontsize=8,
                clip_on = False
            )
    ax.set_xticks(x)
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=9)
    if ax not in axes[-ncols:]:
        ax.set_xticklabels([])
        ax.set_xlabel('')
    else:
        ax.set_xticklabels(rats, rotation=45, ha='right')
  

    
    if ax not in axes[::ncols]:
        ax.set_ylabel('')
    else:
        ax.set_ylabel('Beta (angle_n-1)')

    ax.set_title(f'{tr} All rats', fontsize=10)
    
    

    fig.subplots_adjust(
    bottom=0.25,   # space for x tick labels (last row)
    left=0.12,     # space for y ticks
    wspace=0.25,
    hspace=0.35
)



plt.show()