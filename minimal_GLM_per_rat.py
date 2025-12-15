
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

an = SerialDependenceAnalyzer(history_depth=1, save_figures=False)
an.load_and_preprocess_data()
df_processed = an.create_lagged_features()

rat_results = an.analyze_individual_rats(minimal_glm=True)  # modality: 1=touch, 2=vision, 3=vt

# estrai betas per rat
rows = []
for rat_id, res in rat_results.items():
    names = res['feature_names']
    coefs = res['coefficients']
    if 'angle_n-1' not in names:
        continue
    beta = coefs[names.index('angle_n-1')]
    rows.append((rat_id, beta))

beta_df = pd.DataFrame(rows, columns=['rat', 'beta_prev_angle']).sort_values('rat')

# colori per segno
colors = np.where(beta_df['beta_prev_angle'] > 0, 'red',
         np.where(beta_df['beta_prev_angle'] < 0, 'blue', 'gray'))

rats = beta_df['rat'].tolist()
x = np.arange(len(rats))
y = beta_df['beta_prev_angle'].values
plt.figure(figsize=(max(8, 0.6*len(beta_df)), 4))
plt.axhline(0, color='k', ls='--', alpha=0.4)
#plt.scatter(x, beta_df['beta_prev_angle'], c=colors, s=80)
plt.bar(
    x,
    y,
    color=colors,
    width=0.6
)

# add text labels
for xi, yi in zip(x, y):
    if yi > 0:
        plt.text(
            xi,
            yi + 0.02 * max(abs(y)),   # slightly above bar
            f"{yi:.3f}",
            ha='center',
            va='bottom',
            fontsize=9
        )
    elif yi < 0:
        plt.text(
            xi,
            yi - 0.02 * max(abs(y)),   # slightly below bar
            f"{yi:.3f}",
            ha='center',
            va='top',
            fontsize=9
        )
plt.xticks(ticks=x,labels=rats,rotation=45, ha='right')
# vertical lines from 0 to each beta
'''
plt.vlines(
    x,
    ymin=0,
    ymax=y,
    colors=colors,
    linewidth=2,
    alpha=0.8
)
'''
plt.ylabel('Beta (angle_n-1)')
plt.xlabel('Rat')
plt.title('Betas per rat: previous-trial angle (angle_n-1)')
plt.tight_layout()
plt.show()