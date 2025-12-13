
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

'''
df_processed, angle_to_label_N_1 = fa.bin_by_unique_angles(
    df_processed, name_new_col=BIN_COL_N_1, n_groups=2,
    angle_col='angle_n-1', exclude_angle=45
)
df_processed, angle_to_label_N  = fa.bin_by_unique_angles(
    df_processed, name_new_col=BIN_COL_N,   n_groups=2,
    angle_col='angle', exclude_angle=45
)
'''

rat_results = an.analyze_individual_rats(minimal_glm=True, transition = (1,2))  # modality: 1=touch, 2=vision, 3=vt

print (rat_results.keys())
for rat_id in rat_results.keys():
    #print("Rat", rat_id, "feature names:", rat_results[rat_id]['feature_names'])
    names = rat_results[rat_id]['feature_names']
    coefs = rat_results[rat_id]['coefficients']

    beta_prev_angle = coefs[names.index('angle_n-1')]
    print("Rat", rat_id, "beta(angle_n-1) =", beta_prev_angle)
'''

input("Press Enter to close plots...")

'''

# mo devo tipo plottare sti cosi qui per ogni ratto e sto point fare le analisi anche per modalità (tatto, vis, vt)








