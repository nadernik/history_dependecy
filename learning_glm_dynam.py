# this script run the same glm analysis at the session level, to see how the beta coefficients evolve over learning.
# you can select one target rat (changeable at the top of the script), and the script will run a glm analysis on each session of that rat,
# and plot the beta coefficients of the features of interest as a function of session index (relative to the first session), with a smoothed line to see the trend.
# there is one pannel per feature, and the raw beta values are plotted as faded lines in the background, to show the variability across sessions, 
# while the smoothed line is the main story.


# =============================
# CONFIG
# =============================
RAT_ID = 9          # pick one rat by its ID as it appears in the 'rat' column
MIN_TRIALS = 50      # minimum number of trials in a session to include it

FEATURES = ['angle', 'action_n-1', 'angle_n-1', 'failure_n-1']  # features_17
HISTORY_DEPTH = 1  # how many past trials to include as features (e.g., 1 means only n-1 features)

from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from functions_Ale import filter_training_trials, flatten_model_results
   
from serial_dependence_analysis import SerialDependenceAnalyzer

# =============================
# CONFIG
# =============================
RAT_COL        = "rat"
ANGLE_COL      = "angle"          # trial n angle
ACTION_COL     = "action"         # trial n choice (0/1)
ANGLE_N1_COL   = "angle_n-1"      # trial n-1 angle
ACTION_N1_COL  = "action_n-1"     # trial n-1 choice (0/1)
BOUNDARY_ANGLE = 45
HITMISS_N1_COL = "hitmiss_n-1"    # trial n-1 outcome (0/1)



# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=HISTORY_DEPTH, save_figures=False)
an.load_and_preprocess_data()
df = an.create_lagged_features()
df_processed  = filter_training_trials(df_processed, method='criterion', criterion=0.80, min_consec=2)


# re-code action_n-1 to be -1 for left and +1 for right, instead of 0/1, so that it can be used in the model as a signed variable (e.g. for interaction with hit_n-1)
df['action_n-1'] = df['action_n-1']*2 -1
#df['hitmiss_n-1'] = df['hitmiss_n-1']*2 -1
df['success_n-1'] = df['hit_n-1']*df['action_n-1'] # 1 if previous trial was a hit and the action was right (1*1), -1 if hit and left (1*-1), 0 if miss (0*anything)
df['failure_n-1'] = (1 - df['hit_n-1'])*df['action_n-1'] # 1 if previous trial was a miss and the action was right (1*1), -1 if miss and left (1*-1), 0 if hit (0*anything)
#df['angle_hit_n-1'] = df['angle_n-1']*df['hit_n-1'] # angle of the previous trial if it was a hit, 0 if it was a miss
#df['angle_miss_n-1'] = df['angle_n-1']*(1 - df['hit_n-1']) # angle of the previous trial if it was a miss, 0 if it was a hit
df['angle'] = df['angle'] - 45
df['angle_n-1'] = df['angle_n-1'] - 45
df['hit_n-1'] = 1 - df['hit_n-1'] # TMEPORARY
# =============================
# FILTER TO ONE RAT
# =============================
df_rat = df[df['rat'] == RAT_ID].copy()
df_rat = df_rat[df_rat['mod'] == 1].copy()  # TEMPORARY: filter to tactile trials only, to increase power for now, we can look at modality differences later
# =============================
# GET SESSIONS (sorted chronologically)
# =============================
sessions = df_rat['date'].unique()
print(f"Rat {RAT_ID}: {sessions} sessions found before filtering by trial count")
sessions.sort()  # ensure sessions are in chronological order
print(f"Rat {RAT_ID}: {sessions} sessions found")


rows = []

for session in sessions:
    df_session = df_rat[df_rat['date'] == session].copy()
    
    if len(df_session) < MIN_TRIALS:
        print(f"  Skipping session {session}: only {len(df_session)} trials")
        continue
    
    result = an.analyze_individual_rats_ale_N(
        features_keys=FEATURES,
        df_processed=df_session,
        k=None,
        scaling='cont_only', 
        cv_split=False,  # set to True to get cross-validated scores (but it will break here because we are fitting one glm per session and cv needs at least 5 sessions to split into folds)
    )
    
    values = result[RAT_ID]['coefficients']
    keys = result[RAT_ID]['feature_names']
    row = {key: value for key, value in zip(keys, values)} | {'session': session}
    rows.append(row)

df_betas = pd.DataFrame(rows)


# =============================
# BLOCK 3: PLOT BETA TRAJECTORIES
# =============================

# convert session datenums to session index preserving gaps
# hint: if session 1 is at datenum 735549 and session 2 is at 735556,
# the gap between them is 7 days — how would you turn the 'session' 
# column into a relative day index starting at 0?
# =============================
# BLOCK 3: PLOT
# =============================
SMOOTH_WINDOW = 10  # rolling mean window in sessions, adjust to taste

df_betas['session_idx'] = df_betas['session'] - df_betas['session'].min()
beta_cols = [col for col in df_betas.columns if col not in ('session', 'session_idx')]

fig, axes = plt.subplots(len(beta_cols), 1, figsize=(14, 3 * len(beta_cols)), sharex=True)

colors = ['#e07b7b', '#c8a227', '#4a9e6b', '#3a8fbf']  # one per beta

for ax, col, color in zip(axes, beta_cols, colors):
    # raw line — faded background
    ax.plot(df_betas['session_idx'], df_betas[col],
            color=color, alpha=0.2, linewidth=0.8)
    
    # smoothed line — main story
    smoothed = df_betas[col].rolling(window=SMOOTH_WINDOW, center=True).mean()
    ax.plot(df_betas['session_idx'], smoothed,
            color=color, linewidth=2.5, label=col)
    
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_ylabel('Beta')
    ax.legend(loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)

axes[-1].set_xlabel('Session (days from start)')
fig.suptitle(f'Rat {RAT_ID} — beta trajectories over learning', fontsize=13)
plt.tight_layout()
plt.show()


'''
df_betas['session_idx'] = df_betas['session'] - df_betas['session'].min()

beta_cols = [col for col in df_betas.columns if col != 'session' and col != 'session_idx']

fig, ax = plt.subplots(figsize=(12, 5))

for col in beta_cols:
    ax.plot(df_betas['session_idx'], df_betas[col], label=col)

ax.axhline(0, color='black', linestyle='--', linewidth=0.8)  # what line would be useful here?
ax.set_xlabel('Session (days from start)')
ax.set_ylabel('Beta coefficient')
ax.set_title(f'Rat {RAT_ID} — beta trajectories over learning')
ax.legend()
plt.tight_layout()
plt.show()
'''


