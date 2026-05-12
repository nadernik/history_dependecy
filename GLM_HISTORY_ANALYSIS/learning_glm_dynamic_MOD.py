# this script run the same glm analysis at the session level, to see how the beta coefficients evolve over learning.
# you can select one target rat (changeable at the top of the script), and the script will run a glm analysis on each session of that rat,
# and plot the beta coefficients of the features of interest as a function of session index (relative to the first session), with a smoothed line to see the trend.
# there is one pannel per feature, and the raw beta values are plotted as faded lines in the background, to show the variability across sessions, 
# while the smoothed line is the main story. The analysis is done separately for each modality, to see if there are differences in the learning dynamics between modalities.
# So each pannel will have three smoothed lines, one per modality, and the raw beta values will also be colored by modality.


# =============================
# CONFIG
# =============================
RAT_ID        = 9
MIN_TRIALS    = 200
SMOOTH_WINDOW = 10
HISTORY_DEPTH = 1
FEATURES      = ['angle', 'action_n-1', 'angle_n-1', 'failure_n-1']  # features_17
MODALITIES    = {1: 'tactile', 2: 'visual', 3: 'visuotactile'}
MOD_COLORS    = {
    'tactile':      '#e07b7b',
    'visual':       '#4a9e6b',
    'visuotactile': '#3a8fbf'
}

# =============================
# IMPORTS
# =============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions_Ale import filter_training_trials
from serial_dependence_analysis import SerialDependenceAnalyzer

# =============================
# BLOCK 1: LOAD + PREPROCESS
# =============================
an = SerialDependenceAnalyzer(history_depth=HISTORY_DEPTH, save_figures=False)
an.load_and_preprocess_data()
df = an.create_lagged_features()
df_processed  = filter_training_trials(df, method='criterion', criterion=0.80, min_consec=2)

df['action_n-1']  = df['action_n-1'] * 2 - 1
df['success_n-1'] = df['hit_n-1'] * df['action_n-1']
df['failure_n-1'] = (1 - df['hit_n-1']) * df['action_n-1']
df['angle']       = df['angle'] - 45
df['angle_n-1']   = df['angle_n-1'] - 45

df_rat   = df[df['rat'] == RAT_ID].copy()
sessions = sorted(df_rat['date'].unique())
print(f"Rat {RAT_ID}: {len(sessions)} sessions found")

# =============================
# BLOCK 2: LOOP OVER MODALITIES + SESSIONS
# =============================
def get_modality_chunks(df_rat, sessions, modality, min_trials):
    """
    For a given modality, merge consecutive sessions until min_trials
    is reached. Returns a list of (representative_session, df_chunk) tuples.
    """
    chunks = []
    i = 0
    while i < len(sessions):
        accumulated = df_rat[
            (df_rat['date'] == sessions[i]) & (df_rat['mod'] == modality)
        ].copy()
        representative_session = sessions[i]
        j = i + 1

        while len(accumulated) < min_trials and j < len(sessions):
            next_trials = df_rat[
                (df_rat['date'] == sessions[j]) & (df_rat['mod'] == modality)
            ].copy()
            accumulated = pd.concat([accumulated, next_trials])
            j += 1

        if len(accumulated) >= min_trials:
            chunks.append((representative_session, accumulated))

        i = j  # no overlap — jump to where this chunk ended

    return chunks


all_betas = {}

for mod_id, mod_name in MODALITIES.items():
    print(f"\nProcessing modality: {mod_name}")
    rows   = []
    chunks = get_modality_chunks(df_rat, sessions, modality=mod_id, min_trials=MIN_TRIALS)
    print(f"  {len(chunks)} chunks found")

    for representative_session, df_chunk in chunks:
        try:
            result = an.analyze_individual_rats_ale_N(
                features_keys=FEATURES,
                df_processed=df_chunk,
                k=None,
                scaling='cont_only', 
                cv_split=False,
            )
            values = result[RAT_ID]['coefficients']
            keys   = result[RAT_ID]['feature_names']
            row    = {key: value for key, value in zip(keys, values)} | {'session': representative_session}
            rows.append(row)
        except ValueError as e:
            print(f"  Skipping chunk at session {representative_session}: {e}")
            continue

    all_betas[mod_name] = pd.DataFrame(rows)

# =============================
# BLOCK 3: PLOT
# =============================
for mod_name, df_mod in all_betas.items():
    df_mod['session_idx'] = df_mod['session'] - df_rat['date'].min()

beta_cols = [col for col in all_betas['tactile'].columns if col not in ('session', 'session_idx')]

fig, axes = plt.subplots(len(beta_cols), 1, figsize=(14, 3 * len(beta_cols)), sharex=True)

for ax, col in zip(axes, beta_cols):
    for mod_name, df_mod in all_betas.items():
        color = MOD_COLORS[mod_name]
        ax.plot(df_mod['session_idx'], df_mod[col],
                color=color, alpha=0.2, linewidth=0.8)
        smoothed = df_mod[col].rolling(window=SMOOTH_WINDOW, center=True).mean()
        ax.plot(df_mod['session_idx'], smoothed,
                color=color, linewidth=2.5, label=mod_name)

    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_ylabel(col)
    ax.legend(loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)

axes[-1].set_xlabel('Session (days from start)')
fig.suptitle(f'Rat {RAT_ID} — beta trajectories over learning', fontsize=13)
plt.tight_layout()
plt.show()