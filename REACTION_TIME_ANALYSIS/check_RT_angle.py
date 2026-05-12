
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from serial_dependence_analysis import SerialDependenceAnalyzer
HISTORY_DEPTH = 1


# -----------------------------
# LOAD + PREPROCESS
# -----------------------------
an = SerialDependenceAnalyzer(history_depth=HISTORY_DEPTH, save_figures=False)
an.load_and_preprocess_data(fold = False)  # Set fold=False to keep all angles
df = an.create_lagged_features()
print(df.columns.tolist())
# =============================
# RT per angle analysis — per rat
# =============================

# --- clean RT first ---
# --- clean RT first ---
def clean_rt(df_input):
    result = []
    for rat_id in df_input['rat'].unique():
        group = df_input[df_input['rat'] == rat_id].copy()
        valid = group[group['RT'] > 0]
        if len(valid) == 0:
            continue
        mean, std = valid['RT'].mean(), valid['RT'].std()
        cleaned = group[(group['RT'] > 0) & (group['RT'] < mean + 3*std)]
        result.append(cleaned)
    return pd.concat(result, ignore_index=True)

df_rt_clean = clean_rt(df)


print('rat' in df_rt_clean.columns)  # should print True
print(df_rt_clean.columns.tolist())
print(df_rt_clean.head(2))
# --- convert 270-360 range to negative angles ---
df_rt_clean['angle'] = df_rt_clean['angle'].apply(
    lambda x: x - 360 if x > 180 else x
)

# --- filter out angles with fewer than 50 trials (pooled across rats) ---
angle_counts = df_rt_clean['angle'].value_counts()
valid_angles = angle_counts[angle_counts >= 50].index
df_rt_clean = df_rt_clean[df_rt_clean['angle'].isin(valid_angles)]

angles_sorted = sorted(df_rt_clean['angle'].unique())
boundary_idx = angles_sorted.index(45) if 45 in angles_sorted else None
print(f"Valid angles: {angles_sorted}")

# --- per rat plots ---
rats = sorted(df_rt_clean['rat'].unique())

# --- compute summary stats per rat ---
all_rt_by_angle = []
for rat_id in rats:
    df_rat = df_rt_clean[df_rt_clean['rat'] == rat_id]

    rat_angle_counts = df_rat['angle'].value_counts()
    rat_valid_angles = rat_angle_counts[rat_angle_counts >= 50].index
    df_rat = df_rat[df_rat['angle'].isin(rat_valid_angles)]

    if df_rat.empty:
        print(f"Rat {rat_id}: no valid angles after filtering, skipping")
        continue

    rt_by_angle = (
        df_rat.groupby('angle')['RT']
        .agg(mean='mean', se=lambda x: x.std() / np.sqrt(len(x)))
        .reset_index()
        .sort_values('angle')
    )
    rt_by_angle['rat'] = rat_id
    all_rt_by_angle.append(rt_by_angle)

df_all = pd.concat(all_rt_by_angle, ignore_index=True)

# --- one subplot per rat, all in one figure ---
n_rats = len(rats)
n_cols = 4
n_rows = int(np.ceil(n_rats / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)
axes = axes.flatten()  # make it easy to index with a single number

palette = sns.color_palette("tab10", n_colors=n_rats)

for i, rat_id in enumerate(rats):
    ax = axes[i]
    rat_data = df_all[df_all['rat'] == rat_id].sort_values('angle')

    ax.errorbar(
        rat_data['angle'], rat_data['mean'],
        yerr=rat_data['se'],
        fmt='o-', color=palette[i], capsize=3,
        linewidth=1.5, markersize=4
    )
    ax.axvline(x=45, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='blue', linestyle='--', alpha=0.5)
    ax.axvline(x=90, color='red', linestyle='--', alpha=0.5)
    ax.set_title(f'Rat {rat_id}', fontsize=11)
    ax.set_xlabel('Angle')
    ax.set_ylabel('Mean RT (ms)')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.tick_params(axis='x', rotation=45)

# hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Mean RT per angle — per rat (± SEM)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()