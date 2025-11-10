'functions _ Ale'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def bin_by_unique_angles(df, name_new_col, n_groups, angle_col, exclude_angle=45, min_count=100):
    """
    Bin by *unique angle values* so each bin has (nearly) the same count of distinct angles.
    Rows with angle==exclude are removed first.

    label options:
      - 'range' : labels like '0–16°'
      - 'center': numeric labels = group midpoint (mean of angles in the group)
      - 'index' : integer bin index 0..n_groups-1
    """

  # === COUNT OCCURRENCES OF EACH ANGLE ===
    counts = df[angle_col].value_counts()

    # === KEEP ONLY ANGLES THAT APPEAR AT LEAST `min_count` TIMES ===
    valid_angles = counts[counts >= min_count].index

    # === EXCLUDE 45° AND GET UNIQUE ANGLES ===
    unique_angles = np.sort(df.loc[(df[angle_col].isin(valid_angles)) & (df[angle_col] != exclude_angle), angle_col].unique())

    # === SPLIT UNIQUE ANGLES INTO ~EQUAL-SIZED GROUPS ===
    groups = np.array_split(unique_angles, n_groups)

    # === ASSIGN LABELS (RANGE LABELS, e.g., '0–16°') ===
    labels = [f"{g[0]:.0f}–{g[-1]:.0f}°" for g in groups]

    # === MAP EACH ANGLE VALUE TO ITS BIN LABEL ===
    angle_to_label = {a: labels[i] for i, g in enumerate(groups) for a in g}

    # === CREATE NEW COLUMN WITH BIN LABEL ===
    df[name_new_col] = df[angle_col].map(angle_to_label)

    #print(df[[angle_col, name_new_col]].sample(15, random_state=42))
    
    return df, angle_to_label

def midpoint(interval):
    if hasattr(interval, 'mid'):
        return interval.mid  # for pd.Interval bins
    try:
        # for string labels like "0–15°"
        parts = [float(x.replace('°','')) for x in str(interval).replace('–','-').split('-')]
        return np.mean(parts)
    except:
        return np.nan


def color_bin(bin_means): 
        
        bins_sorted = sorted(bin_means, key=bin_means.get)
        blue_bins = [b for b in bins_sorted if bin_means[b] < 45]
        red_bins  = [b for b in bins_sorted if bin_means[b] >= 45]

        blues = plt.cm.Blues(np.linspace(0.25, 0.85, len(blue_bins)))
        reds  = plt.cm.Reds(np.linspace(0.25, 0.85, len(red_bins)))

        color_by_bin = {b: c for b, c in zip(blue_bins, blues)}
        color_by_bin.update({b: c for b, c in zip(red_bins, reds)})

        return color_by_bin

def aggregate_data(dfc, BIN_COL, ANGLE_COL, RESP_COL):
    agg = (dfc.groupby([BIN_COL, ANGLE_COL])[RESP_COL]
            .agg(mean='mean', n='size')
            .reset_index())
    agg = agg[agg['n'] >= 5]  # or 10, depending on your data
    agg['se'] = np.sqrt(agg['mean'] * (1 - agg['mean']) / agg['n'])
    return agg

        