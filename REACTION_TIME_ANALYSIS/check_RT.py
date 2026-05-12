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
an.load_and_preprocess_data()
df = an.create_lagged_features()


# Quick diagnostic on all timing-related columns
timing_cols = ['RT', 'PT', 'samplingTime', 'penaltyTime', 'rewardDelay']

print("=== VALUE COUNTS (top 5) ===")
for col in timing_cols:
    print(f"\n{col}:")
    print(df[col].value_counts().head())

print("\n=== DESCRIPTIVE STATS ===")
print(df[timing_cols].describe())

print("\n=== ZEROS COUNT ===")
print((df[timing_cols] == 0).sum())

print("\n=== NaN COUNT ===")
print(df[timing_cols].isna().sum())

# Diagnostic: what does the valid RT distribution look like?
rt_valid = df[(df['RT'] > 0) & (df['RT'] < 10000)]  # adjust upper bound if needed
print(f"Valid RT trials: {len(rt_valid)} / {len(df)} ({100*len(rt_valid)/len(df):.1f}%)")
print(rt_valid['RT'].describe())

# Plot distribution
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(rt_valid['RT'], bins=100, color='steelblue', edgecolor='none')
plt.xlabel('RT (ms?)'); plt.title('RT distribution (valid trials)')

plt.subplot(1, 2, 2)
plt.hist(rt_valid['RT'], bins=100, color='steelblue', edgecolor='none', log=True)
plt.xlabel('RT (ms?)'); plt.title('RT distribution (log scale)')
plt.tight_layout()
plt.show()

print(df[df['RT'] > 0]['RT'].quantile([0.90, 0.95, 0.99, 0.999]))
print(f"Trials > 5000ms: {(rt_valid['RT'] > 5000).sum()}")
print(f"Trials > 10000ms: {(df['RT'] > 10000).sum()}")