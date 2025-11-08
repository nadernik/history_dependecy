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

# coping from behavior_exploration.ipynb

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
import seaborn as sns
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("Libraries imported successfully!")


# Load the behavioral data
def load_behavior_data(filepath):
    """Load and structure the behavioral data from .mat file"""
    data = scipy.io.loadmat(filepath)
    trial_data = data['TrialNADER'][0, 0]
    
    # Convert to dictionary for easier access
    trial_dict = {}
    for field in trial_data.dtype.names:
        trial_dict[field] = trial_data[field][0].flatten()
    
    # Create a pandas DataFrame for easier manipulation
    df = pd.DataFrame(trial_dict)
    
    # Convert numeric columns to proper types
    numeric_columns = ['rat', 'trialID', 'date', 'oricat', 'action', 'hitmiss', 'penalty', 
                      'penaltyTime', 'samplingTime', 'theta', 'cumPerformance', 'mod', 'angle', 
                      'alpha', 'stimColor', 'RT', 'PT', 'rewardDelay', 'trialReset', 
                      'brightness', 'soa', 'soundQ', 'soundQType']
    
    for col in numeric_columns:
        if col in df.columns:
            # First convert to string, then to numeric to handle any data type issues
            df[col] = pd.to_numeric(df[col].astype(str), errors='coerce')
    
    # Remove rows with NaN values in critical columns
    df = df.dropna(subset=['rat', 'action', 'hitmiss', 'mod', 'angle'])
    
    return df, trial_dict

# Load the data
df, trial_data = load_behavior_data('data/behavior_data.mat')

# SIMPLIFIED DATA PROCESSING
print("=== SIMPLIFIED DATA PROCESSING ===")

# 1. Filter rats: exclude [8,14,15,17:21]
rats = df['rat'].unique()
rats = rats[~np.isin(rats, [8,14,15,17,18,19,20,21])]
df = df[df['rat'].isin(rats)]

# 2. Ensure action is binary 0 or 1
df['action'] = df['action'].astype(int)
df = df[df['action'].isin([0, 1])]

# 3. Ensure hitmiss is binary: 0=fail, 1=hit
df['hitmiss'] = df['hitmiss'].astype(int)
df = df[df['hitmiss'].isin([0, 1])]

# 4. Map all angles to 0-90 range
# Rule: angle>180 -> angle-180; angle>90 -> (90-(angle-90))+90
df['angle'] = df['angle'].astype(float)
#print(f"Angle range: {df['angle'].min():.2f}° to {df['angle'].max():.2f}°")


df.loc[df['angle'] > 180, 'angle'] = df.loc[df['angle'] > 180, 'angle'] - 180 # they turn out as[135 140 141 143 145 150 151 155 159 160 165 169 170 175]

mask = (df['angle'] > 90) & (df['angle'] < 134)
df.loc[mask, 'angle'] = (90 - (df.loc[mask, 'angle'] - 90))
df.loc[df['angle'] > 134, 'angle'] = -(df.loc[df['angle'] > 134, 'angle'] - 180)

unique_angles = np.sort(df['angle'].unique()) # 43 unique angles after mapping - 1 (bc we exclude 45°) = 42, binning options: 6, 7, 14, 21
print("Unique angles:", unique_angles)
print('angle range successfully mapped angles to 0-90° range!')

# Binning my angles 
from functions_Ale import bin_by_unique_angles # fix this import path later

df, angle_to_label= bin_by_unique_angles(df, n_groups=6, angle_col='angle', exclude_angle=45)
print(angle_to_label)































'''
# Save original  angles between 90° and 134° and their indices --> check that the transformation is correct
in the future maybe we want to plot original vs mirrored angles to be see if they behave as we expected ro not
mask_check = (df['angle'] > 90) & (df['angle'] < 134)
original_angles = df.loc[mask_check, 'angle'].copy()

# Apply the transformation
df.loc[mask_check, 'angle'] = (90 - (df.loc[mask_check, 'angle'] - 90))

# Compare old vs new
comparison = pd.DataFrame({
    'original_angle': original_angles,
    'mirrored_angle': df.loc[mask_check, 'angle']
}).reset_index(drop=True)

print(comparison)
'''

'''
# 5. Filter to only angles 0-90
df = df[(df['angle'] >= 0) & (df['angle'] <= 90)]

# 6. Define reversed rule rats
rev_rule_rats = [6, 7, 14, 15, 16, 17]

# 7. Define colors for modalities
modality_colors = [[0, 2/3, 0], [0, 0.4470, 0.7410], [1, 0, 0], [0, 0, 0]]

print(f"Data loaded and simplified successfully!")
print(f"Total trials after filtering: {len(df)}")
print(f"Rats included: {sorted(rats)}")
print(f"Angle range: {df['angle'].min():.1f} to {df['angle'].max():.1f}")
print(f"Action values: {sorted(df['action'].unique())}")
print(f"Hit/Miss values: {sorted(df['hitmiss'].unique())}")
print(f"Modalities: {sorted(df['mod'].unique())}")
print(f"Reversed rule rats: {rev_rule_rats}")
print(f"\nFirst few rows:")
print(df[['rat', 'action', 'hitmiss', 'mod', 'angle']].head())



# Plot simplified distributions
fig, axes = plt.subplots(2,2 figsize=(15, 10))

# Angle distribution (0-90)
axes[0,0].hist(df['angle'], bins=20, alpha=0.7, edgecolor='black', color='black')
axes[0,0].set_title('Distribution of Angles (0-90°)')
axes[0,0].set_xlabel('Angle (degrees)')
axes[0,0].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
'''