'''
I made chatGBT test these functions bc they were important for the script exploratory_psych_curves.py to work properly, but everything was fine. 

'''
#from functions_Ale import midpoint

import pandas as pd
import numpy as np

# --- Your function ---
from functions_Ale import midpoint#

# --- Test cases ---
test_cases = [
    pd.Interval(0, 15, closed='right'),   # proper pandas interval
    "0–15°",                              # en dash
    "15-30°",                             # plain dash
    "(30, 45]",                           # parentheses style
    "foo-bar",                            # invalid
    np.nan                                # missing
]

# --- Run tests ---
for t in test_cases:
    print(f"test results: {t!r:>15} → {midpoint(t)}")


import numpy as np
import matplotlib.pyplot as plt

# --- Your function ---
from functions_Ale import color_bin


# --- Fake test input ---
bin_means = {
    "(0,15]": 7.5,
    "(15,30]": 22.5,
    "(30,45]": 37.5,
    "(45,60]": 52.5,
    "(60,75]": 67.5,
    "(75,90]": 82.5
}

# --- Run the function ---
color_by_bin = color_bin(bin_means)

# --- Print result ---
#for b, c in color_by_bin.items():
    #print(f"{b:>8} → {c}")

# --- Quick visual check ---
plt.figure(figsize=(6,1))
for i, (b, c) in enumerate(color_by_bin.items()):
    plt.bar(i, 1, color=c)
plt.xticks(range(len(color_by_bin)), color_by_bin.keys(), rotation=45)
plt.yticks([])
plt.title("Color map from color_bin()")
plt.show()

