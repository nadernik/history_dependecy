
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
import seaborn as sns
import matplotlib.pyplot as plt


# -----------------------------
# LOAD BOOTSTRAP RESULTS + SUMMARY
# -----------------------------

boot_df= pd.read_csv("rat13_allmodalities_bootstrap_psychometric_params.csv")

# -----------------------------
# COMPUTE VIOLINE (one per rat)
# -----------------------------


param = "mu"  # or "sigma","mu" "gamma", "lapse"

# Ensure we are only comparing two N-1 conditions
bins = sorted(boot_df["prev_bin"].dropna().unique())
assert len(bins) == 2, f"Expected exactly 2 prev_bin values, got {bins}"

bin_A, bin_B = bins[0], bins[1]  # define subtraction order: B - A

# Pivot so each row is (rat, bootstrap) and columns are the two prev_bin conditions
wide = (
    boot_df
    .pivot(index=["md_mode", "bootstrap"], columns="prev_bin", values=param)
    .dropna(subset=[bin_A, bin_B])  # keep only paired bootstrap reps
    .reset_index()
)
nps = boot_df.groupby("md_mode")["n_per_side"].first()  # get n per side for reference

# Paired difference
wide["delta"] = wide[bin_B] - wide[bin_A]
wide["param"] = param
wide["contrast"] = f"{bin_B} − {bin_A}"
# ci of delta
delta_summary = (
    wide.groupby("md_mode")["delta"]
        .agg(
            med="median",
            low=lambda x: x.quantile(0.025),
            high=lambda x: x.quantile(0.975),
            p_pos=lambda x: np.mean(x > 0),  # proportion of bootstraps where Δ>0
            n="size"
        )
        .reset_index()
)

delta_summary["ci_excludes_0"] = (delta_summary["low"] > 0) | (delta_summary["high"] < 0)
delta_summary["n_per_side"] = delta_summary["md_mode"].map(nps)
delta_summary["n_total_per_rep"] = 2 * delta_summary["n_per_side"]

print(delta_summary)
# ---------- PLOTTING ----------
order = list(wide["md_mode"].unique())   # lock order used by seaborn
plt.figure(figsize=(max(10, 0.8 * len(order)), 4))
ax = plt.gca()

sns.violinplot(data=wide, x="md_mode", y="delta", inner="quartile", cut=0)
sns.stripplot(data=wide, x="md_mode", y="delta", jitter=0.25, size=2, alpha=0.25)


# mark modalities where CI excludes 0
sig_md = delta_summary.loc[delta_summary["ci_excludes_0"], "md_mode"]
for i, md in enumerate(wide["md_mode"].unique()):
    if md in set(sig_md):
        plt.text(i, wide["delta"].max(), "*", ha="center", va="bottom", fontsize=14)

plt.axhline(0, color="k", linestyle="--", alpha=0.5)
plt.xlabel("Transition Type")
plt.ylabel(f"Δ{param} ({bin_B} − {bin_A})")
plt.title(f"Bootstrap Δ{param} per transition (paired bootstrap)")

# ---------- ADD TEXT UNDER EACH VIOLIN ----------
# Align summary rows with plotted order
ds = delta_summary.set_index("md_mode").reindex(order).reset_index()

ymin, ymax = ax.get_ylim()
yrng = ymax - ymin

# add extra room below and define y position for text
ax.set_ylim(ymin - 0.30 * yrng, ymax)
y_text = ymin - 0.8 * yrng

for i, row in ds.iterrows():
    txt = (
        f"p(Δ>0)={row['p_pos']:.2f}\n"
        f"CI[{row['low']:.2f},{row['high']:.2f}]\n"
        f"boot={int(row['n'])}\n"
        f"n/side={int(row['n_per_side'])}"
    )
    ax.text(i, y_text, txt, ha="center", va="top", fontsize=8)

plt.tight_layout()
plt.show()

# Save figure
#plt.savefig(f"rat9_bootstrap_delta_{param}_per_transition_violin_plot.png", dpi=300)


        