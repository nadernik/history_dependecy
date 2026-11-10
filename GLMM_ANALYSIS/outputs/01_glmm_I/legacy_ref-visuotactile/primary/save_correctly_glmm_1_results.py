import shutil
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR   = Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_fits\glmm_I\glmm_I_primary")
CONSOLE_LOG   = Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_fits\glmm_I\glmm_I_console_log.txt")

# Paste the full terminal output (from "[22:53:02] Dropping..." to "All done in 690.5 minutes total")
# into a plain text file here so it's preserved as-is.
# If you already have it in the clipboard, save it manually to CONSOLE_LOG first.
if not CONSOLE_LOG.exists():
    print(f"⚠  No console log found at {CONSOLE_LOG}")
    print("   Save your terminal output to that path, then re-run this snippet.")
else:
    # Copy the console log into the results folder as well, so everything is together
    shutil.copy2(CONSOLE_LOG, RESULTS_DIR / "console_log.txt")
    print(f"✓ Console log copied into {RESULTS_DIR / 'console_log.txt'}")

# ─────────────────────────────────────────────────────────────────────────────
# Verify every extraction file R was supposed to produce is on disk
# ─────────────────────────────────────────────────────────────────────────────
EXPECTED_FILES = [
    "fixed_effects.csv",
    "fixed_vcov.csv",
    "ranef_variances.csv",
    "predictions.csv",
    "contrasts.csv",
    "heterogeneity.csv",
    "per_rat_coords.csv",
    "modality_baselines.csv",
    "meta.txt",
    "formula.txt",
    "session_info.txt",
    "blups_rat.csv",
    "blups_rat_date.csv",
]

print("\n── Files in results folder ──")
present = {f.name for f in RESULTS_DIR.iterdir() if f.is_file()}
missing = []
for fname in EXPECTED_FILES:
    if fname in present:
        size_kb = (RESULTS_DIR / fname).stat().st_size / 1024
        print(f"  ✓ {fname:35s}  {size_kb:>10.1f} KB")
    else:
        missing.append(fname)
        print(f"  ✗ {fname:35s}  MISSING")

unexpected = present - set(EXPECTED_FILES) - {"console_log.txt", "data.csv", "fit.R", "r_console.log"}
if unexpected:
    print("\n── Unexpected files (also present) ──")
    for f in sorted(unexpected):
        print(f"  ? {f}")

if missing:
    raise RuntimeError(f"Missing extraction files: {missing}. "
                       "Do NOT proceed — the R fit output is incomplete.")

# ─────────────────────────────────────────────────────────────────────────────
# Load all CSVs into a single dict
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Loading extracted results ──")
results = {}

for name in ["fixed_effects", "fixed_vcov", "ranef_variances", "contrasts",
             "heterogeneity", "per_rat_coords", "modality_baselines",
             "blups_rat", "blups_rat_date"]:
    df = pd.read_csv(RESULTS_DIR / f"{name}.csv")
    results[name] = df
    print(f"  loaded {name}: {df.shape[0]} rows × {df.shape[1]} cols")

# predictions.csv is large — load lazily or downsample if you don't need it now
preds_path = RESULTS_DIR / "predictions.csv"
print(f"  predictions.csv on disk ({preds_path.stat().st_size / 1024**2:.1f} MB) "
      f"— load on demand with pd.read_csv when needed")

# meta.txt as a dict
meta = {}
with open(RESULTS_DIR / "meta.txt") as f:
    for line in f:
        line = line.strip()
        if "=" in line:
            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip()
results["meta"] = meta
print(f"  loaded meta: {len(meta)} keys")

# formula and session info as strings
results["formula"]      = (RESULTS_DIR / "formula.txt").read_text().strip()
results["session_info"] = (RESULTS_DIR / "session_info.txt").read_text()
print(f"  loaded formula and session_info")

# ─────────────────────────────────────────────────────────────────────────────
# Cross-check loaded values against what the console printed
# (catches silent CSV corruption or a mismatched run)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Cross-checks against console output ──")

def _get_fe(term):
    row = results["fixed_effects"].query("term == @term").iloc[0]
    return float(row["estimate"]), float(row["p"])

expected_fe = {
    "(Intercept)":              (0.0905, 0.3473),
    "angle_c":                  (0.0914, 0.0000),
    "modalitytactile":          (-0.0665, 0.0000),
    "angle_prev_c":             (0.0046, 0.0000),
    "failure_prev":             (0.0773, 0.0338),
    "success_prev":             (-0.0290, 0.6576),
}
for term, (exp_est, exp_p) in expected_fe.items():
    got_est, got_p = _get_fe(term)
    match_est = abs(got_est - exp_est) < 0.001
    match_p   = abs(got_p   - exp_p)   < 0.01
    marker    = "✓" if match_est and match_p else "✗"
    print(f"  {marker} {term:25s}  est={got_est:+.4f} (exp {exp_est:+.4f}), "
          f"p={got_p:.4f} (exp {exp_p:.4f})")

# meta check
exp_meta = {"n_obs": "588267", "n_rats": "12", "n_sessions": "812",
            "singular": "FALSE", "converged": "TRUE"}
for k, v in exp_meta.items():
    marker = "✓" if meta.get(k) == v else "✗"
    print(f"  {marker} meta.{k:12s} = {meta.get(k)}  (expected {v})")

# ─────────────────────────────────────────────────────────────────────────────
# Save a consolidated snapshot for the thesis -- commented out for now, so I don't have a ton of snapshots (change if needed)
# ─────────────────────────────────────────────────────────────────────────────

#SNAPSHOT_DIR = Path(r"C:\dev\projects\Thesis_SISSA\Rats\history_dependecy\thesis_fits\glmm_I\snapshot_" 
#                    + datetime.now().strftime("%Y%m%d_%H%M%S"))
#SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# Copy everything used-facing into a timestamped snapshot folder — protects against
# accidental overwrite if you rerun the pipeline later
#for f in RESULTS_DIR.iterdir():
#    if f.is_file():
#        shutil.copy2(f, SNAPSHOT_DIR / f.name)
#print(f"\n✓ Full snapshot copied to {SNAPSHOT_DIR}")



# One-line JSON summary of the headline numbers (handy for the LaTeX \newcommand{} pattern)
headline = {
    "n_obs":              int(meta["n_obs"]),
    "n_rats":             int(meta["n_rats"]),
    "n_sessions":         int(meta["n_sessions"]),
    "singular":           meta["singular"] == "TRUE",
    "converged":          meta["converged"] == "TRUE",
    "fit_minutes":        float(meta["fit_minutes"]),
    "aic":                float(meta["aic"]),
    "bic":                float(meta["bic"]),
    "logLik":             float(meta["logLik"]),
    "angle_prev_c":       _get_fe("angle_prev_c"),
    "success_prev":       _get_fe("success_prev"),
    "failure_prev":       _get_fe("failure_prev"),
    "S_beta_est":         float(results["contrasts"].query("contrast=='S_beta'")["estimate"].iloc[0]),
    "S_beta_p":           float(results["contrasts"].query("contrast=='S_beta'")["p"].iloc[0]),
    "D_beta_est":         float(results["contrasts"].query("contrast=='D_beta'")["estimate"].iloc[0]),
    "D_beta_p":           float(results["contrasts"].query("contrast=='D_beta'")["p"].iloc[0]),
    "tau_S":              float(results["heterogeneity"].query("quantity=='tau_S'")["value"].iloc[0]),
    "tau_F":              float(results["heterogeneity"].query("quantity=='tau_F'")["value"].iloc[0]),
    "cor_SF":             float(results["heterogeneity"].query("quantity=='cor_SF'")["value"].iloc[0]),
    "tau_S_beta":         float(results["heterogeneity"].query("quantity=='tau_S_beta'")["value"].iloc[0]),
    "tau_D_beta":         float(results["heterogeneity"].query("quantity=='tau_D_beta'")["value"].iloc[0]),
}
#with open(SNAPSHOT_DIR / "headline_numbers.json", "w") as f:
#    json.dump(headline, f, indent=2, default=str)
#print(f"✓ Headline numbers written to {SNAPSHOT_DIR / 'headline_numbers.json'}")

print("\n── All done ──")
print(f"Primary results folder:  {RESULTS_DIR}")
#print(f"Backup snapshot folder:  {SNAPSHOT_DIR}")
print(f"Console log preserved:   {CONSOLE_LOG}")