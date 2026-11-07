"""Runner — regenerate all four Section-4 pooled figures for one or more subsets.

Usage
-----
Regenerate everything for pooled (default):
    python run_all_pooled_figures.py

Regenerate everything for the three modality subsets:
    python run_all_pooled_figures.py --subsets tactile visual visuotactile

Regenerate a single figure across all subsets:
    python run_all_pooled_figures.py --scripts figure_p1_sd_plane.py \
        --subsets pooled tactile visual visuotactile

Pass extra flags to the individual scripts by putting them after ``--``:
    python run_all_pooled_figures.py --subsets pooled -- --overlay-psychometric
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_SCRIPTS = (
    "figure_p1_sd_plane.py",
    "figure_p2_coefficient_forest.py",
    "figure_p3_sdp_forest.py",
    "table_p4_profile_comparison.py",
)
DEFAULT_SUBSETS = ("pooled",)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subsets", nargs="+", default=list(DEFAULT_SUBSETS),
        help="Subsets to run through (default: pooled). "
             "Any subset present in the data works.",
    )
    parser.add_argument(
        "--scripts", nargs="+", default=list(DEFAULT_SCRIPTS),
        help="Scripts to run (default: all four).",
    )
    parser.add_argument(
        "extra", nargs=argparse.REMAINDER,
        help="Extra flags passed through to each script (put after --).",
    )
    args = parser.parse_args()

    # argparse.REMAINDER captures a leading '--' as its first element on
    # some Python versions; drop it so we don't pass '--' to the scripts.
    extra = list(args.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]

    here = Path(__file__).resolve().parent
    fail: list[tuple[str, str, int]] = []
    for subset in args.subsets:
        for script in args.scripts:
            script_path = here / script
            if not script_path.exists():
                print(f"[ERROR] script not found: {script_path}", file=sys.stderr)
                fail.append((subset, script, -1))
                continue
            cmd = [sys.executable, str(script_path), "--subset", subset, *extra]
            banner = f"=== {subset:<12} :: {script} ==="
            print("\n" + banner)
            print(" ".join(cmd))
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"[FAIL] {script} for subset={subset} "
                      f"(exit code {result.returncode})", file=sys.stderr)
                fail.append((subset, script, result.returncode))

    print("\n" + "=" * 60)
    n_runs = len(args.subsets) * len(args.scripts)
    n_ok = n_runs - len(fail)
    print(f"Completed: {n_ok}/{n_runs}")
    if fail:
        print("Failures:")
        for subset, script, code in fail:
            print(f"  - subset={subset}, script={script}, exit={code}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())