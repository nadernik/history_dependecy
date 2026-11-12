# DEBUG re-run (12 Sep 2026) - not thesis results

Partial re-run of task_performance_check_thesis.py (n_boot = 500, rats 5 and 7,
unfiltered only), made to investigate the quirk in ../task_check_output_boot.
Original (pre-reclassify) settings: fit_degenerate_sigma = 45.1, good/medium/poor tiers.
Known difference: rat_5_1 point estimate popt differs strongly from task_check_output_boot
([67.6, 33.3, 0.23, 0.12] vs [43.5, 28.6, 0.15, 0.30]) while the bootstrap draws agree.
