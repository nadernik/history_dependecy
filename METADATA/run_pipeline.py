# Full analysis (includes all rat-specific plots)
analyzer = SerialDependenceAnalyzer()
results = analyzer.run_complete_analysis()

# Focus on specific rat
analyzer.plot_specific_rat(rat_id=2, show_coefficients=True)

# Get summary for all rats
summaries = analyzer.get_rat_summary()

# Create just the heatmap
analyzer.plot_rat_coefficients_heatmap()