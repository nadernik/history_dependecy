#!/usr/bin/env python3
"""
Demo script showing how to use the enhanced rat-specific analysis features
"""

from serial_dependence_analysis import SerialDependenceAnalyzer
import matplotlib.pyplot as plt

def main():
    """Demo of rat-specific analysis features"""
    print("="*60)
    print("DEMO: RAT-SPECIFIC ANALYSIS FEATURES")
    print("="*60)
    
    # Initialize analyzer
    analyzer = SerialDependenceAnalyzer(
        data_path='data/behavior_data.mat',
        history_depth=3,
        csv_path='processed_behavior_data.csv'
    )
    
    # Load data (will use CSV if available)
    print("\n1. Loading data...")
    if analyzer.check_and_load_csv() is None:
        print("No CSV found, processing from scratch...")
        if analyzer.load_and_preprocess_data() is None:
            print("Failed to load data!")
            return
        if analyzer.create_lagged_features() is None:
            print("Failed to create features!")
            return
    
    print(f"Data loaded: {len(analyzer.df_processed)} trials")
    
    # Analyze individual rats
    print("\n2. Analyzing individual rats...")
    rat_results = analyzer.analyze_individual_rats()
    
    if not rat_results:
        print("No rat results available!")
        return
    
    # Show summary for all rats
    print("\n3. Rat summaries:")
    all_summaries = analyzer.get_rat_summary()
    for rat_id, summary in all_summaries.items():
        print(f"  Rat {rat_id}: {summary['n_trials']} trials, "
              f"accuracy = {summary['accuracy']:.3f}")
        print(f"    Top feature: {summary['top_features'][0]['feature']} "
              f"(β = {summary['top_features'][0]['coefficient']:.3f})")
    
    # Plot results for first few rats
    print("\n4. Creating individual rat plots...")
    rats_to_plot = sorted(list(rat_results.keys()))[:3]  # Plot first 3 rats
    
    for rat_id in rats_to_plot:
        print(f"  Plotting Rat {rat_id}...")
        analyzer.plot_specific_rat(rat_id, show_coefficients=True)
    
    # Create coefficient heatmap
    print("\n5. Creating coefficient heatmap...")
    heatmap_fig = analyzer.plot_rat_coefficients_heatmap(rat_results)
    if heatmap_fig:
        plt.show()
    
    # Create summary stats
    print("\n6. Creating summary statistics...")
    summary_fig = analyzer._plot_rat_summary_stats(rat_results)
    if summary_fig:
        plt.show()
    
    print("\n" + "="*60)
    print("DEMO COMPLETED!")
    print("="*60)
    
    return analyzer, rat_results

if __name__ == "__main__":
    analyzer, results = main()
    
    print("\nAvailable functions for further analysis:")
    print("- analyzer.plot_specific_rat(rat_id)")
    print("- analyzer.get_rat_summary(rat_id)")
    print("- analyzer.plot_rat_coefficients_heatmap()")
    print("- analyzer.create_comprehensive_rat_dashboard()")
