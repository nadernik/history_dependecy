"""
Main Entry Point for Serial Dependence Analysis
===============================================

This script provides the main entry point for running the complete
serial dependence analysis pipeline.
"""

import matplotlib.pyplot as plt
from analyzer import SerialDependenceAnalyzer
from logging_utils import AnalysisLogger


def main():
    """
    Main function - runs comprehensive serial dependence analysis.
    
    This function orchestrates the complete analysis pipeline including:
    1. Data loading and preprocessing
    2. Statistical modeling
    3. Individual rat analysis
    4. Modality sequence analysis
    5. Visualization creation
    """
    
    # Initialize logger for the complete analysis
    with AnalysisLogger(log_dir='logs', enable_logging=True) as logger:
        print("🧠 Starting comprehensive serial dependence analysis...")
        print("=" * 80)
        
        # Initialize analyzer with desired parameters
        analyzer = SerialDependenceAnalyzer(
            data_path='data/behavior_data.mat',
            history_depth=5,  # Can be changed as needed
            save_figures=True,
            figures_dir='figures',
            enable_logging=False  # Logger already active
        )
        
        try:
            # Run the complete analysis pipeline
            print("🚀 Launching analysis pipeline...")
            model, history_effects, rat_results = analyzer.run_complete_analysis()
            
            print("\n" + "🎯" * 20)
            print("ANALYSIS RESULTS SUMMARY")
            print("🎯" * 20)
            
            # Print key findings
            if history_effects:
                print(f"\n📊 Key History Effects Identified:")
                effect_types = {}
                for effect_type, lag, feature, coeff in history_effects:
                    if effect_type not in effect_types:
                        effect_types[effect_type] = []
                    effect_types[effect_type].append((lag, coeff))
                
                for effect_type, effects in effect_types.items():
                    print(f"\n  {effect_type.upper()} Effects:")
                    for lag, coeff in sorted(effects):
                        print(f"    n-{lag}: β = {coeff:.4f}")
            
            if rat_results:
                accuracies = [r['accuracy'] for r in rat_results.values()]
                print(f"\n🐭 Individual Rat Performance:")
                print(f"    Number of rats: {len(rat_results)}")
                print(f"    Accuracy range: {min(accuracies):.3f} - {max(accuracies):.3f}")
                print(f"    Mean accuracy: {sum(accuracies)/len(accuracies):.3f}")
            
            print(f"\n✅ Analysis completed successfully!")
            print(f"📁 All figures saved to: figures/")
            print(f"📋 Complete log saved to: logs/")
            
            # Display plots
            print(f"\n🖼️ Displaying visualization plots...")
            print("   All plots will remain open for interaction.")
            print("   Close plot windows when finished reviewing results.")
            
            # Keep all plots open for user interaction
            try:
                import matplotlib
                if matplotlib.get_backend() != 'Agg':  # Only if not headless
                    plt.show(block=True)  # This will keep all figures open
            except Exception as e:
                print(f"Note: Could not display plots interactively: {e}")
            
            return analyzer, (model, history_effects, rat_results)
            
        except Exception as e:
            print(f"\n❌ Analysis failed with error: {e}")
            print("Check the log files for detailed error information.")
            raise


def run_quick_analysis():
    """
    Run a quick analysis with default parameters for testing.
    """
    print("🚀 Running quick analysis (k=3)...")
    
    analyzer = SerialDependenceAnalyzer(
        history_depth=3,
        save_figures=True,
        enable_logging=False
    )
    
    try:
        # Load data
        analyzer.load_data()
        
        # Build model and analyze coefficients
        analyzer.build_model()
        analyzer.analyze_coefficients()
        
        # Create basic visualizations
        if analyzer.df_processed is not None:
            summary_fig = analyzer.plotter.plot_summary_psychometric_by_modality(analyzer.df_processed)
            
        if analyzer.history_effects:
            effects_fig = analyzer.plotter.create_psychometric_plots(analyzer.history_effects, analyzer.df_processed)
        
        print("✅ Quick analysis complete!")
        plt.show()
        
        return analyzer
        
    except Exception as e:
        print(f"❌ Quick analysis failed: {e}")
        raise


def analyze_specific_rat(rat_id: int, history_depth: int = 3):
    """
    Run analysis focused on a specific rat.
    
    Args:
        rat_id: ID of the rat to analyze
        history_depth: History depth for analysis
    """
    print(f"🐭 Analyzing rat {rat_id} (k={history_depth})...")
    
    analyzer = SerialDependenceAnalyzer(
        history_depth=history_depth,
        save_figures=True,
        enable_logging=False
    )
    
    try:
        # Load data and run individual analysis
        analyzer.load_data()
        analyzer.analyze_individual_rats()
        
        # Plot specific rat
        analyzer.plot_specific_rat(rat_id)
        
        # Get summary
        summary = analyzer.get_rat_summary(rat_id)
        print(f"\nRat {rat_id} Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        return analyzer
        
    except Exception as e:
        print(f"❌ Rat analysis failed: {e}")
        raise


if __name__ == "__main__":
    """
    Entry point when script is run directly.
    
    Uncomment different function calls below to run different types of analysis:
    - main(): Complete comprehensive analysis
    - run_quick_analysis(): Quick test with basic features
    - analyze_specific_rat(rat_id): Focus on individual rat
    """
    
    # Run the complete analysis
    analyzer, results = main()
    
    # Alternative: run quick analysis for testing
    # analyzer = run_quick_analysis()
    
    # Alternative: analyze specific rat
    # analyzer = analyze_specific_rat(rat_id=2, history_depth=3)
