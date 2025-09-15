"""
Main Serial Dependence Analyzer - Orchestrator Class
====================================================

This module contains the main SerialDependenceAnalyzer class that orchestrates
all components of the analysis pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any

from config import (
    DEFAULT_DATA_PATH, DEFAULT_CSV_PATH, DEFAULT_FIGURES_DIR, 
    DEFAULT_LOG_DIR, DEFAULT_HISTORY_DEPTH, ANALYSIS_CONFIG
)
from data_loader import DataLoader
from models import SerialDependenceModels
from plotting import SerialDependencePlotter
from logging_utils import AnalysisLogger
from utils import get_top_features


class SerialDependenceAnalyzer:
    """
    Main analyzer class that orchestrates the complete serial dependence analysis pipeline.
    
    This class integrates data loading, statistical modeling, and visualization
    to provide a comprehensive analysis of serial dependence effects.
    """
    
    def __init__(self, 
                 data_path: str = DEFAULT_DATA_PATH,
                 history_depth: int = DEFAULT_HISTORY_DEPTH,
                 csv_path: str = DEFAULT_CSV_PATH,
                 save_figures: bool = True,
                 figures_dir: str = DEFAULT_FIGURES_DIR,
                 enable_logging: bool = False,
                 log_dir: str = DEFAULT_LOG_DIR):
        """
        Initialize the Serial Dependence Analyzer.
        
        Args:
            data_path: Path to the MAT file containing behavioral data
            history_depth: Number of historical trials to include (k parameter)
            csv_path: Path for CSV cache file
            save_figures: Whether to save figures to disk
            figures_dir: Directory to save figures
            enable_logging: Whether to enable logging to files
            log_dir: Directory for log files
        """
        self.data_path = data_path
        self.history_depth = history_depth
        self.csv_path = csv_path
        self.save_figures = save_figures
        self.figures_dir = figures_dir
        self.enable_logging = enable_logging
        self.log_dir = log_dir
        
        # Initialize components
        self.data_loader = DataLoader(data_path, csv_path, history_depth)
        self.models = SerialDependenceModels(history_depth)
        self.plotter = SerialDependencePlotter(save_figures, figures_dir, history_depth)
        
        # Analysis results storage
        self.df_processed: Optional[pd.DataFrame] = None
        self.main_model = None
        self.feature_names: List[str] = []
        self.history_effects: List[Tuple[str, int, str, float]] = []
        self.rat_results: Dict[int, Dict[str, Any]] = {}
        self.modality_results: Dict[str, Any] = {}
        self.modality_history_results: Dict[str, Any] = {}
        
        # Logging
        self.logger: Optional[AnalysisLogger] = None
        
        print(f"Serial Dependence Analyzer initialized")
        print(f"  History depth (k): {self.history_depth}")
        print(f"  Data path: {self.data_path}")
        print(f"  CSV cache: {self.csv_path}")
        if self.save_figures:
            print(f"  Figures directory: {self.figures_dir}")
        if self.enable_logging:
            print(f"  Logging directory: {self.log_dir}")
    
    def start_logging(self) -> None:
        """Start logging analysis output to file"""
        if self.enable_logging and self.logger is None:
            self.logger = AnalysisLogger(self.log_dir, True)
            self.logger.start_logging()
    
    def stop_logging(self) -> None:
        """Stop logging and close log file"""
        if self.logger is not None:
            self.logger.stop_logging()
            self.logger = None
    
    def load_data(self) -> pd.DataFrame:
        """
        Load and preprocess data, using CSV cache if available.
        
        Returns:
            Processed DataFrame with lagged features
        """
        print("\n" + "=" * 60)
        print("STEP 1: DATA LOADING AND PREPROCESSING")
        print("=" * 60)
        
        # Try to load from CSV cache first
        cached_data = self.data_loader.check_and_load_csv()
        
        if cached_data is not None:
            self.df_processed = cached_data
            print(f"✅ Data loaded from cache: {len(self.df_processed):,} trials")
        else:
            # Load from MAT file and create features
            print("Loading from MAT file and creating features...")
            
            # Load raw data
            self.data_loader.load_and_preprocess_data()
            
            # Create lagged features
            self.df_processed = self.data_loader.create_lagged_features()
            
            # Save to CSV for future use
            self.data_loader.save_processed_data_to_csv()
            
            print(f"✅ Data processed: {len(self.df_processed):,} trials with complete history")
        
        # Print data summary
        summary = self.data_loader.get_data_summary()
        print(f"\nData Summary:")
        print(f"  Total trials: {summary['total_trials']:,}")
        print(f"  Number of rats: {summary['n_rats']}")
        print(f"  Rat IDs: {summary['rat_ids']}")
        print(f"  Modalities: {summary['modalities']}")
        print(f"  Overall accuracy: {summary['accuracy_overall']:.3f}")
        print(f"  Features: {summary['n_features']}")
        
        return self.df_processed
    
    def build_model(self) -> Tuple[Any, List[str]]:
        """
        Build the main logistic regression model.
        
        Returns:
            Tuple of (model, feature_names)
        """
        print("\n" + "=" * 60)
        print("STEP 2: STATISTICAL MODELING")
        print("=" * 60)
        
        if self.df_processed is None:
            raise ValueError("Must load data first")
        
        # Build main model
        self.main_model, self.feature_names = self.models.build_simple_model(self.df_processed)
        
        print(f"✅ Main model built with {len(self.feature_names)} features")
        
        return self.main_model, self.feature_names
    
    def analyze_coefficients(self) -> List[Tuple[str, int, str, float]]:
        """
        Analyze model coefficients to identify significant history effects.
        
        Returns:
            List of (effect_type, lag, feature_name, coefficient) tuples
        """
        if self.main_model is None:
            raise ValueError("Must build model first")
        
        self.history_effects = self.models.analyze_coefficients(self.main_model, self.feature_names)
        
        print(f"✅ Identified {len(self.history_effects)} significant history effects")
        
        return self.history_effects
    
    def analyze_individual_rats(self) -> Dict[int, Dict[str, Any]]:
        """
        Perform individual rat analysis.
        
        Returns:
            Dictionary with individual rat results
        """
        print("\n" + "=" * 60)
        print("STEP 3: INDIVIDUAL RAT ANALYSIS")
        print("=" * 60)
        
        if self.df_processed is None:
            raise ValueError("Must load data first")
        
        self.rat_results = self.models.analyze_individual_rats(self.df_processed)
        
        print(f"✅ Individual analysis complete for {len(self.rat_results)} rats")
        
        return self.rat_results
    
    def analyze_modality_sequences(self) -> Dict[str, Any]:
        """
        Analyze modality-specific sequences (homo vs hetero-modal).
        
        Returns:
            Dictionary with modality sequence results
        """
        print("\n" + "=" * 60)
        print("STEP 4: MODALITY SEQUENCE ANALYSIS")
        print("=" * 60)
        
        if self.df_processed is None:
            raise ValueError("Must load data first")
        
        self.modality_results = self.models.analyze_modality_sequences(self.df_processed)
        
        if 'error' not in self.modality_results:
            print(f"✅ Modality sequence analysis complete")
            print(f"  Homo-modal trials: {self.modality_results['homo_count']:,}")
            print(f"  Hetero-modal trials: {self.modality_results['hetero_count']:,}")
        else:
            print(f"⚠️ Modality analysis failed: {self.modality_results['error']}")
        
        return self.modality_results
    
    def analyze_modality_specific_history_effects(self) -> Dict[str, Any]:
        """
        Analyze history effects specific to each modality.
        
        Returns:
            Dictionary with modality-specific history effects
        """
        print("\n" + "=" * 60)
        print("STEP 5: MODALITY-SPECIFIC HISTORY EFFECTS")
        print("=" * 60)
        
        if self.df_processed is None:
            raise ValueError("Must load data first")
        
        self.modality_history_results = self.models.analyze_modality_specific_history_effects(self.df_processed)
        
        print(f"✅ Modality-specific history analysis complete")
        
        return self.modality_history_results
    
    def create_visualizations(self) -> List[Any]:
        """
        Create all visualization plots.
        
        Returns:
            List of matplotlib figures
        """
        print("\n" + "=" * 60)
        print("STEP 6: VISUALIZATION CREATION")
        print("=" * 60)
        
        figures = []
        
        # Summary psychometric by modality
        if self.df_processed is not None:
            print("Creating summary psychometric curves...")
            summary_fig = self.plotter.plot_summary_psychometric_by_modality(self.df_processed)
            if summary_fig:
                figures.append(summary_fig)
        
        # History effects plots
        if self.history_effects:
            print("Creating history effects plots...")
            effects_fig = self.plotter.create_psychometric_plots(self.history_effects, self.df_processed)
            if effects_fig:
                figures.append(effects_fig)
        
        # Individual rat results
        if self.rat_results:
            print("Creating individual rat plots...")
            rat_figs = self.plotter.plot_individual_rat_results(self.rat_results)
            figures.extend(rat_figs)
            
            # Coefficients heatmap
            print("Creating coefficients heatmap...")
            heatmap_fig = self.plotter.plot_rat_coefficients_heatmap(self.rat_results)
            if heatmap_fig:
                figures.append(heatmap_fig)
        
        # Modality sequence results
        if self.modality_results and 'error' not in self.modality_results:
            print("Creating modality sequence plots...")
            modality_fig = self.plotter.plot_modality_sequence_results(self.modality_results)
            if modality_fig:
                figures.append(modality_fig)
        
        # Modality-specific history effects
        if self.modality_history_results:
            print("Creating modality-specific history plots...")
            mod_history_fig = self.plotter.plot_modality_specific_history_effects(self.modality_history_results)
            if mod_history_fig:
                figures.append(mod_history_fig)
        
        # Temporal patterns
        if self.history_effects:
            print("Creating temporal pattern plots...")
            temporal_fig = self.plotter.plot_temporal_history_pattern(self.history_effects)
            if temporal_fig:
                figures.append(temporal_fig)
        
        print(f"✅ Created {len(figures)} visualization figures")
        
        return figures
    
    def run_complete_analysis(self) -> Tuple[Any, List[Tuple[str, int, str, float]], Dict[int, Dict[str, Any]]]:
        """
        Run the complete analysis pipeline.
        
        Returns:
            Tuple of (main_model, history_effects, rat_results)
        """
        print("\n" + "🧠" * 20)
        print("SERIAL DEPENDENCE ANALYSIS - COMPLETE PIPELINE")
        print("🧠" * 20)
        
        try:
            # Step 1: Load data
            self.load_data()
            
            # Step 2: Build main model
            self.build_model()
            
            # Step 3: Analyze coefficients
            self.analyze_coefficients()
            
            # Step 4: Individual rat analysis
            self.analyze_individual_rats()
            
            # Step 5: Modality sequence analysis
            self.analyze_modality_sequences()
            
            # Step 6: Modality-specific history effects
            self.analyze_modality_specific_history_effects()
            
            # Step 7: Create visualizations
            self.create_visualizations()
            
            print("\n" + "🎉" * 20)
            print("ANALYSIS COMPLETE!")
            print("🎉" * 20)
            
            # Print summary
            self._print_analysis_summary()
            
            return self.main_model, self.history_effects, self.rat_results
            
        except Exception as e:
            print(f"\n❌ Analysis failed: {e}")
            raise
    
    def _print_analysis_summary(self) -> None:
        """Print a summary of the analysis results"""
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        
        if self.df_processed is not None:
            print(f"📊 Data: {len(self.df_processed):,} trials from {self.df_processed['rat'].nunique()} rats")
        
        if self.main_model is not None:
            # Calculate overall accuracy
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            
            # Prepare features for accuracy calculation
            features = []
            for feature_name in self.feature_names:
                if feature_name == 'angle':
                    features.append(self.df_processed['angle'].values)
                elif feature_name.startswith('mod_'):
                    mod_id = int(feature_name.split('_')[1])
                    mod_feature = (self.df_processed['mod'] == mod_id).astype(int)
                    features.append(mod_feature.values)
                elif feature_name in self.df_processed.columns:
                    features.append(self.df_processed[feature_name].values)
            
            if features:
                X = np.column_stack(features)
                X_scaled = scaler.fit_transform(X)
                y = self.df_processed['action'].values
                accuracy = self.main_model.score(X_scaled, y)
                print(f"🎯 Model Accuracy: {accuracy:.3f}")
        
        if self.history_effects:
            print(f"🔍 History Effects: {len(self.history_effects)} significant effects identified")
            
            # Group by effect type
            effect_counts = {}
            for effect_type, lag, feature, coeff in self.history_effects:
                effect_counts[effect_type] = effect_counts.get(effect_type, 0) + 1
            
            for effect_type, count in effect_counts.items():
                print(f"   - {effect_type.capitalize()}: {count} effects")
        
        if self.rat_results:
            accuracies = [r['accuracy'] for r in self.rat_results.values()]
            print(f"🐭 Individual Rats: {len(self.rat_results)} rats analyzed")
            print(f"   - Accuracy range: {min(accuracies):.3f} - {max(accuracies):.3f}")
            print(f"   - Mean accuracy: {np.mean(accuracies):.3f}")
        
        if self.modality_results and 'error' not in self.modality_results:
            total_tv = self.modality_results['homo_count'] + self.modality_results['hetero_count']
            homo_pct = self.modality_results['homo_count'] / total_tv * 100
            print(f"🔄 Modality Sequences: {total_tv:,} Touch-Vision trials")
            print(f"   - Homo-modal: {homo_pct:.1f}%")
            print(f"   - Hetero-modal: {100-homo_pct:.1f}%")
        
        print("\n✅ All analyses completed successfully!")
    
    def get_rat_summary(self, rat_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Get summary statistics for a specific rat or all rats.
        
        Args:
            rat_id: Specific rat ID, or None for all rats
            
        Returns:
            Dictionary with summary statistics
        """
        if not self.rat_results:
            return {"error": "No rat results available"}
        
        if rat_id is not None:
            if rat_id not in self.rat_results:
                return {"error": f"Rat {rat_id} not found"}
            
            result = self.rat_results[rat_id]
            top_features = get_top_features(result['coefficients'], result['feature_names'], n=5)
            
            return {
                "rat_id": rat_id,
                "accuracy": result['accuracy'],
                "n_trials": result['n_trials'],
                "top_features": top_features
            }
        else:
            # Summary for all rats
            rats = sorted(self.rat_results.keys())
            accuracies = [self.rat_results[r]['accuracy'] for r in rats]
            n_trials = [self.rat_results[r]['n_trials'] for r in rats]
            
            return {
                "n_rats": len(rats),
                "rat_ids": rats,
                "accuracy_mean": np.mean(accuracies),
                "accuracy_std": np.std(accuracies),
                "accuracy_range": (min(accuracies), max(accuracies)),
                "trials_mean": np.mean(n_trials),
                "trials_total": sum(n_trials)
            }
    
    def plot_specific_rat(self, rat_id: int, show_coefficients: bool = True) -> None:
        """
        Create detailed plots for a specific rat.
        
        Args:
            rat_id: ID of the rat to plot
            show_coefficients: Whether to show coefficient details
        """
        if rat_id not in self.rat_results:
            print(f"Rat {rat_id} not found in results")
            return
        
        print(f"\n=== DETAILED ANALYSIS FOR RAT {rat_id} ===")
        
        rat_result = self.rat_results[rat_id]
        
        # Create individual psychometric plot
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        self.plotter._plot_rat_psychometric(ax, rat_id, rat_result)
        plt.tight_layout()
        
        if self.save_figures:
            from utils import save_figure
            save_figure(fig, f'rat_{rat_id}_psychometric_k{self.history_depth}.png', self.figures_dir)
        
        # Print summary
        print(f"Trials: {rat_result['n_trials']:,}")
        print(f"Accuracy: {rat_result['accuracy']:.3f}")
        
        if show_coefficients:
            print("\nTop 5 features:")
            for feature, coeff in rat_result['top_features']:
                print(f"  {feature}: β = {coeff:.4f}")
        
        plt.show()
    
    def __enter__(self):
        """Context manager entry"""
        self.start_logging()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_logging()
