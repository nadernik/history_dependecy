"""
Plotting and Visualization for Serial Dependence Analysis
=========================================================

This module contains all plotting and visualization functions for
psychometric curves, coefficient analysis, and comprehensive dashboards.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats

from config import (
    MODALITY_NAMES, MODALITY_COLORS, DEFAULT_FIGURE_SIZE, 
    FIGURE_NAME_PATTERNS, DEFAULT_MAX_RATS_PER_FIGURE
)
from utils import fit_psychometric_curve, save_figure, create_angle_bins


class SerialDependencePlotter:
    """Class containing all plotting methods for serial dependence analysis"""
    
    def __init__(self, save_figures: bool = True, figures_dir: str = 'figures', 
                 history_depth: int = 3):
        """
        Initialize the plotter.
        
        Args:
            save_figures: Whether to save figures to disk
            figures_dir: Directory to save figures
            history_depth: History depth for naming files
        """
        self.save_figures = save_figures
        self.figures_dir = figures_dir
        self.history_depth = history_depth
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = DEFAULT_FIGURE_SIZE
    
    def create_psychometric_plots(self, history_effects: List[Tuple[str, int, str, float]], 
                                df_processed: pd.DataFrame) -> plt.Figure:
        """
        Create psychometric plots showing history effects.
        
        Args:
            history_effects: List of (effect_type, lag, feature_name, coefficient) tuples
            df_processed: Processed DataFrame
            
        Returns:
            Matplotlib figure
        """
        print("\n=== CREATING PSYCHOMETRIC PLOTS ===")
        
        if not history_effects:
            print("No history effects to plot")
            return None
        
        # Organize effects by type
        effect_types = {}
        for effect_type, lag, feature, coeff in history_effects:
            if effect_type not in effect_types:
                effect_types[effect_type] = []
            effect_types[effect_type].append((lag, feature, coeff))
        
        # Create subplots
        n_types = len(effect_types)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        
        # Plot each effect type
        for effect_type in ['action', 'angle', 'hitmiss', 'difficulty']:
            if effect_type in effect_types and plot_idx < 4:
                ax = axes[plot_idx]
                
                if effect_type == 'action':
                    self._plot_action_effect(ax, effect_types[effect_type], df_processed)
                elif effect_type == 'angle':
                    self._plot_angle_effect(ax, effect_types[effect_type], df_processed)
                elif effect_type == 'hitmiss':
                    self._plot_hitmiss_effect(ax, effect_types[effect_type], df_processed)
                elif effect_type == 'difficulty':
                    self._plot_difficulty_effect(ax, effect_types[effect_type], df_processed)
                
                plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        if self.save_figures:
            filename = FIGURE_NAME_PATTERNS['effects'].format(self.history_depth)
            save_figure(fig, filename, self.figures_dir)
        
        return fig
    
    def _plot_action_effect(self, ax: plt.Axes, effects: List[Tuple[int, str, float]], 
                          df_processed: pd.DataFrame) -> None:
        """Plot action (choice) history effects"""
        angle_bins, angle_centers = create_angle_bins()
        
        for lag, feature, coeff in effects:
            # Create bins based on previous choice
            choice_0_data = df_processed[df_processed[f'action_n-{lag}'] == 0]
            choice_1_data = df_processed[df_processed[f'action_n-{lag}'] == 1]
            
            # Calculate psychometric curves
            props_0, props_1 = [], []
            
            for i in range(len(angle_bins) - 1):
                bin_mask = (df_processed['angle'] >= angle_bins[i]) & (df_processed['angle'] < angle_bins[i + 1])
                
                # Choice = 0 condition
                subset_0 = choice_0_data[choice_0_data['angle'].between(angle_bins[i], angle_bins[i + 1])]
                if len(subset_0) > 5:
                    props_0.append(subset_0['action'].mean())
                else:
                    props_0.append(np.nan)
                
                # Choice = 1 condition
                subset_1 = choice_1_data[choice_1_data['angle'].between(angle_bins[i], angle_bins[i + 1])]
                if len(subset_1) > 5:
                    props_1.append(subset_1['action'].mean())
                else:
                    props_1.append(np.nan)
            
            # Plot curves
            ax.plot(angle_centers, props_0, 'o-', label=f'Previous Left (n-{lag})', alpha=0.7)
            ax.plot(angle_centers, props_1, 's-', label=f'Previous Right (n-{lag})', alpha=0.7)
        
        ax.set_xlabel('Stimulus Angle (degrees)')
        ax.set_ylabel('P(Choose Right)')
        ax.set_title(f'Choice History Effects (β={effects[0][2]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_angle_effect(self, ax: plt.Axes, effects: List[Tuple[int, str, float]], 
                         df_processed: pd.DataFrame) -> None:
        """Plot angle (perceptual) history effects"""
        angle_bins, angle_centers = create_angle_bins()
        
        for lag, feature, coeff in effects:
            # Split by previous angle (low vs high)
            median_angle = df_processed[f'angle_n-{lag}'].median()
            low_angle_data = df_processed[df_processed[f'angle_n-{lag}'] <= median_angle]
            high_angle_data = df_processed[df_processed[f'angle_n-{lag}'] > median_angle]
            
            # Calculate psychometric curves
            props_low, props_high = [], []
            
            for i in range(len(angle_bins) - 1):
                # Low previous angle condition
                subset_low = low_angle_data[low_angle_data['angle'].between(angle_bins[i], angle_bins[i + 1])]
                if len(subset_low) > 5:
                    props_low.append(subset_low['action'].mean())
                else:
                    props_low.append(np.nan)
                
                # High previous angle condition
                subset_high = high_angle_data[high_angle_data['angle'].between(angle_bins[i], angle_bins[i + 1])]
                if len(subset_high) > 5:
                    props_high.append(subset_high['action'].mean())
                else:
                    props_high.append(np.nan)
            
            # Plot curves
            ax.plot(angle_centers, props_low, 'o-', label=f'Previous Low Angle (n-{lag})', alpha=0.7)
            ax.plot(angle_centers, props_high, 's-', label=f'Previous High Angle (n-{lag})', alpha=0.7)
        
        ax.set_xlabel('Stimulus Angle (degrees)')
        ax.set_ylabel('P(Choose Right)')
        ax.set_title(f'Perceptual History Effects (β={effects[0][2]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_hitmiss_effect(self, ax: plt.Axes, effects: List[Tuple[int, str, float]], 
                           df_processed: pd.DataFrame) -> None:
        """Plot hit/miss (outcome) history effects"""
        angle_bins, angle_centers = create_angle_bins()
        
        for lag, feature, coeff in effects:
            # Split by previous outcome
            hit_data = df_processed[df_processed[f'hitmiss_n-{lag}'] == 1]
            miss_data = df_processed[df_processed[f'hitmiss_n-{lag}'] == 0]
            
            # Calculate psychometric curves
            props_hit, props_miss = [], []
            
            for i in range(len(angle_bins) - 1):
                # Previous hit condition
                subset_hit = hit_data[hit_data['angle'].between(angle_bins[i], angle_bins[i + 1])]
                if len(subset_hit) > 5:
                    props_hit.append(subset_hit['action'].mean())
                else:
                    props_hit.append(np.nan)
                
                # Previous miss condition
                subset_miss = miss_data[miss_data['angle'].between(angle_bins[i], angle_bins[i + 1])]
                if len(subset_miss) > 5:
                    props_miss.append(subset_miss['action'].mean())
                else:
                    props_miss.append(np.nan)
            
            # Plot curves
            ax.plot(angle_centers, props_hit, 'o-', label=f'Previous Hit (n-{lag})', alpha=0.7)
            ax.plot(angle_centers, props_miss, 's-', label=f'Previous Miss (n-{lag})', alpha=0.7)
        
        ax.set_xlabel('Stimulus Angle (degrees)')
        ax.set_ylabel('P(Choose Right)')
        ax.set_title(f'Outcome History Effects (β={effects[0][2]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_difficulty_effect(self, ax: plt.Axes, effects: List[Tuple[int, str, float]], 
                              df_processed: pd.DataFrame) -> None:
        """Plot difficulty (confidence) history effects"""
        angle_bins, angle_centers = create_angle_bins()
        
        for lag, feature, coeff in effects:
            # Split by previous difficulty (easy vs hard)
            median_difficulty = df_processed[f'difficulty_n-{lag}'].median()
            easy_data = df_processed[df_processed[f'difficulty_n-{lag}'] >= median_difficulty]  # Far from boundary
            hard_data = df_processed[df_processed[f'difficulty_n-{lag}'] < median_difficulty]   # Near boundary
            
            # Calculate psychometric curves
            props_easy, props_hard = [], []
            
            for i in range(len(angle_bins) - 1):
                # Previous easy trial condition
                subset_easy = easy_data[easy_data['angle'].between(angle_bins[i], angle_bins[i + 1])]
                if len(subset_easy) > 5:
                    props_easy.append(subset_easy['action'].mean())
                else:
                    props_easy.append(np.nan)
                
                # Previous hard trial condition
                subset_hard = hard_data[hard_data['angle'].between(angle_bins[i], angle_bins[i + 1])]
                if len(subset_hard) > 5:
                    props_hard.append(subset_hard['action'].mean())
                else:
                    props_hard.append(np.nan)
            
            # Plot curves
            ax.plot(angle_centers, props_easy, 'o-', label=f'Previous Easy (n-{lag})', alpha=0.7)
            ax.plot(angle_centers, props_hard, 's-', label=f'Previous Hard (n-{lag})', alpha=0.7)
        
        ax.set_xlabel('Stimulus Angle (degrees)')
        ax.set_ylabel('P(Choose Right)')
        ax.set_title(f'Difficulty History Effects (β={effects[0][2]:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_individual_rat_results(self, rat_results: Dict[int, Dict[str, Any]], 
                                   max_rats_per_figure: int = DEFAULT_MAX_RATS_PER_FIGURE) -> List[plt.Figure]:
        """
        Plot psychometric results for individual rats.
        
        Args:
            rat_results: Dictionary with individual rat analysis results
            max_rats_per_figure: Maximum number of rats per figure
            
        Returns:
            List of matplotlib figures
        """
        print(f"\n=== PLOTTING INDIVIDUAL RAT RESULTS ===")
        
        if not rat_results:
            print("No rat results to plot")
            return []
        
        rats = sorted(rat_results.keys())
        n_rats = len(rats)
        n_figures = (n_rats + max_rats_per_figure - 1) // max_rats_per_figure
        
        figures = []
        
        for fig_idx in range(n_figures):
            start_idx = fig_idx * max_rats_per_figure
            end_idx = min(start_idx + max_rats_per_figure, n_rats)
            figure_rats = rats[start_idx:end_idx]
            
            n_rats_this_fig = len(figure_rats)
            
            # Create subplot grid
            if n_rats_this_fig <= 3:
                rows, cols = 1, n_rats_this_fig
                figsize = (5 * n_rats_this_fig, 5)
            else:
                rows, cols = 2, 3
                figsize = (15, 10)
            
            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            if n_rats_this_fig == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            # Plot each rat
            for i, rat_id in enumerate(figure_rats):
                ax = axes[i]
                self._plot_rat_psychometric(ax, rat_id, rat_results[rat_id])
            
            # Hide unused subplots
            for i in range(n_rats_this_fig, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Save figure
            if self.save_figures:
                set_num = fig_idx + 1
                filename = FIGURE_NAME_PATTERNS[f'individual_set{set_num}'].format(self.history_depth)
                save_figure(fig, filename, self.figures_dir)
            
            figures.append(fig)
        
        print(f"✅ Created {len(figures)} individual rat figures")
        return figures
    
    def _plot_rat_psychometric(self, ax: plt.Axes, rat_id: int, rat_result: Dict[str, Any]) -> None:
        """Plot psychometric curve for a single rat"""
        rat_data = rat_result['data']
        
        # Create angle bins and calculate performance
        angle_bins, angle_centers = create_angle_bins()
        
        proportions = []
        errors = []
        
        for i in range(len(angle_bins) - 1):
            mask = (rat_data['angle'] >= angle_bins[i]) & (rat_data['angle'] < angle_bins[i + 1])
            if np.sum(mask) >= 5:
                prop = np.mean(rat_data[mask]['action'])
                error = np.sqrt(prop * (1 - prop) / np.sum(mask))
                proportions.append(prop)
                errors.append(error)
            else:
                proportions.append(np.nan)
                errors.append(0)
        
        proportions = np.array(proportions)
        errors = np.array(errors)
        
        # Plot data points
        valid_mask = ~np.isnan(proportions)
        ax.errorbar(angle_centers[valid_mask], proportions[valid_mask], 
                   yerr=errors[valid_mask], fmt='ko', alpha=0.7, capsize=3)
        
        # Fit and plot psychometric curve
        angles_all = rat_data['angle'].values
        responses_all = rat_data['action'].values
        
        fit_params, fit_success, x_fit, y_fit = fit_psychometric_curve(angles_all, responses_all)
        
        if fit_success and x_fit is not None:
            ax.plot(x_fit, y_fit, 'r-', linewidth=2, alpha=0.8)
        
        # Formatting
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('P(Choose Right)')
        ax.set_title(f'Rat {rat_id}\nAcc: {rat_result["accuracy"]:.3f}, N: {rat_result["n_trials"]:,}')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
    
    def plot_rat_coefficients_heatmap(self, rat_results: Dict[int, Dict[str, Any]]) -> plt.Figure:
        """
        Create a heatmap of coefficients across rats.
        
        Args:
            rat_results: Dictionary with individual rat analysis results
            
        Returns:
            Matplotlib figure
        """
        print("\n=== CREATING COEFFICIENTS HEATMAP ===")
        
        if not rat_results:
            print("No rat results for heatmap")
            return None
        
        # Collect coefficients from all rats
        rats = sorted(rat_results.keys())
        feature_names = rat_results[rats[0]]['feature_names']
        
        coeff_matrix = []
        for rat_id in rats:
            coeff_matrix.append(rat_results[rat_id]['coefficients'])
        
        coeff_matrix = np.array(coeff_matrix)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(coeff_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
        
        # Set ticks and labels
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_yticks(range(len(rats)))
        ax.set_yticklabels([f'Rat {r}' for r in rats])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Coefficient Value')
        
        ax.set_title('Model Coefficients Across Rats')
        plt.tight_layout()
        
        # Save figure
        if self.save_figures:
            filename = FIGURE_NAME_PATTERNS['heatmap'].format(self.history_depth)
            save_figure(fig, filename, self.figures_dir)
        
        return fig
    
    def plot_summary_psychometric_by_modality(self, df_processed: pd.DataFrame) -> plt.Figure:
        """
        Plot summary psychometric curves by modality.
        
        Args:
            df_processed: Processed DataFrame
            
        Returns:
            Matplotlib figure
        """
        print("\n=== CREATING SUMMARY PSYCHOMETRIC BY MODALITY ===")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        modalities = [1, 2, 3]  # Touch, Vision, Visual-Tactile
        
        for i, mod in enumerate(modalities):
            ax = axes[i]
            mod_data = df_processed[df_processed['mod'] == mod]
            
            if len(mod_data) == 0:
                ax.text(0.5, 0.5, f'No data for\nModality {mod}', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Calculate psychometric curve
            angle_bins, angle_centers = create_angle_bins()
            proportions = []
            errors = []
            
            for j in range(len(angle_bins) - 1):
                mask = (mod_data['angle'] >= angle_bins[j]) & (mod_data['angle'] < angle_bins[j + 1])
                if np.sum(mask) >= 10:
                    prop = np.mean(mod_data[mask]['action'])
                    error = np.sqrt(prop * (1 - prop) / np.sum(mask))
                    proportions.append(prop)
                    errors.append(error)
                else:
                    proportions.append(np.nan)
                    errors.append(0)
            
            proportions = np.array(proportions)
            errors = np.array(errors)
            valid_mask = ~np.isnan(proportions)
            
            # Plot data points
            color = MODALITY_COLORS.get(MODALITY_NAMES[mod], 'black')
            ax.errorbar(angle_centers[valid_mask], proportions[valid_mask], 
                       yerr=errors[valid_mask], fmt='o', color=color, 
                       alpha=0.7, capsize=3, markersize=8)
            
            # Fit and plot curve
            fit_params, fit_success, x_fit, y_fit = fit_psychometric_curve(
                mod_data['angle'].values, mod_data['action'].values)
            
            if fit_success and x_fit is not None:
                ax.plot(x_fit, y_fit, '-', color=color, linewidth=3, alpha=0.8)
            
            # Calculate overall accuracy
            accuracy = mod_data['hitmiss'].mean()
            
            # Formatting
            ax.set_xlabel('Stimulus Angle (degrees)')
            ax.set_ylabel('P(Choose Right)')
            ax.set_title(f'{MODALITY_NAMES[mod]} (N={len(mod_data):,})\nAccuracy: {accuracy:.3f}')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        if self.save_figures:
            filename = FIGURE_NAME_PATTERNS['summary'].format(self.history_depth)
            save_figure(fig, filename, self.figures_dir)
        
        return fig
    
    def plot_modality_sequence_results(self, modality_results: Dict[str, Any]) -> plt.Figure:
        """
        Plot results from modality sequence analysis.
        
        Args:
            modality_results: Results from analyze_modality_sequences
            
        Returns:
            Matplotlib figure
        """
        print("\n=== PLOTTING MODALITY SEQUENCE RESULTS ===")
        
        if 'error' in modality_results:
            print(f"Cannot plot: {modality_results['error']}")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Sequence type distribution
        ax = axes[0, 0]
        counts = [modality_results['homo_count'], modality_results['hetero_count']]
        labels = ['Homo-modal', 'Hetero-modal']
        colors = ['skyblue', 'lightcoral']
        
        bars = ax.bar(labels, counts, color=colors)
        ax.set_ylabel('Number of Trials')
        ax.set_title('Sequence Type Distribution')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                   f'{count:,}', ha='center', va='bottom')
        
        # Plot 2: Perceptual effects comparison
        ax = axes[0, 1]
        perceptual = modality_results.get('perceptual_results', {})
        if perceptual:
            seq_types = list(perceptual.keys())
            effects = [perceptual[st]['angle_effect'] for st in seq_types]
            
            bars = ax.bar(seq_types, effects, color=['skyblue', 'lightcoral'])
            ax.set_ylabel('Perceptual Effect (β)')
            ax.set_title('Perceptual History Effects')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Add value labels
            for bar, effect in zip(bars, effects):
                ax.text(bar.get_x() + bar.get_width()/2, 
                       bar.get_height() + 0.01 if effect > 0 else bar.get_height() - 0.01,
                       f'{effect:.3f}', ha='center', va='bottom' if effect > 0 else 'top')
        
        # Plot 3: Choice effects comparison
        ax = axes[1, 0]
        choice = modality_results.get('choice_results', {})
        if choice:
            seq_types = list(choice.keys())
            effects = [choice[st]['choice_effect'] for st in seq_types]
            
            bars = ax.bar(seq_types, effects, color=['skyblue', 'lightcoral'])
            ax.set_ylabel('Choice Effect (β)')
            ax.set_title('Sequential Choice Effects')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Add value labels
            for bar, effect in zip(bars, effects):
                ax.text(bar.get_x() + bar.get_width()/2,
                       bar.get_height() + 0.01 if effect > 0 else bar.get_height() - 0.01,
                       f'{effect:.3f}', ha='center', va='bottom' if effect > 0 else 'top')
        
        # Plot 4: Hypothesis test results
        ax = axes[1, 1]
        hyp = modality_results.get('hypothesis_results', {})
        if hyp:
            correlations = [hyp.get('perceptual_correlation', 0), hyp.get('choice_correlation', 0)]
            labels = ['Perceptual\nCorrelation', 'Choice\nCorrelation']
            
            bars = ax.bar(labels, correlations, color=['lightgreen', 'orange'])
            ax.set_ylabel('Correlation Coefficient')
            ax.set_title('Hetero-modal Correlations')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Add value labels
            for bar, corr in zip(bars, correlations):
                ax.text(bar.get_x() + bar.get_width()/2,
                       bar.get_height() + 0.005 if corr > 0 else bar.get_height() - 0.005,
                       f'{corr:.3f}', ha='center', va='bottom' if corr > 0 else 'top')
        
        plt.tight_layout()
        
        # Save figure
        if self.save_figures:
            filename = FIGURE_NAME_PATTERNS['modality_test'].format(self.history_depth)
            save_figure(fig, filename, self.figures_dir)
        
        return fig
    
    def plot_modality_specific_history_effects(self, modality_history_results: Dict[str, Any]) -> plt.Figure:
        """
        Plot modality-specific history effects.
        
        Args:
            modality_history_results: Results from analyze_modality_specific_history_effects
            
        Returns:
            Matplotlib figure
        """
        print("\n=== PLOTTING MODALITY-SPECIFIC HISTORY EFFECTS ===")
        
        if not modality_history_results:
            print("No modality history results to plot")
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        modalities = [1, 2, 3]
        
        for i, mod in enumerate(modalities):
            ax = axes[i]
            
            if mod not in modality_history_results:
                ax.text(0.5, 0.5, f'No data for\nModality {MODALITY_NAMES[mod]}', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            mod_results = modality_history_results[mod]
            lag_results = mod_results.get('lag_results', {})
            
            if not lag_results:
                continue
            
            lags = sorted(lag_results.keys())
            coefficients = [lag_results[lag]['coefficient'] for lag in lags]
            
            # Plot coefficients
            color = MODALITY_COLORS.get(MODALITY_NAMES[mod], 'black')
            bars = ax.bar([f'n-{lag}' for lag in lags], coefficients, color=color, alpha=0.7)
            
            # Add value labels
            for bar, coeff in zip(bars, coefficients):
                ax.text(bar.get_x() + bar.get_width()/2,
                       bar.get_height() + 0.005 if coeff > 0 else bar.get_height() - 0.005,
                       f'{coeff:.3f}', ha='center', va='bottom' if coeff > 0 else 'top')
            
            ax.set_ylabel('Choice Effect (β)')
            ax.set_title(f'{MODALITY_NAMES[mod]} Modality\n(N={mod_results["n_trials"]:,})')
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Trial Lag')
        
        plt.tight_layout()
        
        # Save figure
        if self.save_figures:
            filename = FIGURE_NAME_PATTERNS['modality_history'].format(self.history_depth)
            save_figure(fig, filename, self.figures_dir)
        
        return fig
    
    def plot_temporal_history_pattern(self, history_effects: List[Tuple[str, int, str, float]]) -> plt.Figure:
        """
        Plot temporal patterns of history effects.
        
        Args:
            history_effects: List of (effect_type, lag, feature_name, coefficient) tuples
            
        Returns:
            Matplotlib figure
        """
        print("\n=== PLOTTING TEMPORAL HISTORY PATTERNS ===")
        
        if not history_effects:
            print("No history effects to plot")
            return None
        
        # Organize effects by type and lag
        effect_data = {}
        for effect_type, lag, feature, coeff in history_effects:
            if effect_type not in effect_data:
                effect_data[effect_type] = {}
            effect_data[effect_type][lag] = coeff
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = {'action': 'blue', 'angle': 'red', 'hitmiss': 'green', 'difficulty': 'orange'}
        markers = {'action': 'o', 'angle': 's', 'hitmiss': '^', 'difficulty': 'D'}
        
        for effect_type, lag_coeffs in effect_data.items():
            lags = sorted(lag_coeffs.keys())
            coeffs = [lag_coeffs[lag] for lag in lags]
            
            ax.plot(lags, coeffs, marker=markers.get(effect_type, 'o'), 
                   color=colors.get(effect_type, 'black'), 
                   linewidth=2, markersize=8, label=effect_type.capitalize())
            
            # Add value labels
            for lag, coeff in zip(lags, coeffs):
                ax.annotate(f'{coeff:.3f}', (lag, coeff), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('Trial Lag')
        ax.set_ylabel('Effect Strength (β)')
        ax.set_title('Temporal Patterns of Serial Dependence Effects')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save figure
        if self.save_figures:
            filename = FIGURE_NAME_PATTERNS['temporal'].format(self.history_depth)
            save_figure(fig, filename, self.figures_dir)
        
        return fig
    
    def create_comprehensive_rat_dashboard(self, rat_results: Dict[int, Dict[str, Any]]) -> List[plt.Figure]:
        """
        Create a comprehensive dashboard with all rat visualization.
        
        Args:
            rat_results: Dictionary with individual rat analysis results
            
        Returns:
            List of matplotlib figures
        """
        print("\n=== CREATING COMPREHENSIVE RAT DASHBOARD ===")
        
        if not rat_results:
            print("No rat results for dashboard")
            return []
        
        figures = []
        
        # Individual rat psychometric plots
        individual_figs = self.plot_individual_rat_results(rat_results)
        figures.extend(individual_figs)
        
        # Coefficients heatmap
        heatmap_fig = self.plot_rat_coefficients_heatmap(rat_results)
        if heatmap_fig:
            figures.append(heatmap_fig)
        
        # Summary statistics
        summary_fig = self._plot_rat_summary_stats(rat_results)
        if summary_fig:
            figures.append(summary_fig)
        
        print(f"✅ Created comprehensive dashboard with {len(figures)} figures")
        return figures
    
    def _plot_rat_summary_stats(self, rat_results: Dict[int, Dict[str, Any]]) -> plt.Figure:
        """Plot summary statistics across rats"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        rats = sorted(rat_results.keys())
        accuracies = [rat_results[rat]['accuracy'] for rat in rats]
        n_trials = [rat_results[rat]['n_trials'] for rat in rats]
        
        # Accuracy distribution
        ax = axes[0]
        ax.hist(accuracies, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('Number of Rats')
        ax.set_title('Accuracy Distribution')
        ax.axvline(np.mean(accuracies), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(accuracies):.3f}')
        ax.legend()
        
        # Trial count distribution
        ax = axes[1]
        ax.hist(n_trials, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
        ax.set_xlabel('Number of Trials')
        ax.set_ylabel('Number of Rats')
        ax.set_title('Trial Count Distribution')
        ax.axvline(np.mean(n_trials), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(n_trials):,.0f}')
        ax.legend()
        
        # Accuracy vs Trial count
        ax = axes[2]
        ax.scatter(n_trials, accuracies, alpha=0.7, s=60)
        ax.set_xlabel('Number of Trials')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs Trial Count')
        
        # Add correlation
        corr = np.corrcoef(n_trials, accuracies)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save figure
        if self.save_figures:
            filename = FIGURE_NAME_PATTERNS['summary_stats'].format(self.history_depth)
            save_figure(fig, filename, self.figures_dir)
        
        return fig
