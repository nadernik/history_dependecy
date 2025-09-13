#!/usr/bin/env python3
"""
Serial Dependence Analysis - Enhanced Version with Individual Rat Analysis
==========================================================================

This script analyzes serial dependence in behavioral data with the following features:

1. Automatic CSV caching for faster repeated analysis
2. Individual rat-specific model fitting and analysis
3. Comprehensive visualization dashboard for each rat
4. Coefficient heatmaps across rats
5. Detailed psychometric curves with history effects

New Methods:
- analyze_individual_rats(): Fit models for each rat separately
- plot_individual_rat_results(): Create psychometric plots for all rats
- plot_specific_rat(rat_id): Plot detailed results for a specific rat
- plot_rat_coefficients_heatmap(): Show coefficient patterns across rats
- create_comprehensive_rat_dashboard(): Generate complete visualization suite
- get_rat_summary(): Get summary statistics for rats

Usage:
    analyzer = SerialDependenceAnalyzer()
    results = analyzer.run_complete_analysis()  # Full analysis with rat details
    
    # Or for specific rat analysis:
    rat_results = analyzer.analyze_individual_rats()
    analyzer.plot_specific_rat(rat_id=2)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
import seaborn as sns
import warnings
import os
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

class SerialDependenceAnalyzer:
    """
    Simplified analyzer using sklearn instead of statsmodels
    """
    
    def __init__(self, data_path='data/behavior_data.mat', history_depth=3, csv_path='processed_behavior_data.csv'):
        self.data_path = data_path
        self.csv_path = csv_path
        self.k = history_depth
        self.df = None
        self.df_processed = None
        self.model = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.rat_results = None
        
        # Define modality mappings
        self.modality_names = {1: 'T', 2: 'V', 3: 'VT', 4: 'Control'}
        self.rev_rule_rats = [6, 7, 14, 15, 16, 17]
        
        print(f"Serial Dependence Analyzer initialized with k={self.k}")
        print(f"MAT file: {self.data_path}")
        print(f"CSV file: {self.csv_path}")
    
    def check_and_load_csv(self):
        """Check if preprocessed CSV file exists and load it"""
        if os.path.exists(self.csv_path):
            print(f"Found preprocessed data file: {self.csv_path}")
            print("Loading preprocessed data from CSV...")
            
            try:
                # Load the CSV file
                df_processed = pd.read_csv(self.csv_path)
                
                # Verify it has the expected columns for our history depth
                expected_lag_columns = []
                for i in range(1, self.k + 1):
                    expected_lag_columns.extend([
                        f'action_n-{i}', f'angle_n-{i}', f'hitmiss_n-{i}', 
                        f'mod_n-{i}', f'mod_transition_n-{i}'
                    ])
                
                missing_columns = [col for col in expected_lag_columns if col not in df_processed.columns]
                
                if missing_columns:
                    print(f"Warning: CSV file missing expected columns: {missing_columns}")
                    print("CSV file may not match current history depth. Will reprocess from MAT file.")
                    return None
                
                # Set both df and df_processed since CSV contains fully processed data
                self.df_processed = df_processed
                
                # Create a basic df for compatibility (subset of processed data)
                basic_columns = ['rat', 'trialID', 'date', 'action', 'hitmiss', 'mod', 'angle', 'trial_number']
                available_basic_columns = [col for col in basic_columns if col in df_processed.columns]
                self.df = df_processed[available_basic_columns].copy()
                
                print(f"Successfully loaded preprocessed data!")
                print(f"Total trials with complete history: {len(self.df_processed)}")
                print(f"Rats included: {sorted(self.df_processed['rat'].unique())}")
                print(f"Modalities: {sorted(self.df_processed['mod'].unique())}")
                
                return self.df_processed
                
            except Exception as e:
                print(f"Error loading CSV file: {e}")
                print("Will reprocess from MAT file.")
                return None
        else:
            print(f"No preprocessed CSV file found at: {self.csv_path}")
            return None
    
    def save_processed_data_to_csv(self):
        """Save the processed data to CSV for future use"""
        if self.df_processed is not None:
            try:
                print(f"Saving processed data to: {self.csv_path}")
                self.df_processed.to_csv(self.csv_path, index=False)
                print("Processed data saved successfully!")
            except Exception as e:
                print(f"Error saving processed data: {e}")
    
    def load_and_preprocess_data(self):
        """Load and preprocess data (same as before)"""
        print("=== STEP 1: DATA LOADING AND PREPROCESSING ===")
        
        try:
            # Load data
            data = scipy.io.loadmat(self.data_path)
            trial_data = data['TrialNADER'][0, 0]
            
            # Convert to dictionary
            trial_dict = {}
            for field in trial_data.dtype.names:
                trial_dict[field] = trial_data[field][0].flatten()
            
            # Create DataFrame
            self.df = pd.DataFrame(trial_dict)
            
            # Convert numeric columns
            numeric_columns = ['rat', 'trialID', 'date', 'action', 'hitmiss', 'mod', 'angle']
            
            for col in numeric_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col].astype(str), errors='coerce')
            
            # Remove NaN values
            self.df = self.df.dropna(subset=['rat', 'action', 'hitmiss', 'mod', 'angle'])
            
            # Apply filtering
            print("Applying data filters...")
            
            # Filter rats
            excluded_rats = [8, 14, 15, 17, 18, 19, 20, 21]
            rats = self.df['rat'].unique()
            rats = rats[~np.isin(rats, excluded_rats)]
            self.df = self.df[self.df['rat'].isin(rats)]
            
            # Binary variables
            self.df['action'] = self.df['action'].astype(int)
            self.df = self.df[self.df['action'].isin([0, 1])]
            
            self.df['hitmiss'] = self.df['hitmiss'].astype(int)
            self.df = self.df[self.df['hitmiss'].isin([0, 1])]
            
            # Angle processing
            self.df['angle'] = self.df['angle'].astype(float)
            self.df.loc[self.df['angle'] > 180, 'angle'] = self.df.loc[self.df['angle'] > 180, 'angle'] - 180
            self.df.loc[self.df['angle'] > 90, 'angle'] = (90 - (self.df.loc[self.df['angle'] > 90, 'angle'] - 90)) + 90
            
            # Filter angles and modalities
            self.df = self.df[(self.df['angle'] >= 0) & (self.df['angle'] <= 90)]
            self.df = self.df[self.df['mod'] != 4]  # Exclude control
            
            # Handle reversed rule rats
            for rat_id in self.rev_rule_rats:
                if rat_id in self.df['rat'].values:
                    rat_mask = self.df['rat'] == rat_id
                    self.df.loc[rat_mask, 'action'] = 1 - self.df.loc[rat_mask, 'action']
            
            # Add trial numbering
            self.df = self.df.sort_values(['rat', 'date', 'trialID'])
            self.df['trial_number'] = self.df.groupby(['rat', 'date']).cumcount() + 1
            
            print(f"Data loaded successfully!")
            print(f"Total trials: {len(self.df)}")
            print(f"Rats included: {sorted(self.df['rat'].unique())}")
            print(f"Modalities: {sorted(self.df['mod'].unique())}")
            
            return self.df
            
        except Exception as e:
            print(f"Data loading failed: {e}")
            return None
    
    def create_lagged_features(self):
        """Create lagged features"""
        print(f"\n=== STEP 2: CREATING LAGGED FEATURES (k={self.k}) ===")
        
        try:
            df_lag = self.df.copy()
            df_lag = df_lag.sort_values(['rat', 'date', 'trial_number'])
            
            # Create lagged features
            for i in range(1, self.k + 1):
                print(f"Creating lag {i} features...")
                
                lag_columns = ['action', 'angle', 'hitmiss', 'mod']
                
                for col in lag_columns:
                    new_col = f"{col}_n-{i}"
                    df_lag[new_col] = df_lag.groupby(['rat', 'date'])[col].shift(i)
                
                # Create modality transitions
                current_mod = df_lag['mod']
                past_mod = df_lag[f'mod_n-{i}']
                
                current_mod_names = current_mod.map(self.modality_names)
                past_mod_names = past_mod.map(self.modality_names)
                
                df_lag[f'mod_transition_n-{i}'] = past_mod_names + '->' + current_mod_names
            
            # Remove rows with NaN values
            print("Removing trials without sufficient history...")
            df_lag = df_lag.dropna()
            
            self.df_processed = df_lag
            
            print(f"Lagged features created successfully!")
            print(f"Trials with complete history: {len(self.df_processed)}")
            
            # Save processed data to CSV for future use
            self.save_processed_data_to_csv()
            
            return self.df_processed
            
        except Exception as e:
            print(f"Feature creation failed: {e}")
            return None
    
    def build_simple_model(self):
        """Build a simple logistic regression model using sklearn"""
        print(f"\n=== STEP 3: SIMPLE LOGISTIC REGRESSION MODEL ===")
        
        try:
            if self.df_processed is None:
                raise ValueError("No processed data available.")
            
            print("Preparing features...")
            
            # Select key features for the model
            features = []
            feature_names = []
            
            # Current trial features
            features.append(self.df_processed['angle'].values)
            feature_names.append('angle')
            
            # One-hot encode current modality
            for mod in [1, 2, 3]:
                mod_feature = (self.df_processed['mod'] == mod).astype(int)
                features.append(mod_feature.values)
                feature_names.append(f'mod_{mod}')
            
            # History features (simplified)
            for i in range(1, self.k + 1):
                features.append(self.df_processed[f'angle_n-{i}'].values)
                feature_names.append(f'angle_n-{i}')
                
                features.append(self.df_processed[f'action_n-{i}'].values)
                feature_names.append(f'action_n-{i}')
                
                features.append(self.df_processed[f'hitmiss_n-{i}'].values)
                feature_names.append(f'hitmiss_n-{i}')
            
            # Stack features
            X = np.column_stack(features)
            y = self.df_processed['action'].values
            
            self.feature_names = feature_names
            
            print(f"Feature matrix shape: {X.shape}")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Fit model
            print("Fitting logistic regression model...")
            self.model = LogisticRegression(
                penalty='l2',
                C=1.0,
                max_iter=1000,
                random_state=42
            )
            
            self.model.fit(X_scaled, y)
            
            # Calculate accuracy
            train_accuracy = self.model.score(X_scaled, y)
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
            
            print("Model fitted successfully!")
            print(f"Training accuracy: {train_accuracy:.4f}")
            print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            return self.model
            
        except Exception as e:
            print(f"Model fitting failed: {e}")
            return None
    
    def analyze_coefficients(self):
        """Analyze model coefficients to identify important history effects"""
        if self.model is None:
            print("No model available.")
            return []
        
        print(f"\n=== MODEL COEFFICIENTS ANALYSIS ===")
        
        coefficients = self.model.coef_[0]
        
        # Create summary
        coef_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })
        
        coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
        
        print("Top 15 most important features:")
        print(coef_df.head(15).to_string(index=False, float_format='%.4f'))
        
        # Find significant history effects
        history_effects = []
        threshold = np.percentile(np.abs(coefficients), 80)  # Top 20%
        
        for feature, coeff in zip(self.feature_names, coefficients):
            if abs(coeff) > threshold and any(f"n-{i}" in feature for i in range(1, self.k + 1)):
                if 'action_n-' in feature:
                    lag = int(feature.split('action_n-')[1])
                    history_effects.append(('action', lag, feature, coeff))
                elif 'angle_n-' in feature:
                    lag = int(feature.split('angle_n-')[1])
                    history_effects.append(('angle', lag, feature, coeff))
        
        print(f"\nSignificant history effects:")
        for effect_type, lag, feature, coeff in history_effects:
            print(f"  {feature}: β={coeff:.4f}")
        
        return history_effects
    
    def create_psychometric_plots(self, history_effects=None):
        """Create visualization plots"""
        print(f"\n=== STEP 4: VISUALIZATION ===")
        
        if history_effects is None:
            history_effects = self.analyze_coefficients()
        
        if not history_effects:
            print("No significant history effects to plot.")
            return
        
        print(f"Creating plots for {len(history_effects)} effects...")
        
        # Set up plotting
        n_plots = min(len(history_effects), 6)  # Limit to 6 plots
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_cols > 1 else [axes]
        else:
            axes = axes.flatten()
        
        angle_bins = np.arange(0, 91, 15)  # Wider bins for more data
        angle_centers = angle_bins[:-1] + 7.5
        
        for plot_idx, (effect_type, lag, feature, coeff) in enumerate(history_effects[:n_plots]):
            ax = axes[plot_idx]
            
            if effect_type == 'action':
                self._plot_action_effect(ax, lag, angle_bins, angle_centers)
                ax.set_title(f'Previous Choice Effect (n-{lag})\nβ={coeff:.3f}')
            elif effect_type == 'angle':
                self._plot_angle_effect(ax, lag, angle_bins, angle_centers)
                ax.set_title(f'Previous Stimulus Effect (n-{lag})\nβ={coeff:.3f}')
            
            ax.set_xlabel('Current Stimulus Angle (degrees)')
            ax.set_ylabel('P(Turn Right)')
            ax.set_xlim(0, 90)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        # Hide unused subplots
        for i in range(n_plots, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def _plot_action_effect(self, ax, lag, angle_bins, angle_centers):
        """Plot previous action effect"""
        action_col = f'action_n-{lag}'
        
        # Overall baseline
        overall_psycho = []
        for i in range(len(angle_bins) - 1):
            mask = (self.df_processed['angle'] >= angle_bins[i]) & \
                   (self.df_processed['angle'] < angle_bins[i+1])
            if mask.sum() > 10:  # Minimum data requirement
                prob = self.df_processed.loc[mask, 'action'].mean()
                overall_psycho.append(prob)
            else:
                overall_psycho.append(np.nan)
        
        ax.plot(angle_centers, overall_psycho, 'k--', alpha=0.7, 
                label='Overall', linewidth=2)
        
        # Previous action effects
        colors = ['blue', 'red']
        labels = ['Prev: Left', 'Prev: Right']
        
        for prev_action in [0, 1]:
            psycho_curve = []
            
            for i in range(len(angle_bins) - 1):
                mask = (self.df_processed['angle'] >= angle_bins[i]) & \
                       (self.df_processed['angle'] < angle_bins[i+1]) & \
                       (self.df_processed[action_col] == prev_action)
                
                if mask.sum() > 5:  # Minimum data requirement
                    prob = self.df_processed.loc[mask, 'action'].mean()
                    psycho_curve.append(prob)
                else:
                    psycho_curve.append(np.nan)
            
            ax.plot(angle_centers, psycho_curve, 'o-', 
                   color=colors[prev_action], label=labels[prev_action], 
                   linewidth=2, markersize=5)
    
    def _plot_angle_effect(self, ax, lag, angle_bins, angle_centers):
        """Plot previous angle effect"""
        angle_col = f'angle_n-{lag}'
        
        # Overall baseline
        overall_psycho = []
        for i in range(len(angle_bins) - 1):
            mask = (self.df_processed['angle'] >= angle_bins[i]) & \
                   (self.df_processed['angle'] < angle_bins[i+1])
            if mask.sum() > 10:
                prob = self.df_processed.loc[mask, 'action'].mean()
                overall_psycho.append(prob)
            else:
                overall_psycho.append(np.nan)
        
        ax.plot(angle_centers, overall_psycho, 'k--', alpha=0.7, 
                label='Overall', linewidth=2)
        
        # Previous angle effects
        left_mask = self.df_processed[angle_col] < 45
        right_mask = self.df_processed[angle_col] >= 45
        
        colors = ['blue', 'red']
        labels = ['Prev: Left-tilted', 'Prev: Right-tilted']
        masks = [left_mask, right_mask]
        
        for mask, color, label in zip(masks, colors, labels):
            psycho_curve = []
            
            for i in range(len(angle_bins) - 1):
                angle_mask = (self.df_processed['angle'] >= angle_bins[i]) & \
                            (self.df_processed['angle'] < angle_bins[i+1]) & mask
                
                if angle_mask.sum() > 5:
                    prob = self.df_processed.loc[angle_mask, 'action'].mean()
                    psycho_curve.append(prob)
                else:
                    psycho_curve.append(np.nan)
            
            ax.plot(angle_centers, psycho_curve, 'o-', 
                   color=color, label=label, linewidth=2, markersize=5)
    
    def analyze_individual_rats(self):
        """Analyze each rat individually"""
        print(f"\n=== INDIVIDUAL RAT ANALYSIS ===")
        
        if self.df_processed is None:
            print("No processed data available.")
            return {}
        
        rats = sorted(self.df_processed['rat'].unique())
        rat_results = {}
        
        print(f"Analyzing {len(rats)} rats individually...")
        
        for rat_id in rats:
            print(f"\nAnalyzing Rat {rat_id}...")
            rat_data = self.df_processed[self.df_processed['rat'] == rat_id].copy()
            
            if len(rat_data) < 50:  # Minimum data requirement
                print(f"  Insufficient data for Rat {rat_id} ({len(rat_data)} trials)")
                continue
            
            try:
                # Prepare features for this rat
                features = []
                feature_names = []
                
                # Current trial features
                features.append(rat_data['angle'].values)
                feature_names.append('angle')
                
                # One-hot encode current modality
                for mod in [1, 2, 3]:
                    mod_feature = (rat_data['mod'] == mod).astype(int)
                    features.append(mod_feature.values)
                    feature_names.append(f'mod_{mod}')
                
                # History features
                for i in range(1, self.k + 1):
                    features.append(rat_data[f'angle_n-{i}'].values)
                    feature_names.append(f'angle_n-{i}')
                    
                    features.append(rat_data[f'action_n-{i}'].values)
                    feature_names.append(f'action_n-{i}')
                    
                    features.append(rat_data[f'hitmiss_n-{i}'].values)
                    feature_names.append(f'hitmiss_n-{i}')
                
                # Stack features
                X = np.column_stack(features)
                y = rat_data['action'].values
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Fit individual model
                model = LogisticRegression(
                    penalty='l2',
                    C=1.0,
                    max_iter=1000,
                    random_state=42
                )
                
                model.fit(X_scaled, y)
                
                # Calculate metrics
                train_accuracy = model.score(X_scaled, y)
                
                # Store results
                rat_results[rat_id] = {
                    'data': rat_data,
                    'model': model,
                    'scaler': scaler,
                    'feature_names': feature_names,
                    'accuracy': train_accuracy,
                    'n_trials': len(rat_data),
                    'coefficients': model.coef_[0],
                    'X': X,
                    'y': y,
                    'X_scaled': X_scaled
                }
                
                print(f"  Rat {rat_id}: {len(rat_data)} trials, accuracy = {train_accuracy:.3f}")
                
            except Exception as e:
                print(f"  Error analyzing Rat {rat_id}: {e}")
        
        print(f"\nSuccessfully analyzed {len(rat_results)} rats")
        return rat_results
    
    def plot_individual_rat_results(self, rat_results=None, max_rats_per_figure=6):
        """Create detailed plots for each rat"""
        print(f"\n=== INDIVIDUAL RAT VISUALIZATION ===")
        
        if rat_results is None:
            rat_results = self.analyze_individual_rats()
        
        if not rat_results:
            print("No rat results available for plotting.")
            return []
        
        rats = list(rat_results.keys())
        n_rats = len(rats)
        
        # Create multiple figures if needed
        figures = []
        n_figures = (n_rats + max_rats_per_figure - 1) // max_rats_per_figure
        
        for fig_idx in range(n_figures):
            start_idx = fig_idx * max_rats_per_figure
            end_idx = min((fig_idx + 1) * max_rats_per_figure, n_rats)
            rats_in_figure = rats[start_idx:end_idx]
            
            print(f"Creating figure {fig_idx + 1}/{n_figures} for rats: {rats_in_figure}")
            
            n_rats_fig = len(rats_in_figure)
            n_cols = min(3, n_rats_fig)
            n_rows = (n_rats_fig + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
            fig.suptitle(f'Individual Rat Analysis - Figure {fig_idx + 1}', fontsize=16, fontweight='bold')
            
            if n_rats_fig == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes if n_cols > 1 else [axes]
            else:
                axes = axes.flatten()
            
            for i, rat_id in enumerate(rats_in_figure):
                ax = axes[i]
                self._plot_rat_psychometric(ax, rat_id, rat_results[rat_id])
            
            # Hide unused subplots
            for i in range(n_rats_fig, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            figures.append(fig)
        
        return figures
    
    def _plot_rat_psychometric(self, ax, rat_id, rat_result):
        """Plot psychometric curve for individual rat with history effects"""
        data = rat_result['data']
        model = rat_result['model']
        accuracy = rat_result['accuracy']
        n_trials = rat_result['n_trials']
        
        # Define angle bins
        angle_bins = np.arange(0, 91, 15)
        angle_centers = angle_bins[:-1] + 7.5
        
        # Overall psychometric curve
        overall_psycho = []
        for i in range(len(angle_bins) - 1):
            mask = (data['angle'] >= angle_bins[i]) & (data['angle'] < angle_bins[i+1])
            if mask.sum() > 3:
                prob = data.loc[mask, 'action'].mean()
                overall_psycho.append(prob)
            else:
                overall_psycho.append(np.nan)
        
        ax.plot(angle_centers, overall_psycho, 'k-', linewidth=3, 
                label='Overall', alpha=0.8)
        
        # Previous choice effect (most recent)
        prev_action_col = 'action_n-1'
        if prev_action_col in data.columns:
            colors = ['blue', 'red']
            labels = ['After Left', 'After Right']
            
            for prev_action in [0, 1]:
                psycho_curve = []
                
                for i in range(len(angle_bins) - 1):
                    mask = (data['angle'] >= angle_bins[i]) & \
                           (data['angle'] < angle_bins[i+1]) & \
                           (data[prev_action_col] == prev_action)
                    
                    if mask.sum() > 2:
                        prob = data.loc[mask, 'action'].mean()
                        psycho_curve.append(prob)
                    else:
                        psycho_curve.append(np.nan)
                
                ax.plot(angle_centers, psycho_curve, 'o-', 
                       color=colors[prev_action], label=labels[prev_action], 
                       linewidth=2, markersize=4, alpha=0.7)
        
        # Formatting
        ax.set_title(f'Rat {rat_id}\n{n_trials} trials, Acc: {accuracy:.3f}', 
                    fontweight='bold')
        ax.set_xlabel('Stimulus Angle (degrees)')
        ax.set_ylabel('P(Turn Right)')
        ax.set_xlim(0, 90)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        
        # Add performance info
        ax.text(0.02, 0.98, f'N = {n_trials}', transform=ax.transAxes, 
                verticalalignment='top', fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def plot_rat_coefficients_heatmap(self, rat_results=None):
        """Create heatmap of coefficients across rats"""
        print(f"\n=== COEFFICIENT HEATMAP ACROSS RATS ===")
        
        if rat_results is None:
            rat_results = self.analyze_individual_rats()
        
        if not rat_results:
            print("No rat results available.")
            return None
        
        # Collect coefficients
        rats = sorted(rat_results.keys())
        feature_names = rat_results[rats[0]]['feature_names']
        
        coeff_matrix = []
        for rat_id in rats:
            coeff_matrix.append(rat_results[rat_id]['coefficients'])
        
        coeff_matrix = np.array(coeff_matrix)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, max(8, len(rats) * 0.4)))
        
        im = ax.imshow(coeff_matrix, cmap='RdBu_r', aspect='auto', 
                      vmin=-np.abs(coeff_matrix).max(), 
                      vmax=np.abs(coeff_matrix).max())
        
        # Set ticks
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_yticks(range(len(rats)))
        ax.set_yticklabels([f'Rat {r}' for r in rats])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Coefficient Value', rotation=270, labelpad=20)
        
        # Add text annotations for significant coefficients
        threshold = np.percentile(np.abs(coeff_matrix), 75)
        for i in range(len(rats)):
            for j in range(len(feature_names)):
                if abs(coeff_matrix[i, j]) > threshold:
                    text = f'{coeff_matrix[i, j]:.2f}'
                    ax.text(j, i, text, ha='center', va='center', 
                           color='white' if abs(coeff_matrix[i, j]) > threshold * 1.5 else 'black',
                           fontsize=8, fontweight='bold')
        
        ax.set_title('Model Coefficients Across Rats', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        return fig
    
    def analyze_modality_sequences(self):
        """
        Analyze homo-modality vs hetero-modality sequences to test:
        1. Perceptual history effects in homo vs hetero-modal sequences
        2. Sequential choice effects across modalities
        """
        print(f"\n=== MODALITY SEQUENCE ANALYSIS ===")
        
        if self.df_processed is None:
            print("No processed data available. Run analysis first.")
            return None
            
        df = self.df_processed.copy()
        
        # Focus on T (1) and V (2) modalities for clear homo/hetero distinction
        df_tv = df[df['mod'].isin([1, 2]) & df['mod_n-1'].isin([1, 2])]
        
        results = {}
        
        # 1. Classify sequences as homo or hetero-modal
        df_tv['sequence_type'] = 'hetero'
        homo_mask = df_tv['mod'] == df_tv['mod_n-1']
        df_tv.loc[homo_mask, 'sequence_type'] = 'homo'
        
        print(f"Analyzing {len(df_tv)} T-V trials:")
        print(f"  Homo-modal sequences: {sum(homo_mask)} ({100*sum(homo_mask)/len(df_tv):.1f}%)")
        print(f"  Hetero-modal sequences: {sum(~homo_mask)} ({100*sum(~homo_mask)/len(df_tv):.1f}%)")
        
        # 2. Analyze perceptual history effects (stimulus angle effects)
        print("\n--- PERCEPTUAL HISTORY EFFECTS ---")
        perceptual_results = self._analyze_perceptual_history_effects(df_tv)
        results['perceptual_effects'] = perceptual_results
        
        # 3. Analyze sequential choice effects
        print("\n--- SEQUENTIAL CHOICE EFFECTS ---")
        choice_results = self._analyze_sequential_choice_effects(df_tv)
        results['choice_effects'] = choice_results
        
        # 4. Test specific hypotheses
        print("\n--- HYPOTHESIS TESTING ---")
        hypothesis_results = self._test_modality_hypotheses(df_tv)
        results['hypothesis_tests'] = hypothesis_results
        
        return results
    
    def _analyze_perceptual_history_effects(self, df_tv):
        """Analyze how previous stimulus angles affect current choices in homo vs hetero sequences"""
        results = {}
        
        for seq_type in ['homo', 'hetero']:
            seq_data = df_tv[df_tv['sequence_type'] == seq_type].copy()
            
            # Create modality indicator (Vision = 1, Touch = 0)
            seq_data['is_vision'] = (seq_data['mod'] == 2).astype(int)
            
            # Fit model with angle history effects
            features = ['angle', 'angle_n-1', 'is_vision']
            X = seq_data[features].values
            y = seq_data['action'].values
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit logistic regression
            model = LogisticRegression(random_state=42)
            model.fit(X_scaled, y)
            
            # Store results
            results[seq_type] = {
                'n_trials': len(seq_data),
                'coefficients': dict(zip(features, model.coef_[0])),
                'angle_n1_effect': model.coef_[0][1],  # Previous angle effect
                'accuracy': model.score(X_scaled, y)
            }
            
            print(f"  {seq_type.capitalize()}-modal sequences:")
            print(f"    N trials: {len(seq_data)}")
            print(f"    Previous angle effect (β): {model.coef_[0][1]:.4f}")
            print(f"    Model accuracy: {model.score(X_scaled, y):.3f}")
        
        # Test for difference in angle history effects
        homo_effect = results['homo']['angle_n1_effect']
        hetero_effect = results['hetero']['angle_n1_effect']
        
        print(f"\n  Comparison:")
        print(f"    Homo-modal angle effect: {homo_effect:.4f}")
        print(f"    Hetero-modal angle effect: {hetero_effect:.4f}")
        print(f"    Difference: {hetero_effect - homo_effect:.4f}")
        
        return results
    
    def _analyze_sequential_choice_effects(self, df_tv):
        """Analyze how previous choices affect current choices in homo vs hetero sequences"""
        results = {}
        
        for seq_type in ['homo', 'hetero']:
            seq_data = df_tv[df_tv['sequence_type'] == seq_type].copy()
            
            # Create modality indicator (Vision = 1, Touch = 0)
            seq_data['is_vision'] = (seq_data['mod'] == 2).astype(int)
            
            # Fit model with choice history effects
            features = ['angle', 'action_n-1', 'is_vision']  # action_n-1 is previous choice
            X = seq_data[features].values
            y = seq_data['action'].values
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit logistic regression
            model = LogisticRegression(random_state=42)
            model.fit(X_scaled, y)
            
            # Store results
            results[seq_type] = {
                'n_trials': len(seq_data),
                'coefficients': dict(zip(features, model.coef_[0])),
                'action_n1_effect': model.coef_[0][1],  # Previous choice effect
                'accuracy': model.score(X_scaled, y)
            }
            
            print(f"  {seq_type.capitalize()}-modal sequences:")
            print(f"    N trials: {len(seq_data)}")
            print(f"    Previous choice effect (β): {model.coef_[0][1]:.4f}")
            print(f"    Model accuracy: {model.score(X_scaled, y):.3f}")
        
        # Test for difference in choice history effects
        homo_effect = results['homo']['action_n1_effect']
        hetero_effect = results['hetero']['action_n1_effect']
        
        print(f"\n  Comparison:")
        print(f"    Homo-modal choice effect: {homo_effect:.4f}")
        print(f"    Hetero-modal choice effect: {hetero_effect:.4f}")
        print(f"    Difference: {hetero_effect - homo_effect:.4f}")
        
        return results
    
    def _test_modality_hypotheses(self, df_tv):
        """Test specific hypotheses about modality effects"""
        results = {}
        
        # Hypothesis 1: Hetero-modal sequences show repulsion (negative angle effect)
        hetero_data = df_tv[df_tv['sequence_type'] == 'hetero']
        
        # Simple correlation between previous angle and current choice
        angle_choice_corr = np.corrcoef(hetero_data['angle_n-1'], hetero_data['action'])[0,1]
        
        print(f"Hypothesis 1 - Hetero-modal repulsion:")
        print(f"  Correlation between previous angle and current choice: {angle_choice_corr:.4f}")
        print(f"  Repulsion evidence: {'YES' if angle_choice_corr < 0 else 'NO'}")
        
        # Hypothesis 2: Attractive choice effects occur even for hetero-modal sequences
        choice_choice_corr = np.corrcoef(hetero_data['action_n-1'], hetero_data['action'])[0,1]
        
        print(f"\nHypothesis 2 - Hetero-modal choice attraction:")
        print(f"  Correlation between previous and current choice: {choice_choice_corr:.4f}")
        print(f"  Attraction evidence: {'YES' if choice_choice_corr > 0 else 'NO'}")
        
        results['hetero_repulsion'] = angle_choice_corr < 0
        results['hetero_choice_attraction'] = choice_choice_corr > 0
        results['angle_choice_corr'] = angle_choice_corr
        results['choice_choice_corr'] = choice_choice_corr
        
        return results
    
    def plot_modality_sequence_results(self, modality_results=None):
        """Plot comprehensive results of modality sequence analysis"""
        if modality_results is None:
            modality_results = self.analyze_modality_sequences()
        
        if modality_results is None:
            print("No modality results to plot")
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Modality Sequence Analysis Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Perceptual history effects comparison
        perceptual = modality_results['perceptual_effects']
        homo_angle = perceptual['homo']['angle_n1_effect']
        hetero_angle = perceptual['hetero']['angle_n1_effect']
        
        axes[0,0].bar(['Homo-modal', 'Hetero-modal'], [homo_angle, hetero_angle], 
                     color=['skyblue', 'lightcoral'])
        axes[0,0].set_title('Previous Angle Effect (Perceptual History)')
        axes[0,0].set_ylabel('Coefficient (β)')
        axes[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add significance indicators
        if hetero_angle < 0:
            axes[0,0].text(1, hetero_angle-0.01, 'Repulsion', ha='center', fontweight='bold')
        
        # Plot 2: Choice history effects comparison  
        choice = modality_results['choice_effects']
        homo_choice = choice['homo']['action_n1_effect']
        hetero_choice = choice['hetero']['action_n1_effect']
        
        axes[0,1].bar(['Homo-modal', 'Hetero-modal'], [homo_choice, hetero_choice],
                     color=['skyblue', 'lightcoral'])
        axes[0,1].set_title('Previous Choice Effect (Sequential Choice)')
        axes[0,1].set_ylabel('Coefficient (β)')
        axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add significance indicators
        if hetero_choice > 0:
            axes[0,1].text(1, hetero_choice+0.01, 'Attraction', ha='center', fontweight='bold')
        
        # Plot 3: Trial counts
        homo_trials = perceptual['homo']['n_trials']
        hetero_trials = perceptual['hetero']['n_trials']
        
        axes[1,0].pie([homo_trials, hetero_trials], labels=['Homo-modal', 'Hetero-modal'],
                     autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
        axes[1,0].set_title('Distribution of Sequence Types')
        
        # Plot 4: Hypothesis test results
        hyp = modality_results['hypothesis_tests']
        
        # Create a summary plot
        axes[1,1].text(0.1, 0.8, 'HYPOTHESIS TEST RESULTS', fontsize=14, fontweight='bold')
        axes[1,1].text(0.1, 0.6, f'H1: Hetero-modal repulsion', fontsize=12)
        axes[1,1].text(0.1, 0.5, f'    Evidence: {"✓" if hyp["hetero_repulsion"] else "✗"}', fontsize=12)
        axes[1,1].text(0.1, 0.4, f'    Correlation: {hyp["angle_choice_corr"]:.3f}', fontsize=12)
        
        axes[1,1].text(0.1, 0.2, f'H2: Hetero-modal choice attraction', fontsize=12)
        axes[1,1].text(0.1, 0.1, f'    Evidence: {"✓" if hyp["hetero_choice_attraction"] else "✗"}', fontsize=12)
        axes[1,1].text(0.1, 0.0, f'    Correlation: {hyp["choice_choice_corr"]:.3f}', fontsize=12)
        
        axes[1,1].set_xlim(0, 1)
        axes[1,1].set_ylim(-0.1, 1)
        axes[1,1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def analyze_individual_rat_modality_effects(self, rat_id):
        """Analyze modality-specific effects for a single rat"""
        if self.df_processed is None:
            print("No processed data available. Run analysis first.")
            return None
            
        rat_data = self.df_processed[self.df_processed['rat'] == rat_id]
        
        # Focus on T-V sequences
        rat_tv = rat_data[rat_data['mod'].isin([1, 2]) & rat_data['mod_n-1'].isin([1, 2])]
        
        if len(rat_tv) < 50:  # Need minimum trials
            print(f"Insufficient T-V trials for rat {rat_id}: {len(rat_tv)}")
            return None
            
        # Classify sequences
        rat_tv['sequence_type'] = 'hetero'
        homo_mask = rat_tv['mod'] == rat_tv['mod_n-1']
        rat_tv.loc[homo_mask, 'sequence_type'] = 'homo'
        
        results = {
            'rat_id': rat_id,
            'total_trials': len(rat_tv),
            'homo_trials': sum(homo_mask),
            'hetero_trials': sum(~homo_mask)
        }
        
        # Test both sequence types if sufficient data
        for seq_type in ['homo', 'hetero']:
            seq_data = rat_tv[rat_tv['sequence_type'] == seq_type]
            
            if len(seq_data) < 20:  # Minimum for reliable analysis
                continue
                
            # Perceptual history effect
            angle_choice_corr = np.corrcoef(seq_data['angle_n-1'], seq_data['action'])[0,1]
            
            # Choice history effect  
            choice_choice_corr = np.corrcoef(seq_data['action_n-1'], seq_data['action'])[0,1]
            
            results[f'{seq_type}_angle_effect'] = angle_choice_corr
            results[f'{seq_type}_choice_effect'] = choice_choice_corr
            results[f'{seq_type}_trials'] = len(seq_data)
        
        return results

    def create_comprehensive_rat_dashboard(self, rat_results=None):
        """Create a comprehensive dashboard for all rats"""
        print(f"\n=== COMPREHENSIVE RAT DASHBOARD ===")
        
        if rat_results is None:
            rat_results = self.analyze_individual_rats()
        
        if not rat_results:
            print("No rat results available.")
            return []
        
        figures = []
        
        # 1. Individual psychometric curves
        print("Creating individual psychometric plots...")
        individual_figs = self.plot_individual_rat_results(rat_results)
        figures.extend(individual_figs)
        
        # 2. Coefficient heatmap
        print("Creating coefficient heatmap...")
        heatmap_fig = self.plot_rat_coefficients_heatmap(rat_results)
        if heatmap_fig:
            figures.append(heatmap_fig)
        
        # 3. Summary statistics plot
        print("Creating summary statistics...")
        summary_fig = self._plot_rat_summary_stats(rat_results)
        if summary_fig:
            figures.append(summary_fig)
        
        # 4. Modality sequence analysis
        print("Creating modality sequence analysis...")
        modality_fig = self.plot_modality_sequence_results()
        if modality_fig:
            figures.append(modality_fig)
        
        plt.show()
        return figures
    
    def _plot_rat_summary_stats(self, rat_results):
        """Plot summary statistics across rats"""
        rats = sorted(rat_results.keys())
        accuracies = [rat_results[r]['accuracy'] for r in rats]
        n_trials = [rat_results[r]['n_trials'] for r in rats]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy plot
        bars1 = ax1.bar(range(len(rats)), accuracies, color='skyblue', alpha=0.7)
        ax1.set_xlabel('Rat ID')
        ax1.set_ylabel('Model Accuracy')
        ax1.set_title('Model Accuracy by Rat')
        ax1.set_xticks(range(len(rats)))
        ax1.set_xticklabels([f'R{r}' for r in rats], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Trial count plot
        bars2 = ax2.bar(range(len(rats)), n_trials, color='lightcoral', alpha=0.7)
        ax2.set_xlabel('Rat ID')
        ax2.set_ylabel('Number of Trials')
        ax2.set_title('Data Availability by Rat')
        ax2.set_xticks(range(len(rats)))
        ax2.set_xticklabels([f'R{r}' for r in rats], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, n in zip(bars2, n_trials):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(n_trials)*0.01,
                    f'{n}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        return fig
    
    def plot_specific_rat(self, rat_id, show_coefficients=True):
        """Plot detailed results for a specific rat"""
        if self.rat_results is None:
            print("No individual rat results available. Run analysis first.")
            return None
        
        if rat_id not in self.rat_results:
            print(f"Rat {rat_id} not found in results. Available rats: {list(self.rat_results.keys())}")
            return None
        
        rat_result = self.rat_results[rat_id]
        
        # Create detailed plot for this rat
        if show_coefficients:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        
        # Psychometric plot
        self._plot_rat_psychometric(ax1, rat_id, rat_result)
        
        if show_coefficients:
            # Coefficient plot
            coefficients = rat_result['coefficients']
            feature_names = rat_result['feature_names']
            
            # Sort by absolute magnitude
            sorted_indices = np.argsort(np.abs(coefficients))[::-1]
            top_n = min(15, len(coefficients))  # Show top 15 features
            
            top_coeffs = coefficients[sorted_indices[:top_n]]
            top_features = [feature_names[i] for i in sorted_indices[:top_n]]
            
            colors = ['red' if c > 0 else 'blue' for c in top_coeffs]
            bars = ax2.barh(range(len(top_coeffs)), top_coeffs, color=colors, alpha=0.7)
            
            ax2.set_yticks(range(len(top_coeffs)))
            ax2.set_yticklabels(top_features)
            ax2.set_xlabel('Coefficient Value')
            ax2.set_title(f'Top Model Features - Rat {rat_id}')
            ax2.grid(True, alpha=0.3)
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Add value labels
            for bar, coeff in zip(bars, top_coeffs):
                width = bar.get_width()
                ax2.text(width + (0.01 if width > 0 else -0.01), bar.get_y() + bar.get_height()/2,
                        f'{coeff:.3f}', ha='left' if width > 0 else 'right', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.show()
        return fig
    
    def get_rat_summary(self, rat_id=None):
        """Get summary statistics for a specific rat or all rats"""
        if self.rat_results is None:
            print("No individual rat results available. Run analysis first.")
            return None
        
        if rat_id is not None:
            if rat_id not in self.rat_results:
                print(f"Rat {rat_id} not found in results.")
                return None
            
            result = self.rat_results[rat_id]
            return {
                'rat_id': rat_id,
                'n_trials': result['n_trials'],
                'accuracy': result['accuracy'],
                'top_features': self._get_top_features(result['coefficients'], result['feature_names'], n=5)
            }
        else:
            # Return summary for all rats
            summary = {}
            for rat_id in sorted(self.rat_results.keys()):
                result = self.rat_results[rat_id]
                summary[rat_id] = {
                    'n_trials': result['n_trials'],
                    'accuracy': result['accuracy'],
                    'top_features': self._get_top_features(result['coefficients'], result['feature_names'], n=3)
                }
            return summary
    
    def _get_top_features(self, coefficients, feature_names, n=5):
        """Get top n features by absolute coefficient value"""
        sorted_indices = np.argsort(np.abs(coefficients))[::-1]
        top_n = min(n, len(coefficients))
        
        top_features = []
        for i in sorted_indices[:top_n]:
            top_features.append({
                'feature': feature_names[i],
                'coefficient': coefficients[i],
                'abs_coefficient': abs(coefficients[i])
            })
        
        return top_features
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("="*60)
        print("SERIAL DEPENDENCE ANALYSIS - SIMPLIFIED VERSION")
        print("="*60)
        
        # Step 1: Try to load preprocessed data from CSV first
        print("=== CHECKING FOR PREPROCESSED DATA ===")
        if self.check_and_load_csv() is not None:
            print("Using preprocessed data from CSV file!")
        else:
            print("No valid preprocessed data found. Processing from scratch...")
            
            # Step 1a: Load data from MAT file
            if self.load_and_preprocess_data() is None:
                print("Data loading failed!")
                return None
            
            # Step 1b: Create features
            if self.create_lagged_features() is None:
                print("Feature creation failed!")
                return None
        
        # Step 2: Build model
        model = self.build_simple_model()
        if model is None:
            print("Model building failed!")
            return None
        
        # Step 3: Analyze and visualize - Overall model
        history_effects = self.analyze_coefficients()
        self.create_psychometric_plots(history_effects)
        
        # Step 4: Individual rat analysis and detailed visualization
        print(f"\n=== DETAILED RAT-SPECIFIC ANALYSIS ===")
        rat_results = self.analyze_individual_rats()
        
        if rat_results:
            # Create comprehensive dashboard
            dashboard_figures = self.create_comprehensive_rat_dashboard(rat_results)
            
            # Store rat results for further analysis
            self.rat_results = rat_results
            
            print(f"Created {len(dashboard_figures)} detailed visualization figures")
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return model


def main():
    """Main function - runs comprehensive analysis and plots results for each rat"""
    print("Starting comprehensive serial dependence analysis...")
    
    analyzer = SerialDependenceAnalyzer(
        data_path='data/behavior_data.mat',
        history_depth=3
    )
    
    # Run the complete analysis to get overall results
    print("Running complete analysis...")
    results = analyzer.run_complete_analysis()
    
    # Create comprehensive dashboard with all rat results
    print("Creating comprehensive rat dashboard...")
    figures = analyzer.create_comprehensive_rat_dashboard()
    
    print(f"\nAnalysis complete! Generated {len(figures)} figures.")
    print("All plots are now displayed showing comprehensive results for each rat.")
    
    return analyzer, results, figures


if __name__ == "__main__":
    analyzer, results, figures = main()
