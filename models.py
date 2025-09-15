"""
Statistical Modeling for Serial Dependence Analysis
===================================================

This module contains all statistical modeling functions including
logistic regression, individual rat analysis, and modality-specific analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from scipy import stats

from config import DEFAULT_CV_FOLDS, DEFAULT_REGULARIZATION_C, EFFECT_SIZE_THRESHOLD
from utils import get_top_features


class SerialDependenceModels:
    """Class containing all statistical modeling methods"""
    
    def __init__(self, history_depth: int = 3):
        """
        Initialize the models class.
        
        Args:
            history_depth: Number of historical trials to include
        """
        self.history_depth = history_depth
        self.scaler = StandardScaler()
        
    def build_simple_model(self, df_processed: pd.DataFrame) -> Tuple[LogisticRegression, List[str]]:
        """
        Build a simple logistic regression model using sklearn.
        
        Args:
            df_processed: Processed DataFrame with lagged features
            
        Returns:
            Tuple of (fitted_model, feature_names)
        """
        print(f"\n=== BUILDING LOGISTIC REGRESSION MODEL ===")
        
        if df_processed is None:
            raise ValueError("No processed data available.")
        
        print("Preparing features...")
        
        # Select key features for the model
        features = []
        feature_names = []
        
        # Current trial features
        features.append(df_processed['angle'].values)
        feature_names.append('angle')
        
        # One-hot encode current modality
        for mod in [1, 2, 3]:
            mod_feature = (df_processed['mod'] == mod).astype(int)
            features.append(mod_feature.values)
            feature_names.append(f'mod_{mod}')
        
        # History features
        for i in range(1, self.history_depth + 1):
            features.append(df_processed[f'angle_n-{i}'].values)
            feature_names.append(f'angle_n-{i}')
            
            features.append(df_processed[f'action_n-{i}'].values)
            feature_names.append(f'action_n-{i}')
            
            features.append(df_processed[f'hitmiss_n-{i}'].values)
            feature_names.append(f'hitmiss_n-{i}')
            
            # Add difficulty features if available
            if f'difficulty_n-{i}' in df_processed.columns:
                features.append(df_processed[f'difficulty_n-{i}'].values)
                feature_names.append(f'difficulty_n-{i}')
        
        # Stack features
        X = np.column_stack(features)
        y = df_processed['action'].values
        
        print(f"Feature matrix shape: {X.shape}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        print("Fitting logistic regression model...")
        model = LogisticRegression(
            penalty='l2',
            C=DEFAULT_REGULARIZATION_C,
            max_iter=1000,
            random_state=42
        )
        
        model.fit(X_scaled, y)
        
        # Calculate accuracy
        train_accuracy = model.score(X_scaled, y)
        cv_scores = cross_val_score(model, X_scaled, y, cv=DEFAULT_CV_FOLDS)
        
        print("Model fitted successfully!")
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return model, feature_names
    
    def analyze_coefficients(self, model: LogisticRegression, feature_names: List[str]) -> List[Tuple[str, int, str, float]]:
        """
        Analyze model coefficients to identify important history effects.
        
        Args:
            model: Fitted logistic regression model
            feature_names: List of feature names
            
        Returns:
            List of (effect_type, lag, feature_name, coefficient) tuples
        """
        if model is None:
            print("No model available.")
            return []
        
        print(f"\n=== MODEL COEFFICIENTS ANALYSIS ===")
        
        coefficients = model.coef_[0]
        
        # Create summary
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        })
        
        coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
        
        print("Top 15 most important features:")
        print(coef_df.head(15).to_string(index=False, float_format='%.4f'))
        
        # Find significant history effects
        history_effects = []
        threshold = np.percentile(np.abs(coefficients), 80)  # Top 20%
        
        # Add effects above threshold
        for feature, coeff in zip(feature_names, coefficients):
            if abs(coeff) > threshold and any(f"n-{i}" in feature for i in range(1, self.history_depth + 1)):
                if 'action_n-' in feature:
                    lag = int(feature.split('action_n-')[1])
                    history_effects.append(('action', lag, feature, coeff))
                elif 'angle_n-' in feature:
                    lag = int(feature.split('angle_n-')[1])
                    history_effects.append(('angle', lag, feature, coeff))
                elif 'hitmiss_n-' in feature:
                    lag = int(feature.split('hitmiss_n-')[1])
                    history_effects.append(('hitmiss', lag, feature, coeff))
                elif 'difficulty_n-' in feature:
                    lag = int(feature.split('difficulty_n-')[1])
                    history_effects.append(('difficulty', lag, feature, coeff))
        
        # Always include top hit/miss effects
        hitmiss_effects = []
        for feature, coeff in zip(feature_names, coefficients):
            if 'hitmiss_n-' in feature and any(f"n-{i}" in feature for i in range(1, self.history_depth + 1)):
                lag = int(feature.split('hitmiss_n-')[1])
                hitmiss_effects.append((abs(coeff), 'hitmiss', lag, feature, coeff))
        
        hitmiss_effects.sort(reverse=True)
        existing_hitmiss = [effect for effect in history_effects if effect[0] == 'hitmiss']
        
        for _, effect_type, lag, feature, coeff in hitmiss_effects[:2]:
            if not any(e[1] == lag and e[0] == 'hitmiss' for e in existing_hitmiss):
                history_effects.append((effect_type, lag, feature, coeff))
                print(f"  Added top hit/miss effect: {feature}: β={coeff:.4f}")
        
        # Always include top difficulty effects
        difficulty_effects = []
        for feature, coeff in zip(feature_names, coefficients):
            if 'difficulty_n-' in feature and any(f"n-{i}" in feature for i in range(1, self.history_depth + 1)):
                lag = int(feature.split('difficulty_n-')[1])
                difficulty_effects.append((abs(coeff), 'difficulty', lag, feature, coeff))
        
        difficulty_effects.sort(reverse=True)
        existing_difficulty = [effect for effect in history_effects if effect[0] == 'difficulty']
        
        for _, effect_type, lag, feature, coeff in difficulty_effects[:2]:
            if not any(e[1] == lag and e[0] == 'difficulty' for e in existing_difficulty):
                history_effects.append((effect_type, lag, feature, coeff))
                print(f"  Added top difficulty effect: {feature}: β={coeff:.4f}")
        
        # Sort by lag for consistent ordering
        history_effects.sort(key=lambda x: (x[0], x[1]))
        
        print(f"\nIdentified {len(history_effects)} significant history effects:")
        for effect_type, lag, feature, coeff in history_effects:
            print(f"  {feature}: β={coeff:.4f}")
        
        return history_effects
    
    def analyze_individual_rats(self, df_processed: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
        """
        Analyze each rat individually with separate models.
        
        Args:
            df_processed: Processed DataFrame with lagged features
            
        Returns:
            Dictionary mapping rat_id to analysis results
        """
        print(f"\n=== INDIVIDUAL RAT ANALYSIS ===")
        
        rat_results = {}
        rats = sorted(df_processed['rat'].unique())
        
        print(f"Analyzing {len(rats)} rats individually...")
        
        for rat_id in rats:
            print(f"\n--- Analyzing Rat {rat_id} ---")
            
            # Filter data for this rat
            rat_data = df_processed[df_processed['rat'] == rat_id].copy()
            
            if len(rat_data) < 100:  # Skip rats with too few trials
                print(f"  Skipping rat {rat_id}: only {len(rat_data)} trials")
                continue
            
            try:
                # Build model for this rat
                model, feature_names = self.build_simple_model(rat_data)
                
                # Get coefficients
                coefficients = model.coef_[0]
                
                # Calculate accuracy
                X = self._prepare_features(rat_data, feature_names)
                X_scaled = self.scaler.fit_transform(X)
                y = rat_data['action'].values
                accuracy = model.score(X_scaled, y)
                
                # Get top features
                top_features = get_top_features(coefficients, feature_names, n=5)
                
                # Store results
                rat_results[rat_id] = {
                    'model': model,
                    'coefficients': coefficients,
                    'feature_names': feature_names,
                    'accuracy': accuracy,
                    'n_trials': len(rat_data),
                    'top_features': top_features,
                    'data': rat_data
                }
                
                print(f"  Trials: {len(rat_data):,}")
                print(f"  Accuracy: {accuracy:.3f}")
                print(f"  Top feature: {top_features[0][0]} (β={top_features[0][1]:.3f})")
                
            except Exception as e:
                print(f"  Failed to analyze rat {rat_id}: {e}")
                continue
        
        print(f"\n✅ Individual rat analysis complete for {len(rat_results)} rats")
        return rat_results
    
    def analyze_modality_sequences(self, df_processed: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze modality-specific sequences (homo vs hetero-modal).
        
        Args:
            df_processed: Processed DataFrame with lagged features
            
        Returns:
            Dictionary with modality sequence analysis results
        """
        print(f"\n=== MODALITY SEQUENCE ANALYSIS ===")
        
        # Filter to Touch-Vision sequences only (both current and previous modalities must be T or V)
        df_tv = df_processed[df_processed['mod'].isin([1, 2]) & df_processed['mod_n-1'].isin([1, 2])].copy()
        print(f"Analyzing {len(df_tv)} T-V trials:")
        
        if len(df_tv) == 0:
            return {"error": "No Touch-Vision sequences found"}
        
        # Classify sequences as homo or hetero-modal
        df_tv['sequence_type'] = 'hetero'
        homo_mask = df_tv['mod'] == df_tv['mod_n-1']
        df_tv.loc[homo_mask, 'sequence_type'] = 'homo'
        
        homo_count = homo_mask.sum()
        hetero_count = len(df_tv) - homo_count
        
        print(f"  Homo-modal sequences: {homo_count} ({homo_count/len(df_tv)*100:.1f}%)")
        print(f"  Hetero-modal sequences: {hetero_count} ({hetero_count/len(df_tv)*100:.1f}%)")
        
        # Analyze perceptual and choice effects
        perceptual_results = self._analyze_perceptual_history_effects(df_tv)
        choice_results = self._analyze_sequential_choice_effects(df_tv)
        hypothesis_results = self._test_modality_hypotheses(df_tv)
        
        return {
            'df_tv': df_tv,
            'homo_count': homo_count,
            'hetero_count': hetero_count,
            'perceptual_results': perceptual_results,
            'choice_results': choice_results,
            'hypothesis_results': hypothesis_results
        }
    
    def _analyze_perceptual_history_effects(self, df_tv: pd.DataFrame) -> Dict[str, Any]:
        """Analyze perceptual history effects in homo vs hetero-modal sequences"""
        print("\n--- PERCEPTUAL HISTORY EFFECTS ---")
        
        results = {}
        
        for seq_type in ['homo', 'hetero']:
            data = df_tv[df_tv['sequence_type'] == seq_type].copy()
            
            if len(data) < 100:
                continue
            
            # Prepare features for perceptual model
            features = ['angle', 'angle_n-1', 'is_vision']
            data['is_vision'] = (data['mod'] == 2).astype(int)
            
            X = data[features].values
            y = data['action'].values
            
            # Fit model
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = LogisticRegression(C=1.0, random_state=42)
            model.fit(X_scaled, y)
            
            accuracy = model.score(X_scaled, y)
            angle_coeff = model.coef_[0][1]  # angle_n-1 coefficient
            
            results[seq_type] = {
                'model': model,
                'accuracy': accuracy,
                'angle_effect': angle_coeff,
                'angle_n1_effect': angle_coeff,  # For compatibility with plotting
                'n_trials': len(data)
            }
            
            print(f"  {seq_type.capitalize()}-modal sequences:")
            print(f"    N trials: {len(data)}")
            print(f"    Previous angle effect (β): {angle_coeff:.4f}")
            print(f"    Model accuracy: {accuracy:.3f}")
        
        # Test for difference in angle history effects
        if 'homo' in results and 'hetero' in results:
            homo_effect = results['homo']['angle_effect']
            hetero_effect = results['hetero']['angle_effect']
            
            print(f"\n  Comparison:")
            print(f"    Homo-modal angle effect: {homo_effect:.4f}")
            print(f"    Hetero-modal angle effect: {hetero_effect:.4f}")
            print(f"    Difference: {hetero_effect - homo_effect:.4f}")
        
        return results
    
    def _analyze_sequential_choice_effects(self, df_tv: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sequential choice effects in homo vs hetero-modal sequences"""
        print("\n--- SEQUENTIAL CHOICE EFFECTS ---")
        
        results = {}
        
        for seq_type in ['homo', 'hetero']:
            data = df_tv[df_tv['sequence_type'] == seq_type].copy()
            
            if len(data) < 100:
                continue
            
            # Prepare features for choice model
            features = ['angle', 'action_n-1', 'is_vision']
            data['is_vision'] = (data['mod'] == 2).astype(int)
            
            X = data[features].values
            y = data['action'].values
            
            # Fit model
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = LogisticRegression(C=1.0, random_state=42)
            model.fit(X_scaled, y)
            
            accuracy = model.score(X_scaled, y)
            choice_coeff = model.coef_[0][1]  # action_n-1 coefficient
            
            results[seq_type] = {
                'model': model,
                'accuracy': accuracy,
                'choice_effect': choice_coeff,
                'action_n1_effect': choice_coeff,  # For compatibility with plotting
                'n_trials': len(data)
            }
            
            print(f"  {seq_type.capitalize()}-modal sequences:")
            print(f"    N trials: {len(data)}")
            print(f"    Previous choice effect (β): {choice_coeff:.4f}")
            print(f"    Model accuracy: {accuracy:.3f}")
        
        # Test for difference in choice history effects
        if 'homo' in results and 'hetero' in results:
            homo_effect = results['homo']['choice_effect']
            hetero_effect = results['hetero']['choice_effect']
            
            print(f"\n  Comparison:")
            print(f"    Homo-modal choice effect: {homo_effect:.4f}")
            print(f"    Hetero-modal choice effect: {hetero_effect:.4f}")
            print(f"    Difference: {hetero_effect - homo_effect:.4f}")
        
        return results
    
    def _test_modality_hypotheses(self, df_tv: pd.DataFrame) -> Dict[str, Any]:
        """Test specific hypotheses about modality effects"""
        print("\n--- HYPOTHESIS TESTING ---")
        
        # Test correlation between previous angle and current choice for hetero-modal
        hetero_data = df_tv[df_tv['sequence_type'] == 'hetero']
        
        if len(hetero_data) > 0:
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
            
            return {
                'hetero_repulsion': angle_choice_corr < 0,
                'hetero_choice_attraction': choice_choice_corr > 0,
                'angle_choice_corr': angle_choice_corr,
                'choice_choice_corr': choice_choice_corr,
                'perceptual_correlation': angle_choice_corr,  # For backward compatibility
                'choice_correlation': choice_choice_corr  # For backward compatibility
            }
        
        return {}
    
    def analyze_individual_rat_modality_effects(self, df_processed: pd.DataFrame, rat_id: int) -> Dict[str, Any]:
        """
        Analyze modality effects for a specific rat.
        
        Args:
            df_processed: Processed DataFrame
            rat_id: ID of the rat to analyze
            
        Returns:
            Dictionary with rat-specific modality analysis
        """
        print(f"\n--- Rat {rat_id} Modality Effects ---")
        
        rat_data = df_processed[df_processed['rat'] == rat_id].copy()
        
        if len(rat_data) < 100:
            return {"error": f"Insufficient data for rat {rat_id}"}
        
        # Analyze by modality
        modality_results = {}
        
        for mod in rat_data['mod'].unique():
            mod_data = rat_data[rat_data['mod'] == mod]
            
            if len(mod_data) < 50:
                continue
            
            # Build simple model for this modality
            try:
                model, feature_names = self.build_simple_model(mod_data)
                
                X = self._prepare_features(mod_data, feature_names)
                X_scaled = self.scaler.fit_transform(X)
                y = mod_data['action'].values
                accuracy = model.score(X_scaled, y)
                
                modality_results[mod] = {
                    'accuracy': accuracy,
                    'n_trials': len(mod_data),
                    'coefficients': model.coef_[0],
                    'feature_names': feature_names
                }
                
            except Exception as e:
                print(f"  Failed to analyze modality {mod}: {e}")
        
        return modality_results
    
    def analyze_modality_specific_history_effects(self, df_processed: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze history effects specific to each modality.
        
        Args:
            df_processed: Processed DataFrame with lagged features
            
        Returns:
            Dictionary with modality-specific history effects
        """
        print(f"\n=== MODALITY-SPECIFIC HISTORY EFFECTS ===")
        
        modality_results = {}
        
        for modality in [1, 2, 3]:  # Touch, Vision, Visual-Tactile
            mod_data = df_processed[df_processed['mod'] == modality].copy()
            
            if len(mod_data) < 1000:  # Need sufficient data
                print(f"Skipping modality {modality}: only {len(mod_data)} trials")
                continue
            
            print(f"\n--- Modality {modality} Analysis ---")
            print(f"Total trials: {len(mod_data):,}")
            
            # Analyze effects for each lag
            lag_results = {}
            
            for lag in range(1, self.history_depth + 1):
                # Build model for this lag
                features = ['angle', f'action_n-{lag}']
                
                if all(col in mod_data.columns for col in features):
                    X = mod_data[features].values
                    y = mod_data['action'].values
                    
                    # Fit model
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    model = LogisticRegression(C=1.0, random_state=42)
                    model.fit(X_scaled, y)
                    
                    accuracy = model.score(X_scaled, y)
                    action_coeff = model.coef_[0][1]  # action_n-lag coefficient
                    
                    lag_results[lag] = {
                        'coefficient': action_coeff,
                        'accuracy': accuracy
                    }
                    
                    print(f"  Lag {lag}: β={action_coeff:.4f}, accuracy={accuracy:.3f}")
            
            modality_results[modality] = {
                'n_trials': len(mod_data),
                'lag_results': lag_results
            }
        
        return modality_results
    
    def _prepare_features(self, df: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
        """
        Prepare feature matrix from DataFrame.
        
        Args:
            df: DataFrame with features
            feature_names: List of feature names to extract
            
        Returns:
            Feature matrix
        """
        features = []
        
        for feature_name in feature_names:
            if feature_name == 'angle':
                features.append(df['angle'].values)
            elif feature_name.startswith('mod_'):
                mod_id = int(feature_name.split('_')[1])
                mod_feature = (df['mod'] == mod_id).astype(int)
                features.append(mod_feature.values)
            elif feature_name in df.columns:
                features.append(df[feature_name].values)
            else:
                # Handle missing features with zeros
                features.append(np.zeros(len(df)))
        
        return np.column_stack(features)
