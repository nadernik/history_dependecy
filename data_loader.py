"""
Data Loading and Preprocessing for Serial Dependence Analysis
=============================================================

This module handles loading behavioral data from MAT files, preprocessing,
creating lagged features, and managing CSV caching for faster analysis.
"""

import numpy as np
import pandas as pd
import scipy.io
import os
from typing import Optional, Tuple, Dict, Any
from config import (
    REVERSED_RULE_RATS, 
    DEFAULT_DATA_PATH, 
    DEFAULT_CSV_PATH,
    VALIDATION_RANGES,
    get_feature_names
)
from utils import calculate_difficulty, validate_data_ranges


class DataLoader:
    """Handles loading and preprocessing of behavioral data"""
    
    def __init__(self, data_path: str = DEFAULT_DATA_PATH, 
                 csv_path: str = DEFAULT_CSV_PATH, 
                 history_depth: int = 3):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the MAT file
            csv_path: Path to the CSV cache file
            history_depth: Number of historical trials to include
        """
        self.data_path = data_path
        self.csv_path = csv_path
        self.history_depth = history_depth
        self.df = None
        self.df_processed = None
        
    def check_and_load_csv(self) -> Optional[pd.DataFrame]:
        """
        Check if processed CSV exists and is compatible, load if available.
        
        Returns:
            DataFrame if CSV loaded successfully, None otherwise
        """
        if not os.path.exists(self.csv_path):
            print(f"CSV file not found: {self.csv_path}")
            return None
        
        try:
            # Load the CSV file (skip comment lines if present)
            df = pd.read_csv(self.csv_path, comment='#')
            print(f"Found existing CSV: {self.csv_path}")
            print(f"CSV shape: {df.shape}")
            
            # Detect the history depth of the existing CSV file
            existing_history_columns = [col for col in df.columns if '_n-' in col]
            if existing_history_columns:
                # Extract the maximum lag number from column names
                max_lag_in_csv = 0
                for col in existing_history_columns:
                    if '_n-' in col:
                        try:
                            lag = int(col.split('_n-')[1])
                            max_lag_in_csv = max(max_lag_in_csv, lag)
                        except:
                            continue
                
                if max_lag_in_csv != self.history_depth:
                    print(f"⚠️  HISTORY DEPTH MISMATCH DETECTED:")
                    print(f"   Current history_depth: {self.history_depth}")
                    print(f"   CSV file history_depth: {max_lag_in_csv}")
                    print(f"   CSV file needs to be regenerated with history_depth={self.history_depth}")
                    print(f"   Will reprocess from MAT file to create correct history columns...")
                    return None
            
            # Verify it has all the expected columns for our history depth
            expected_lag_columns = []
            for i in range(1, self.history_depth + 1):
                expected_lag_columns.extend([
                    f'action_n-{i}', f'angle_n-{i}', f'hitmiss_n-{i}', 
                    f'mod_n-{i}', f'mod_transition_n-{i}'
                ])
            
            missing_columns = [col for col in expected_lag_columns if col not in df.columns]
            
            if missing_columns:
                print(f"⚠️  CSV file missing expected columns: {missing_columns}")
                print(f"   This indicates the CSV doesn't match the current history_depth={self.history_depth}")
                print(f"   Will reprocess from MAT file to generate correct columns...")
                return None
            
            print("✅ CSV is compatible with current analysis settings")
            print(f"   History depth matches: {self.history_depth} (CSV is compatible)")
            print(f"   Total trials with complete history: {len(df)}")
            if 'rat' in df.columns:
                print(f"   Rats included: {sorted(df['rat'].unique())}")
            if 'mod' in df.columns:
                print(f"   Modalities: {sorted(df['mod'].unique())}")
            
            self.df_processed = df
            return df
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None
    
    def save_processed_data_to_csv(self) -> None:
        """Save the processed DataFrame to CSV for future use (with metadata)"""
        if self.df_processed is None:
            print("No processed data to save")
            return
            
        try:
            print(f"Saving processed data to: {self.csv_path}")
            
            # Create a temporary file with metadata header
            import tempfile
            import shutil
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_file:
                # Write metadata as comments at the top
                temp_file.write(f"# Serial Dependence Analysis - Processed Data\n")
                temp_file.write(f"# History Depth (k): {self.history_depth}\n")
                temp_file.write(f"# Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                temp_file.write(f"# Total trials: {len(self.df_processed)}\n")
                if 'rat' in self.df_processed.columns:
                    temp_file.write(f"# Rats: {sorted(self.df_processed['rat'].unique())}\n")
                if 'mod' in self.df_processed.columns:
                    temp_file.write(f"# Modalities: {sorted(self.df_processed['mod'].unique())}\n")
                temp_file.write(f"#\n")
                
                # Write the actual CSV data
                self.df_processed.to_csv(temp_file, index=False)
            
            # Move temp file to final location
            shutil.move(temp_file.name, self.csv_path)
            print(f"✅ Processed data saved successfully with history_depth={self.history_depth} metadata!")
            
            # Show file size
            file_size = os.path.getsize(self.csv_path) / (1024 * 1024)  # MB
            print(f"   Shape: {self.df_processed.shape}")
            print(f"   Features: {len(self.df_processed.columns)} columns")
            print(f"   File size: {file_size:.1f} MB")
            
        except Exception as e:
            print(f"Error saving CSV: {e}")
    
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """
        Load and preprocess data from MAT file.
        
        Returns:
            Preprocessed DataFrame
        """
        print(f"Loading data from: {self.data_path}")
        
        try:
            # Load MAT file
            mat_data = scipy.io.loadmat(self.data_path)
            print("✅ MAT file loaded successfully")
            
            # Extract the main data structure (following original code pattern)
            if 'TrialNADER' in mat_data:
                trial_data = mat_data['TrialNADER'][0, 0]
                print("Using data key: TrialNADER")
            else:
                # Try to find the main data array
                possible_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                if possible_keys:
                    trial_data = mat_data[possible_keys[0]]
                    if hasattr(trial_data, 'shape') and len(trial_data.shape) > 0:
                        trial_data = trial_data[0, 0] if trial_data.shape == (1, 1) else trial_data
                    print(f"Using data key: {possible_keys[0]}")
                else:
                    raise ValueError("Could not find data in MAT file")
            
            # Convert structured array to dictionary
            trial_dict = {}
            for field in trial_data.dtype.names:
                trial_dict[field] = trial_data[field][0].flatten()
            
            # Create DataFrame
            df = pd.DataFrame(trial_dict)
            
            print(f"DataFrame created with shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Data preprocessing
            df = self._preprocess_data(df)
            
            self.df = df
            print(f"✅ Data preprocessing complete. Final shape: {df.shape}")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing steps to the raw data (following original code logic).
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        print("Applying data preprocessing...")
        
        # Convert numeric columns (following original code)
        numeric_columns = ['rat', 'trialID', 'date', 'action', 'hitmiss', 'mod', 'angle']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str), errors='coerce')
        
        # Remove NaN values
        df = df.dropna(subset=['rat', 'action', 'hitmiss', 'mod', 'angle'])
        
        # Apply filtering (following original code)
        print("Applying data filters...")
        
        # Filter rats
        excluded_rats = [8, 14, 15, 17, 18, 19, 20, 21]
        rats = df['rat'].unique()
        rats = rats[~np.isin(rats, excluded_rats)]
        df = df[df['rat'].isin(rats)]
        
        # Binary variables
        df['action'] = df['action'].astype(int)
        df = df[df['action'].isin([0, 1])]
        
        df['hitmiss'] = df['hitmiss'].astype(int)
        df = df[df['hitmiss'].isin([0, 1])]
        
        # Angle processing (following original code logic)
        df['angle'] = df['angle'].astype(float)
        df.loc[df['angle'] > 180, 'angle'] = df.loc[df['angle'] > 180, 'angle'] - 180
        df.loc[df['angle'] > 90, 'angle'] = (90 - (df.loc[df['angle'] > 90, 'angle'] - 90)) + 90
        
        # Filter angles and modalities
        df = df[(df['angle'] >= 0) & (df['angle'] <= 90)]
        df = df[df['mod'] != 4]  # Exclude control
        
        # Handle reversed rule rats (following original code)
        rev_rule_rats = REVERSED_RULE_RATS
        for rat_id in rev_rule_rats:
            if rat_id in df['rat'].values:
                rat_mask = df['rat'] == rat_id
                df.loc[rat_mask, 'action'] = 1 - df.loc[rat_mask, 'action']
                print(f"  Flipped actions for reversed rule rat {rat_id}: {rat_mask.sum()} trials")
        
        # Add trial numbering (following original code)
        if 'trialID' in df.columns:
            df = df.sort_values(['rat', 'date', 'trialID'])
            df['trial_number'] = df.groupby(['rat', 'date']).cumcount() + 1
        else:
            # If no trialID, just number sequentially
            df = df.sort_values(['rat', 'date'])
            df['trial_number'] = df.groupby(['rat', 'date']).cumcount() + 1
        
        # Add difficulty feature (distance from category boundary at 45°)
        df['difficulty'] = np.abs(df['angle'] - 45.0)
        
        print(f"Data loaded successfully!")
        print(f"Total trials: {len(df)}")
        print(f"Rats included: {sorted(df['rat'].unique())}")
        print(f"Modalities: {sorted(df['mod'].unique())}")
        print(f"Added difficulty feature (range: {df['difficulty'].min():.1f}° - {df['difficulty'].max():.1f}°)")
        
        return df
    
    def create_lagged_features(self) -> pd.DataFrame:
        """
        Create lagged features for serial dependence analysis (following original code).
        
        Returns:
            DataFrame with lagged features
        """
        if self.df is None:
            raise ValueError("Must load data first")
        
        print(f"\n=== CREATING LAGGED FEATURES (k={self.history_depth}) ===")
        
        df_lag = self.df.copy()
        df_lag = df_lag.sort_values(['rat', 'date', 'trial_number'])
        
        # Add perceptual difficulty feature if not already present
        if 'difficulty' not in df_lag.columns:
            df_lag['difficulty'] = np.abs(df_lag['angle'] - 45.0)
            print(f"Added perceptual difficulty feature (distance from 45°)")
        
        print(f"Difficulty range: {df_lag['difficulty'].min():.1f}° - {df_lag['difficulty'].max():.1f}°")
        
        # Create lagged features (following original code approach)
        for i in range(1, self.history_depth + 1):
            print(f"Creating lag {i} features...")
            
            lag_columns = ['action', 'angle', 'hitmiss', 'mod', 'difficulty']
            
            for col in lag_columns:
                new_col = f"{col}_n-{i}"
                df_lag[new_col] = df_lag.groupby(['rat', 'date'])[col].shift(i)
            
            # Create modality transitions
            current_mod = df_lag['mod']
            past_mod = df_lag[f'mod_n-{i}']
            
            # Create transition indicator
            df_lag[f'mod_transition_n-{i}'] = (current_mod != past_mod).astype(int)
            
            print(f"  Added {len(lag_columns)} lag-{i} features + transition indicator")
        
        # Remove rows with incomplete history
        print(f"Filtering trials with complete {self.history_depth}-trial history...")
        
        # Check for complete history
        complete_mask = pd.Series(True, index=df_lag.index)
        for i in range(1, self.history_depth + 1):
            complete_mask &= df_lag[f'action_n-{i}'].notna()
        
        df_complete = df_lag[complete_mask].copy()
        
        print(f"✅ Lagged features created successfully!")
        print(f"   Original trials: {len(df_lag):,}")
        print(f"   Complete history trials: {len(df_complete):,}")
        print(f"   Retention rate: {len(df_complete)/len(df_lag)*100:.1f}%")
        print(f"   Features per trial: {len(df_complete.columns)}")
        
        # Show feature summary
        feature_types = {
            'Current': ['rat', 'date', 'trial_number', 'action', 'angle', 'hitmiss', 'mod', 'difficulty'],
            'History': [col for col in df_complete.columns if '_n-' in col]
        }
        
        for ftype, features in feature_types.items():
            available_features = [f for f in features if f in df_complete.columns]
            print(f"   {ftype} features: {len(available_features)}")
        
        # Validate feature creation
        self._validate_lagged_features(df_complete)
        
        self.df_processed = df_complete
        return df_complete
    
    def _validate_lagged_features(self, df: pd.DataFrame) -> None:
        """
        Validate that lagged features were created correctly.
        
        Args:
            df: DataFrame with lagged features
        """
        print("Validating lagged features...")
        
        # Check that no lagged features have NaN values
        for i in range(1, self.history_depth + 1):
            for feature_base in ['action', 'angle', 'hitmiss', 'difficulty', 'mod']:
                feature_name = f'{feature_base}_n-{i}'
                if feature_name in df.columns:
                    nan_count = df[feature_name].isna().sum()
                    if nan_count > 0:
                        print(f"Warning: {feature_name} has {nan_count} NaN values")
        
        # Check reasonable value ranges for lagged features
        for i in range(1, self.history_depth + 1):
            action_col = f'action_n-{i}'
            if action_col in df.columns:
                unique_actions = df[action_col].unique()
                if not all(a in [0, 1] for a in unique_actions if not np.isnan(a)):
                    print(f"Warning: {action_col} has unexpected values: {unique_actions}")
        
        print("✅ Lagged feature validation complete")
    
    def get_processed_data(self) -> Optional[pd.DataFrame]:
        """
        Get the processed data with lagged features.
        
        Returns:
            Processed DataFrame or None if not available
        """
        return self.df_processed
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics about the loaded data.
        
        Returns:
            Dictionary with data summary
        """
        if self.df_processed is None:
            return {"error": "No processed data available"}
        
        df = self.df_processed
        
        return {
            "total_trials": len(df),
            "n_rats": df['rat'].nunique(),
            "rat_ids": sorted(df['rat'].unique()),
            "modalities": sorted(df['mod'].unique()),
            "date_range": (df['date'].min(), df['date'].max()) if 'date' in df.columns else None,
            "angle_range": (df['angle'].min(), df['angle'].max()),
            "accuracy_overall": df['hitmiss'].mean(),
            "history_depth": self.history_depth,
            "n_features": len(df.columns)
        }
