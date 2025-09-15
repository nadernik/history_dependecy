"""
Configuration and Constants for Serial Dependence Analysis
==========================================================

This module contains all configuration settings, constants, and default parameters
used throughout the serial dependence analysis pipeline.
"""

from typing import Dict, List, Any

# Modality mappings
MODALITY_NAMES: Dict[int, str] = {
    1: 'T',      # Touch
    2: 'V',      # Vision  
    3: 'VT',     # Visual-Tactile
    4: 'Control' # Control condition
}

# Rats with reversed rule (need special handling)
REVERSED_RULE_RATS: List[int] = [6, 7, 14, 15, 16, 17]

# Default file paths
DEFAULT_DATA_PATH: str = 'data/behavior_data.mat'
DEFAULT_CSV_PATH: str = 'processed_behavior_data.csv'
DEFAULT_FIGURES_DIR: str = 'figures'
DEFAULT_LOG_DIR: str = 'logs'

# Analysis parameters
DEFAULT_HISTORY_DEPTH: int = 3
DEFAULT_MIN_TRIALS: int = 5

# Model parameters
DEFAULT_CV_FOLDS: int = 5
DEFAULT_REGULARIZATION_C: float = 1.0

# Plotting parameters
DEFAULT_DPI: int = 300
DEFAULT_FIGURE_SIZE: tuple = (12, 8)
DEFAULT_MAX_RATS_PER_FIGURE: int = 6

# Color schemes for modalities
MODALITY_COLORS: Dict[str, tuple] = {
    'T': (0, 2/3, 0),           # Touch - Green
    'V': (0, 0.4470, 0.7410),   # Vision - Blue  
    'VT': (1, 0, 0),            # Visual-Tactile - Red
    'Control': (0.5, 0.5, 0.5)  # Control - Gray
}

# Statistical thresholds
SIGNIFICANCE_THRESHOLD: float = 0.05
EFFECT_SIZE_THRESHOLD: float = 0.1753  # 80th percentile threshold

# File naming patterns
FIGURE_NAME_PATTERNS: Dict[str, str] = {
    'summary': '01_summary_psychometric_by_modality_k{}.png',
    'effects': '02_serial_dependence_effects_k{}.png', 
    'individual_set1': '03_individual_rats_set1_k{}.png',
    'individual_set2': '03_individual_rats_set2_k{}.png',
    'heatmap': '04_coefficients_heatmap_k{}.png',
    'temporal': '05_temporal_patterns_all_effects_k{}.png',
    'modality_history': '06_modality_specific_history_effects_k{}.png',
    'modality_comparison': '07_modality_psychometric_comparison_k{}.png',
    'modality_test': '08_hetero_homo_modality_test_k{}.png',
    'summary_stats': '10_rat_summary_stats_k{}.png'
}

# Analysis configuration
ANALYSIS_CONFIG: Dict[str, Any] = {
    'enable_logging': False,
    'save_figures': True,
    'show_progress': True,
    'use_csv_cache': True,
    'validate_data': True
}

# Feature names for lagged analysis
def get_feature_names(history_depth: int) -> List[str]:
    """
    Generate feature names for lagged analysis.
    
    Args:
        history_depth: Number of historical trials to include
        
    Returns:
        List of feature names
    """
    features = ['angle']
    
    # Add modality indicators
    for mod_id in MODALITY_NAMES.keys():
        if mod_id != 4:  # Skip control
            features.append(f'mod_{mod_id}')
    
    # Add lagged features
    for i in range(1, history_depth + 1):
        features.extend([
            f'action_n-{i}',
            f'angle_n-{i}', 
            f'hitmiss_n-{i}',
            f'difficulty_n-{i}'
        ])
    
    return features

# Validation ranges
VALIDATION_RANGES: Dict[str, tuple] = {
    'angle': (0, 90),
    'action': (0, 1),
    'mod': (1, 4),
    'rat': (1, 20),
    'hitmiss': (0, 1)
}
