"""
Utility Functions for Serial Dependence Analysis
================================================

This module contains utility functions for statistical analysis,
curve fitting, and general helper functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple, Optional, List, Dict, Any
from scipy.special import erf
from scipy.optimize import curve_fit
from config import DEFAULT_DPI


def cumulative_gaussian_lapse(x: np.ndarray, mu: float, sigma: float, 
                            gamma: float, lambda_param: float) -> np.ndarray:
    """
    Cumulative Gaussian function with lapse parameters for psychometric curve fitting.
    
    Args:
        x: Input values (stimulus angles)
        mu: Mean of the Gaussian (bias parameter)
        sigma: Standard deviation of the Gaussian (sensitivity parameter)
        gamma: Lower asymptote (guess rate)
        lambda_param: Upper asymptote lapse rate
        
    Returns:
        Cumulative Gaussian values with lapse parameters
    """
    return gamma + (1 - gamma - lambda_param) * 0.5 * (1 + erf((x - mu) / (sigma * np.sqrt(2))))


def sigma_error(p: float, n: int) -> float:
    """
    Calculate standard error for binomial proportion.
    
    Args:
        p: Proportion (between 0 and 1)
        n: Sample size
        
    Returns:
        Standard error of the proportion
    """
    if n <= 0:
        return 0
    return np.sqrt(p * (1 - p) / n)


def fit_psychometric_curve(angles: np.ndarray, responses: np.ndarray, 
                          min_trials: int = 5) -> Tuple[Optional[np.ndarray], bool, 
                                                       Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Fit a psychometric curve to behavioral data using cumulative Gaussian with lapse.
    
    Args:
        angles: Stimulus angles
        responses: Binary responses (0/1)
        min_trials: Minimum number of trials required per angle bin
        
    Returns:
        Tuple of (fitted_params, fit_success, x_fit, y_fit)
        - fitted_params: [mu, sigma, gamma, lambda] parameters or None if fit failed
        - fit_success: Boolean indicating if fit was successful
        - x_fit: X values for fitted curve or None
        - y_fit: Y values for fitted curve or None
    """
    try:
        # Create angle bins
        angle_bins = np.linspace(0, 90, 10)
        angle_centers = (angle_bins[:-1] + angle_bins[1:]) / 2
        
        # Calculate proportion correct for each bin
        proportions = []
        counts = []
        
        for i in range(len(angle_bins) - 1):
            mask = (angles >= angle_bins[i]) & (angles < angle_bins[i + 1])
            if np.sum(mask) >= min_trials:
                prop = np.mean(responses[mask])
                proportions.append(prop)
                counts.append(np.sum(mask))
            else:
                proportions.append(np.nan)
                counts.append(0)
        
        proportions = np.array(proportions)
        valid_mask = ~np.isnan(proportions)
        
        if np.sum(valid_mask) < 4:  # Need at least 4 points for fitting
            return None, False, None, None
        
        # Fit cumulative Gaussian with lapse
        x_data = angle_centers[valid_mask]
        y_data = proportions[valid_mask]
        
        # Initial parameter guesses
        mu_init = 45  # Middle of angle range
        sigma_init = 15  # Reasonable sensitivity
        gamma_init = 0.05  # Small guess rate
        lambda_init = 0.05  # Small lapse rate
        
        # Parameter bounds
        bounds = ([0, 1, 0, 0],      # Lower bounds
                 [90, 50, 0.5, 0.5])  # Upper bounds
        
        popt, _ = curve_fit(cumulative_gaussian_lapse, x_data, y_data,
                           p0=[mu_init, sigma_init, gamma_init, lambda_init],
                           bounds=bounds, maxfev=2000)
        
        # Generate fitted curve
        x_fit = np.linspace(0, 90, 100)
        y_fit = cumulative_gaussian_lapse(x_fit, *popt)
        
        return popt, True, x_fit, y_fit
        
    except Exception as e:
        print(f"Curve fitting failed: {e}")
        return None, False, None, None


def save_figure(fig: plt.Figure, filename: str, figures_dir: str = 'figures', 
               dpi: int = DEFAULT_DPI) -> None:
    """
    Save a matplotlib figure to the specified directory.
    
    Args:
        fig: Matplotlib figure object
        filename: Name of the file to save
        figures_dir: Directory to save figures in
        dpi: Resolution for saved figure
    """
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
        
    filepath = os.path.join(figures_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved: {filepath}")


def get_top_features(coefficients: np.ndarray, feature_names: List[str], 
                    n: int = 5) -> List[Tuple[str, float]]:
    """
    Get the top N features by absolute coefficient value.
    
    Args:
        coefficients: Array of coefficient values
        feature_names: List of feature names
        n: Number of top features to return
        
    Returns:
        List of (feature_name, coefficient) tuples sorted by absolute value
    """
    if len(coefficients) != len(feature_names):
        raise ValueError("Coefficients and feature names must have same length")
    
    # Create list of (name, coef) tuples
    feature_coefs = list(zip(feature_names, coefficients))
    
    # Sort by absolute coefficient value (descending)
    feature_coefs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return feature_coefs[:n]


def validate_data_ranges(df, validation_ranges: Dict[str, Tuple[float, float]]) -> bool:
    """
    Validate that data columns fall within expected ranges.
    
    Args:
        df: DataFrame to validate
        validation_ranges: Dictionary mapping column names to (min, max) tuples
        
    Returns:
        True if all validations pass, False otherwise
    """
    for column, (min_val, max_val) in validation_ranges.items():
        if column in df.columns:
            col_min = df[column].min()
            col_max = df[column].max()
            
            if col_min < min_val or col_max > max_val:
                print(f"Warning: {column} values ({col_min}-{col_max}) outside expected range ({min_val}-{max_val})")
                return False
    
    return True


def calculate_difficulty(angles: np.ndarray, boundary: float = 45.0) -> np.ndarray:
    """
    Calculate perceptual difficulty as distance from category boundary.
    
    Args:
        angles: Array of stimulus angles
        boundary: Category boundary angle
        
    Returns:
        Array of difficulty values (distance from boundary)
    """
    return np.abs(angles - boundary)


def create_angle_bins(n_bins: int = 9, angle_range: Tuple[float, float] = (0, 90)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create angle bins for psychometric analysis.
    
    Args:
        n_bins: Number of bins to create
        angle_range: (min, max) angle range
        
    Returns:
        Tuple of (bin_edges, bin_centers)
    """
    bin_edges = np.linspace(angle_range[0], angle_range[1], n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_edges, bin_centers


def format_coefficient_text(coef: float, threshold: float = 0.001) -> str:
    """
    Format coefficient value for display.
    
    Args:
        coef: Coefficient value
        threshold: Minimum absolute value to display (smaller values shown as 0)
        
    Returns:
        Formatted string representation
    """
    if abs(coef) < threshold:
        return "0.000"
    else:
        return f"{coef:.3f}"


def calculate_model_accuracy_stats(accuracies: List[float]) -> Dict[str, float]:
    """
    Calculate summary statistics for model accuracies.
    
    Args:
        accuracies: List of accuracy values
        
    Returns:
        Dictionary with mean, std, min, max accuracy
    """
    accuracies = np.array(accuracies)
    return {
        'mean': np.mean(accuracies),
        'std': np.std(accuracies),
        'min': np.min(accuracies),
        'max': np.max(accuracies),
        'median': np.median(accuracies)
    }
