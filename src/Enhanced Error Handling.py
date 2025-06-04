import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy.stats import pearsonr, spearmanr
import logging
from dataclasses import dataclass

@dataclass
class CorrelationResult:
    """Data class for correlation analysis results"""
    correlation: float
    p_value: float
    sample_size: int
    method: str
    confidence_interval: Tuple[float, float]

class CorrelationAnalyzer:
    """Advanced correlation analysis with robust error handling"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def calculate_correlation(self, 
                            x: pd.Series, 
                            y: pd.Series, 
                            method: str = 'pearson',
                            min_samples: int = 10) -> Optional[CorrelationResult]:
        """Calculate correlation with comprehensive error handling"""
        
        try:
            # Input validation
            if not isinstance(x, pd.Series) or not isinstance(y, pd.Series):
                raise ValueError("Inputs must be pandas Series")
                
            if len(x) != len(y):
                raise ValueError("Series must have the same length")
                
            # Remove NaN values
            valid_data = pd.DataFrame({'x': x, 'y': y}).dropna()
            
            if len(valid_data) < min_samples:
                self.logger.warning(f"Insufficient data points: {len(valid_data)} < {min_samples}")
                return None
                
            x_clean = valid_data['x']
            y_clean = valid_data['y']
            
            # Check for constant values
            if x_clean.std() == 0 or y_clean.std() == 0:
                self.logger.warning("One or both series have zero variance")
                return None
                
            # Calculate correlation
            if method == 'pearson':
                corr, p_value = pearsonr(x_clean, y_clean)
            elif method == 'spearman':
                corr, p_value = spearmanr(x_clean, y_clean)
            else:
                raise ValueError(f"Unsupported correlation method: {method}")
                
            # Calculate confidence interval (Fisher transformation for Pearson)
            if method == 'pearson':
                ci = self._calculate_confidence_interval(corr, len(x_clean))
            else:
                ci = (np.nan, np.nan)
                
            return CorrelationResult(
                correlation=corr,
                p_value=p_value,
                sample_size=len(x_clean),
                method=method,
                confidence_interval=ci
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation: {str(e)}")
            raise
    
    def _calculate_confidence_interval(self, r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
        """Calculate confidence interval using Fisher transformation"""
        try:
            from scipy.stats import norm
            
            # Fisher transformation
            z = 0.5 * np.log((1 + r) / (1 - r))
            se = 1 / np.sqrt(n - 3)
            
            # Critical value
            z_crit = norm.ppf(1 - alpha/2)
            
            # Confidence interval in z-space
            z_lower = z - z_crit * se
            z_upper = z + z_crit * se
            
            # Transform back to correlation space
            r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
            r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
            
            return (r_lower, r_upper)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence interval: {str(e)}")
            return (np.nan, np.nan)