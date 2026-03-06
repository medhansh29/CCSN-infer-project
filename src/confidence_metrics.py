#!/usr/bin/env python3
"""
Confidence Metrics for CCSN Inference

Computes raw relative uncertainties for each parameter from the model output.
No composite scores - just the raw data for the user to interpret.
"""

import json
import numpy as np
from typing import Dict, Optional, Any


class ConfidenceMetrics:
    """
    Compute per-parameter relative uncertainties from JSON parameters.
    
    Usage:
        metrics = ConfidenceMetrics(json_file_path)
        results = metrics.compute_all()
    """
    
    # Parameters to compute uncertainties for
    ALL_PARAMS = ['zams', 'k_energy', 'mloss_rate', 'beta', '56Ni', 'texp', 'A_v']
    
    def __init__(self, json_file: str = None, json_data: dict = None):
        """
        Initialize with JSON file path or pre-loaded JSON data.
        """
        if json_data is not None:
            self.data = json_data
        elif json_file is not None:
            with open(json_file, 'r') as f:
                self.data = json.load(f)
        else:
            raise ValueError("Must provide either json_file or json_data")
        
        self.params = self.data.get('parameters', {})
    
    def calculate_relative_uncertainty(self, param_name: str) -> Optional[float]:
        """
        Calculate relative uncertainty for a parameter.
        
        Returns:
            Relative uncertainty (avg_error / |median|), or None if not available
        """
        if param_name not in self.params:
            return None
            
        values = self.params[param_name]
        if not isinstance(values, list) or len(values) < 3:
            return None
            
        median, upper_err, lower_err = values[0], values[1], values[2]
        
        if median == 0:
            return float('inf') if (upper_err + lower_err) > 0 else 0
            
        avg_error = (upper_err + lower_err) / 2
        return avg_error / abs(median)
    
    def calculate_asymmetry_index(self, param_name: str) -> Optional[float]:
        """
        Calculate asymmetry between upper and lower uncertainties.
        
        Returns:
            Asymmetry index (0 = symmetric, 1 = highly asymmetric)
        """
        if param_name not in self.params:
            return None
            
        values = self.params[param_name]
        if not isinstance(values, list) or len(values) < 3:
            return None
            
        upper_err, lower_err = values[1], values[2]
        total = upper_err + lower_err
        
        if total == 0:
            return 0
            
        return abs(upper_err - lower_err) / total
    
    def calculate_posterior_predictive_spread(self) -> float:
        """
        Calculate the spread of posterior predictive samples.
        
        Returns:
            Average 1-sigma spread across time points (in magnitudes)
        """
        mag_arr = self.data.get('mag_arr', [])
        if not mag_arr or len(mag_arr) < 2:
            return 0.0
        
        mag_array = np.array(mag_arr)
        stds = np.std(mag_array, axis=0)
        return float(np.mean(stds))
    
    def extract_log_evidence(self) -> Optional[float]:
        """Extract log evidence (log Z) from parameters."""
        logz = self.params.get('logZ')
        if logz is None:
            return None
        if isinstance(logz, list):
            return logz[0]
        return logz
    
    def compute_all(self) -> Dict[str, Any]:
        """
        Compute all per-parameter relative uncertainties.
        
        Returns:
            Dictionary with relative uncertainties for each parameter
        """
        results = {}
        
        # Per-parameter metrics
        for param in self.ALL_PARAMS:
            rel_unc = self.calculate_relative_uncertainty(param)
            asym = self.calculate_asymmetry_index(param)
            
            results[f'{param}_rel_uncertainty'] = rel_unc
            results[f'{param}_asymmetry_index'] = asym
        
        # Additional useful metrics
        results['log_evidence'] = self.extract_log_evidence()
        results['posterior_predictive_spread'] = self.calculate_posterior_predictive_spread()
        
        return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        metrics = ConfidenceMetrics(json_file=json_file)
        results = metrics.compute_all()
        
        print("\n=== Parameter Relative Uncertainties ===\n")
        for param in ConfidenceMetrics.ALL_PARAMS:
            rel_unc = results.get(f'{param}_rel_uncertainty')
            if rel_unc is not None:
                print(f"{param:15}: {rel_unc*100:6.1f}% relative uncertainty")
    else:
        print("Usage: python3 confidence_metrics.py <json_file>")
