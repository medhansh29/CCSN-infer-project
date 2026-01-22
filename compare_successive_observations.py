#!/usr/bin/env python3
"""
Phase 1: Compare Successive Observations for CCSN Inference Analysis

Analyzes parameter convergence across multiple observations of the same object,
calculating metrics like N_90 efficiency, volatility, and fit accuracy.

Based on TL_DR CCSN Infer.md strategy.
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from fetch_successive_jsons import JSONFetcher


class ConvergenceAnalyzer:
    """Analyzes parameter convergence for objects with successive observations."""
    
    def __init__(self, object_id: str, fetcher: JSONFetcher):
        self.object_id = object_id
        self.fetcher = fetcher
        self.timeline_df = fetcher.get_object_timeline(object_id)
        
        # Remove duplicate filter observations from same date (keep first)
        self.timeline_df = self.timeline_df.drop_duplicates(subset=['date'], keep='first')
        self.timeline_df = self.timeline_df.sort_values('Phase').reset_index(drop=True)
        
    def calculate_n90_efficiency(self, param: str = 'zams', tolerance: float = 0.1) -> Dict:
        """
        Calculate N_90 efficiency: days until parameter stays within tolerance of final value.
        
        Args:
            param: Parameter name ('zams', 'mloss_rate', or '56Ni')
            tolerance: Percentage tolerance (0.1 = 10%)
            
        Returns:
            Dictionary with N_90 metrics
        """
        values = self.timeline_df[param].dropna()
        phases = self.timeline_df.loc[values.index, 'Phase']
        
        if len(values) < 2:
            return {
                'param': param,
                'n90_days': None,
                'n90_phase': None,
                'convergence_achieved': False,
                'final_value': None,
                'reason': 'Insufficient data'
            }
        
        final_value = values.iloc[-1]
        lower_bound = final_value * (1 - tolerance)
        upper_bound = final_value * (1 + tolerance)
        
        # Find first index where value enters 10% band and stays there
        n90_idx = None
        for i in range(len(values)):
            # Check if this value and all subsequent values are within bounds
            remaining_values = values.iloc[i:]
            if all((remaining_values >= lower_bound) & (remaining_values <= upper_bound)):
                n90_idx = i
                break
        
        if n90_idx is not None:
            n90_phase = phases.iloc[n90_idx]
            first_phase = phases.iloc[0]
            n90_days = n90_phase - first_phase
            
            return {
                'param': param,
                'n90_days': n90_days,
                'n90_phase': n90_phase,
                'convergence_achieved': True,
                'final_value': final_value,
                'convergence_index': n90_idx,
                'total_observations': len(values)
            }
        else:
            return {
                'param': param,
                'n90_days': None,
                'n90_phase': None,
                'convergence_achieved': False,
                'final_value': final_value,
                'reason': 'Never stabilized within tolerance',
                'total_observations': len(values)
            }
    
    def calculate_volatility(self, param: str = 'zams') -> Dict:
        """
        Calculate volatility: measure of "jitter" between successive observations.
        
        Args:
            param: Parameter name
            
        Returns:
            Dictionary with volatility metrics
        """
        values = self.timeline_df[param].dropna()
        
        if len(values) < 2:
            return {
                'param': param,
                'volatility_std': None,
                'volatility_mean_abs_change': None,
                'max_jump': None,
                'reason': 'Insufficient data'
            }
        
        # Calculate successive differences
        diffs = values.diff().dropna()
        
        # Calculate percentage changes
        pct_changes = values.pct_change().dropna() * 100
        
        return {
            'param': param,
            'volatility_std': diffs.std(),
            'volatility_mean_abs_change': diffs.abs().mean(),
            'volatility_pct_std': pct_changes.std(),
            'max_jump': diffs.abs().max(),
            'max_pct_jump': pct_changes.abs().max(),
            'num_changes': len(diffs)
        }
    
    def calculate_residuals(self, early_idx: int = 0, final_idx: int = -1) -> Dict:
        """
        Calculate residuals between early and final mag_arr predictions.
        
        Args:
            early_idx: Index of early observation
            final_idx: Index of final observation
            
        Returns:
            Dictionary with residual metrics
        """
        if len(self.timeline_df) < 2:
            return {'rmse': None, 'reason': 'Insufficient data'}
        
        early_file = self.timeline_df.iloc[early_idx]['filepath']
        final_file = self.timeline_df.iloc[final_idx]['filepath']
        
        # Load full JSON data
        with open(early_file) as f:
            early_data = json.load(f)
        with open(final_file) as f:
            final_data = json.load(f)
        
        # Extract mag_arr (nested list of samples)
        early_mag_samples = early_data.get('mag_arr', [])
        final_mag_samples = final_data.get('mag_arr', [])
        
        if not early_mag_samples or not final_mag_samples:
            return {'rmse': None, 'reason': 'Missing mag_arr data'}
        
        # Calculate mean magnitude for each MJD point across samples
        early_mag_mean = np.mean(early_mag_samples, axis=0)
        final_mag_mean = np.mean(final_mag_samples, axis=0)
        
        # Ensure same length (should be, but check)
        min_len = min(len(early_mag_mean), len(final_mag_mean))
        early_mag_mean = early_mag_mean[:min_len]
        final_mag_mean = final_mag_mean[:min_len]
        
        # Calculate residuals
        residuals = early_mag_mean - final_mag_mean
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        
        return {
            'rmse': rmse,
            'mae': mae,
            'max_residual': np.max(np.abs(residuals)),
            'mean_residual': np.mean(residuals),
            'early_phase': self.timeline_df.iloc[early_idx]['Phase'],
            'final_phase': self.timeline_df.iloc[final_idx]['Phase'],
            'num_points': len(residuals)
        }
    
    def plot_parameter_trajectory(self, params: List[str] = None, 
                                  save_path: Optional[str] = None):
        """
        Plot parameter evolution over time.
        
        Args:
            params: List of parameters to plot (default: ['zams', 'mloss_rate', '56Ni'])
            save_path: Optional path to save figure
        """
        if params is None:
            params = ['zams', 'mloss_rate', '56Ni']
        
        fig, axes = plt.subplots(len(params), 1, figsize=(10, 3*len(params)))
        if len(params) == 1:
            axes = [axes]
        
        for ax, param in zip(axes, params):
            data = self.timeline_df[[param, f'{param}_std', 'Phase']].dropna()
            
            if len(data) == 0:
                ax.text(0.5, 0.5, f'No data for {param}', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Plot main trajectory
            ax.plot(data['Phase'], data[param], 'o-', label=param, markersize=6)
            
            # Add error bars if available
            if f'{param}_std' in data.columns and not data[f'{param}_std'].isna().all():
                ax.fill_between(data['Phase'], 
                               data[param] - data[f'{param}_std'],
                               data[param] + data[f'{param}_std'],
                               alpha=0.3)
            
            # Mark final value with horizontal line
            final_value = data[param].iloc[-1]
            ax.axhline(final_value, color='red', linestyle='--', 
                      alpha=0.5, label=f'Final: {final_value:.3f}')
            
            # Mark 10% tolerance band
            ax.axhline(final_value * 1.1, color='gray', linestyle=':', alpha=0.3)
            ax.axhline(final_value * 0.9, color='gray', linestyle=':', alpha=0.3)
            ax.fill_between([data['Phase'].min(), data['Phase'].max()],
                           final_value * 0.9, final_value * 1.1,
                           color='green', alpha=0.1, label='±10% band')
            
            ax.set_xlabel('Phase (days since explosion)')
            ax.set_ylabel(param)
            ax.set_title(f'{param} Evolution - {self.object_id}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved trajectory plot to: {save_path}")
        else:
            plt.show()
    
    def generate_report(self) -> Dict:
        """
        Generate comprehensive convergence report for this object.
        
        Returns:
            Dictionary with all metrics
        """
        report = {
            'object_id': self.object_id,
            'num_observations': len(self.timeline_df),
            'phase_range': (self.timeline_df['Phase'].min(), 
                           self.timeline_df['Phase'].max()),
            'date_range': (self.timeline_df['date'].min(), 
                          self.timeline_df['date'].max()),
        }
        
        # Calculate metrics for key parameters
        for param in ['zams', 'mloss_rate', '56Ni']:
            n90 = self.calculate_n90_efficiency(param)
            vol = self.calculate_volatility(param)
            
            report[f'{param}_n90'] = n90
            report[f'{param}_volatility'] = vol
        
        # Calculate residuals
        if len(self.timeline_df) >= 2:
            residuals = self.calculate_residuals()
            report['residuals'] = residuals
        
        return report


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Analyze parameter convergence for CCSN objects'
    )
    parser.add_argument('--object', type=str, 
                       help='Object ID to analyze (e.g., ZTF25acfwklu)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate trajectory plots')
    parser.add_argument('--save-plots', type=str,
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Initialize fetcher
    fetcher = JSONFetcher()
    fetcher.scan_directories()
    
    if args.object:
        # Analyze single object
        if args.object not in fetcher.get_all_object_ids():
            print(f"Error: Object {args.object} not found")
            return
        
        analyzer = ConvergenceAnalyzer(args.object, fetcher)
        report = analyzer.generate_report()
        
        # Print report
        print(f"\n{'='*70}")
        print(f"CONVERGENCE ANALYSIS: {args.object}")
        print(f"{'='*70}")
        print(f"Observations: {report['num_observations']}")
        print(f"Phase Range: {report['phase_range'][0]:.1f} - {report['phase_range'][1]:.1f} days")
        print(f"Date Range: {report['date_range'][0]} to {report['date_range'][1]}")
        
        print(f"\n{'N_90 EFFICIENCY (days to 10% convergence)':-^70}")
        for param in ['zams', 'mloss_rate', '56Ni']:
            n90 = report[f'{param}_n90']
            if n90['convergence_achieved']:
                print(f"  {param:12} : {n90['n90_days']:6.1f} days (Phase {n90['n90_phase']:.1f}, obs {n90['convergence_index']+1}/{n90['total_observations']})")
            else:
                print(f"  {param:12} : Not achieved ({n90.get('reason', 'Unknown')})")
        
        print(f"\n{'VOLATILITY (parameter stability)':-^70}")
        for param in ['zams', 'mloss_rate', '56Ni']:
            vol = report[f'{param}_volatility']
            if vol.get('volatility_std') is not None:
                print(f"  {param:12} : σ={vol['volatility_std']:.3f}, "
                     f"mean_Δ={vol['volatility_mean_abs_change']:.3f}, "
                     f"max_jump={vol['max_jump']:.3f}")
        
        if 'residuals' in report:
            res = report['residuals']
            print(f"\n{'PREDICTION ACCURACY (mag_arr residuals)':-^70}")
            if res.get('rmse') is not None:
                print(f"  RMSE (early vs final): {res['rmse']:.4f} mag")
                print(f"  MAE:  {res['mae']:.4f} mag")
                print(f"  Max residual: {res['max_residual']:.4f} mag")
        
        # Generate plots if requested
        if args.plot or args.save_plots:
            save_path = None
            if args.save_plots:
                Path(args.save_plots).mkdir(exist_ok=True)
                save_path = f"{args.save_plots}/{args.object}_trajectory.png"
            
            analyzer.plot_parameter_trajectory(save_path=save_path)
    
    else:
        # Show available objects
        multi_obs = fetcher.get_objects_with_multiple_obs(min_obs=5)
        print(f"\nAvailable objects with 5+ observations: {len(multi_obs)}")
        sorted_objects = sorted(multi_obs.items(), key=lambda x: x[1], reverse=True)[:20]
        
        print("\nTop 20 objects by observation count:")
        for obj_id, count in sorted_objects:
            print(f"  {obj_id}: {count} observations")
        
        print("\nUsage: python3 compare_successive_observations.py --object ZTF25acfwklu --plot")


if __name__ == "__main__":
    main()
