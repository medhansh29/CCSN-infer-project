#!/usr/bin/env python3
"""
Advanced Statistical Analysis for CCSN Inference

Implements sophisticated metrics for parameter degeneracy, phase-specific
residuals, and prediction lead time analysis.

Based on Subrayan et al. insight about parameter correlations and physical phases.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from fetch_successive_jsons import JSONFetcher


class AdvancedAnalyzer:
    """Advanced statistical analysis for CCSN inference convergence."""
    
    def __init__(self, object_id: str, fetcher: JSONFetcher):
        self.object_id = object_id
        self.fetcher = fetcher
        self.timeline_df = fetcher.get_object_timeline(object_id)
        
        # Remove duplicate filter observations
        self.timeline_df = self.timeline_df.drop_duplicates(subset=['date'], keep='first')
        self.timeline_df = self.timeline_df.sort_values('Phase').reset_index(drop=True)
    
    # ========================================================================
    # 1. Parameter Correlation Heatmaps Over Time
    # ========================================================================
    
    def calculate_rolling_correlation(self, param1: str, param2: str, 
                                      window: int = 3) -> pd.DataFrame:
        """
        Calculate rolling Pearson correlation between two parameters.
        
        Detects when degeneracy breaks (correlation drops from ~1.0).
        
        Args:
            param1: First parameter (e.g., 'zams')
            param2: Second parameter (e.g., 'k_energy')
            window: Rolling window size (number of observations)
            
        Returns:
            DataFrame with Phase and correlation coefficient
        """
        data = self.timeline_df[[param1, param2, 'Phase']].dropna()
        
        if len(data) < window:
            return pd.DataFrame({'Phase': [], 'correlation': [], 't_break': []})
        
        correlations = []
        phases = []
        
        for i in range(window, len(data) + 1):
            window_data = data.iloc[i-window:i]
            
            try:
                corr, _ = pearsonr(window_data[param1], window_data[param2])
                correlations.append(corr)
                phases.append(window_data['Phase'].iloc[-1])
            except:
                continue
        
        corr_df = pd.DataFrame({
            'Phase': phases,
            'correlation': correlations
        })
        
        # Detect t_break: first time correlation drops below 0.8
        if len(corr_df) > 0:
            break_points = corr_df[abs(corr_df['correlation']) < 0.8]
            t_break = break_points['Phase'].iloc[0] if len(break_points) > 0 else None
            corr_df['t_break'] = t_break
        
        return corr_df
    
    def plot_correlation_heatmap(self, param_pairs: List[Tuple[str, str]] = None,
                                save_path: Optional[str] = None):
        """
        Create heatmap showing parameter correlations over time.
        
        Args:
            param_pairs: List of parameter pairs to analyze
            save_path: Optional path to save figure
        """
        if param_pairs is None:
            param_pairs = [
                ('zams', 'k_energy'),
                ('zams', 'mloss_rate'),
                ('mloss_rate', '56Ni')
            ]
        
        fig, axes = plt.subplots(len(param_pairs), 1, figsize=(12, 4*len(param_pairs)))
        if len(param_pairs) == 1:
            axes = [axes]
        
        for ax, (p1, p2) in zip(axes, param_pairs):
            corr_df = self.calculate_rolling_correlation(p1, p2)
            
            if len(corr_df) == 0:
                ax.text(0.5, 0.5, 'Insufficient data', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Plot correlation over time
            ax.plot(corr_df['Phase'], corr_df['correlation'], 
                   'o-', linewidth=2, markersize=8, color='steelblue')
            
            # Mark t_break
            if corr_df['t_break'].iloc[0] is not None:
                t_break = corr_df['t_break'].iloc[0]
                ax.axvline(t_break, color='red', linestyle='--', linewidth=2,
                          label=f'Degeneracy Break: {t_break:.1f} days')
            
            # Reference lines
            ax.axhline(0.8, color='orange', linestyle=':', alpha=0.5, label='Threshold (r=0.8)')
            ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
            
            ax.set_xlabel('Phase (days since explosion)', fontsize=11)
            ax.set_ylabel('Pearson Correlation', fontsize=11)
            ax.set_title(f'{p1} vs {p2} Degeneracy - {self.object_id}', 
                        fontsize=12, fontweight='bold')
            ax.set_ylim([-1.05, 1.05])
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved correlation heatmap to: {save_path}")
        else:
            plt.show()
    
    # ========================================================================
    # 2. Residual Phase-Space Analysis
    # ========================================================================
    
    def calculate_binned_residuals(self, early_idx: int = 0, final_idx: int = -1,
                                   phase_bins: List[Tuple[float, float]] = None) -> Dict:
        """
        Calculate RMSE binned by physical SN phases.
        
        Args:
            early_idx: Index of early observation
            final_idx: Index of final observation
            phase_bins: List of (start, end) tuples for phases
                       Default: Shock (0-20), Plateau (20-100), Tail (100+)
                       
        Returns:
            Dictionary with binned residuals
        """
        if phase_bins is None:
            # Standard SN IIP phases
            phase_bins = [
                (0, 20),      # Shock cooling
                (20, 100),    # Plateau
                (100, 300)    # Radioactive tail
            ]
        
        phase_names = ['Shock Cooling', 'Plateau', 'Radioactive Tail']
        
        if len(self.timeline_df) < 2:
            return {'error': 'Insufficient data'}
        
        early_file = self.timeline_df.iloc[early_idx]['filepath']
        final_file = self.timeline_df.iloc[final_idx]['filepath']
        
        # Load JSON data
        with open(early_file) as f:
            early_data = json.load(f)
        with open(final_file) as f:
            final_data = json.load(f)
        
        # Get mag_arr and mjd_arr
        early_mag = np.mean(early_data.get('mag_arr', []), axis=0)
        final_mag = np.mean(final_data.get('mag_arr', []), axis=0)
        mjd_arr = np.array(early_data.get('mjd_arr', []))
        
        # Convert MJD to Phase (need explosion time)
        # Use the Phase from the JSON
        explosion_mjd = mjd_arr[0] - early_data.get('parameters', {}).get('Phase', 0)
        phases = mjd_arr - explosion_mjd
        
        # Calculate residuals
        residuals = early_mag - final_mag
        
        # Bin by phase
        results = {
            'phases': phases,
            'residuals': residuals,
            'binned_rmse': {},
            'binned_mae': {}
        }
        
        for (start, end), name in zip(phase_bins, phase_names):
            mask = (phases >= start) & (phases < end)
            if mask.sum() > 0:
                bin_residuals = residuals[mask]
                results['binned_rmse'][name] = np.sqrt(np.mean(bin_residuals**2))
                results['binned_mae'][name] = np.mean(np.abs(bin_residuals))
            else:
                results['binned_rmse'][name] = None
                results['binned_mae'][name] = None
        
        return results
    
    def plot_residual_gradient(self, save_path: Optional[str] = None):
        """
        Plot residuals vs Phase showing phase-specific errors.
        
        Args:
            save_path: Optional path to save figure
        """
        results = self.calculate_binned_residuals()
        
        if 'error' in results:
            print(f"Cannot plot: {results['error']}")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        phases = results['phases']
        residuals = results['residuals']
        
        # Top: Residuals over time
        scatter = ax1.scatter(phases, residuals, c=np.abs(residuals), 
                             cmap='RdYlBu_r', s=30, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax1.axhline(0, color='black', linestyle='-', linewidth=1)
        
        # Mark phase boundaries
        for phase_bound in [20, 100]:
            ax1.axvline(phase_bound, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        
        # Add phase labels
        ax1.text(10, ax1.get_ylim()[1]*0.9, 'Shock\nCooling', 
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax1.text(60, ax1.get_ylim()[1]*0.9, 'Plateau', 
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax1.text(150, ax1.get_ylim()[1]*0.9, 'Radioactive\nTail', 
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        
        plt.colorbar(scatter, ax=ax1, label='|Residual| (mag)')
        ax1.set_xlabel('Phase (days since explosion)', fontsize=11)
        ax1.set_ylabel('Residual (mag)', fontsize=11)
        ax1.set_title(f'Residual Gradient Analysis - {self.object_id}', 
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Bottom: Binned RMSE
        bin_names = list(results['binned_rmse'].keys())
        rmse_values = [results['binned_rmse'][name] for name in bin_names]
        rmse_values = [v if v is not None else 0 for v in rmse_values]
        
        colors = ['wheat', 'lightblue', 'lightcoral']
        bars = ax2.bar(bin_names, rmse_values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, rmse_values):
            if val > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax2.set_ylabel('RMSE (mag)', fontsize=11)
        ax2.set_title('Phase-Binned Prediction Error', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved residual gradient to: {save_path}")
        else:
            plt.show()
    
    # ========================================================================
    # 3. Discovery vs Stability Delta (Prediction Lead Time)
    # ========================================================================
    
    def calculate_prediction_lead_time(self, param: str = 'zams', 
                                      accuracy_threshold: float = 0.1) -> Dict:
        """
        Calculate L_90: prediction lead time showing when model achieves
        90% accuracy before SN finishes.
        
        Args:
            param: Parameter to analyze
            accuracy_threshold: Accuracy threshold (0.1 = 10%)
            
        Returns:
            Dictionary with lead time metrics
        """
        values = self.timeline_df[param].dropna()
        phases = self.timeline_df.loc[values.index, 'Phase']
        
        if len(values) < 2:
            return {'error': 'Insufficient data'}
        
        final_value = values.iloc[-1]
        final_phase = phases.iloc[-1]
        
        # Find when prediction reaches 90% accuracy
        lower_bound = final_value * (1 - accuracy_threshold)
        upper_bound = final_value * (1 + accuracy_threshold)
        
        # Find first stable point
        stable_idx = None
        for i in range(len(values)):
            remaining = values.iloc[i:]
            if all((remaining >= lower_bound) & (remaining <= upper_bound)):
                stable_idx = i
                break
        
        if stable_idx is not None:
            stable_phase = phases.iloc[stable_idx]
            lead_time = final_phase - stable_phase
            
            return {
                'param': param,
                'stable_phase': stable_phase,
                'final_phase': final_phase,
                'lead_time_L90': lead_time,
                'stable_value': values.iloc[stable_idx],
                'final_value': final_value,
                'percent_early': (lead_time / final_phase) * 100 if final_phase > 0 else 0
            }
        else:
            return {
                'param': param,
                'lead_time_L90': None,
                'reason': 'Never stabilized'
            }
    
    def plot_confidence_growth(self, param: str = 'zams', 
                              save_path: Optional[str] = None):
        """
        Plot prediction confidence growth over time with shrinking ribbon.
        
        Shows the "value add": knowing the parameter X days before SN finished.
        
        Args:
            param: Parameter to analyze
            save_path: Optional path to save figure
        """
        data = self.timeline_df[[param, f'{param}_std', 'Phase']].dropna()
        
        if len(data) < 2:
            print("Insufficient data for confidence growth plot")
            return
        
        lead_time_result = self.calculate_prediction_lead_time(param)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot mean prediction
        ax.plot(data['Phase'], data[param], 'o-', linewidth=2, markersize=8,
               color='steelblue', label=f'{param} prediction')
        
        # Plot confidence ribbon (shrinking over time)
        if f'{param}_std' in data.columns and not data[f'{param}_std'].isna().all():
            ax.fill_between(data['Phase'],
                           data[param] - data[f'{param}_std'],
                           data[param] + data[f'{param}_std'],
                           alpha=0.3, color='steelblue', label='Uncertainty (1σ)')
        
        # Mark final value
        final_value = data[param].iloc[-1]
        final_phase = data['Phase'].iloc[-1]
        ax.axhline(final_value, color='red', linestyle='--', linewidth=2,
                  alpha=0.7, label=f'Final value: {final_value:.2f}')
        
        # Mark stable point and lead time
        if lead_time_result.get('lead_time_L90') is not None:
            stable_phase = lead_time_result['stable_phase']
            lead_time = lead_time_result['lead_time_L90']
            
            ax.axvline(stable_phase, color='green', linestyle='--', linewidth=2,
                      label=f'Stability achieved: {stable_phase:.1f} days')
            
            # Add arrow showing lead time
            ax.annotate('', xy=(final_phase, final_value * 1.05), 
                       xytext=(stable_phase, final_value * 1.05),
                       arrowprops=dict(arrowstyle='<->', color='green', lw=2))
            ax.text((stable_phase + final_phase)/2, final_value * 1.08,
                   f'Lead Time: {lead_time:.1f} days\n({lead_time_result["percent_early"]:.1f}% early)',
                   ha='center', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        ax.set_xlabel('Phase (days since explosion)', fontsize=12)
        ax.set_ylabel(param, fontsize=12)
        ax.set_title(f'Confidence Growth Curve - {self.object_id}\n"We knew the {param} {lead_time_result.get("lead_time_L90", 0):.1f} days before the SN finished"',
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved confidence growth to: {save_path}")
        else:
            plt.show()


def main():
    """Example usage of advanced analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced CCSN inference analysis')
    parser.add_argument('--object', type=str, required=True,
                       help='Object ID to analyze')
    parser.add_argument('--output-dir', type=str, default='advanced_plots',
                       help='Directory for output plots')
    
    args = parser.parse_args()
    
    # Initialize
    fetcher = JSONFetcher()
    fetcher.scan_directories()
    
    analyzer = AdvancedAnalyzer(args.object, fetcher)
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Generate all plots
    print(f"\nGenerating advanced analysis for {args.object}...")
    
    analyzer.plot_correlation_heatmap(
        save_path=f'{args.output_dir}/{args.object}_correlation_heatmap.png'
    )
    
    analyzer.plot_residual_gradient(
        save_path=f'{args.output_dir}/{args.object}_residual_gradient.png'
    )
    
    analyzer.plot_confidence_growth(
        save_path=f'{args.output_dir}/{args.object}_confidence_growth.png'
    )
    
    # Print lead time results
    print("\nPrediction Lead Time Analysis:")
    for param in ['zams', 'mloss_rate', '56Ni']:
        result = analyzer.calculate_prediction_lead_time(param)
        if result.get('lead_time_L90'):
            print(f"  {param:12}: L_90 = {result['lead_time_L90']:.1f} days "
                  f"({result['percent_early']:.1f}% early)")
    
    print(f"\n✅ Plots saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
