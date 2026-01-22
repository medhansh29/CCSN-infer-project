#!/usr/bin/env python3
"""
Batch Analysis: Process Multiple Objects for Convergence Metrics

Runs convergence analysis on all objects with sufficient observations
and generates summary statistics and visualizations.
"""

import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from fetch_successive_jsons import JSONFetcher
from compare_successive_observations import ConvergenceAnalyzer


def batch_analyze(min_obs: int = 5, save_plots: bool = True, 
                 plot_dir: str = 'convergence_plots'):
    """
    Run convergence analysis on all objects with minimum observations.
    
    Args:
        min_obs: Minimum number of observations required
        save_plots: Whether to save trajectory plots
        plot_dir: Directory to save plots
    """
    # Initialize fetcher
    print("Scanning directories...")
    fetcher = JSONFetcher()
    fetcher.scan_directories()
    
    # Get objects with sufficient observations
    multi_obs = fetcher.get_objects_with_multiple_obs(min_obs=min_obs)
    print(f"\nFound {len(multi_obs)} objects with {min_obs}+ observations")
    
    # Create plot directory if needed
    if save_plots:
        Path(plot_dir).mkdir(exist_ok=True)
        print(f"Plots will be saved to: {plot_dir}/")
    
    # Process each object
    results = []
    
    for obj_id in tqdm(sorted(multi_obs.keys()), desc="Analyzing objects"):
        try:
            analyzer = ConvergenceAnalyzer(obj_id, fetcher)
            report = analyzer.generate_report()
            
            # Flatten report for DataFrame
            row = {
                'object_id': obj_id,
                'num_observations': report['num_observations'],
                'phase_start': report['phase_range'][0],
                'phase_end': report['phase_range'][1],
                'phase_span': report['phase_range'][1] - report['phase_range'][0],
                'date_start': report['date_range'][0],
                'date_end': report['date_range'][1],
            }
            
            # Add N_90 metrics
            for param in ['zams', 'mloss_rate', '56Ni']:
                n90 = report[f'{param}_n90']
                row[f'{param}_n90_days'] = n90['n90_days'] if n90['convergence_achieved'] else None
                row[f'{param}_n90_phase'] = n90['n90_phase'] if n90['convergence_achieved'] else None
                row[f'{param}_converged'] = n90['convergence_achieved']
                row[f'{param}_final'] = n90['final_value']
            
            # Add volatility metrics
            for param in ['zams', 'mloss_rate', '56Ni']:
                vol = report[f'{param}_volatility']
                row[f'{param}_volatility_std'] = vol.get('volatility_std')
                row[f'{param}_volatility_mean_abs'] = vol.get('volatility_mean_abs_change')
                row[f'{param}_max_jump'] = vol.get('max_jump')
            
            # Add residual metrics
            if 'residuals' in report:
                res = report['residuals']
                row['mag_arr_rmse'] = res.get('rmse')
                row['mag_arr_mae'] = res.get('mae')
                row['mag_arr_max_residual'] = res.get('max_residual')
            
            results.append(row)
            
            # Generate plot if requested
            if save_plots:
                plot_path = f"{plot_dir}/{obj_id}_trajectory.png"
                analyzer.plot_parameter_trajectory(save_path=plot_path)
                # Close plot to avoid memory issues
                import matplotlib.pyplot as plt
                plt.close('all')
                
        except Exception as e:
            print(f"  Error processing {obj_id}: {str(e)}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    output_file = 'convergence_metrics.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved convergence metrics to: {output_file}")
    
    return df


def print_summary_stats(df: pd.DataFrame):
    """Print summary statistics from batch analysis."""
    
    print(f"\n{'='*70}")
    print("BATCH ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"Total objects analyzed: {len(df)}")
    
    print(f"\n{'CONVERGENCE RATES':-^70}")
    for param in ['zams', 'mloss_rate', '56Ni']:
        converged = df[f'{param}_converged'].sum()
        total = len(df)
        pct = (converged / total) * 100
        print(f"  {param:12} : {converged}/{total} ({pct:.1f}%) achieved convergence")
        
        # Average N_90 for converged objects
        converged_df = df[df[f'{param}_converged']]
        if len(converged_df) > 0:
            avg_n90 = converged_df[f'{param}_n90_days'].mean()
            median_n90 = converged_df[f'{param}_n90_days'].median()
            print(f"               Average N_90: {avg_n90:.1f} days (median: {median_n90:.1f})")
    
    print(f"\n{'VOLATILITY STATISTICS':-^70}")
    for param in ['zams', 'mloss_rate', '56Ni']:
        vol_std = df[f'{param}_volatility_std'].dropna()
        if len(vol_std) > 0:
            print(f"  {param:12} : mean σ={vol_std.mean():.3f}, "
                 f"median σ={vol_std.median():.3f}, "
                 f"max σ={vol_std.max():.3f}")
    
    print(f"\n{'PREDICTION ACCURACY':-^70}")
    rmse = df['mag_arr_rmse'].dropna()
    if len(rmse) > 0:
        print(f"  mag_arr RMSE: mean={rmse.mean():.4f}, median={rmse.median():.4f}, max={rmse.max():.4f}")
    
    # Best performers
    print(f"\n{'FASTEST CONVERGING OBJECTS (by zams N_90)':-^70}")
    converged = df[df['zams_converged']].sort_values('zams_n90_days')
    if len(converged) > 0:
        for idx, row in converged.head(5).iterrows():
            print(f"  {row['object_id']}: {row['zams_n90_days']:.1f} days")
    
    print(f"\n{'MOST STABLE OBJECTS (by zams volatility)':-^70}")
    stable = df.dropna(subset=['zams_volatility_std']).sort_values('zams_volatility_std')
    if len(stable) > 0:
        for idx, row in stable.head(5).iterrows():
            print(f"  {row['object_id']}: σ={row['zams_volatility_std']:.3f}")
    
    print(f"{'='*70}\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Batch analysis of convergence metrics for multiple objects'
    )
    parser.add_argument('--min-obs', type=int, default=5,
                       help='Minimum number of observations required (default: 5)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating trajectory plots')
    parser.add_argument('--plot-dir', type=str, default='convergence_plots',
                       help='Directory to save plots (default: convergence_plots)')
    
    args = parser.parse_args()
    
    # Run batch analysis
    df = batch_analyze(
        min_obs=args.min_obs,
        save_plots=not args.no_plots,
        plot_dir=args.plot_dir
    )
    
    # Print summary statistics
    print_summary_stats(df)
    
    print(f"Results saved to: convergence_metrics.csv")
    if not args.no_plots:
        print(f"Plots saved to: {args.plot_dir}/")


if __name__ == "__main__":
    main()
