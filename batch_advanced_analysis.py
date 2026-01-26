#!/usr/bin/env python3
"""
Batch Advanced Analysis: Process Multiple Objects

Runs advanced statistical analysis on all objects with sufficient observations.
"""

import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from fetch_successive_jsons import JSONFetcher
from advanced_analysis import AdvancedAnalyzer


def batch_advanced_analysis(min_obs: int = 5, save_plots: bool = True,
                            plot_dir: str = 'advanced_plots'):
    """
    Run advanced analysis on all objects.
    
    Args:
        min_obs: Minimum observations required
        save_plots: Whether to save plots
        plot_dir: Directory for plots
    """
    # Initialize
    print("Scanning directories...")
    fetcher = JSONFetcher()
    fetcher.scan_directories()
    
    multi_obs = fetcher.get_objects_with_multiple_obs(min_obs=min_obs)
    print(f"\nFound {len(multi_obs)} objects with {min_obs}+ observations")
    
    if save_plots:
        Path(plot_dir).mkdir(exist_ok=True)
        print(f"Plots will be saved to: {plot_dir}/")
    
    results = []
    
    for obj_id in tqdm(sorted(multi_obs.keys()), desc="Advanced analysis"):
        try:
            analyzer = AdvancedAnalyzer(obj_id, fetcher)
            
            row = {'object_id': obj_id}
            
            # Calculate prediction lead times
            for param in ['zams', 'mloss_rate', '56Ni']:
                lead_result = analyzer.calculate_prediction_lead_time(param)
                row[f'{param}_L90'] = lead_result.get('lead_time_L90')
                row[f'{param}_percent_early'] = lead_result.get('percent_early')
            
            # Calculate correlation breakpoints
            param_pairs = [
                ('zams', 'k_energy'),
                ('zams', 'mloss_rate'),
                ('mloss_rate', '56Ni')
            ]
            
            for p1, p2 in param_pairs:
                corr_df = analyzer.calculate_rolling_correlation(p1, p2)
                if len(corr_df) > 0:
                    row[f't_break_{p1}_{p2}'] = corr_df['t_break'].iloc[0]
            
            # Calculate phase-binned residuals
            res_result = analyzer.calculate_binned_residuals()
            if 'binned_rmse' in res_result:
                for phase_name, rmse in res_result['binned_rmse'].items():
                    row[f'rmse_{phase_name.lower().replace(" ", "_")}'] = rmse
            
            results.append(row)
            
            # Generate plots if requested
            if save_plots:
                analyzer.plot_correlation_heatmap(
                    param_pairs=param_pairs,
                    save_path=f'{plot_dir}/{obj_id}_correlation.png'
                )
                analyzer.plot_residual_gradient(
                    save_path=f'{plot_dir}/{obj_id}_residuals.png'
                )
                analyzer.plot_confidence_growth(
                    param='zams',
                    save_path=f'{plot_dir}/{obj_id}_confidence.png'
                )
                
                import matplotlib.pyplot as plt
                plt.close('all')
                
        except Exception as e:
            print(f"  Error processing {obj_id}: {str(e)}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    output_file = 'advanced_metrics.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Saved advanced metrics to: {output_file}")
    
    return df


def print_advanced_summary(df: pd.DataFrame):
    """Print summary statistics from advanced analysis."""
    
    print(f"\n{'='*70}")
    print("ADVANCED ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"Total objects analyzed: {len(df)}")
    
    print(f"\n{'PREDICTION LEAD TIME (L_90)':-^70}")
    for param in ['zams', 'mloss_rate', '56Ni']:
        lead_times = df[f'{param}_L90'].dropna()
        if len(lead_times) > 0:
            print(f"  {param:12} : mean={lead_times.mean():.1f} days, "
                  f"median={lead_times.median():.1f} days, "
                  f"max={lead_times.max():.1f} days")
            
            percent_early = df[f'{param}_percent_early'].dropna()
            print(f"               On average: {percent_early.mean():.1f}% early prediction")
    
    print(f"\n{'DEGENERACY BREAKING POINT (t_break)':-^70}")
    param_pairs = [
        ('zams', 'k_energy'),
        ('zams', 'mloss_rate'),
        ('mloss_rate', '56Ni')
    ]
    
    for p1, p2 in param_pairs:
        col = f't_break_{p1}_{p2}'
        if col in df.columns:
            breaks = df[col].dropna()
            if len(breaks) > 0:
                print(f"  {p1:12} vs {p2:12} : mean={breaks.mean():.1f} days, "
                      f"median={breaks.median():.1f} days")
    
    print(f"\n{'PHASE-SPECIFIC PREDICTION ERRORS':-^70}")
    phases = ['shock_cooling', 'plateau', 'radioactive_tail']
    for phase in phases:
        col = f'rmse_{phase}'
        if col in df.columns:
            errors = df[col].dropna()
            if len(errors) > 0:
                print(f"  {phase.replace('_', ' ').title():20} : "
                      f"mean RMSE={errors.mean():.3f} mag, "
                      f"median={errors.median():.3f} mag")
    
    print(f"{'='*70}\n")


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Batch advanced analysis for multiple objects'
    )
    parser.add_argument('--min-obs', type=int, default=10,
                       help='Minimum observations (default: 10)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plots')
    parser.add_argument('--plot-dir', type=str, default='advanced_plots',
                       help='Plot directory')
    
    args = parser.parse_args()
    
    df = batch_advanced_analysis(
        min_obs=args.min_obs,
        save_plots=not args.no_plots,
        plot_dir=args.plot_dir
    )
    
    print_advanced_summary(df)
    
    print(f"Results saved to: advanced_metrics.csv")
    if not args.no_plots:
        print(f"Plots saved to: {args.plot_dir}/")


if __name__ == "__main__":
    main()
