#!/usr/bin/env python3
"""
Batch Analysis: Process Multiple Objects for Convergence Metrics

Runs convergence analysis on all objects with sufficient observations
and generates summary statistics and visualizations.
"""

import argparse
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from fetch_successive_jsons import JSONFetcher
from compare_successive_observations import ConvergenceAnalyzer
from lightcurve_completeness import LightCurveCompletenessChecker

from confidence_metrics import ConfidenceMetrics
from tns_classifier import TNSClassifier, generate_flagged_csv


def batch_analyze(min_obs: int = 5):
    """
    Run convergence analysis on all objects with minimum observations.
    
    Args:
        min_obs: Minimum number of observations required
    """
    # Initialize fetcher
    print("Scanning directories...")
    fetcher = JSONFetcher()
    fetcher.scan_directories()
    
    # Get objects with sufficient observations
    multi_obs = fetcher.get_objects_with_multiple_obs(min_obs=min_obs)
    print(f"\nFound {len(multi_obs)} objects with {min_obs}+ observations")
    
    # --- TNS Classification Filter ---
    print("\n🔍 Running TNS classification filter...")
    tns_clf = TNSClassifier()
    all_ids = sorted(multi_obs.keys())
    class_df = tns_clf.classify_batch(all_ids)
    flagged_ids = tns_clf.get_flagged_ids(all_ids)
    
    # Generate flagged objects CSV with convergence metrics
    generate_flagged_csv(all_ids)
    
    print(f"  🚩 Flagged {len(flagged_ids)} non-IIP objects (excluded from analysis)")
    # ------------------------------------
    
    # Process each object
    results = []
    skipped_incomplete = 0
    skipped_classification = 0
    
    for obj_id in tqdm(sorted(multi_obs.keys()), desc="Analyzing objects"):
        try:
            # Skip objects flagged as non-II by TNS
            if obj_id in flagged_ids:
                skipped_classification += 1
                continue
            
            # Get timeline for this object
            timeline = fetcher.get_object_timeline(obj_id)
            
            # Check completeness of final observation
            # object_index stores: (date, filter, filepath)
            observations = fetcher.object_index[obj_id]
            final_obs_file = observations[-1][2]  # Get filepath from last observation
            
            checker = LightCurveCompletenessChecker(json_file=final_obs_file)
            completeness_score = checker.check_completeness()
            
            # Skip objects with incomplete light curves
            if completeness_score.overall_status == "Incomplete":
                skipped_incomplete += 1
                continue
            
            # Run convergence analysis
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
            for param in ['zams', 'mloss_rate', '56Ni', 'k_energy', 'beta', 'texp', 'A_v']:
                n90 = report[f'{param}_n90']
                row[f'{param}_n90_days'] = n90['n90_days'] if n90['convergence_achieved'] else None
                row[f'{param}_n90_phase'] = n90['n90_phase'] if n90['convergence_achieved'] else None
                row[f'{param}_converged'] = n90['convergence_achieved']
                row[f'{param}_final'] = n90['final_value']
            
            # Add volatility metrics
            for param in ['zams', 'mloss_rate', '56Ni', 'k_energy', 'beta', 'texp', 'A_v']:
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
            
            # Add completeness information from SNCosmo checker
            row['completeness_status'] = completeness_score.overall_status
            row['latest_phase'] = completeness_score.latest_phase
            row['phase_category'] = completeness_score.phase_category
            row['fit_success'] = completeness_score.fit_success
            row['template_name'] = completeness_score.template_name
            row['chi_squared_reduced'] = completeness_score.chi_squared_reduced
            row['t0_fitted'] = completeness_score.t0_fitted
            
            # --- Confidence Metrics (raw per-parameter uncertainties) ---
            try:
                conf_metrics = ConfidenceMetrics(json_file=final_obs_file)
                conf_results = conf_metrics.compute_all()
                # Add all confidence metrics to the row
                for key, value in conf_results.items():
                    row[key] = value
            except Exception as e:
                # If confidence metrics fail, continue without them
                pass
            # ---------------------------
            
            results.append(row)
            
        except Exception as e:
            print(f"  Error processing {obj_id}: {str(e)}")
            continue
    
    # Create full DataFrame
    df = pd.DataFrame(results)
    
    # Print summary of completeness filtering
    print(f"\n📊 Filtering Results:")
    print(f"  • Total objects with {min_obs}+ observations: {len(multi_obs)}")
    print(f"  • Skipped (Non-IIP classification): {skipped_classification}")
    print(f"  • Skipped (Incomplete light curves): {skipped_incomplete}")
    print(f"  • Analyzed (Clean IIP, Validated/Partial): {len(results)}")
    
    if len(df) == 0:
        print("\n⚠️  No objects to analyze.")
        return df
    
    # ------------------------------------------------------------------ #
    #  Integrate frequency analysis                                       #
    # ------------------------------------------------------------------ #
    print("\n📈 Computing run frequency metrics...")
    freq_rows = []
    for obj_id in df['object_id']:
        observations = fetcher.object_index[obj_id]
        dates = sorted(set(obs[0] for obs in observations))
        num_runs = len(dates)
        filters_used = ','.join(sorted(set(obs[1] for obs in observations)))
        
        time_diffs = []
        if num_runs > 1:
            for i in range(1, len(dates)):
                prev = datetime.strptime(dates[i-1], '%Y-%m-%d')
                curr = datetime.strptime(dates[i], '%Y-%m-%d')
                time_diffs.append((curr - prev).days)
        
        freq_rows.append({
            'object_id': obj_id,
            'total_runs': num_runs,
            'first_run': dates[0] if dates else None,
            'last_run': dates[-1] if dates else None,
            'total_span_days': sum(time_diffs) if time_diffs else 0,
            'avg_interval_days': sum(time_diffs) / len(time_diffs) if time_diffs else None,
            'min_interval_days': min(time_diffs) if time_diffs else None,
            'max_interval_days': max(time_diffs) if time_diffs else None,
            'filters': filters_used,
        })
    
    freq_df = pd.DataFrame(freq_rows)
    df = df.merge(freq_df, on='object_id', how='left')
    
    # ------------------------------------------------------------------ #
    #  Split into 3 CSVs                                                  #
    # ------------------------------------------------------------------ #
    
    # --- 1. Convergence Metrics CSV ---
    # Convergence + volatility + frequency + completeness + advanced metrics
    convergence_cols = ['object_id']
    # Frequency columns
    convergence_cols += ['total_runs', 'first_run', 'last_run',
                         'total_span_days', 'avg_interval_days',
                         'min_interval_days', 'max_interval_days', 'filters']
    # Basic info
    convergence_cols += ['num_observations', 'phase_start', 'phase_end',
                         'phase_span', 'date_start', 'date_end']
    # Per-param convergence (n90, converged, final)
    for param in ['zams', 'mloss_rate', '56Ni', 'k_energy', 'beta', 'texp', 'A_v']:
        convergence_cols += [f'{param}_n90_days', f'{param}_n90_phase',
                             f'{param}_converged', f'{param}_final']
    # Per-param volatility
    for param in ['zams', 'mloss_rate', '56Ni', 'k_energy', 'beta', 'texp', 'A_v']:
        convergence_cols += [f'{param}_volatility_std',
                             f'{param}_volatility_mean_abs', f'{param}_max_jump']
    # Residuals and completeness
    convergence_cols += ['mag_arr_rmse', 'mag_arr_mae', 'mag_arr_max_residual',
                         'completeness_status', 'latest_phase',
                         'phase_category', 'fit_success', 'template_name',
                         'chi_squared_reduced', 't0_fitted']
    
    # Keep only columns that actually exist
    convergence_cols = [c for c in convergence_cols if c in df.columns]
    conv_df = df[convergence_cols]
    conv_df.to_csv('data/convergence_metrics.csv', index=False)
    print(f"\n✅ Saved data/convergence_metrics.csv ({len(conv_df)} objects, {len(convergence_cols)} columns)")
    
    # --- 2. Uncertainty Metrics CSV ---
    uncertainty_cols = ['object_id']
    for param in ['zams', 'k_energy', 'mloss_rate', 'beta', '56Ni', 'texp', 'A_v']:
        uncertainty_cols += [f'{param}_rel_uncertainty', f'{param}_asymmetry_index']
    uncertainty_cols += ['log_evidence', 'posterior_predictive_spread']
    # Also include final values for context
    for param in ['zams', 'mloss_rate', '56Ni', 'k_energy', 'beta', 'texp', 'A_v']:
        uncertainty_cols.append(f'{param}_final')
    
    uncertainty_cols = [c for c in uncertainty_cols if c in df.columns]
    unc_df = df[uncertainty_cols]
    unc_df.to_csv('data/uncertainty_metrics.csv', index=False)
    print(f"✅ Saved data/uncertainty_metrics.csv ({len(unc_df)} objects, {len(uncertainty_cols)} columns)")
    
    # --- 3. Flagged CSV --- (already generated by generate_flagged_csv above)
    print(f"✅ flagged_non_iip_objects.csv (already generated)")
    
    return df


def print_summary_stats(df: pd.DataFrame):
    """Print summary statistics from batch analysis."""
    
    print(f"\n{'='*70}")
    print("BATCH ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"Total objects analyzed: {len(df)}")
    
    # Completeness breakdown
    if 'completeness_status' in df.columns:
        print(f"\n{'LIGHT CURVE COMPLETENESS BREAKDOWN':-^70}")
        status_counts = df['completeness_status'].value_counts()
        for status in ['Validated', 'Partial', 'Incomplete']:
            count = status_counts.get(status, 0)
            pct = (count / len(df)) * 100 if len(df) > 0 else 0
            print(f"  {status:12} : {count:3} objects ({pct:5.1f}%)")
        
        # Show phase categories
        if 'phase_category' in df.columns:
            print(f"\n  Phase Categories:")
            phase_counts = df['phase_category'].value_counts()
            for cat in ['Validated', 'Transitional', 'Preliminary']:
                count = phase_counts.get(cat, 0)
                print(f"    {cat:12} : {count:3} objects")
        
        # Filter to validated for statistics
        df_validated = df[df['completeness_status'] == 'Validated']
        df_all = df.copy()
        
        if len(df_validated) > 0:
            print(f"\n  ⚠️  Statistics below calculated on VALIDATED objects only ({len(df_validated)}/{len(df)})")
            print(f"     Non-validated objects may show false convergence!")
            df = df_validated
        else:
            print(f"\n  ⚠️  WARNING: No validated light curves found!")
            print(f"     All statistics are from potentially incomplete light curves")
    
    print(f"\n{'CONVERGENCE RATES':-^70}")
    for param in ['zams', 'mloss_rate', '56Ni', 'k_energy', 'beta', 'texp', 'A_v']:
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
    for param in ['zams', 'mloss_rate', '56Ni', 'k_energy', 'beta', 'texp', 'A_v']:
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
    parser = argparse.ArgumentParser(description='Batch process SN II-P LCs')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='Base directory containing YYYY-MM-DD observation folders')
    parser.add_argument('--min-obs', type=int, default=5,
                       help='Minimum number of observations required (default: 5)')
    
    args = parser.parse_args()
    
    # Run batch analysis
    df = batch_analyze(
        min_obs=args.min_obs
    )
    
    # Print summary statistics
    print_summary_stats(df)
    
    print(f"Results saved to: convergence_metrics.csv")


if __name__ == "__main__":
    main()
