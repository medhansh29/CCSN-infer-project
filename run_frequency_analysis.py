#!/usr/bin/env python3
"""
Run Frequency Analysis: Analyze REFITT Model Run Statistics

Creates a comprehensive table showing:
- How many times each object has been run through REFITT
- Time differences between consecutive runs for each object
- Statistics on run frequency and intervals
"""

import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
from fetch_successive_jsons import JSONFetcher


def analyze_run_frequency(fetcher: JSONFetcher) -> pd.DataFrame:
    """
    Analyze run frequency for all objects.
    
    Returns:
        DataFrame with run statistics for each object
    """
    results = []
    
    for obj_id in sorted(fetcher.get_all_object_ids()):
        observations = fetcher.object_index[obj_id]
        
        # Get unique dates (sorted)
        dates = sorted(set(obs[0] for obs in observations))
        num_runs = len(dates)
        
        # Calculate time differences between consecutive runs
        time_diffs = []
        if num_runs > 1:
            for i in range(1, len(dates)):
                prev_date = datetime.strptime(dates[i-1], '%Y-%m-%d')
                curr_date = datetime.strptime(dates[i], '%Y-%m-%d')
                diff_days = (curr_date - prev_date).days
                time_diffs.append(diff_days)
        
        # Get filters used
        filters = sorted(set(obs[1] for obs in observations))
        
        # Build row
        row = {
            'object_id': obj_id,
            'total_runs': num_runs,
            'first_run': dates[0] if dates else None,
            'last_run': dates[-1] if dates else None,
            'total_span_days': time_diffs[0] if len(time_diffs) == 1 else sum(time_diffs) if time_diffs else 0,
            'filters': ','.join(filters),
            'total_observations': len(observations),  # Including different filters
        }
        
        # Add individual time differences as separate columns
        if time_diffs:
            row['avg_interval_days'] = sum(time_diffs) / len(time_diffs)
            row['min_interval_days'] = min(time_diffs)
            row['max_interval_days'] = max(time_diffs)
            row['median_interval_days'] = sorted(time_diffs)[len(time_diffs) // 2]
            
            # Store all intervals as a string for reference
            row['all_intervals'] = ','.join(map(str, time_diffs))
            
            # Store all run dates
            row['all_run_dates'] = ','.join(dates)
        else:
            row['avg_interval_days'] = None
            row['min_interval_days'] = None
            row['max_interval_days'] = None
            row['median_interval_days'] = None
            row['all_intervals'] = ''
            row['all_run_dates'] = dates[0] if dates else ''
        
        results.append(row)
    
    return pd.DataFrame(results)


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics about run frequency."""
    
    print(f"\n{'='*80}")
    print("REFITT RUN FREQUENCY ANALYSIS")
    print(f"{'='*80}")
    print(f"Total unique objects: {len(df)}")
    print(f"Total runs across all objects: {df['total_runs'].sum()}")
    print(f"Total observations (including filter variations): {df['total_observations'].sum()}")
    
    # Run count distribution
    print(f"\n{'RUN COUNT DISTRIBUTION':-^80}")
    run_counts = df['total_runs'].value_counts().sort_index()
    for count in sorted(run_counts.index, reverse=True):
        num_objects = run_counts[count]
        pct = (num_objects / len(df)) * 100
        print(f"  {count:2} runs: {num_objects:2} objects ({pct:5.1f}%)")
    
    # Statistics for objects with multiple runs
    multi_run_df = df[df['total_runs'] > 1]
    if len(multi_run_df) > 0:
        print(f"\n{'INTERVAL STATISTICS (for objects with multiple runs)':-^80}")
        print(f"Number of objects with multiple runs: {len(multi_run_df)}")
        print(f"\nAverage interval between runs:")
        print(f"  Mean: {multi_run_df['avg_interval_days'].mean():.1f} days")
        print(f"  Median: {multi_run_df['median_interval_days'].median():.1f} days")
        print(f"  Min: {multi_run_df['min_interval_days'].min():.0f} days")
        print(f"  Max: {multi_run_df['max_interval_days'].max():.0f} days")
    
    # Most frequently run objects
    print(f"\n{'MOST FREQUENTLY RUN OBJECTS (Top 10)':-^80}")
    top_runs = df.nlargest(10, 'total_runs')
    for idx, row in top_runs.iterrows():
        interval_info = ""
        if row['total_runs'] > 1:
            interval_info = f", avg interval: {row['avg_interval_days']:.1f} days"
        print(f"  {row['object_id']:15} : {row['total_runs']:2} runs{interval_info}")
    
    # Objects with shortest average intervals (most frequently updated)
    print(f"\n{'MOST FREQUENTLY UPDATED OBJECTS (Shortest avg interval)':-^80}")
    frequent_updates = multi_run_df.nsmallest(10, 'avg_interval_days')
    for idx, row in frequent_updates.iterrows():
        print(f"  {row['object_id']:15} : avg {row['avg_interval_days']:.1f} days " +
              f"({row['total_runs']} runs)")
    
    # Objects with longest tracking time
    print(f"\n{'LONGEST TRACKED OBJECTS (by total span)':-^80}")
    longest_tracked = df.nlargest(10, 'total_span_days')
    for idx, row in longest_tracked.iterrows():
        print(f"  {row['object_id']:15} : {row['total_span_days']:3.0f} days " +
              f"({row['first_run']} to {row['last_run']})")
    
    print(f"{'='*80}\n")


def create_detailed_interval_report(df: pd.DataFrame, output_file: str = 'run_intervals_detailed.csv'):
    """
    Create a detailed report with one row per interval.
    
    This expands the data so each consecutive run pair gets its own row.
    """
    detailed_data = []
    
    for idx, row in df.iterrows():
        if not row['all_intervals']:
            continue
            
        obj_id = row['object_id']
        dates = row['all_run_dates'].split(',')
        intervals = list(map(int, row['all_intervals'].split(',')))
        
        for i, interval in enumerate(intervals):
            detailed_data.append({
                'object_id': obj_id,
                'run_number': i + 1,
                'date_from': dates[i],
                'date_to': dates[i + 1],
                'interval_days': interval,
            })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_df.to_csv(output_file, index=False)
    print(f"✅ Detailed interval report saved to: {output_file}")
    
    return detailed_df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Analyze REFITT run frequency and intervals for all objects'
    )
    parser.add_argument('--detailed', action='store_true',
                       help='Also generate detailed interval report')
    
    args = parser.parse_args()
    
    # Initialize fetcher and scan directories
    print("Scanning directories...")
    fetcher = JSONFetcher()
    fetcher.scan_directories()
    
    # Analyze run frequency
    print("Analyzing run frequency...")
    df = analyze_run_frequency(fetcher)
    
    # Save main summary table
    output_file = 'run_frequency_summary.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Run frequency summary saved to: {output_file}")
    
    # Print statistics
    print_summary_statistics(df)
    
    # Create detailed interval report if requested
    if args.detailed:
        create_detailed_interval_report(df)
    
    # Display the table
    print(f"\n{'COMPLETE RUN FREQUENCY TABLE':-^80}")
    display_cols = ['object_id', 'total_runs', 'first_run', 'last_run', 
                   'total_span_days', 'avg_interval_days', 'filters']
    print(df[display_cols].to_string(index=False))
    
    print(f"\nFull results saved to: {output_file}")


if __name__ == "__main__":
    main()
