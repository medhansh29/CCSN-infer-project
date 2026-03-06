#!/usr/bin/env python3
"""
Main pipeline for REFITT CCSN Inference Analysis

Runs the complete analysis pipeline:
1. Index all JSON files by object ID
2. Batch analyze convergence metrics
3. Generate summary visualizations
4. Print comprehensive report

Usage:
    python3 main.py [--min-obs N] [--no-plots]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add 'src' directory to Python path so internal modules can resolve each other
src_path = str(Path(__file__).resolve().parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import analysis modules
from fetch_successive_jsons import JSONFetcher
from batch_analyze_objects import batch_analyze, print_summary_stats
from create_summary_plots import create_summary_plots
from find_outliers import find_all_outliers
from red_alert import main as run_red_alerts
from report_generator import main as generate_report

def print_header(text, char='='):
    """Print formatted section header."""
    width = 70
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}\n")


def main():
    """Run complete analysis pipeline."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Complete CCSN Inference Analysis Pipeline'
    )
    parser.add_argument('--min-obs', type=int, default=5,
                       help='Minimum number of observations required (default: 5)')
    parser.add_argument('--summary-dir', type=str, default='data/summary_plots',
                       help='Directory for summary plots (default: data/summary_plots)')

    
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    print_header("REFITT CCSN INFERENCE ANALYSIS PIPELINE")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Minimum observations: {args.min_obs}")
    
    # ====================================================================
    # STEP 1: Index JSON Files
    # ====================================================================
    print_header("STEP 1: Indexing JSON Files", '-')
    
    try:
        fetcher = JSONFetcher()
        object_index = fetcher.scan_directories()
        fetcher.print_summary()
        
        # Save object index
        multi_obs = fetcher.get_objects_with_multiple_obs(min_obs=1)
        print(f"✅ Indexed {len(multi_obs)} unique objects")
        
    except Exception as e:
        print(f"❌ Error indexing files: {str(e)}")
        sys.exit(1)
    
    # ====================================================================
    # STEP 2: Batch Convergence Analysis
    # ====================================================================
    print_header("STEP 2: Batch Convergence Analysis", '-')
    
    try:
        df = batch_analyze(
            min_obs=args.min_obs
        )
        
        if len(df) == 0:
            print(f"⚠️  No objects found with {args.min_obs}+ observations")
            print("Try reducing --min-obs parameter")
            sys.exit(0)
        
        print(f"✅ Analyzed {len(df)} objects")
        
    except Exception as e:
        print(f"❌ Error in batch analysis: {str(e)}")
        sys.exit(1)
    
    # ====================================================================
    # STEP 3: Generate Summary Visualizations
    # ====================================================================
    print_header("STEP 3: Generating Summary Visualizations", '-')
    
    try:
        create_summary_plots(
            metrics_file='data/convergence_metrics.csv',
            output_dir=args.summary_dir
        )
        print(f"✅ Summary plots saved to: {args.summary_dir}/")
        
    except Exception as e:
        print(f"❌ Error generating plots: {str(e)}")
        print("Continuing without visualizations...")

    # ====================================================================
    # STEP 4: Find Scattter Outliers
    # ====================================================================
    print_header("STEP 4: Finding Scatter Outliers", '-')
    
    try:
        find_all_outliers(metrics_file='data/convergence_metrics.csv')
    except Exception as e:
        print(f"❌ Error generating scatter outliers: {str(e)}")
        
    # ====================================================================
    # STEP 5: Run Integrated Anomaly Detection Engine
    # ====================================================================
    print_header("STEP 5: Running Anomaly Detection (Physics, ML, Percentiles)", '-')
    
    try:
        # red_alert.main() handles the argument parsing if we run via CLI, 
        # but here we can just call it (it uses argparse defaults if passed no args!)
        # Actually it parses sys.argv. Let's patch sys.argv temporarily.
        original_argv = sys.argv
        sys.argv = ['red_alert.py', '--convergence', 'data/convergence_metrics.csv', '--uncertainties', 'data/uncertainty_metrics.csv', '--alerts-output', 'data/red_alerts.csv']
        run_red_alerts()
        sys.argv = original_argv
    except Exception as e:
        print(f"❌ Error running anomaly detection: {str(e)}")
        
    # ====================================================================
    # STEP 6: Generate Final Diagnostic Report
    # ====================================================================
    print_header("STEP 6: Generating Diagnostic Report", '-')
    
    try:
        generate_report()
    except Exception as e:
        print(f"❌ Error generating report: {str(e)}")
    
    # ====================================================================
    # STEP 7: Print Final Summary
    # ====================================================================
    print_header("STEP 7: Analysis Console Summary", '-')
    
    try:
        print_summary_stats(df)
    except Exception as e:
        print(f"❌ Error printing summary: {str(e)}")
    
    # ====================================================================
    # Pipeline Complete
    # ====================================================================
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print_header("PIPELINE COMPLETE", '=')
    print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    print(f"\n📊 Results saved:")
    print(f"  • data/convergence_metrics.csv")
    print(f"  • data/uncertainty_metrics.csv")
    print(f"  • data/red_alerts.csv - Critical physics and ML anomalies")
    print(f"  • data/scatter_outliers.csv - Trendline violations")
    print(f"  • data/diagnostic_report.pdf - 🔥 MASTER DASHBOARD (PDF)")
    print(f"  • {args.summary_dir}/ - Global Plot directory")
    
    print("\n✨ Analysis pipeline completed successfully!\n")


if __name__ == "__main__":
    main()
