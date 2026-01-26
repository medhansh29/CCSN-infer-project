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

# Import analysis modules
from fetch_successive_jsons import JSONFetcher
from batch_analyze_objects import batch_analyze, print_summary_stats
from create_summary_plots import create_summary_plots
from batch_advanced_analysis import batch_advanced_analysis, print_advanced_summary


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
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots (faster for large datasets)')
    parser.add_argument('--plot-dir', type=str, default='convergence_plots',
                       help='Directory for individual plots (default: convergence_plots)')
    parser.add_argument('--summary-dir', type=str, default='summary_plots',
                       help='Directory for summary plots (default: summary_plots)')
    parser.add_argument('--advanced-dir', type=str, default='advanced_plots',
                       help='Directory for advanced plots (default: advanced_plots)')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    
    print_header("REFITT CCSN INFERENCE ANALYSIS PIPELINE")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Minimum observations: {args.min_obs}")
    print(f"Generate plots: {not args.no_plots}")
    
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
            min_obs=args.min_obs,
            save_plots=not args.no_plots,
            plot_dir=args.plot_dir
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
    # STEP 3: Advanced Statistical Analysis
    # ====================================================================
    print_header("STEP 3: Advanced Statistical Analysis", '-')
    
    try:
        df_advanced = batch_advanced_analysis(
            min_obs=args.min_obs,
            save_plots=not args.no_plots,
            plot_dir=args.advanced_dir
        )
        print(f"✅ Advanced analysis complete for {len(df_advanced)} objects")
        
    except Exception as e:
        print(f"❌ Error in advanced analysis: {str(e)}")
        print("Continuing without advanced analysis...")
    
    # ====================================================================
    # STEP 4: Generate Summary Visualizations
    # ====================================================================
    if not args.no_plots:
        print_header("STEP 4: Generating Summary Visualizations", '-')
        
        try:
            create_summary_plots(
                metrics_file='convergence_metrics.csv',
                output_dir=args.summary_dir
            )
            print(f"✅ Summary plots saved to: {args.summary_dir}/")
            
        except Exception as e:
            print(f"❌ Error generating plots: {str(e)}")
            print("Continuing without visualizations...")
    
    # ====================================================================
    # STEP 5: Print Final Summary
    # ====================================================================
    print_header("STEP 5: Analysis Summary", '-')
    
    try:
        print_summary_stats(df)
        
        # Print advanced summary
        try:
            df_advanced = pd.read_csv('advanced_metrics.csv')
            print_advanced_summary(df_advanced)
        except:
            pass
        
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
    print(f"  • convergence_metrics.csv - Full metrics for {len(df)} objects")
    print(f"  • advanced_metrics.csv - Advanced statistical metrics")
    print(f"  • object_index_summary.csv - Overview of all objects")
    
    if not args.no_plots:
        print(f"  • {args.summary_dir}/ - Summary visualizations (including advanced metrics)")
        print(f"  • {args.plot_dir}/ - Individual trajectory plots")
        print(f"  • {args.advanced_dir}/ - Advanced analysis plots")
    
    print("\n✨ Analysis pipeline completed successfully!\n")


if __name__ == "__main__":
    main()
