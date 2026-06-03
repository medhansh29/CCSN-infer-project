#!/usr/bin/env python3
"""
Main entry point for the REFITT CCSN Inference Analysis Pipeline.

This script orchestrates the full analytical workflow:
1. Indexing: Scans local directories for REFITT inference JSON outputs.
2. Batch Analysis: Performs physics-based feature extraction and convergence checks.
3. Visualization: Generates batch-level summary plots.
4. Diagnostic Report: Produces a concise, physically-grounded outlier PDF.

Usage:
    python3 main.py [--min-obs N] [--summary-dir DIR]
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add 'src' directory to Python path
src_path = str(Path(__file__).resolve().parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import consolidated pipeline modules
from src.fetch_successive_jsons import JSONFetcher
from src.batch_analyze_objects import batch_analyze, print_summary_stats
from src.create_summary_plots import create_summary_plots
from src.report_generator import generate_report

def print_header(text, char='='):
    """Print formatted section header."""
    width = 70
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}\n")

def main():
    """Run automated analysis pipeline."""
    
    parser = argparse.ArgumentParser(
        description='REFITT CCSN Inference Unified Pipeline'
    )
    parser.add_argument('--min-obs', type=int, default=12,
                       help='Minimum observations for reliable analysis (default: 12)')
    parser.add_argument('--summary-dir', type=str, default='data/summary_plots',
                       help='Directory for summary plots (default: data/summary_plots)')
    parser.add_argument('--fix-params', action='store_true',
                       help='[TEMP FIX] Correct misaligned JSON parameters using samples.txt')
    
    args = parser.parse_args()
    
    start_time = datetime.now()
    pipeline_errors = []
    
    print_header("REFITT CCSN INFERENCE: UNIFIED PIPELINE")
    print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Minimum reliable observations: {args.min_obs}")
    
    if args.fix_params:
        from src.fix_json_params import fix_json_parameters
        fix_json_parameters()
    
    # --- STEP 1: Indexing ---
    print_header("STEP 1: Indexing JSON Outputs", '-')
    try:
        fetcher = JSONFetcher()
        fetcher.scan_directories()
        fetcher.print_summary()
    except Exception as e:
        print(f"❌ Critical Error in indexing: {str(e)}")
        sys.exit(1)
    
    # --- STEP 2: Batch Analysis ---
    print_header("STEP 2: Physics-Based Batch Analysis", '-')
    try:
        df = batch_analyze(min_obs=args.min_obs)
        if len(df) == 0:
            print(f"⚠️ No objects found with {args.min_obs}+ observations.")
            sys.exit(0)
        print(f"✅ Analyzed {len(df)} objects.")
    except Exception as e:
        print(f"❌ Error in batch analysis: {str(e)}")
        sys.exit(1)
    
    # --- STEP 3: Summary Plots ---
    print_header("STEP 3: Generating Summary Visualizations", '-')
    try:
        create_summary_plots(
            metrics_file='data/convergence_metrics.csv',
            output_dir=args.summary_dir
        )
        
        # Cleanup redundant files (legacy per-object plots)
        allowed_files = [
            'n90_distributions.png', 'volatility_vs_n90.png', 'n90_correlations.png',
            'parameter_correlations.png', 'overall_summary.png', 'confidence_grades.png',
            'relative_uncertainties.png', 'confidence_components.png', 'confidence_vs_data.png',
            'parameter_scatter_grid.png'
        ]
        print(f"🧹 Cleaning up redundant files in {args.summary_dir}...")
        for f in os.listdir(args.summary_dir):
            if f not in allowed_files and f.endswith('.png'):
                os.remove(os.path.join(args.summary_dir, f))
                
        print(f"✅ Summary plots saved to: {args.summary_dir}/")
    except Exception as e:
        err_msg = f"Error generating plots: {str(e)}"
        print(f"❌ {err_msg}")
        pipeline_errors.append(err_msg)

    # --- STEP 4: Diagnostic Report (PDF) ---
    print_header("STEP 4: Generating Master Diagnostic Report", '-')
    try:
        generate_report()
    except Exception as e:
        err_msg = f"Error generating PDF report: {str(e)}"
        print(f"❌ {err_msg}")
        pipeline_errors.append(err_msg)
    
    # --- Final Summary ---
    print_header("PIPELINE COMPLETE", '=')
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {duration:.2f} seconds")
    
    if os.path.exists('data/diagnostic_report.pdf'):
        print(f"\n🚀 MASTER REPORT GENERATED: data/diagnostic_report.pdf")
    
    if pipeline_errors:
        print("\n" + "!" * 70)
        print("⚠️  PIPELINE FINISHED WITH NON-CRITICAL ERRORS")
        for err in pipeline_errors:
            print(f"  - {err}")
        print("!" * 70 + "\n")
    else:
        print("\n✨ Pipeline executed successfully!\n")

if __name__ == "__main__":
    main()
