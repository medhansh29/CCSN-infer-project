#!/usr/bin/env python3
"""
Standalone tool: generate publication-quality figures from existing
REFITT-CCSN-Infer pipeline output. Not part of main.py — run this
separately, after main.py has already produced the metrics CSVs.

Usage:
    python3 generate_paper_figures.py [--output-dir DIR]
"""
import argparse
import os
import sys
from pathlib import Path

src_path = str(Path(__file__).resolve().parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.create_paper_figures import create_paper_figures

REQUIRED = ['data/convergence_metrics.csv', 'data/uncertainty_metrics.csv',
            'data/scatter_outliers.csv']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=str, default='data/paper_figures')
    args = parser.parse_args()

    missing = [f for f in REQUIRED if not os.path.exists(f)]
    if missing:
        print(f"❌ Missing required input file(s): {', '.join(missing)}")
        print("   Run `python3 main.py` first to generate pipeline output.")
        sys.exit(1)

    create_paper_figures(
        convergence_file='data/convergence_metrics.csv',
        uncertainty_file='data/uncertainty_metrics.csv',
        outliers_file='data/scatter_outliers.csv',
        output_dir=args.output_dir,
    )
    print(f"✅ Paper figures saved to: {args.output_dir}/")
