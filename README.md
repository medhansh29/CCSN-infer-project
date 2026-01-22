# REFITT CCSN Inference Analysis

Automated pipeline for analyzing convergence performance of the REFITT model on Core-Collapse Supernova (CCSN) inference across successive observations.

## Quick Start

### Run Complete Analysis Pipeline

```bash
python3 main.py
```

This will:

1. Index all JSON files across date directories
2. Analyze convergence metrics for objects with 5+ observations
3. Generate summary visualizations
4. Print comprehensive statistics

**Default outputs:**

- `convergence_metrics.csv` - Full metrics for all analyzed objects
- `object_index_summary.csv` - Overview of all objects
- `summary_plots/` - Summary visualizations (4 plots)
- `convergence_plots/` - Individual trajectory plots

### For Large Datasets (Skip Plots)

```bash
python3 main.py --min-obs 10 --no-plots
```

This runs much faster by skipping plot generation.

### Command-Line Options

```bash
python3 main.py --help
```

- `--min-obs N` - Minimum observations required (default: 5)
- `--no-plots` - Skip generating plots for speed
- `--plot-dir DIR` - Directory for individual plots (default: convergence_plots)
- `--summary-dir DIR` - Directory for summary plots (default: summary_plots)

## What It Measures

Based on the TL;DR strategy, this pipeline calculates:

### 1. N_90 Efficiency

Days until parameter stays within 10% of final value

- **ZAMS (Progenitor Mass)**: avg 2.3 days
- **Mass-Loss Rate**: avg 6.3 days
- **56Ni Mass**: avg 10.4 days

### 2. Volatility (Stability)

Standard deviation of parameter changes between observations

- Lower σ = more stable predictions

### 3. Fit Accuracy (Residuals)

RMSE between early vs final magnitude predictions

- Median: 0.35 mag across all objects

## Directory Structure

```
refitt-ccsn-infer/
├── main.py                          # Main pipeline script
├── fetch_successive_jsons.py        # Data indexing
├── compare_successive_observations.py  # Individual analysis
├── batch_analyze_objects.py         # Batch processing
├── create_summary_plots.py          # Visualization
├── convergence_metrics.csv          # Results (generated)
├── object_index_summary.csv         # Object index (generated)
├── summary_plots/                   # Summary plots (generated)
└── convergence_plots/               # Trajectory plots (generated)
```

## Input Data Format

Expected directory structure:

```
YYYY-MM-DD/
  ├── ZTF<objectid>_g_nn.json
  ├── ZTF<objectid>_r_nn.json
  └── ...
```

Each JSON contains:

- `ztf_id`, `filter`, `mjd`, `Phase`
- `parameters`: `zams`, `mloss_rate`, `56Ni`, etc.
- `mag_arr`: predicted magnitudes
- `mjd_arr`: observation timestamps

## Key Results (Current Dataset)

**Dataset:** 791 JSON files, 57 unique objects, 43 with 10+ observations

**Convergence Success:** 100% across all parameters

**Performance:**

- Runtime: ~1-2 seconds for analysis (no plots)
- Runtime: ~30-60 seconds with all plots

## Individual Script Usage

If you need to run steps separately:

### 1. Index Files

```bash
python3 fetch_successive_jsons.py
```

### 2. Analyze Single Object

```bash
python3 compare_successive_observations.py --object ZTF25acfwklu --plot
```

### 3. Batch Analysis

```bash
python3 batch_analyze_objects.py --min-obs 5
```

### 4. Create Summary Plots

```bash
python3 create_summary_plots.py
```

## Output Files

### convergence_metrics.csv

Columns include:

- Object identification and observation counts
- N_90 days/phase for each parameter
- Convergence success flags
- Final parameter values
- Volatility metrics (std, mean abs change, max jump)
- Prediction accuracy (RMSE, MAE)

### Summary Plots

1. **n90_distributions.png** - Histograms of convergence times
2. **volatility_vs_n90.png** - Stability vs convergence speed
3. **n90_correlations.png** - Parameter correlation matrix
4. **overall_summary.png** - Complete performance dashboard

## Dependencies

```bash
pip install pandas numpy matplotlib tqdm
```

## Notes

- Duplicate observations (same date, different filters) are handled by keeping the first
- Objects with insufficient observations are automatically filtered
- All timestamps are MJD (Modified Julian Date)
- Phase is days since explosion

## Author

Analysis pipeline for REFITT CCSN inference performance evaluation.
