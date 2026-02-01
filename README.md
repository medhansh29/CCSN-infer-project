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
3. **Run advanced statistical analysis** (correlation, phase-binned errors, lead time)
4. Generate summary visualizations
5. Print comprehensive statistics

**Default outputs:**

- `convergence_metrics.csv` - Full metrics for all analyzed objects
- `advanced_metrics.csv` - Advanced statistical metrics
- `object_index_summary.csv` - Overview of all objects
- `summary_plots/` - 8 summary visualizations (standard + advanced)
- `convergence_plots/` - Individual trajectory plots
- `advanced_plots/` - Individual advanced analysis plots

### For Large Datasets (Skip Plots)

```bash
python3 main.py --min-obs 10 --no-plots
```

This runs much faster by skipping plot generation.

## What It Measures

This pipeline analyzes the stability and convergence of **all 7 physical model parameters**:

1.  `zams` (Progenitor Mass)
2.  `k_energy` (Explosion Energy)
3.  `mloss_rate` (Mass-Loss Rate)
4.  `beta` (Density Profile Slope)
5.  `56Ni` (Nickel Mass)
6.  `texp` (Explosion Time)
7.  `A_v` (Extinction)

### Standard Metrics

#### 1. N_90 Efficiency (Convergence Speed)

Days until parameter stays within 10% of final value.

- **Key Finding**: 67% of objects converge immediately (N_90 = 0 days) because they are often first observed at late phases (>60 days) when the model is already stable.

#### 2. Volatility (Stability)

Standard deviation of parameter changes between observations.

- Low volatility (< 0.1) indicates self-consistent predictions.

#### 3. Fit Accuracy (Residuals)

RMSE between early vs final magnitude predictions.

- **Median**: 0.35 mag across all objects.

### Advanced Metrics

#### 4. Parameter Degeneracy Breaking (t_break)

Detects when the rolling correlation between two parameters (e.g., ZAMS vs Energy) drops, indicating the model has distinguished them.

#### 5. Phase-Binned Residual Analysis

RMSE calculated separately for distinct SN IIP phases:

- **Plateau** (20-100 days): median RMSE ~0.35 mag
- **Radioactive Tail** (100+ days): median RMSE ~0.25 mag

#### 6. Prediction Lead Time (L_90)

Days between 90% accuracy achievement and SN completion.

- **ZAMS**: avg 27 days early.

## Analysis Visualizations

The pipeline generates 8 summary plots in `summary_plots/`:

1.  **`n90_distributions.png`**: Histograms of convergence times.
2.  **`parameter_correlations.png`**: **(NEW)** Physical parameter degeneracies (Pearson correlation of final values). Comparable to literature.
3.  **`n90_correlations.png`**: Correlation of _convergence times_ (do params converge simultaneously?).
4.  **`lead_time_distributions.png`**: How early can we predict parameters?
5.  **`degeneracy_breaking.png`**: When do key degeneracies resolve?
6.  **`phase_binned_errors.png`**: Model accuracy by physical phase.
7.  **`volatility_vs_n90.png`**: Stability vs Speed.
8.  **`overall_summary.png`**: Consolidated dashboard.

## Light Curve Completeness

The pipeline validates light curves using **SNCosmo template fitting**:

- **Validated**: Physically complete (Phase > 100d + good fit).
- **Partial**: Approaching completeness.
- **Incomplete**: Early-phase.

**Note**: Statistics in summaries are filtered to **Validated** objects only to ensure reliability.

## Directory Structure

```
refitt-ccsn-infer/
├── main.py                            # Orchestrator
├── fetch_successive_jsons.py          # Data Loading
├── compare_successive_observations.py # Analysis Core
├── batch_analyze_objects.py           # Metric Calculation
├── create_summary_plots.py            # Visualization
├── convergence_metrics.csv            # Final Results Table
├── summary_plots/                     # Output Images
└── convergence_plots/                 # Individual Trajectories
└── run_frequency_analysis.py.         # Frequency metrics
```

## Dependencies

```bash
pip install pandas numpy matplotlib tqdm sncosmo
```

## Authors

Medhansh Garg: Purdue Physics and Astronomy
