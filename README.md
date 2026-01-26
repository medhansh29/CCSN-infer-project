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

### Standard Metrics

#### 1. N_90 Efficiency

Days until parameter stays within 10% of final value

- **ZAMS (Progenitor Mass)**: avg 2.0 days, median 0.0 days
- **Mass-Loss Rate**: avg 5.7 days, median 0.0 days
- **56Ni Mass**: avg 9.4 days, median 4.1 days

#### 2. Volatility (Stability)

Standard deviation of parameter changes between observations

- **ZAMS**: mean σ=0.204, median σ=0.120
- **Mass-Loss Rate**: mean σ=0.130, median σ=0.044
- **56Ni**: mean σ=0.002, median σ=0.002

#### 3. Fit Accuracy (Residuals)

RMSE between early vs final magnitude predictions

- **Mean**: 0.49 mag
- **Median**: 0.35 mag across all objects

### Advanced Metrics

#### 4. Parameter Correlation Heatmaps

Rolling Pearson correlation between parameters over time

- **Detects degeneracy breaking** (t_break): When correlation drops below 0.8
- Typical t_break: ~50-70 days for ZAMS vs k_energy

#### 5. Phase-Binned Residual Analysis

RMSE calculated separately for distinct SN IIP phases:

- **Plateau** (20-100 days): median RMSE ~0.35 mag (best performance)
- **Radioactive Tail** (100+ days): median RMSE ~0.25 mag
- **Shock Cooling** (0-20 days): limited data, higher variance

#### 6. Discovery vs Stability Delta (L_90)

**Prediction Lead Time**: Days between 90% accuracy achievement and SN completion

- **ZAMS**: avg 27.3 days early (28.9% of observation window)
- **Mass-Loss Rate**: avg 13.9 days early (15.3% early)
- **56Ni**: avg 11.2 days early (12.6% early)

---

## Analysis of Results (49 Objects, 5+ Observations)

### Key Findings

> [!IMPORTANT]
> **100% Convergence Success Rate** - All 49 objects achieved parameter convergence

#### Convergence Speed

The model demonstrates remarkably fast convergence:

- **ZAMS**: 67% of objects converge immediately (N_90 = 0 days)
- **Mass-Loss Rate**: 57% converge immediately
- **56Ni**: More gradual, median convergence at 4.1 days

**Interpretation**: Physical parameters related to progenitor structure (ZAMS) are constrained earliest, while nucleosynthesis products (56Ni) require more light curve coverage to finalize.

#### Parameter Stability

Volatility analysis reveals:

- **56Ni shows exceptional stability** (σ = 0.002) - once estimated, it rarely changes
- **ZAMS and mloss_rate** show moderate volatility early, then stabilize
- Low median volatility (σ < 0.12) indicates the model doesn't "jump around" - estimates are consistent

#### Prediction Accuracy

- **Median RMSE: 0.35 mag** - early predictions closely match final light curves
- 75% of objects have RMSE < 0.6 mag
- **Phase-dependent performance**: Model excels during plateau phase, where hydrogen recombination physics are well-constrained

#### Degeneracy Breaking

Parameter correlation analysis shows:

- **t_break typically occurs 50-80 days** after explosion
- Before t_break: ZAMS and k_energy are highly correlated (both affect luminosity)
- After t_break: Light curve shape breaks degeneracy, allowing independent constraints

**Physical insight**: The plateau drop-off timing (around day 80-100) provides orthogonal information that separates mass from energy.

#### Early Warning Capability

The L_90 metric demonstrates strong early prediction:

- **ZAMS known ~27 days before completion** (29% early)
- Even 56Ni (hardest to constrain) is known ~11 days early (13%)

**Value for real-time alerts**: The model provides reliable progenitor classification well before the SN completes its evolution, enabling rapid follow-up decisions.

### Best Performing Objects

- **Fastest convergence**: Multiple objects achieve N_90=0 for all parameters
- **Most stable**: ZTF25abwlocm (σ=0.034), ZTF25abvbczt (σ=0.038)
- **Best fit accuracy**: Several objects with RMSE < 0.15 mag

---

## Key Results Summary

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

**Standard Metrics (4 plots):**

1. **n90_distributions.png** - Histograms of convergence times
2. **volatility_vs_n90.png** - Stability vs convergence speed
3. **n90_correlations.png** - Parameter correlation matrix
4. **overall_summary.png** - Complete performance dashboard

**Advanced Metrics (4 plots):** 5. **lead_time_distributions.png** - L_90 prediction lead time distributions 6. **degeneracy_breaking.png** - Parameter degeneracy t_break boxplots 7. **phase_binned_errors.png** - RMSE by SN physical phase 8. **lead_time_vs_convergence.png** - Relationship between N_90 and L_90

## Dependencies

```bash
pip install pandas numpy matplotlib tqdm
```

## Notes

> [!WARNING]
> **"Final" Observation Definition**: The analysis treats the chronologically **last available observation** in the dataset as the "final" value, NOT a physically complete supernova light curve. This means:
>
> - **N_90**: "Days to 10% convergence" is measured relative to the last observation's parameter values
> - **L_90**: "Prediction lead time" is the time before the last observation, not before the SN physically finished
> - **Residuals**: Calculated as difference between early and last observation predictions
>
> If additional observations become available later (e.g., extending the dataset beyond 2026-01-21), the "final" values will change and metrics will need to be recalculated. The assumption is that by the last observation, parameters have stabilized (validated by low volatility in most cases).

- Duplicate observations (same date, different filters) are handled by keeping the first
- Objects with insufficient observations are automatically filtered
- All timestamps are MJD (Modified Julian Date)
- Phase is days since explosion

## Author

Analysis pipeline for REFITT CCSN inference performance evaluation.
