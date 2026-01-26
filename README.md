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

## Physical Light Curve Completeness Validation

> [!IMPORTANT]
> **NEW**: The pipeline now validates light curve completeness using **physical Type IIP supernova criteria** instead of treating the last observation as "truth."

### Why Completeness Matters

Without validation, the analysis could show "false convergence" - parameters appear stable simply because observations ended early, not because the supernova reached a physically meaningful state.

### Four Physical Criteria

The completeness checker (`lightcurve_completeness.py`) implements four quantitative methods:

#### 1. **Slope-Break Detection** (Plateau Drop-Off)

- **Metric**: Rolling derivative of magnitude (Δmag/Δt)
- **Trigger**: Slope exceeds 0.1 mag/day for 10+ days, then flattens
- **Physics**: Marks end of hydrogen recombination plateau (~80-120 days)
- **Why it matters**: Progenitor mass (M_ZAMS) is degenerate with plateau duration

#### 2. **Radioactive Tail Alignment**

- **Metric**: Linear fit to last 15-20 days of data
- **Trigger**: Slope = 0.0098 ± 0.002 mag/day
- **Physics**: Co-56 radioactive decay has immutable decay rate
- **Why it matters**: Only phase where 56Ni mass can be accurately measured

#### 3. **Phase-Window Filter**

- **Categories**:
  - Preliminary: < 70 days (too early)
  - Transitional: 70-120 days (approaching completeness)
  - Validated: > 120 days (mature light curve)
- **Physics**: Progenitor properties "lock in" by late-plateau phase

#### 4. **Flux Ratio Threshold**

- **Metric**: Current magnitude - Peak magnitude
- **Trigger**: ≥ 2.5 mag dimming from peak
- **Physics**: Ensures model has seen information gain from recombination ending

### Completeness Status

Objects receive one of three overall statuses:

| Status         | Criteria                                    | Interpretation             |
| -------------- | ------------------------------------------- | -------------------------- |
| **Validated**  | All criteria met OR (on tail + phase >120d) | True physical convergence  |
| **Partial**    | Some criteria met, phase >70d               | Approaching completeness   |
| **Incomplete** | Early phase or missing features             | May show false convergence |

### How Objects Are Flagged

**Example: ZTF25acfwklu**

```
Phase: 58.7 days → Status: Incomplete

Physical Criteria:
  • Plateau Drop-off: ✗ (not detected)
  • Radioactive Tail: ✗ (slope too steep)
  • Phase Category: Preliminary (< 70 days)
  • Sufficient Dimming: ✗ (0.36 mag, need 2.5 mag)

⚠️ WARNING: Light curve incomplete - metrics may not reflect true convergence
```

The object is flagged as **Incomplete** because:

1. Phase < 70 days (Preliminary)
2. No plateau drop-off detected
3. Not on radioactive tail
4. Insufficient dimming

**Statistics Impact**: Only **Validated** objects contribute to summary statistics to avoid false convergence bias.

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

## Directory Structure

```
refitt-ccsn-infer/
├── main.py                            # Main pipeline orchestrator
├── fetch_successive_jsons.py          # Data indexing and organization
├── compare_successive_observations.py # Individual object analysis
├── batch_analyze_objects.py           # Batch processing
├── lightcurve_completeness.py         # Physical completeness validation (NEW)
├── advanced_analysis.py               # Advanced statistical metrics
├── batch_advanced_analysis.py         # Batch advanced analysis
├── create_summary_plots.py            # Visualization generation
├── convergence_metrics.csv            # Results (generated)
├── advanced_metrics.csv               # Advanced metrics (generated)
├── object_index_summary.csv           # Object index (generated)
├── summary_plots/                     # Summary visualizations (generated)
├── convergence_plots/                 # Individual trajectories (generated)
└── advanced_plots/                    # Advanced analysis plots (generated)
```

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

### 4. Check Light Curve Completeness

```bash
python3 lightcurve_completeness.py 2026-01-21/ZTF25acfwklu_r_nn.json
```

Outputs physical completeness assessment for a single observation.

### 5. Advanced Analysis (Single Object)

```bash
python3 advanced_analysis.py --object ZTF25acfwklu --output-dir advanced_plots
```

### 6. Create Summary Plots

```bash
python3 create_summary_plots.py
```

## Output Files

### convergence_metrics.csv

Columns include:

**Object Identification:**

- Object ID, observation counts, phase/date ranges

**Convergence Metrics (N_90):**

- N_90 days/phase for each parameter (zams, mloss_rate, 56Ni)
- Convergence success flags
- Final parameter values

**Volatility Metrics:**

- Standard deviation, mean absolute change, max jump

**Prediction Accuracy:**

- RMSE, MAE between early and final predictions

**Completeness Validation (NEW):**

- `completeness_status`: Validated / Partial / Incomplete
- `has_plateau_dropoff`: Boolean - plateau drop-off detected
- `on_radioactive_tail`: Boolean - Co-56 decay slope confirmed
- `phase_category`: Preliminary / Transitional / Validated
- `sufficient_dimming`: Boolean - ≥2.5 mag from peak
- `total_dimming_mag`: Total dimming from peak (magnitudes)
- `tail_slope`: Measured late-time slope (mag/day)
- `dropoff_phase`: Phase where plateau drop-off occurs (days)

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
> **Light Curve Completeness**: The analysis uses **physical Type IIP supernova criteria** to validate completeness, not just the last available observation. See "Physical Light Curve Completeness Validation" section above for details.
>
> Objects are categorized as:
>
> - **Validated**: Physically complete (plateau drop-off + sufficient phase/dimming)
> - **Partial**: Approaching completeness (some criteria met)
> - **Incomplete**: Early-phase or missing key features
>
> **Summary statistics are filtered to Validated objects only** to avoid false convergence from incomplete light curves.

- Duplicate observations (same date, different filters) are handled by keeping the first
- Objects with insufficient observations are automatically filtered
- All timestamps are MJD (Modified Julian Date)
- Phase is days since explosion

## Author

Analysis pipeline for REFITT CCSN inference performance evaluation.
