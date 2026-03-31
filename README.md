# REFITT CCSN Inference Analysis Pipeline

A modular pipeline for analyzing Core-Collapse Supernova (CCSN) parameter convergence from [REFITT](https://refitt.org/) model outputs. The pipeline processes successive JSON observation files, evaluates how quickly physical parameters converge to stable values, and produces publication-ready metrics and visualizations.

## Overview

REFITT produces daily inference outputs for active ZTF transients, each containing posterior estimates for 7 physical parameters (ZAMS mass, mass-loss rate, ⁵⁶Ni mass, kinetic energy, density profile slope, explosion time, and dust extinction). This pipeline tracks these parameters over time to answer: **how quickly and reliably does the model converge to the correct answer?**

### What the Pipeline Does

1. **Indexes** all JSON observation files across date directories, organized by ZTF object ID
2. **Filters** out contaminating objects (SN IIn, SN IIb) using official TNS spectroscopic classifications
3. **Validates** light curve completeness based on the model's Phase parameter
4. **Computes** convergence metrics (N₉₀, volatility), prediction residuals, and parameter uncertainties
5. **Generates** per-object trajectory plots and batch summary visualizations
6. **Detects** mathematical and physical anomalies (ML & Rule-based alerts)
7. **Compiles** a comprehensive PDF diagnostic report of all outputs

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/medhansh29/refitt-ccsn-infer.git
cd refitt-ccsn-infer
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Dependencies: `pandas`, `numpy`, `matplotlib`, `tqdm`, `requests`, `python-dotenv`, `sncosmo`, `astropy`, `iminuit`

### 3. Configure TNS Credentials

The pipeline queries the [Transient Name Server (TNS)](https://www.wis-tns.org/) to classify objects. You need API credentials:

```bash
cp .env.example .env
```

Then edit `.env` with the team's credentials:

```
TNS_API_KEY = "your_api_key"
TNS_BOT_NAME = "your_bot_name"
TNS_BOT_ID = "your_bot_id"
```

> **Note:** Register a bot at [wis-tns.org/bots](https://www.wis-tns.org/bots) if you need new credentials. The `.env` file is gitignored and must never be committed.

### 4. Add Data

Place REFITT JSON observation data in date-stamped directories in the project root:

```
refitt-ccsn-infer/
├── 2025-10-31/
│   ├── ZTF25abfuicb_r.json
│   ├── ZTF25abfuicb_g.json
│   └── ...
├── 2025-11-01/
│   └── ...
└── ...
```

Each JSON file contains REFITT's posterior parameter estimates for one object in one filter on one date.

---

## Running the Pipeline

### Full Pipeline (Recommended)

```bash
python3 main.py
```

This sequentially executes all pipeline modules:

1. **Index** — Scans all date directories and indexes JSON files by ZTF object ID
2. **Analyze** — TNS classification filtering → light curve completeness → convergence metrics → CSV output
3. **Visualize** — Generates summary plots from `data/convergence_metrics.csv`
4. **Scatter Outliers** — Detects trendline deviations from parameter distributions
5. **Anomaly Detection** — Uses physics constraints and Isolation Forests to flag weird fits
6. **Report Generation** — Converts all CSV metrics and plot images into a clean PDF `data/diagnostic_report.pdf`

### CLI Options

| Flag                | Default             | Description                                        |
| :------------------ | :------------------ | :------------------------------------------------- |
| `--min-obs N`       | `5`                 | Minimum number of observations required per object |
| `--no-plots`        | off                 | Skip generating all plots (faster)                 |
| `--plot-dir DIR`    | `convergence_plots` | Output directory for per-object trajectory plots   |
| `--summary-dir DIR` | `summary_plots`     | Output directory for batch summary visualizations  |

### Examples

```bash
# Full pipeline with all plots
python3 main.py

# Fast run without plots
python3 main.py --no-plots

# Only include objects with 10+ observations
python3 main.py --min-obs 10

# Custom output directories
python3 main.py --plot-dir my_plots --summary-dir my_summary
```

### Running Without `main.py`

You can also run the batch analysis directly (skips the `main.py` wrapper):

```bash
python3 batch_analyze_objects.py
python3 batch_analyze_objects.py --no-plots
```

---

## Directory Structure

```
refitt-ccsn-infer/
│
├── main.py                            # CLI entrypoint — runs the complete pipeline
├── .env.example                       # Template for TNS credentials (committed)
├── .env                               # Actual TNS credentials (gitignored)
├── .tns_cache.json                    # Cached TNS API responses (gitignored)
├── .gitignore                         # Git exclusion rules
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
│
├── src/                               # SOURCE CODE DIRECTORY
│   ├── batch_analyze_objects.py       # Main orchestrator — filtering, analysis, CSV output
│   ├── fetch_successive_jsons.py      # Indexes JSON files by ZTF object ID across dates
│   ├── feature_extractors.py          # Physics engine: GP extraction, mass budget, anomalies
│   ├── compare_successive_observations.py # Convergence: N₉₀, volatility, trajectory plots
│   ├── lightcurve_completeness.py     # LC validation: Phase-based completeness gating
│   ├── alerce_client.py               # Raw data: Fetches ZTF lightcurves via ALeRCE API
│   ├── tns_classifier.py              # Filtering: TNS spectroscopic classification
│   ├── create_summary_plots.py        # Visualization: Batch summary plots
│   └── report_generator.py            # Reporting: PDF diagnostic report generator
│
├── data/                              # PIPELINE OUTPUT DIRECTORY
│   ├── convergence_metrics.csv        # OUTPUT: main metrics, features, and flags
│   ├── uncertainty_metrics.csv        # OUTPUT: per-parameter posterior diagnostics
│   ├── flagged_non_iip_objects.csv    # OUTPUT: TNS-excluded objects
│   ├── diagnostic_report.pdf          # OUTPUT: auto-generated PDF summary
│   └── summary_plots/                 # OUTPUT: batch summary visualizations
│
├── convergence_plots/                 # OUTPUT: per-object trajectory plots
│
├── 202*                               # DATA: date-stamped REFITT JSON directories
└── ...                                # (one per object × filter × date)
```

---

## Pipeline Output

The pipeline produces **five CSVs**, a PDF report, and two optional plot directories.

### 1. `convergence_metrics.csv`

The primary output. One row per analyzed object.

| Column Group                | Example Columns                                                                                                             | Description                                               |
| :-------------------------- | :-------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------- |
| **Identifiers**             | `object_id`                                                                                                                 | ZTF object ID                                             |
| **Frequency**               | `total_runs`, `avg_interval_days`, `min_interval_days`, `max_interval_days`, `first_run`, `last_run`                        | How often REFITT ran inference on this object             |
| **Phase**                   | `phase_start`, `phase_end`, `phase_span`                                                                                    | Observation time range (days since explosion)             |
| **Convergence (×7 params)** | `zams_n90_days`, `zams_converged`, `zams_final`                                                                            | N₉₀: days until parameter stays within 10% of final value        |
| **Morphological (GP)**     | `M_plateau_25d`, `gp_t_fall`, `gp_t_rise`, `gp_gr_slope`, `plateau_duration_days`                                          | Derived from ALeRCE raw lightcurves                              |
| **Mass & Physics**         | `implied_Mej`, `mass_budget_violation`, `prior_pegged`, `is_bimodal`                                                       | Kinetic energy vs ejecta mass and MCMC posterior diagnostics     |
| **Precursor Activity**     | `precursor_status`, `precursor_flag`, `precursor_snr_max`                                                                  | Integration across $[-80, -20]$d window                         |
| **Benchmarking**           | `M_peak_residual`, `plateau_duration_residual`, `is_anomaly`, `population_deviation_score`                                 | Deviation from batch-wide scaling relations and Isolation Forest |

The 7 parameters are: `zams`, `mloss_rate`, `56Ni`, `k_energy`, `beta`, `texp`, `A_v`.

### 2. `uncertainty_metrics.csv`

Per-parameter posterior uncertainties from the final observation. One row per object.

| Column Group                  | Example Columns                               | Description                                 |
| :---------------------------- | :-------------------------------------------- | :------------------------------------------ |
| **Relative Uncertainty (×7)** | `zams_rel_uncertainty`                        | (upper_err + lower_err) / (2 × \|median\|)  |
| **Asymmetry (×7)**            | `zams_asymmetry_index`                        | 0 = symmetric errors, 1 = highly asymmetric |
| **Global**                    | `log_evidence`, `posterior_predictive_spread` | Model quality metrics                       |
| **Final Values (×7)**         | `zams_final`, `mloss_rate_final`, ...         | Final parameter estimates for context       |

### 3. `flagged_non_iip_objects.csv`

Objects excluded from analysis because their TNS spectroscopic type is not SN II/IIP.

| Column        | Description                                 |
| :------------ | :------------------------------------------ |
| `object_id`   | ZTF object ID                               |
| `tns_name`    | Official TNS designation (e.g., SN 2025uue) |
| `tns_type`    | Spectroscopic type (SN IIn, SN IIb, etc.)   |
| `flag_reason` | Human-readable reason for exclusion         |

### 4. `red_alerts.csv`

Anomaly flags generated via physical rule constraints and Isolation Forest ML.

| Column           | Description                                             |
| :--------------- | :------------------------------------------------------ |
| `object_id`      | ZTF object ID                                           |
| `alert_reason`   | Human-readable explanation of why this was flagged      |
| `severity_score` | 3 (Physics), 2 (Machine Learning), 1 (Statistical Skew) |

### 5. `scatter_outliers.csv`

Objects strongly deviating (>1.5σ) from typical linear distributions (e.g. ZAMS vs Ni56).

| Column                          | Description                          |
| :------------------------------ | :----------------------------------- |
| `object_id`                     | ZTF object ID                        |
| `x_param_name` / `y_param_name` | The variables being cross-referenced |
| `direction`                     | E.g. "Below predicted trendline"     |

### Plot Directories

| Directory             | Contents                                                                                                                                                      |
| :-------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `convergence_plots/`  | Per-object parameter trajectory plots — 7 parameters over time with N₉₀ markers                                                                               |
| `data/summary_plots/` | Batch-level visualizations: convergence distributions, volatility box plots, parameter correlations, overall performance dashboard, object stability rankings |

---

## Data Flow

```
Date directories (2025-10-31/, 2025-11-01/, ...)
    └── JSON files per object per filter per date
            │
            ▼
    ┌──────────────────────────┐
    │  fetch_successive_jsons  │  Index & organize by object ID
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │    tns_classifier         │  Query TNS → flag non-SN II objects
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │ lightcurve_completeness  │  Phase-based validation (skip <70d)
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │  Convergence Analysis    │  N₉₀, volatility, residuals
    │  (compare_successive_    │
    │   observations)          │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │   confidence_metrics     │  Posterior uncertainties
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────────────────────────┐
    │                Data Outputs                  │
    │  • convergence_metrics.csv                   │
    │  • uncertainty_metrics.csv                   │
    │  • flagged_non_iip_objects.csv               │
    └──────────┬───────────────────────────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │   Anomaly Detection      │  (red_alert.py, find_outliers.py)
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │   Report Generation      │  (report_generator.py)
    └──────────┬───────────────┘
               │
               ▼
      data/diagnostic_report.pdf
```

---

## Detailed Methodology

### 1. Parameter Convergence ($N_{90}$)
**$N_{90}$** is defined as the number of days from the first observation until a physical parameter enters and remains within a ±10% threshold of its final (latest) value.
- **Population Baseline**: Since REFITT fits can be volatile in early phases, $N_{90}$ is only considered "Validated" if the final observation phase is $\ge 100$ days (radioactive tail phase).
- **Stability**: A parameter is marked as `converged` only if it maintains this threshold for all subsequent observations in the indexed timeline.

### 2. GP Morphological Features
Using Gaussian Process (GP) regression on raw ZTF lightcurves fetched via ALeRCE:
- **$M_{\text{plateau, 25d}}$**: The absolute magnitude evaluated exactly 25 days post-explosion ($t_{\exp} + 25$). This specific phase is chosen to avoid the "shock breakout" or early cooling spike (typically 0–10 days) which often contains non-representative physics for standard IIP classification.
- **$t_{\text{rise}}$ / $t_{\text{fall}}$**: Derived from the first derivative ($\frac{dm}{dt}$) of the GP mean. $t_{\text{fall}}$ corresponds to the epoch of fastest decline, a proxy for the end of the plateau.
- **Color Evolution ($g-r$ slope)**: The linear slope of the $g-r$ color index between days 25 and 60 post-explosion.

### 3. Precursor Activity Scan
Scans the pre-explosion window ($[-80, -20]$ days) for cumulative flux excess.
- **Cumulative Integration**: Rather than single-night 5$\sigma$ alerts, we integrate flux across the window to detect faint, persistent eruptions.
- **Visibility Gate**: A scan is only performed if a canonical $-13$ mag RSG eruption would be brighter than ZTF's limiting magnitude ($20.5$) at the object's distance. If $z < 0.015$ (peculiar velocity regime), the distance modulus is flagged as uncertain.

### 4. Mass Budget & Physics Constraints
- **Implied $M_{\text{ej}}$**: Calculated as $M_{\text{ZAMS}} - 1.5\,M_\odot$ (assuming a $1.5\,M_\odot$ remnant).
- **Mass Budget Violation**: Flagged if the ratio of Kinetic Energy to Ejecta Mass ($\frac{E_k}{M_{\text{ej}}}$) exceeds $1.0$. This indicates a physically inconsistent fit where the energy density is too high for a sustained IIP plateau.

### 5. Dynamic Benchmarking (Population Residuals)
For each batch of analyzed objects, the pipeline builds a population-level baseline:
- **Sample Median**: We calculate the column-wise median for all morphological features across the current "clean" batch (TNS-validated SN IIP).
- **Linear Benchmarking**: We perform linear regression on the batch to predict $M_{\text{plateau, 25d}}$ from $(ZAMS, E_k)$ and Plateau Duration from the scaling relation $(\frac{M_{\text{ej}}^3}{E_k})^{0.25}$.
- **Residuals**: Individual objects are evaluated by their deviation from these population-derived trendlines.

### 6. Anomaly Detection (Isolation Forest)
The **Aggregator** uses an Isolation Forest trained on the **morphological feature space** (not the REFITT parameter space).
- **Morphological Space**: Includes $M_{\text{plateau, 25d}}$, $t_{\text{fall}}$, $g-r$ slope, and Lag-1 Autocorrelation of residuals.
- **Population Deviation Score**: The Euclidean distance of an object from the batch centroid in the standardized (Z-scored) morphological feature space.

---

## Module Reference

### `main.py`

CLI entrypoint. Runs the complete 4-step pipeline: index → analyze → visualize → report. All steps are orchestrated here with error handling and timing.

### `src/feature_extractors.py`
The "Physics Engine" of the pipeline. Implements all advanced feature extraction classes including `GPMorphologicalExtractor` (absolute magnitudes, color slopes), `PrecursorScan` (integrated pre-explosion flux), and the `Aggregator` (population-level Benchmarking and Isolation Forest anomaly detection).

### `src/alerce_client.py`
A robust client for fetching raw ZTF lightcurves from the ALeRCE API. Used by the GP extractions to supplement REFITT's model-processed data with raw observations for accurate morphological analysis.

### `src/compare_successive_observations.py`
Core convergence analyzer. For each object, it:
1. Loads the chronological timeline of JSON observations
2. Extracts the 7 physical parameters at each epoch
3. Computes N₉₀ (days to convergence) and volatility metrics
4. Generates multi-panel trajectory plots showing parameter stabilization

### `src/batch_analyze_objects.py`
The main orchestrator. For each object: loads data via `JSONFetcher`, runs TNS filtering, checks light curve completeness, executes the `feature_extractors`, and computes run frequency. It writes the three primary CSV outputs and coordinates the batch-level `Aggregator` logic.

### `src/report_generator.py`
Ingests all downstream outputs (metrics, features, anomalies, TNS tags) into a Markdown template and compiles the `diagnostic_report.pdf` presenting overall pipeline health. Outputs are saved to `data/`.

---

## Notes

- The pipeline is **idempotent** — re-running overwrites previous outputs
- TNS cache (`.tns_cache.json`) persists across runs — delete it to force re-query
- Add new date directories with REFITT JSONs and re-run to incorporate new observations
- All modules are importable: `from module import ClassName`
