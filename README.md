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
├── main.py                            # CLI entrypoint — runs the complete 7-step pipeline
├── .env.example                       # Template for TNS credentials (committed)
├── .env                               # Actual TNS credentials (NOT committed — gitignored)
├── .tns_cache.json                    # Cached TNS API responses (auto-generated, gitignored)
├── .gitignore                         # Git exclusion rules
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
│
├── src/                               # SOURCE CODE DIRECTORY
│   ├── batch_analyze_objects.py       # Main orchestrator — filtering, analysis, CSV output
│   ├── fetch_successive_jsons.py      # Indexes JSON files by ZTF object ID across dates
│   ├── compare_successive_observations.py # Per-object convergence: N₉₀, volatility, residuals
│   ├── lightcurve_completeness.py     # Light curve completeness validation via Phase
│   ├── confidence_metrics.py          # Posterior uncertainties from final JSON observation
│   ├── tns_classifier.py              # TNS API queries for spectroscopic classification
│   ├── create_summary_plots.py        # Batch summary visualizations
│   ├── find_outliers.py               # Identifies 2D scatter trendline outliers
│   ├── red_alert.py                   # Physics anomaly & ML (Isolation Forest) flagging
│   └── report_generator.py            # Compiles data into the diagnostic PDF report
│
├── data/                              # PIPELINE OUTPUT DIRECTORY
│   ├── convergence_metrics.csv        # OUTPUT: convergence + frequency metrics
│   ├── uncertainty_metrics.csv        # OUTPUT: per-parameter posterior uncertainties
│   ├── flagged_non_iip_objects.csv    # OUTPUT: objects excluded by TNS classification
│   ├── red_alerts.csv                 # OUTPUT: rule-based and ML flagged anomalies
│   ├── scatter_outliers.csv           # OUTPUT: 2D trendline outliers
│   ├── diagnostic_report.pdf          # OUTPUT: auto-generated summary report
│   └── summary_plots/                 # OUTPUT: batch-level summary visualizations
│
├── convergence_plots/                 # OUTPUT: per-object trajectory plots
│
├── 2025-10-31/                        # DATA: date-stamped REFITT JSON directories
├── 2025-11-01/                        #   Each contains ZTF*.json files
├── ...                                #   (one per object × filter × date)
└── 2026-01-31/
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
| **Convergence (×7 params)** | `zams_n90_days`, `zams_n90_phase`, `zams_converged`, `zams_final`                                                           | N₉₀: days until parameter stays within 10% of final value |
| **Volatility (×7 params)**  | `zams_volatility_std`, `zams_volatility_mean_abs`, `zams_max_jump`                                                          | Jitter between successive parameter estimates             |
| **Residuals**               | `mag_arr_rmse`, `mag_arr_mae`, `mag_arr_max_residual`                                                                       | Prediction accuracy vs observed magnitudes                |
| **Completeness**            | `completeness_status`, `latest_phase`, `phase_category`, `fit_success`, `template_name`, `chi_squared_reduced`, `t0_fitted` | Light curve stage (Validated / Partial / Incomplete)      |

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

## Key Metrics Explained

### N₉₀ (Convergence Time)

Days until a parameter first stays within 10% of its final value for all subsequent observations. Lower = faster convergence.

### Volatility (σ)

Standard deviation of step-to-step fractional changes in a parameter: `Δp / p_prev`. Measures prediction stability.

### Completeness Status

Based on the model's Phase parameter (days since explosion):

- **Validated** (≥100d): On the radioactive tail — convergence is reliable
- **Partial** (70–100d): Plateau phase — interpret with caution
- **Incomplete** (<70d): Too early — excluded from analysis

### Relative Uncertainty

From the posterior: `(upper_err + lower_err) / (2 × |median|)`. Values <0.1 indicate tight constraints.

---

## Module Reference

### `main.py`

CLI entrypoint. Runs the complete 4-step pipeline: index → analyze → visualize → report. All steps are orchestrated here with error handling and timing.

### `src/batch_analyze_objects.py`

The main orchestrator. For each object: loads data via `JSONFetcher`, runs TNS filtering, checks light curve completeness, computes convergence metrics, extracts uncertainties, computes run frequency, and writes all three CSVs. Also generates per-object trajectory plots.

### `src/fetch_successive_jsons.py`

Scans all `YYYY-MM-DD/` directories for `ZTF*.json` files. Builds an index mapping each ZTF object ID to its chronological list of observations. Provides the `JSONFetcher` class used by all other modules.

### `src/compare_successive_observations.py`

Core convergence analyzer. For each object, it:

1. Loads the chronological timeline of JSON observations
2. Extracts the 7 physical parameters at each epoch
3. Computes N₉₀ (days to convergence), volatility metrics, and inter-observation residuals
4. Generates multi-panel trajectory plots

**Standalone**: `python3 src/compare_successive_observations.py --object ZTF25abfuicb`

### `src/lightcurve_completeness.py`

Validates light curve completeness using the model's Phase parameter. Classifies objects as Validated (≥100d, on radioactive tail), Partial (70–100d, plateau), or Incomplete (<70d, early). Objects with Phase <70d are excluded from analysis.

**Standalone**: `python3 src/lightcurve_completeness.py <json_file>`

### `src/confidence_metrics.py`

Extracts posterior uncertainties from a single JSON observation file. Computes relative uncertainty, asymmetry index, log evidence, and posterior predictive spread for all 7 parameters.

**Standalone**: `python3 src/confidence_metrics.py <json_file>`

### `src/tns_classifier.py`

Queries the [Transient Name Server](https://www.wis-tns.org/) for official spectroscopic classifications. Only objects classified as **SN II** or **SN IIP** pass through; all others are flagged and excluded. Results are cached in `.tns_cache.json` — subsequent runs make zero API calls.

- **Rate limiting**: 1 request/second (TNS requirement)
- **Standalone**: `python3 src/tns_classifier.py`

### `src/create_summary_plots.py`

Generates batch visualizations from `data/convergence_metrics.csv`:

- Convergence time distributions (N₉₀ histograms per parameter)
- Volatility comparisons (box plots across parameters)
- Parameter correlations (scatter matrix of final values)
- Overall performance summary (combined dashboard)
- Object stability rankings

**Standalone**: `python3 src/create_summary_plots.py`

### `src/red_alert.py`

Flags mathematically and physically anomalous fits. Uses hard physical bounds (e.g. Explosion Energy to Ejecta Mass ratio) and an Isolation Forest to flag objects holding out-of-distribution values but exhibiting paradoxically high confidence scores.

### `src/report_generator.py`

Ingests all downstream outputs (metrics, anomalies, outliers, TNS tags) into a Markdown template and compiles a comprehensive, styled PDF `diagnostic_report.pdf` presenting overall pipeline health alongside specific flags. Outputs to `data/`.

---

## Notes

- The pipeline is **idempotent** — re-running overwrites previous outputs
- TNS cache (`.tns_cache.json`) persists across runs — delete it to force re-query
- Add new date directories with REFITT JSONs and re-run to incorporate new observations
- All modules are importable: `from module import ClassName`
