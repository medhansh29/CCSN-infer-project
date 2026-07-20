# REFITT CCSN Inference Analysis Pipeline

A modular pipeline for analyzing Core-Collapse Supernova (CCSN) parameter convergence from [REFITT](https://refitt.org/) model outputs. The pipeline processes successive JSON observation files, evaluates how quickly physical parameters converge to stable values, detects physical anomalies, and produces a publication-ready PDF diagnostic report.

## Overview

REFITT produces daily inference outputs for active ZTF transients, each containing posterior estimates for 8 physical parameters (ZAMS mass, mass-loss rate, ⁵⁶Ni mass, kinetic energy, density profile β, explosion time, dust extinction, and metallicity logZ). This pipeline tracks these parameters over time to answer: **how quickly and reliably does the model converge to the correct answer?**

### What the Pipeline Does

1. **Indexes** all JSON observation files across date directories, organized by ZTF object ID
2. **Filters** out contaminating objects (SN IIn, SN IIb) using official TNS spectroscopic classifications
3. **Validates** light curve completeness based on the model's Phase parameter
4. **Computes** convergence metrics (N₉₀, volatility), prediction residuals, and parameter uncertainties
15. **Detects** bivariate (2D trendline) and multivariate (3D Mahalanobis) parameter outliers
16. **Generates** per-object trajectory plots, batch summary visualizations, and seaborn corner pairplots
17. **Detects** mathematical and physical anomalies (ML & Rule-based alerts)
18. **Compiles** a comprehensive PDF diagnostic report with interactive navigation, model plot links, and actionable review items
19. **Exports** static JSON payloads for the frontend web application (`summary_index.json` and object-specific lightcurve JSONs)

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

**System Requirement:** The PDF report generation requires a working `pdflatex` installation (e.g., via [TeX Live](https://tug.org/texlive/) or [MacTeX](https://tug.org/mactex/)).

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
│   ├── ZTF25abfuicb_r_nn.json
│   ├── ZTF25abfuicb_g_nn.json
│   └── ...
├── 2025-11-01/
│   └── ...
└── ...
```

Each JSON file contains REFITT's posterior parameter estimates for one object in one filter on one date.

### 5. Add ZTF Forced Photometry Data

The pipeline natively parses local ZTF Forced Photometry (ZFPS) files as its primary lightcurve source, falling back to ALeRCE if the ZFPS coverage is truncated. 

1. Place standard ZFPS `.txt` or `.csv` files inside `data/ztf_forced_photometry/`.
2. The pipeline automatically performs baseline correction (subtracting the median quiescent pre-explosion flux).
3. The pipeline filters out `-99999` sentinel values and `procstatus != 0` abnormal processing warnings.
4. It computes formal upper limits for non-detections (SNR < 3) and applies $\chi^2$ uncertainty rescaling.

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
4. **Diagnostic Report** — Compiles all CSV metrics, plot images, and anomaly flags into a fully interactive PDF at `data/diagnostic_report.pdf`
5. **Static Payloads** — Exports front-end ready JSON files to `data/static_payloads/`

### CLI Options

| Flag                | Default             | Description                                        |
| :------------------ | :------------------ | :------------------------------------------------- |
| `--min-obs N`       | `12`                | Minimum number of observations required per object |
| `--summary-dir DIR` | `data/summary_plots`| Output directory for batch summary visualizations  |
| `--fix-params`      | `False`             | [TEMP FIX] Correct misaligned JSON parameters using samples.txt |

### Examples

```bash
# Full pipeline with all plots
python3 main.py

# Only include objects with 20+ observations
python3 main.py --min-obs 20

# Custom summary plot directory
python3 main.py --summary-dir my_summary
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
│   ├── multivariate_outliers.py       # Outlier detection: 2D bivariate + 3D Mahalanobis
│   ├── ztf_client.py                  # Raw data: Parses local ZTF forced photometry CSVs
│   ├── tns_classifier.py              # Filtering: TNS spectroscopic classification
│   ├── create_summary_plots.py        # Visualization: Batch summary + seaborn corner plots
│   ├── report_generator.py            # Reporting: PDF diagnostic report generator
│   ├── export_static_payloads.py      # Payload generation: JSON index and lightcurve details
│   └── templates/
│       └── report_template.tex        # LaTeX template for the diagnostic PDF
│
├── data/                              # PIPELINE OUTPUT DIRECTORY
│   ├── convergence_metrics.csv        # OUTPUT: main metrics, features, and flags
│   ├── uncertainty_metrics.csv        # OUTPUT: per-parameter posterior diagnostics
│   ├── scatter_outliers.csv           # OUTPUT: bivariate + multivariate outlier flags
│   ├── flagged_non_iip_objects.csv    # OUTPUT: TNS-excluded objects
│   ├── diagnostic_report.pdf          # OUTPUT: auto-generated PDF diagnostic report
│   ├── diagnostic_report.tex          # OUTPUT: generated LaTeX source
│   ├── report_images/                 # OUTPUT: per-OID diagnostic packages
│   │   └── {OID}_{run_date}/          #   Subfolders with plots + raw parameter JSONs
│   ├── static_payloads/               # OUTPUT: JSON payloads for frontend UI
│   │   ├── summary_index.json         #   Unified catalog of objects & parameters
│   │   └── {OID}_lc.json              #   Per-object lightcurves and percentiles
│   ├── ztf_forced_photometry/         # DATA: Directory for raw ZTF FP CSVs
│   └── summary_plots/                 # OUTPUT: batch summary visualizations
│       └── multivariate/              #   Seaborn corner pairplots per physics cluster
│
├── convergence_plots/                 # OUTPUT: per-object trajectory plots
│
├── 202*                               # DATA: date-stamped REFITT JSON directories
└── ...                                # (one per object × filter × date)
```

---

## Pipeline Output

The pipeline produces **four primary CSVs**, a PDF report, a diagnostic metadata package, two plot directories, and the static JSON payloads.

### 1. `convergence_metrics.csv`

The primary output. One row per analyzed object.

| Column Group                | Example Columns                                                                                                             | Description                                               |
| :-------------------------- | :-------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------- |
| **Identifiers**             | `object_id`                                                                                                                 | ZTF object ID                                             |
| **Frequency**               | `total_runs`, `avg_interval_days`, `min_interval_days`, `max_interval_days`, `first_run`, `last_run`                        | How often REFITT ran inference on this object             |
| **Phase**                   | `phase_start`, `phase_end`, `phase_span`                                                                                    | Observation time range (days since explosion)             |
| **Convergence (×8 params)** | `zams_n90_days`, `zams_converged`, `zams_final`                                                                            | N₉₀: days until parameter stays within 10% of final value        |
| **Morphological (GP)**     | `M_plateau_25d`, `gp_t_fall`, `gp_t_rise`, `gp_gr_slope`, `plateau_duration_days`                                          | Derived from raw ZTF forced photometry                            |
| **Mass & Physics**         | `implied_Mej`, `mass_budget_violation`, `prior_pegged`, `is_bimodal`                                                       | Kinetic energy vs ejecta mass and MCMC posterior diagnostics     |
| **Precursor Activity**     | `precursor_status`, `precursor_flag`, `precursor_snr_max`                                                                  | Integration across $[-80, -20]$d window                         |
| **Benchmarking**           | `M_peak_residual`, `plateau_duration_residual`, `is_anomaly`, `population_deviation_score`                                 | Deviation from batch-wide scaling relations and Isolation Forest |

The 8 tracked parameters are: `zams`, `mloss_rate`, `56Ni`, `k_energy`, `beta`, `texp`, `A_v`, `logZ`.

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

### 4. `scatter_outliers.csv`

Combined bivariate and multivariate outlier flags. One row per outlier detection per object.

| Column              | Description                                                        |
| :------------------ | :----------------------------------------------------------------- |
| `object_id`         | ZTF object ID                                                      |
| `outlier_type`      | Detection category (e.g., `Mloss-Ek`, `Energy Engine`)             |
| `x_param_name`      | X-axis parameter column name                                       |
| `y_param_name`      | Y-axis parameter column name                                       |
| `distance_from_trend` | Sigma-distance (2D) or Mahalanobis distance (3D) from population |
| `direction`         | Above/Below trendline (2D) or chi-squared significance (3D)        |

**Bivariate (2D) combinations:** `Mloss-Ek`, `Ek-Ni`, `Texp-Beta`, `logZ-Av`

**Multivariate (3D Mahalanobis) clusters:** `Energy Engine`, `Progenitor Evolution`, `Modeling Degeneracy`, `Ejecta Efficiency`, `LC Morphology`

### Diagnostic Report (`data/diagnostic_report.pdf`)

The automated PDF report provides a comprehensive anomaly analysis with the following sections:

| Section                          | Contents                                                                                     |
| :------------------------------- | :------------------------------------------------------------------------------------------- |
| **Summary**                      | Parameter scatter grids, 3D Mahalanobis cluster table, bivariate outlier table, corner plots |
| **I. Methodology & Definitions** | Mathematical definitions and thresholds for all anomaly categories                           |
| **II. Energetic Deviations**     | Ledger of objects with luminosity excesses, mass budget violations, or nickel overabundance   |
| **III. Morphological Outliers**  | Ledger of objects with extreme plateau length extensions or truncations                      |
| **IV. Progenitor Environment**   | Ledger of objects showing precursor detection, early rise excess, or arrested cooling         |
| **V. Plateau Topography**        | Ledger of objects with rebrightening bumps or linear residual clusters                       |
| **VI. Coupled Anomalies**        | Objects triggering multiple categories ("Bright & Slow" paradox, environmental correlations) |
| **VII. Flagged Object Profiles** | Deep-dive pages with per-object MCMC fits, light curves, and actionable review instructions  |
| **Appendix A**                   | Filtered non-IIP objects from `flagged_non_iip_objects.csv`                                  |

**Report Features:**
- **Interactive Navigation**: TOC links jump to deep-dive profiles; profiles link back to their summary ledger
- **ALeRCE Integration**: Object IDs in profiles link directly to the ALeRCE explorer
- **Model Plot Links**: Clickable `View Plot` links in outlier tables open the model absolute magnitude plots directly
- **Actionable Diagnostics**: Each profile includes specific "Required Review Actions" tailored to its anomaly flags
- **Processing Timestamps**: Each profile displays the source run folder for temporal context

### Diagnostic Metadata Package (`data/report_images/`)

For each flagged object, the pipeline creates a per-OID subfolder containing:
- Best-fit light curve plot (`.png`)
- Posterior corner plot (`.jpg`)
- Raw REFITT parameter JSON files (`_g_nn.json`, `_r_nn.json`)

Subfolder naming convention: `{OID}_{run_date}/` (e.g., `ZTF25abfntiq_2025-11-10/`)

### Static JSON Payloads (`data/static_payloads/`)

The pipeline generates database-free JSON files designed to serve the UI instantly.
- `summary_index.json`: A single lightweight file caching basic information, inferred parameters, and anomaly metrics for all objects. It includes recent asymmetric percentage uncertainties (`_pct_plus`, `_pct_minus`) and a full `parameter_history` array to track model parameter convergence over REFITT runs.
- `[object_id]_lc.json`: Object-specific payloads containing raw observational arrays and the model_fit parameter predictions. The posterior estimates (`mag_arr`) are parsed automatically to calculate the 16th, 50th, and 84th percentiles for the UI.

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
    │   feature_extractors     │  GP morphology, mass budget, precursors
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
    │   Report Generation      │  (report_generator.py + template)
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────────────────────────┐
    │  data/diagnostic_report.pdf                  │
    │  data/report_images/{OID}_{run_date}/        │
    │    ├── {OID}_lc.png                          │
    │    ├── {OID}_corner.jpg                      │
    │    ├── {OID}_g_nn.json                       │
    │    └── {OID}_r_nn.json                       │
    └──────────┬───────────────────────────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │ export_static_payloads   │  Convert CSVs & JSONs to UI format
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────────────────────────┐
    │  data/static_payloads/                       │
    │    ├── summary_index.json                    │
    │    ├── {OID}_lc.json                         │
    └──────────────────────────────────────────────┘
```

---

## Detailed Methodology

### 1. Parameter Convergence ($N_{90}$)
**$N_{90}$** is defined as the number of days from the first observation until a physical parameter enters and remains within an absolute tolerance band of its final (latest) value. The tolerance is computed as `abs(final_value × 0.10)`, which correctly handles parameters that can be negative (such as `logZ`, where multiplicative percentage bounds produce inverted intervals).
- **Population Baseline**: Since REFITT fits can be volatile in early phases, $N_{90}$ is only considered "Validated" if the final observation phase is $\ge 100$ days (radioactive tail phase).
- **Stability**: A parameter is marked as `converged` only if it maintains this threshold for all subsequent observations in the indexed timeline.

### 2. GP Morphological Features
Using Gaussian Process (GP) regression on raw ZTF lightcurves fetched from local forced photometry CSVs:
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

### 7. Bivariate & Multivariate Outlier Detection
The pipeline runs two complementary outlier detectors after batch analysis:
- **Bivariate (2D)**: For each of 4 physics-motivated parameter pairs (`Mloss-Ek`, `Ek-Ni`, `Texp-Beta`, `logZ-Av`), an OLS trendline is fit and the top-5 sigma-distance deviations are flagged.
- **Multivariate (3D Mahalanobis)**: For each of 5 physics clusters (e.g., Energy Engine = $E_k × \dot{M} × ^{56}Ni$), the Mahalanobis distance from the population centroid is computed using the inverse covariance matrix. Events exceeding the $\chi^2_3$ 99th percentile bound are flagged.

Both detectors write to `data/scatter_outliers.csv`. The report generator deduplicates by OID, merging multi-category detections into single rows with comma-separated type labels.

### 8. Diagnostic Report Generation
The report generator loads `convergence_metrics.csv` and applies category-specific thresholds:
- **Energetic Deviations**: $|M_{\text{obs}} - M_{\text{exp}}| > 0.75$ mag, $E_k/M_{\text{ej}} > 1.0$, $M_{\text{Ni}}/M_{\text{ej}} > 0.01$
- **Morphological Outliers**: $|t_{\text{obs}} - t_{\text{exp}}| > 20$ days
- **Progenitor Environment**: $>3\sigma$ precursor flux, $>0.1$ mag early rise excess, arrested cooling
- **Plateau Topography**: Rebrightening bumps ($dm/dt < 0$ for $>5$ days), linear residual clusters ($\ge 0.1$ mag)

All flagged objects receive a deep-dive profile with actionable review instructions and best-fit plot rendering. Plots and raw JSONs are collected into `data/report_images/{OID}_{run_date}/` subfolders for offline inspection. Outlier tables include clickable `View Plot` links (via `file://` URIs) that open the model absolute magnitude plots directly from the PDF.

---

## Module Reference

### `main.py`

CLI entrypoint. Runs the complete 4-step pipeline: index → analyze → visualize → report. All steps are orchestrated here with error handling and timing.

### `src/report_generator.py`
Loads the LaTeX template from `src/templates/report_template.tex`, populates it with data from `convergence_metrics.csv`, `scatter_outliers.csv`, and `flagged_non_iip_objects.csv`, copies diagnostic plots and JSONs into `data/report_images/`, and compiles the final `data/diagnostic_report.pdf` via `pdflatex`. Includes `ensure_model_plot()` which copies model absolute magnitude plots for all outlier OIDs and generates clickable `file://` links in the PDF tables.

### `src/templates/report_template.tex`
The standalone LaTeX template for the diagnostic PDF. Uses Roman numeral sectioning, `longtable` for multi-page ledgers, and `float` package `[H]` placement for anchored figures. Includes physics-motivated explanations for each 3D Mahalanobis cluster and 2D bivariate combination. Editable independently of the Python code.

### `src/feature_extractors.py`
The "Physics Engine" of the pipeline. Implements all advanced feature extraction classes including `GPMorphologicalExtractor` (absolute magnitudes, color slopes), `PrecursorScan` (integrated pre-explosion flux), `PriorsVolatilityCheck` (prior deviation scores for all 8 parameters including `logZ`), and the `Aggregator` (population-level Benchmarking and Isolation Forest anomaly detection).

### `src/multivariate_outliers.py`
Dual-mode outlier detector:
- **`BivariateOutlierDetector`**: Fits OLS trendlines across 4 physics-motivated parameter pairs and flags the top-5 sigma-distance deviations per combination.
- **`MultivariateOutlierDetector`**: Computes Mahalanobis distances in 5 three-dimensional physics clusters using the inverse covariance matrix, flagging events exceeding the χ²₃ 99th percentile.

### `src/ztf_client.py`
A client for parsing local ZTF forced photometry CSVs. It gracefully extracts MJD, flux, magnitude and error arrays from the pipeline's localized `data/ztf_forced_photometry/` directory, acting as a standalone replacement for the deprecated ALeRCE API caching component.

### `src/export_static_payloads.py`
Converts output arrays, classification reports, and inference variables into an index-detail JSON format for instantaneous ingestion by modern web applications. The script handles array mappings and extracts relevant model percentiles directly from multidimensional posterior samples.

### `src/compare_successive_observations.py`
Core convergence analyzer. For each object, it:
1. Loads the chronological timeline of JSON observations
2. Extracts the 8 physical parameters (including `logZ`) at each epoch
3. Computes N₉₀ (days to convergence) using absolute tolerance bounds (correctly handles negative parameters)
4. Computes volatility metrics (std, mean absolute change, max jump)
5. Generates multi-panel trajectory plots showing parameter stabilization

### `src/create_summary_plots.py`
Batch-level visualization engine. Generates convergence distributions, volatility boxplots, parameter correlations, relative uncertainties, and a targeted 2×2 parameter scatter grid matching the 4 bivariate outlier combinations. Also produces seaborn `PairGrid` corner pairplots (with KDE densities and scatter overlays) for each 3D Mahalanobis physics cluster.

### `src/batch_analyze_objects.py`
The main orchestrator. For each object: loads data via `JSONFetcher`, runs TNS filtering, checks light curve completeness, executes the `feature_extractors`, and computes run frequency. It writes the primary CSV outputs, runs both `BivariateOutlierDetector` and `MultivariateOutlierDetector`, and coordinates the batch-level `Aggregator` logic.

---

## Notes

- The pipeline is **idempotent** — re-running overwrites previous outputs
- TNS cache (`.tns_cache.json`) persists across runs — delete it to force re-query
- Add new date directories with REFITT JSONs and re-run to incorporate new observations
- PDF compilation requires `pdflatex`; the report is compiled twice to resolve cross-references and `longtable` widths
- All modules are importable: `from src.module import ClassName`
