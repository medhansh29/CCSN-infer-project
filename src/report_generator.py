import os
import glob
import pandas as pd
from datetime import datetime

REPORT_TEMPLATE = """# Supernova Inference Diagnostics Report

**Run Date:** {run_date} | **Total Objects Processed:** {total_objects}

---

#### **1. Executive Summary**

* **Total Converged:** {pct_converged:.1f}% ({converged_count}/{total_objects})
* **Red Alerts (High Confidence, Bad Physics/Fit):** {red_alert_count}
* **ML-Flagged Anomalies:** {ml_anomaly_count}
* **Relative Anomalies (Extremes):** {relative_anomaly_count}
* **Trendline Outliers:** {trendline_outlier_count}

**Pipeline Health Diagnostics**
*(Auto-generated from the most recent run)*

> **Overall Health:** Convergence rates, phase span correlations, and prediction accuracy. 
![Overall Summary](../data/summary_plots/overall_summary.png)

> **Relative Uncertainties:** Boxplots showing the constrainability of parameters. Parameters >100% are practically unconstrained.
![Relative Uncertainties](../data/summary_plots/relative_uncertainties.png)

> **Convergence Speed (N90):** Histograms reflecting the average number of days data required before posterior width stopped shrinking.
![N90 Timings](../data/summary_plots/n90_distributions.png)

> **Parameter Relationships:** Final parameter medians plotted against physical priors, with out-of-distribution instances marked via scatter regression.
![Parameter Scatter Grid](../data/summary_plots/parameter_scatter_grid.png)

---

#### **2. 🚨 RED ALERTS: Confident but Suspect / Physical Violations**

*Context: Objects where the model reports low uncertainty, but historical volatility is high, evidence is low, or physical constraints are violated.*

{red_alerts_content}

---

#### **3. 👽 Multi-Dimensional Anomalies (Isolation Forest)**

*Context: Objects flagged by machine learning as anomalies across the entire 7-parameter space.*

{ml_anomalies_content}

---

#### **4. 📊 Relative Anomalies (Top 5% Skew / Bottom 5% Evidence)**

*Context: Objects occupying the extreme tails of the current run's distribution.*

{relative_anomalies_content}

---

#### **5. 📈 Trendline Outliers (2D Scatter)**

*Context: Objects deviating significantly from established scaling relations (e.g., ZAMS vs. 56Ni).*

{trendline_outliers_content}

---

#### **6. ⚠️ TNS Misclassifications (Non-IIP Objects)**

*Context: Objects processed by the pipeline but later flagged by the internal TNS query as a different spectroscopic class (e.g., Type Ia, IIn, Ibn). These inferences are likely physically meaningless.*

{non_iip_content}

---

#### **7. 📖 Definitions & Methodology**

**Anomaly & Alert Criteria:**
* **Red Alerts (Severity 3):** Triggered when the model is confident (relative uncertainty < 5%) but the parameters are physically suspect.
  * *Bad Physics:* Defined by hard limits (e.g. Explosion energy to Ejecta mass ratio $E_k/M_{{ej}} > 2.0$, or extremely low nickel $M_{{Ni}} < 0.005$ with high $E_k > 2.0$).
  * *High Volatility / Low Evidence:* The parameter trajectory jumped by $>2\\sigma$ in recent runs, or the overall $\\log Z$ is less than -40 despite high confidence.
* **ML Flags (Severity 2):** Detected using an Isolation Forest (`contamination=0.1`) trained on the final 7 inferred parameters. It spots multi-dimensional outliers that might look normal in 1D/2D projections. The driving parameter is identified via highest absolute Z-score.
* **Relative Anomalies (Severity 1):** Objects sitting in the extreme statistical tails: the bottom 5% of Evidence ($\\log Z$), or the top 5% of posterior asymmetry (measured via median-mode skewness). 
* **Trendline Outliers:** Based on linear regression across specific physical pairs (e.g., ZAMS vs. Ni56). Objects with residuals greater than $1.5\\sigma$ from the best-fit trendline are flagged for investigation.

**Metrics & Terminology:**
* **Phase & Completeness:**
  * *Preliminary:* Lightcurve spans $<30$ days.
  * *Transitional:* $30 \\le \\text{{days}} < 60$.
  * *Validated (Plateau / Tail):* $>60$ days.
* **Convergence & "Final Value":**
  * The "Final Value" reported is the median of the posterior from the *most recent* inference run for that object.
  * An object is globally "Converged" (Fit Success = True) if the pipeline determines the inference is stable and sufficiently constrained.
* **$N_{{90}}$ Lead Time:** The number of days of observations required for a specific parameter's uncertainty to shrink to within 10% of its final, fully-resolved uncertainty. A lower $N_{{90}}$ means that parameter is constrained earlier in the lightcurve evolution.

---
"""

def find_latest_plots(object_id, base_dir='.'):
    """Finds the most recent lightcurve and corner plots for a given object_id."""
    # Look for date-like directories YYYY-MM-DD
    date_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('202')])
    
    # Search backwards from the most recent date
    for d in reversed(date_dirs):
        dir_path = os.path.join(base_dir, d)
        corner_plot = os.path.join(dir_path, f"{object_id}_corner_plot.jpg")
        lc_plot = os.path.join(dir_path, f"{object_id}_model_apparent_nn.png")
        
        if os.path.exists(corner_plot) or os.path.exists(lc_plot):
            # Return relative paths to the report location (data/diagnostic_report.md)
            return {
                'corner': f"../{d}/{object_id}_corner_plot.jpg" if os.path.exists(corner_plot) else None,
                'lightcurve': f"../{d}/{object_id}_model_apparent_nn.png" if os.path.exists(lc_plot) else None
            }
    return {'corner': None, 'lightcurve': None}

def format_alert_section(df, alert_type):
    """Formats a section of the report for a specific alert type."""
    if df.empty:
        return "*No objects flagged in this category.*"
        
    content = ""
    for _, row in df.iterrows():
        obj_id = row['object_id']
        reason = row['alert_reason']
        ek = row.get('k_energy_final', 'N/A')
        ni = row.get('56Ni_final', 'N/A')
        log_e = row.get('log_evidence', 'N/A')
        comp = row.get('completeness_status', 'Unknown')
        
        if isinstance(ek, float) and not pd.isna(ek): ek = f"{ek:.2f}"
        if isinstance(ni, float) and not pd.isna(ni): ni = f"{ni:.4f}"
        if isinstance(log_e, float) and not pd.isna(log_e): log_e = f"{log_e:.2f}"
        
        content += f"**Object ID:** `{obj_id}` ({comp})\n\n"
        content += f"**Flag Reason:** *{reason}*\n\n"
        content += f"**Key Stats:** $E_k$: {ek}, $^{{56}}\\text{{Ni}}$: {ni}, Log Evidence: {log_e}\n\n"
        
        plots = find_latest_plots(obj_id)
        if plots['lightcurve'] or plots['corner']:
            content += "**Diagnostic Plots:**\n"
            content += "<div>\n"
            if plots['lightcurve']:
                content += f"  <img src='{plots['lightcurve']}' width='45%' style='display:inline-block;'/>\n"
            if plots['corner']:
                content += f"  <img src='{plots['corner']}' width='45%' style='display:inline-block;'/>\n"
            content += "</div>\n\n"
        
        content += "---\n\n"
        
    return content

def format_trendline_section(df):
    """Formats the trendline outliers section."""
    if df.empty:
        return "*No objects flagged in this category.*"
        
    content = ""
    for _, row in df.iterrows():
        obj_id = row['object_id']
        content += f"**Object ID:** `{obj_id}`\n\n"
        content += f"**Relation:** {row.get('x_param_name', 'Unknown')} vs {row.get('y_param_name', 'Unknown')}\n\n"
        content += f"**Deviation:** {row.get('direction', 'Unknown')} by {row.get('residual', 0):.1f} residual\n\n"
        
        plots = find_latest_plots(obj_id)
        if plots['lightcurve'] or plots['corner']:
            content += "**Diagnostic Plots:**\n"
            content += "<div>\n"
            if plots['lightcurve']:
                content += f"  <img src='{plots['lightcurve']}' width='45%' style='display:inline-block;'/>\n"
            if plots['corner']:
                content += f"  <img src='{plots['corner']}' width='45%' style='display:inline-block;'/>\n"
            content += "</div>\n\n"
            
        content += "---\n\n"
        
    return content

def format_non_iip_section(df):
    """Formats objects classified by TNS as something other than II/IIP."""
    if df is None or df.empty:
        return "*No non-IIP objects found in TNS.*"
    
    content = "| Object ID | TNS Name | Actual TNS Classification |\n"
    content += "| :--- | :--- | :--- |\n"
    
    for _, row in df.iterrows():
        obj_id = row['object_id']
        tns_type = row.get('tns_type', 'Unknown')
        tns_name = row.get('tns_name', 'Unknown')
        
        content += f"| `{obj_id}` | {tns_name} | {tns_type} |\n"
        
    content += "\n*(Note: Pipeline automatically excludes these from overarching statistics as their fits assume Type IIP physics.)*\n\n"
    
    return content

def main():
    print("Generating Diagnostic Report...")
    
    # Load all the data
    try:
        conv_df = pd.read_csv('data/convergence_metrics.csv')
    except Exception:
        conv_df = pd.DataFrame()
        
    try:
        alerts_df = pd.read_csv('data/red_alerts.csv')
    except Exception:
        alerts_df = pd.DataFrame()
        
    try:
        scatter_df = pd.read_csv('data/scatter_outliers.csv')
    except Exception:
        scatter_df = pd.DataFrame()
        
    try:
        non_iip_df = pd.read_csv('data/flagged_non_iip_objects.csv')
    except Exception:
        non_iip_df = pd.DataFrame()

    # Basic stats
    total_objs = len(conv_df) if not conv_df.empty else 0
    
    converged_count = 0
    pct_converged = 0.0
    if not conv_df.empty:
        if 'fit_success' in conv_df.columns:
            # We treat fit_success=True as overall pipeline 'convergence' 
            # (or you could map this to confidence grades A/B)
            # The column is boolean or string representation of bool
            converged_mask = conv_df['fit_success'] == True
            # Also catch literal string "True" if it was written directly
            converged_mask = converged_mask | (conv_df['fit_success'] == 'True') 
            
            converged_count = converged_mask.sum()
            pct_converged = (converged_count / total_objs * 100) if total_objs > 0 else 0

    # Categorize alerts
    red_alerts = pd.DataFrame()
    ml_anomalies = pd.DataFrame()
    rel_anomalies = pd.DataFrame()
    
    if not alerts_df.empty:
        red_alerts = alerts_df[alerts_df['severity_score'] == 3]
        ml_anomalies = alerts_df[alerts_df['severity_score'] == 2]
        rel_anomalies = alerts_df[alerts_df['severity_score'] == 1]
    
    # Format sections
    red_alerts_content = format_alert_section(red_alerts, "Physics")
    ml_anomalies_content = format_alert_section(ml_anomalies, "ML")
    rel_anomalies_content = format_alert_section(rel_anomalies, "Relative")
    trendline_content = format_trendline_section(scatter_df)
    non_iip_content = format_non_iip_section(non_iip_df)
    
    # Fill template
    report = REPORT_TEMPLATE.format(
        run_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_objects=total_objs,
        converged_count=converged_count,
        pct_converged=pct_converged,
        red_alert_count=len(red_alerts),
        ml_anomaly_count=len(ml_anomalies),
        relative_anomaly_count=len(rel_anomalies),
        trendline_outlier_count=len(scatter_df),
        red_alerts_content=red_alerts_content,
        ml_anomalies_content=ml_anomalies_content,
        relative_anomalies_content=rel_anomalies_content,
        trendline_outliers_content=trendline_content,
        non_iip_content=non_iip_content
    )
    
    # Save report
    os.makedirs('data', exist_ok=True)
    md_path = 'data/diagnostic_report.md'
    pdf_path = 'data/diagnostic_report.pdf'
    
    with open(md_path, 'w') as f:
        f.write(report)
        
    print(f"✅ Generated Markdown report: {md_path}")
    print("Converting to PDF...")
    
    import subprocess
    try:
        # Use npx to implicitly download and run md-to-pdf in one go
        res = subprocess.run(f'npx --yes md-to-pdf {md_path}', shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"✅ Successfully exported diagnostic report to: {pdf_path}")
            # Clean up the intermediate md ONLY if PDF conversion succeeded
            os.remove(md_path)
        else:
            print(f"⚠️ PDF conversion skipped (Node.js/npx not found on system).")
            print(f"✅ Retaining Markdown report at: {md_path}")
    except Exception as e:
        print(f"⚠️ PDF dependencies missing. Keeping Markdown file at: {md_path}")
        raise e

if __name__ == '__main__':
    main()
