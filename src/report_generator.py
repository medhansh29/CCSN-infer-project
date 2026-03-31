import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

REPORT_TEMPLATE = """<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js" onload="renderMathInElement(document.body, {{delimiters: [{{left: '$', right: '$', display: false}}, {{left: '$$', right: '$$', display: true}}]}});"></script>
<style>
    body {{ font-family: 'Inter', -apple-system, sans-serif; line-height: 1.6; color: #333; padding: 40px; }}
    h1, h2 {{ color: #1a202c; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.5rem; }}
    .reason-box {{ background: #f7fafc; border-left: 4px solid #4a5568; padding: 1rem; margin-bottom: 1rem; border-radius: 0 4px 4px 0; }}
    .leaderboard-item {{ margin-bottom: 0.5rem; }}
    .math-inline {{ font-family: 'serif'; }}
</style>

# Supernova Inference Diagnostics & Physics Report

**Run Date:** {run_date} | **Total Clean IIP Objects Analyzed:** {total_objects}

---

## Section 1: Overall Summary

**Outlier Leaderboard (Batch Extremes)**
<div class="leaderboard-item">* **Most Luminous IIP:** [{most_lum}](https://alerce.online/object/{most_lum}) (<span>$M_{{plateau,25d}}$</span> = {most_lum_val:.2f})</div>
<div class="leaderboard-item">* **Highest Nickel Mass:** [{highest_ni}](https://alerce.online/object/{highest_ni}) (<span>$M_{{Ni}}$</span> = {highest_ni_val:.3f} <span>$M_\\odot$</span>)</div>
<div class="leaderboard_item">* **Most Volatile Posterior:** [{most_vol}](https://alerce.online/object/{most_vol}) (Max Variance across runs)</div>

<img src='../data/summary_plots/overall_summary.png' width='100%'/>

---

## Section 2: Brightness Outliers

*Too bright or too dim for their given <span>$ZAMS / E_k$</span> scaling ratio (expected median).*
{s2_content}

---

## Section 3: Plateau Length Outliers

*Abnormally long or short plateaus for their inferred <span>$M_{{ej}}^{{3/4}} / E_k^{{1/4}}$</span> scaling ratio.*
{s3_content}

---

## Section 4: Precursor Activity Detected

*Objects showing significant pre-explosion flux excess.*
{s4_content}

---

## Section 5: Other Outliers & Peculiar Phenomena

*Objects exhibiting specific physical or statistical anomalies: Bimodal Posteriors, Positive Tail Slopes, or Mass Budget Violations. (See Appendix for definitions).*
{s5_content}

---

## Section 6: Appendix — Outlier Methodology

**1. Brightness Outliers:** We establish a batch-wide baseline by linearly regressing the absolute plateau magnitude (<span>$M_{{plateau,25d}}$</span>) against the <span>$ZAMS$</span> and Kinetic Energy (<span>$E_k$</span>) inferred values. An object is flagged here if its absolute magnitude deviates by > 0.75 magnitudes from the scaling prediction.

**2. Plateau Duration Outliers:** Kasen & Woosley (2009) establish that plateau length scales as <span>$t_p \\propto M_{{ej}}^{{3/4}} / E_k^{{1/4}}$</span>. We use the inferred <span>$(ZAMS - 1.5)^{{3/4}} / E_k^{{1/4}}$</span> as a proxy feature, linearly regress the observed plateau durations against it, and flag deviations of > 20 days.

**3. Precursor Activity:** An integration of flux <span>$-80$</span> to <span>$-20$</span> days prior to peak. Detections > 5.0 <span>$\\sigma$</span> cumulative SNR are flagged.

**4. Bimodal Posteriors:** Detected via Gaussian Kernel Density Estimation (KDE) on the raw MCMC samples. If multiple distinct density peaks are found for any physical parameter, it indicates model degeneracy where two separate physical solutions (e.g., High Mass/Low Energy vs. Low Mass/High Energy) both fit the data.

**5. Positive Tail Slopes:** We perform a linear regression on data at Phase > 120 days. Standard Type IIP supernovae should show a steady decline (negative slope) as they follow the <span>$^{{56}}Co$</span> decay. A positive slope (> 0.01) indicates late-time re-brightening, often a signature of Circumstellar Material (CSM) interaction.

**6. Mass Budget Violations:** A physical sanity check on the ratio of Kinetic Energy to Ejecta Mass (<span>$E_k / M_{{ej}} > 1.0$</span>). Significant violations suggest the fit has settled into an unphysical regime to compensate for high luminosity or unusual lightcurve morphology.

---

### Appendix Section C: Multi-Category Outliers

*The following objects were flagged in more than one analytical category, marking them as the highest priority for further investigation.*

{s6_c_content}
"""

def find_latest_json(object_id, base_dir='.'):
    dates = sorted([d for d in os.listdir(base_dir) if os.path.isdir(d) and d.startswith('202')])
    for d in reversed(dates):
        jsons = glob.glob(os.path.join(d, f"{object_id}_*nn.json"))
        if jsons:
            return jsons[0]
    return None

def find_abs_mag_plot(object_id, base_dir='.'):
    dates = sorted([d for d in os.listdir(base_dir) if os.path.isdir(d) and d.startswith('202')])
    for d in reversed(dates):
        plot = os.path.join(d, f"{object_id}_model_absolute_nn.png")
        if os.path.exists(plot):
            return f"../{plot}"
    return None

def find_corner_plot(object_id, base_dir='.'):
    dates = sorted([d for d in os.listdir(base_dir) if os.path.isdir(d) and d.startswith('202')])
    for d in reversed(dates):
        plot = os.path.join(d, f"{object_id}_corner_plot.jpg")
        if os.path.exists(plot):
            return f"../{plot}"
    return None

def format_object_block(row, df, custom_reason=""):
    obj = row['object_id']
    alerce_link = f"[{obj}](https://alerce.online/object/{obj})"
    
    ek = row.get('k_energy_final', 1.0)
    mej = max(row.get('implied_Mej', 10.0), 1.0)
    zams = row.get('zams_final', 15.0)
    
    zams_ek = zams / ek
    duration_ratio = (mej**3 / ek)**0.25
    ni = row.get('56Ni_final', 0.05) if not pd.isna(row.get('56Ni_final')) else 0.05
    ni_mej = ni / mej
    
    content = f"### {alerce_link}\n\n"
    if custom_reason:
        content += f"<div class='reason-box'>{custom_reason}</div>\n\n"
        
    content += f"<div><strong>Ratios:</strong> $ZAMS/E_k$ = {zams_ek:.2f} | $M_{{ej}}^{{3/4}}/E_k^{{1/4}}$ = {duration_ratio:.2f} | $M_{{Ni}}/M_{{ej}}$ = {ni_mej:.3f}</div>\n\n"
    
    phenom = []
    phenom.append("[✓] Prior Pierced" if not pd.isna(row.get('prior_pegged')) and str(row.get('prior_pegged')).strip() != '' else "[✗] Prior Pierced")
    phenom.append("[✓] Bimodal" if str(row.get('is_bimodal', False)) in ['True', 'true'] else "[✗] Bimodal")
    phenom.append("[✓] Precursor" if str(row.get('precursor_flag', False)) in ['True', 'true'] else "[✗] Precursor")
    phenom.append("[✓] Budg. Viol." if str(row.get('mass_budget_violation', False)) in ['True', 'true'] else "[✗] Budg. Viol.")
    
    content += f"**Checklist:** {' | '.join(phenom)}\n\n"
    
    abs_path = find_abs_mag_plot(obj)
    corner_path = find_corner_plot(obj)
    
    content += "<div>\n"
    if abs_path: content += f"  <img src='{abs_path}' width='48%' style='display:inline-block; vertical-align:top;'/>\n"
    if corner_path: content += f"  <img src='{corner_path}' width='48%' style='display:inline-block; vertical-align:top;'/>\n"
    content += "</div>\n\n"
    return content

def main():
    try:
        df = pd.read_csv('data/convergence_metrics.csv')
    except:
        print("No metrics CSV found!")
        return
        
    df = df[df['fit_success'].isin([True, 'True'])]
    # Filter by observation count
    THRESHOLD = 12
    reliable_df = df[df['num_observations'] >= THRESHOLD].copy()
    sparse_df = df[df['num_observations'] < THRESHOLD].copy()
    
    if reliable_df.empty:
        print("No reliable objects to report!")
        return
        
    def get_col(curr_df, col, default):
        if col in curr_df.columns: return curr_df[col]
        return pd.Series(default, index=curr_df.index)
        
    # Leaderboard (Reliable only)
    plateau_col = 'M_plateau_25d'
    if plateau_col in reliable_df.columns:
        valid_plateau = reliable_df[reliable_df[plateau_col].notna()]
        if not valid_plateau.empty:
            most_lum_idx = valid_plateau[plateau_col].idxmin()
            most_lum = reliable_df.loc[most_lum_idx, 'object_id']
            most_lum_val = reliable_df.loc[most_lum_idx, plateau_col]
        else:
            most_lum, most_lum_val = "N/A", 0.0
    else:
        most_lum, most_lum_val = "N/A", 0.0
    
    if '56Ni_final' in reliable_df.columns:
        valid_ni = reliable_df[reliable_df['56Ni_final'].notna()]
        if not valid_ni.empty:
            highest_ni_idx = valid_ni['56Ni_final'].idxmax()
            highest_ni = reliable_df.loc[highest_ni_idx, 'object_id']
            highest_ni_val = reliable_df.loc[highest_ni_idx, '56Ni_final']
        else:
            highest_ni, highest_ni_val = "N/A", 0.0
    else:
        highest_ni, highest_ni_val = "N/A", 0.0
    
    vol_cols = [c for c in reliable_df.columns if 'volatility_score' in c]
    if vol_cols:
        reliable_df['max_vol'] = reliable_df[vol_cols].max(axis=1)
        most_vol_idx = reliable_df['max_vol'].idxmax()
        most_vol = reliable_df.loc[most_vol_idx, 'object_id'] if not pd.isna(most_vol_idx) else "N/A"
    else:
        most_vol = "N/A"

    # Tracking flags for Multi-Category Appendix
    flag_counts = {oid: set() for oid in reliable_df['object_id']}
        
    # Section 2: Brightness (Reliable only)
    m_resid = get_col(reliable_df, 'M_peak_residual', np.nan)
    s2_mask = m_resid.abs() > 0.75
    s2_content_list = []
    for _, row in reliable_df[s2_mask].iterrows():
        diff = row['M_peak_residual']
        obs = row.get('M_plateau_25d', np.nan)
        pred = row.get('M_peak_predicted', np.nan)
        zams_v = row.get('zams_final', 15)
        ek_v = row.get('k_energy_final', 1)
        ratio_str = f"<span>$ZAMS/E_k$</span> ({zams_v/ek_v:.2f})" if not pd.isna(zams_v) and not pd.isna(ek_v) else "N/A"
        
        direction = "Exceptionally bright" if diff < 0 else "Exceptionally dim"
        reason = f"**{direction}.** Observed: {obs:.2f}. Expected: {pred:.2f}. Outlier by {abs(diff):.2f} mag for this {ratio_str}."
            
        if str(row.get('distance_uncertain', '')) in ['True', 'true']:
            reason += " ⚠️ Distance is uncertain (z < 0.015)."
            
        s2_content_list.append(format_object_block(row, reliable_df, reason))
        flag_counts[row['object_id']].add("Brightness")

    s2_content = "\n".join(s2_content_list) if s2_content_list else "*No Brightness Outliers.*"
    
    # Section 3: Plateau Length Outliers (Reliable only)
    dur_resid = get_col(reliable_df, 'plateau_duration_residual', np.nan)
    s3_mask = dur_resid.abs() > 20.0
    s3_content_list = []
    for _, row in reliable_df[s3_mask].iterrows():
        diff = row['plateau_duration_residual']
        obs = row.get('plateau_duration_days', np.nan)
        pred = row.get('plateau_duration_predicted', np.nan)
        
        mej = max(row.get('implied_Mej', 10), 1.0)
        ek = row.get('k_energy_final', 1.0)
        ratio_str = f"<span>$M_{{ej}}^{{3/4}}/E_k^{{1/4}}$</span> ({(mej**3/ek)**0.25:.2f})"
            
        direction = "Abnormally long plateau" if diff > 0 else "Abnormally short plateau"
        reason = f"**{direction}.** Observed: {obs:.1f} days. Expected: {pred:.1f} days. Outlier by {abs(diff):.1f} days for this {ratio_str}."
        
        s3_content_list.append(format_object_block(row, reliable_df, reason))
        flag_counts[row['object_id']].add("Plateau Duration")

    s3_content = "\n".join(s3_content_list) if s3_content_list else "*No Plateau Length Outliers.*"

    # Section 4: Precursors (Reliable only)
    s4_mask = get_col(reliable_df, 'precursor_snr_max', 0) > 5.0
    s4_content_list = []
    for _, row in reliable_df[s4_mask].iterrows():
        snr_max = row.get('precursor_snr_max', 0)
        reason = f"**Pre-Explosion Precursor.** Significant Activity Detected (Cumulative SNR={snr_max:.1f})."
        s4_content_list.append(format_object_block(row, reliable_df, reason))
        flag_counts[row['object_id']].add("Precursor")
    s4_content = "\n".join(s4_content_list) if s4_content_list else "*No Significant Precursor Activity Detected.*"

    # Section 5: Other Phenomena (Reliable only)
    s5_content_list = []
    for _, row in reliable_df.iterrows():
        reasons = []
        if str(row.get('is_bimodal', False)) in ['True', 'true']: 
            reasons.append("Bimodal Posterior")
        if str(row.get('mass_budget_violation', False)) in ['True', 'true']:
            reasons.append("Mass Budget Violation")
        if row.get('tail_slope', 0) > 0.02: 
            reasons.append("Positive Late-Time Tail Slope (Rebrightening/CSM)")
        if row.get('lag1_autocorr', 0) > 0.9:
            reasons.append("Highly Autocorrelated Residuals (LC Bumps)")
            
        if reasons:
            s5_reason = "**Other Anomalies:** " + " | ".join(reasons)
            s5_content_list.append(format_object_block(row, reliable_df, s5_reason))
            for r in reasons: flag_counts[row['object_id']].add(r)

    s5_content = "\n".join(s5_content_list) if s5_content_list else "*No Other Peculiarities.*"

    # Section 6 Appendix C: Multi-Category
    multi_list = []
    for oid, categories in flag_counts.items():
        if len(categories) >= 2:
            multi_list.append(f"| {oid} | {len(categories)} | {', '.join(sorted(categories))} | [ALeRCE](https://alerce.online/object/{oid}) |")
    
    if multi_list:
        s6_c_content = "| Object ID | Flags Count | Categories | Alerce Link |\n"
        s6_c_content += "| :--- | :--- | :--- | :--- |\n"
        s6_c_content += "\n".join(multi_list)
    else:
        s6_c_content = "*No objects flagged in multiple categories.*"

    # Sparse Table
    sparse_table = "\n### Appendix Table: Sparse Data Objects (N < 12)\n\n"
    sparse_table += "| Object ID | Total Obs | Phase Span | Alerce Link |\n"
    sparse_table += "| :--- | :--- | :--- | :--- |\n"
    for _, row in sparse_df.sort_values('num_observations', ascending=False).iterrows():
        oid = row['object_id']
        n_obs = row['num_observations']
        span = f"{row['phase_span']:.1f}d" if not pd.isna(row['phase_span']) else "N/A"
        link = f"[ALeRCE](https://alerce.online/object/{oid})"
        sparse_table += f"| {oid} | {n_obs} | {span} | {link} |\n"

    # Save
    report = REPORT_TEMPLATE.format(
        run_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_objects=len(reliable_df),
        most_lum=most_lum, most_lum_val=most_lum_val,
        highest_ni=highest_ni, highest_ni_val=highest_ni_val,
        most_vol=most_vol,
        s2_content=s2_content,
        s3_content=s3_content,
        s4_content=s4_content,
        s5_content=s5_content,
        s6_c_content=s6_c_content
    )
    report += sparse_table
    
    os.makedirs('data', exist_ok=True)
    with open('data/diagnostic_report.md', 'w') as f:
        f.write(report)
        
    print("✅ Generated Markdown report: data/diagnostic_report.md")
    import subprocess
    try:
        cmd = 'npx --yes md-to-pdf data/diagnostic_report.md --pdf-options \'{"waitUntil": "networkidle0"}\''
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            print("✅ Successfully exported to PDF!")
            os.remove('data/diagnostic_report.md')
        else:
            print(f"⚠️ PDF conversion failed: {res.stderr}")
    except Exception as e:
        print(f"⚠️ PDF conversion error: {e}")

if __name__ == "__main__":
    main()
